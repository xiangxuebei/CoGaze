from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _load_gazeclr_pretrained_weights(gazeclr_path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(gazeclr_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(state)}")

    for k in [
        "projector_inv.0.weight",
        "projector_inv.2.weight",
        "projector_equiv.0.weight",
        "projector_equiv.2.weight",
    ]:
        state.pop(k, None)

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        k2 = k
        for prefix in ("backbone.", "module.backbone.", "module."):
            if k2.startswith(prefix):
                k2 = k2[len(prefix):]
        cleaned[k2] = v
    return cleaned


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNet(nn.Module):
    def __init__(self, block: type[BasicBlock], layers: Tuple[int, int, int, int]):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block: type[BasicBlock], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride), nn.BatchNorm2d(planes))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return {"s1": s1, "s2": s2, "s3": s3, "s4": s4}


def resnet18() -> ResNet:
    return ResNet(BasicBlock, (2, 2, 2, 2))


class ResNet18MultiScale(nn.Module):
    def __init__(self, gazeclr_weights_path: Optional[str] = None):
        super().__init__()
        self.net = resnet18()
        if gazeclr_weights_path:
            self.net.load_state_dict(_load_gazeclr_pretrained_weights(gazeclr_weights_path), strict=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.net.forward_features(x)
