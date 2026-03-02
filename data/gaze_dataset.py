from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class NPZSequenceDataset(Dataset):
    def __init__(self, manifest_jsonl: str | Path, max_seq_len: Optional[int] = None):
        self.items: List[dict] = []
        with Path(manifest_jsonl).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.items)

    def _clip_or_pad(self, arr: np.ndarray, t: int) -> np.ndarray:
        T = arr.shape[0]
        if T >= t:
            return arr[:t]
        pad = np.zeros((t - T,) + arr.shape[1:], dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)

    @staticmethod
    def _safe_float(v) -> float:
        try:
            return float(v)
        except Exception:
            return float("nan")

    def _infer_px_per_cm(self, item: dict, data: np.lib.npyio.NpzFile) -> float:
        if "px_per_cm" in item:
            return self._safe_float(item["px_per_cm"])
        if "px_per_cm" in data:
            arr = np.asarray(data["px_per_cm"]).reshape(-1)
            if arr.size:
                return self._safe_float(arr[0])

        w_px = self._safe_float(item.get("screen_width_px"))
        w_cm = self._safe_float(item.get("screen_width_cm"))
        h_px = self._safe_float(item.get("screen_height_px"))
        h_cm = self._safe_float(item.get("screen_height_cm"))
        if np.isfinite(w_px) and np.isfinite(w_cm) and w_cm > 0:
            return w_px / w_cm
        if np.isfinite(h_px) and np.isfinite(h_cm) and h_cm > 0:
            return h_px / h_cm

        info_json = item.get("info_json")
        if info_json and Path(str(info_json)).exists():
            try:
                obj = json.loads(Path(str(info_json)).read_text(encoding="utf-8"))
                dev = (obj or {}).get("device_info", {}) or {}
                cm = dev.get("screen_size_cm", {}) or {}
                px = dev.get("screen_resolution", {}) or {}
                w_px = self._safe_float(px.get("width"))
                w_cm = self._safe_float(cm.get("width"))
                h_px = self._safe_float(px.get("height"))
                h_cm = self._safe_float(cm.get("height"))
                if np.isfinite(w_px) and np.isfinite(w_cm) and w_cm > 0:
                    return w_px / w_cm
                if np.isfinite(h_px) and np.isfinite(h_cm) and h_cm > 0:
                    return h_px / h_cm
            except Exception:
                pass

        return float("nan")

    def _infer_device_class(self, item: dict) -> str:
        if item.get("device_class"):
            return str(item["device_class"])
        info_json = item.get("info_json")
        if info_json and Path(str(info_json)).exists():
            try:
                obj = json.loads(Path(str(info_json)).read_text(encoding="utf-8"))
                dev = (obj or {}).get("device_info", {}) or {}
                cm = dev.get("screen_size_cm", {}) or {}
                w = self._safe_float(cm.get("width"))
                h = self._safe_float(cm.get("height"))
                if np.isfinite(w) and np.isfinite(h):
                    diag = float(np.hypot(w, h))
                    return "phone" if diag < 20 else "tablet"
            except Exception:
                pass
        return "unknown"

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        npz_path = Path(item["npz"])
        data = np.load(npz_path, allow_pickle=True)

        face = data["face_rgb"].astype(np.float32)
        left = data["left_eye_rgb"].astype(np.float32)
        right = data["right_eye_rgb"].astype(np.float32)

        T = face.shape[0]
        gt = data["gt_pog_cm"].astype(np.float32) if "gt_pog_cm" in data else np.zeros((T, 2), dtype=np.float32)
        valid = data["valid_mask"].astype(np.float32) if "valid_mask" in data else np.ones((T,), dtype=np.float32)
        segment = data["segment_id"].astype(np.int64) if "segment_id" in data else np.zeros((T,), dtype=np.int64)
        time_ms = data["time_ms"].astype(np.float32) if "time_ms" in data else np.arange(T, dtype=np.float32) * 100.0
        px_per_cm = np.float32(self._infer_px_per_cm(item, data))

        if self.max_seq_len is not None:
            face = self._clip_or_pad(face, self.max_seq_len)
            left = self._clip_or_pad(left, self.max_seq_len)
            right = self._clip_or_pad(right, self.max_seq_len)
            gt = self._clip_or_pad(gt, self.max_seq_len)
            valid = self._clip_or_pad(valid, self.max_seq_len)
            segment = self._clip_or_pad(segment, self.max_seq_len)
            time_ms = self._clip_or_pad(time_ms, self.max_seq_len)

        sample_id = item.get("sample_id", npz_path.stem)
        subject = item.get("subject", "unknown")
        task = item.get("task", "unknown")
        phase = item.get("phase", "unknown")
        device_class = self._infer_device_class(item)
        info_json = item.get("info_json")
        dotinfo_json = item.get("dotinfo_json")
        response_json = item.get("response_json")

        return {
            "face_rgb": torch.from_numpy(face),
            "left_eye_rgb": torch.from_numpy(left),
            "right_eye_rgb": torch.from_numpy(right),
            "gt_pog_cm": torch.from_numpy(gt),
            "valid_mask": torch.from_numpy(valid),
            "segment_id": torch.from_numpy(segment),
            "time_ms": torch.from_numpy(time_ms),
            "px_per_cm": torch.tensor(px_per_cm, dtype=torch.float32),
            "sample_id": sample_id,
            "subject": subject,
            "task": task,
            "phase": phase,
            "device_class": device_class,
            "info_json": info_json,
            "dotinfo_json": dotinfo_json,
            "response_json": response_json,
        }
