from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cogaze.config import CoGazeConfig
from cogaze.data.gaze_dataset import NPZSequenceDataset
from cogaze.models.cogaze import CoGaze
from cogaze.utils.checkpoint import load_checkpoint
from cogaze.utils.gaze_metrics import summarize_gaze_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate CoGaze with paper-style gaze metrics")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--cfg-json", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-json", type=str, default="")
    return p.parse_args()


def load_cfg(path: str) -> CoGazeConfig:
    cfg = CoGazeConfig()
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    for k, v in obj.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg_json)

    ds = NPZSequenceDataset(args.manifest, max_seq_len=cfg.max_seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CoGaze(cfg).to(args.device)
    ckpt = load_checkpoint(args.ckpt, "cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    pred_list = []
    gt_list = []
    valid_list = []
    px_list = []
    task_list = []
    device_list = []

    for batch in tqdm(loader, ncols=120):
        tensor_batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        out = model(tensor_batch)

        pred_list.append(out["pred_pog_cm"].cpu())
        gt_list.append(batch["gt_pog_cm"])
        valid_list.append(batch["valid_mask"])
        px_list.append(batch["px_per_cm"])
        task_list.extend([str(x) for x in batch.get("task", [])])
        device_list.extend([str(x) for x in batch.get("device_class", [])])

    pred = torch.cat(pred_list, dim=0)
    gt = torch.cat(gt_list, dim=0)
    valid = torch.cat(valid_list, dim=0)
    px = torch.cat(px_list, dim=0)

    metrics = summarize_gaze_metrics(pred, gt, valid_mask=valid, task_names=task_list, px_per_cm=px)

    for dev in ["phone", "tablet"]:
        idx = [i for i, d in enumerate(device_list) if d.lower() == dev]
        if not idx:
            continue
        sub_idx = torch.tensor(idx, dtype=torch.long)
        sub_metrics = summarize_gaze_metrics(
            pred[sub_idx],
            gt[sub_idx],
            valid_mask=valid[sub_idx],
            task_names=[task_list[i] for i in idx],
            px_per_cm=px[sub_idx],
        )
        for k, v in sub_metrics.items():
            metrics[f"{dev}_{k}"] = v

    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]:.6f}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()

