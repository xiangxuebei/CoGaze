from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cogaze.config import CoGazeConfig
from cogaze.data.gaze_dataset import NPZSequenceDataset
from cogaze.models.cogaze import CoGaze
from cogaze.utils.checkpoint import load_checkpoint
from cogaze.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Infer gaze trajectories with CoGaze")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--cfg-json", type=str, required=True)
    p.add_argument("--outdir", type=str, default="outputs/gaze_pred")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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

    outdir = ensure_dir(args.outdir)

    for batch in tqdm(loader, ncols=120):
        tensor_batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        out = model(tensor_batch)
        pred = out["pred_pog_cm"].cpu().numpy()
        valid = batch["valid_mask"].numpy()
        time_ms = batch["time_ms"].numpy()

        subjects = batch["subject"]
        tasks = batch["task"]
        phases = batch["phase"]
        sample_ids = batch["sample_id"]
        info_jsons = batch.get("info_json", [None] * pred.shape[0])
        dotinfo_jsons = batch.get("dotinfo_json", [None] * pred.shape[0])
        response_jsons = batch.get("response_json", [None] * pred.shape[0])

        for i in range(pred.shape[0]):
            sample_dir = ensure_dir(outdir / str(sample_ids[i]))
            df = pd.DataFrame({
                "frame_index": list(range(pred.shape[1])),
                "time_ms": time_ms[i].tolist(),
                "x_cam": pred[i, :, 0].tolist(),
                "y_cam": pred[i, :, 1].tolist(),
                "valid": valid[i].tolist(),
            })
            df.to_csv(sample_dir / "gaze_pred.csv", index=False)

            meta = {
                "sample_id": str(sample_ids[i]),
                "subject": str(subjects[i]),
                "task": str(tasks[i]),
                "phase": str(phases[i]),
            }
            (sample_dir / "gaze_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            sidecars = [
                (info_jsons[i], sample_dir / "info.json"),
                (dotinfo_jsons[i], sample_dir / "dotInfo.json"),
                (response_jsons[i], sample_dir / "response.json"),
            ]
            for src, dst in sidecars:
                if src is not None and str(src) and Path(str(src)).exists():
                    shutil.copyfile(str(src), dst)


if __name__ == "__main__":
    main()
