from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from cogaze.data.screening_dataset import SubjectBagDataset, build_screening_meta, collate_subject_bags
from cogaze.screening.metrics import classification_metrics
from cogaze.screening.model import CognitiveScreeningMILModel
from cogaze.utils.checkpoint import save_checkpoint
from cogaze.utils.seed import seed_everything


@dataclass
class ScreeningRunMeta:
    gaze_dim: int
    audio_dim: int
    interaction_dim: int
    num_tasks: int
    num_phases: int
    num_classes: int
    task_to_id: Dict[str, int]
    phase_to_id: Dict[str, int]
    gaze_cols: List[str]
    audio_cols: List[str]
    interaction_cols: List[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train subject-level cognitive screening with 5-fold CV")
    p.add_argument("--feature-csv", type=str, required=True)
    p.add_argument("--label-csv", type=str, required=True)
    p.add_argument("--outdir", type=str, default="runs/cogaze_screening")
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device: str, num_classes: int):
    model.eval()
    ys = []
    probs = []
    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        out = model(batch)
        prob = torch.softmax(out["logits"], dim=-1).cpu().numpy()
        probs.append(prob)
        ys.append(batch["label"].cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(probs, axis=0)
    return classification_metrics(y_true, y_prob, num_classes), y_true, y_prob


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feat_df = pd.read_csv(args.feature_csv)
    label_df = pd.read_csv(args.label_csv)
    label_df["subject"] = label_df["subject"].astype(str)

    feat_df["subject"] = feat_df["subject"].astype(str)
    meta = build_screening_meta(feat_df)

    run_meta = ScreeningRunMeta(
        gaze_dim=max(1, len(meta.gaze_cols)),
        audio_dim=max(1, len(meta.audio_cols)),
        interaction_dim=max(1, len(meta.interaction_cols)),
        num_tasks=meta.num_tasks,
        num_phases=meta.num_phases,
        num_classes=args.num_classes,
        task_to_id=meta.task_to_id,
        phase_to_id=meta.phase_to_id,
        gaze_cols=meta.gaze_cols,
        audio_cols=meta.audio_cols,
        interaction_cols=meta.interaction_cols,
    )
    (outdir / "model_meta.json").write_text(json.dumps(asdict(run_meta), ensure_ascii=False, indent=2), encoding="utf-8")

    subjects = label_df["subject"].values
    labels = label_df["label"].values
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    all_fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(subjects, labels)):
        tr_subjects = subjects[tr_idx].tolist()
        va_subjects = subjects[va_idx].tolist()

        tr_ds = SubjectBagDataset(feat_df, label_df, meta, tr_subjects)
        va_ds = SubjectBagDataset(feat_df, label_df, meta, va_subjects)

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_subject_bags)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_subject_bags)

        model = CognitiveScreeningMILModel(
            gaze_dim=max(1, len(meta.gaze_cols)),
            audio_dim=max(1, len(meta.audio_cols)),
            interaction_dim=max(1, len(meta.interaction_cols)),
            num_tasks=meta.num_tasks,
            num_phases=meta.num_phases,
            num_classes=args.num_classes,
        ).to(args.device)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        best_auc = -1.0

        for epoch in range(args.epochs):
            model.train()
            pbar = tqdm(tr_loader, desc=f"fold {fold} epoch {epoch:03d}", ncols=120)
            for batch in pbar:
                batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                out = model(batch)
                loss = criterion(out["logits"], batch["label"])
                loss.backward()
                opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            metrics, _, _ = evaluate(model, va_loader, args.device, args.num_classes)
            print(f"[fold {fold} val] " + " ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))

            if metrics["macro_auc"] > best_auc:
                best_auc = metrics["macro_auc"]
                save_checkpoint(
                    outdir / f"fold{fold}_best.pt",
                    {
                        "model": model.state_dict(),
                        "fold": fold,
                        "metrics": metrics,
                    },
                )

        fold_metrics_path = outdir / f"fold{fold}_metrics.json"
        fold_metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        all_fold_metrics.append(metrics)

    mean_metrics = {k: float(np.mean([m[k] for m in all_fold_metrics])) for k in all_fold_metrics[0].keys()}
    (outdir / "cv_metrics_mean.json").write_text(json.dumps(mean_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("CV mean metrics:", mean_metrics)


if __name__ == "__main__":
    main()
