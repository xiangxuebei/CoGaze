from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cogaze.data.screening_dataset import ScreeningMeta, SubjectBagDataset, collate_subject_bags
from cogaze.screening.model import CognitiveScreeningMILModel
from cogaze.utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Infer subject-level cognitive screening")
    p.add_argument("--feature-csv", type=str, required=True)
    p.add_argument("--label-csv", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--meta", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=str, default="screening_predictions.csv")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    feat_df = pd.read_csv(args.feature_csv)
    label_df = pd.read_csv(args.label_csv)
    meta_json = json.loads(Path(args.meta).read_text(encoding="utf-8"))

    meta = ScreeningMeta(
        gaze_cols=meta_json["gaze_cols"],
        audio_cols=meta_json["audio_cols"],
        interaction_cols=meta_json["interaction_cols"],
        task_to_id=meta_json["task_to_id"],
        phase_to_id=meta_json["phase_to_id"],
        num_tasks=meta_json["num_tasks"],
        num_phases=meta_json["num_phases"],
    )

    ds = SubjectBagDataset(feat_df, label_df, meta)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_subject_bags)

    model = CognitiveScreeningMILModel(
        gaze_dim=max(1, len(meta.gaze_cols)),
        audio_dim=max(1, len(meta.audio_cols)),
        interaction_dim=max(1, len(meta.interaction_cols)),
        num_tasks=meta.num_tasks,
        num_phases=meta.num_phases,
        num_classes=meta_json["num_classes"],
    ).to(args.device)

    ckpt = load_checkpoint(args.ckpt, "cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = []
    for batch in loader:
        batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        out = model(batch)
        prob = torch.softmax(out["logits"], dim=-1).cpu().numpy()
        pred = prob.argmax(axis=1)
        for i, subject in enumerate(batch["subject"]):
            row = {"subject": subject, "pred_label": int(pred[i])}
            for c in range(prob.shape[1]):
                row[f"prob_class{c}"] = float(prob[i, c])
            rows.append(row)

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
