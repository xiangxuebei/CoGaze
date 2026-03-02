from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _numeric_cols(df: pd.DataFrame, prefixes: Sequence[str]) -> List[str]:
    cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in prefixes):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    return cols


@dataclass
class ScreeningMeta:
    gaze_cols: List[str]
    audio_cols: List[str]
    interaction_cols: List[str]
    task_to_id: Dict[str, int]
    phase_to_id: Dict[str, int]
    num_tasks: int
    num_phases: int


def build_screening_meta(df: pd.DataFrame) -> ScreeningMeta:
    gaze_cols = _numeric_cols(df, ["gaze_", "task_"])
    audio_cols = _numeric_cols(df, ["audio_"])
    interaction_cols = _numeric_cols(df, ["inter_", "interaction_"])

    task_vocab = sorted(df["task"].fillna("unknown").astype(str).unique().tolist())
    phase_vocab = sorted(df["phase"].fillna("unknown").astype(str).unique().tolist())

    return ScreeningMeta(
        gaze_cols=gaze_cols,
        audio_cols=audio_cols,
        interaction_cols=interaction_cols,
        task_to_id={k: i for i, k in enumerate(task_vocab)},
        phase_to_id={k: i for i, k in enumerate(phase_vocab)},
        num_tasks=len(task_vocab),
        num_phases=len(phase_vocab),
    )


class SubjectBagDataset(Dataset):
    def __init__(
        self,
        feature_df: pd.DataFrame,
        label_df: pd.DataFrame,
        meta: ScreeningMeta,
        subjects: List[str] | None = None,
        max_instances: int | None = None,
    ):
        self.meta = meta
        label_df = label_df.copy()
        label_df["subject"] = label_df["subject"].astype(str)

        feature_df = feature_df.copy()
        feature_df["subject"] = feature_df["subject"].astype(str)
        feature_df["task"] = feature_df["task"].fillna("unknown").astype(str)
        feature_df["phase"] = feature_df["phase"].fillna("unknown").astype(str)

        if subjects is not None:
            subj_set = set(subjects)
            feature_df = feature_df[feature_df["subject"].isin(subj_set)]
            label_df = label_df[label_df["subject"].isin(subj_set)]

        self.subjects = sorted(label_df["subject"].unique().tolist())
        self.labels = dict(zip(label_df["subject"], label_df["label"]))

        self.instances: Dict[str, pd.DataFrame] = {}
        for s, sdf in feature_df.groupby("subject"):
            sdf = sdf.sort_values(["task", "phase"]).reset_index(drop=True)
            if max_instances is not None:
                sdf = sdf.iloc[:max_instances].copy()
            self.instances[s] = sdf

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> dict:
        subject = self.subjects[idx]
        sdf = self.instances[subject]
        n = len(sdf)

        def arr(cols: List[str]) -> np.ndarray:
            if not cols:
                return np.zeros((n, 1), dtype=np.float32)
            x = sdf[cols].astype(float).fillna(0.0).values.astype(np.float32)
            return x

        gaze = arr(self.meta.gaze_cols)
        audio = arr(self.meta.audio_cols)
        inter = arr(self.meta.interaction_cols)

        task_id = np.array([self.meta.task_to_id.get(t, 0) for t in sdf["task"].tolist()], dtype=np.int64)
        phase_id = np.array([self.meta.phase_to_id.get(p, 0) for p in sdf["phase"].tolist()], dtype=np.int64)

        avail = np.stack(
            [
                np.ones((n,), dtype=np.float32) if self.meta.gaze_cols else np.zeros((n,), dtype=np.float32),
                np.ones((n,), dtype=np.float32) if self.meta.audio_cols else np.zeros((n,), dtype=np.float32),
                np.ones((n,), dtype=np.float32) if self.meta.interaction_cols else np.zeros((n,), dtype=np.float32),
            ],
            axis=-1,
        )
        attn_mask = np.ones((n,), dtype=np.float32)

        return {
            "subject": subject,
            "label": int(self.labels[subject]),
            "gaze": torch.from_numpy(gaze),
            "audio": torch.from_numpy(audio),
            "interaction": torch.from_numpy(inter),
            "avail": torch.from_numpy(avail),
            "task_id": torch.from_numpy(task_id),
            "phase_id": torch.from_numpy(phase_id),
            "attn_mask": torch.from_numpy(attn_mask),
        }


def collate_subject_bags(batch: List[dict]) -> dict:
    max_n = max(x["gaze"].shape[0] for x in batch)

    def pad_tensor(x: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
        n = x.shape[0]
        if n == max_n:
            return x
        pad_shape = (max_n - n,) + x.shape[1:]
        pad = torch.full(pad_shape, pad_value, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    out = {
        "subject": [x["subject"] for x in batch],
        "label": torch.tensor([x["label"] for x in batch], dtype=torch.long),
        "gaze": torch.stack([pad_tensor(x["gaze"]) for x in batch], dim=0),
        "audio": torch.stack([pad_tensor(x["audio"]) for x in batch], dim=0),
        "interaction": torch.stack([pad_tensor(x["interaction"]) for x in batch], dim=0),
        "avail": torch.stack([pad_tensor(x["avail"]) for x in batch], dim=0),
        "task_id": torch.stack([pad_tensor(x["task_id"], 0).long() for x in batch], dim=0),
        "phase_id": torch.stack([pad_tensor(x["phase_id"], 0).long() for x in batch], dim=0),
        "attn_mask": torch.stack([pad_tensor(x["attn_mask"], 0.0) for x in batch], dim=0),
    }
    return out
