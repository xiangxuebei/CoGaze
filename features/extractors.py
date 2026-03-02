from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from cogaze.features.common import (
    default_device_for_subject,
    ensure_dir,
    flatten_features,
    load_json,
    parse_device_info,
)
from cogaze.features.gaze_events import summarize_gaze_record
from cogaze.features.task_aligned_metrics import compute_task_aligned_metrics


@dataclass
class RecordPaths:
    subject: str
    task: str
    phase: str
    gaze_csv: Path
    meta_json: Optional[Path]
    info_json: Optional[Path]
    dotinfo_json: Optional[Path]
    response_json: Optional[Path]


def load_gaze_csv(gaze_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gaze_csv)
    if "valid" not in df.columns:
        df["valid"] = 1
    return df


def iter_records(gaze_root: Path) -> Iterable[RecordPaths]:
    for sample_dir in gaze_root.iterdir():
        if not sample_dir.is_dir():
            continue
        gaze_csv = sample_dir / "gaze_pred.csv"
        meta_json = sample_dir / "gaze_meta.json"
        if not gaze_csv.exists():
            continue

        subject = "unknown"
        task = "unknown"
        phase = "unknown"
        if meta_json.exists():
            meta = load_json(meta_json)
            subject = meta.get("subject", subject)
            task = meta.get("task", task)
            phase = meta.get("phase", phase)

        info_json = sample_dir / "info.json"
        dotinfo_json = sample_dir / "dotInfo.json"
        response_json = sample_dir / "response.json"

        yield RecordPaths(
            subject=subject,
            task=task,
            phase=phase,
            gaze_csv=gaze_csv,
            meta_json=meta_json if meta_json.exists() else None,
            info_json=info_json if info_json.exists() else None,
            dotinfo_json=dotinfo_json if dotinfo_json.exists() else None,
            response_json=response_json if response_json.exists() else None,
        )


def extract_record_features(paths: RecordPaths) -> Tuple[dict, dict]:
    gaze_df = load_gaze_csv(paths.gaze_csv)

    base = {"subject": paths.subject, "task": paths.task, "phase": paths.phase}

    if paths.info_json:
        device_meta = parse_device_info(load_json(paths.info_json))
    else:
        device_meta = default_device_for_subject(paths.subject) or parse_device_info(None)

    dotinfo = load_json(paths.dotinfo_json) if paths.dotinfo_json and paths.dotinfo_json.exists() else None

    gaze_feat = summarize_gaze_record(gaze_df, device_meta)
    task_feat = compute_task_aligned_metrics(gaze_df, dotinfo, device_meta)

    summary = {
        **base,
        **{k: v for k, v in device_meta.items() if k != "device_model"},
        **flatten_features("gaze_", gaze_feat),
        **flatten_features("task_", task_feat),
    }
    detail = {
        "gaze_csv": str(paths.gaze_csv),
        "meta_json": str(paths.meta_json) if paths.meta_json else None,
        "dotinfo_json": str(paths.dotinfo_json) if paths.dotinfo_json else None,
        "response_json": str(paths.response_json) if paths.response_json else None,
    }
    return summary, detail


def export_feature_records(gaze_root: Path, output_root: Path) -> Tuple[Path, Path]:
    ensure_dir(output_root)
    rows: List[dict] = []
    detail_path = output_root / "feature_detail.jsonl"

    with detail_path.open("w", encoding="utf-8") as wf:
        for rec in iter_records(gaze_root):
            summary, detail = extract_record_features(rec)
            rows.append(summary)
            wf.write(json.dumps({"summary": summary, "detail": detail}, ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    csv_path = output_root / "feature_records.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, detail_path
