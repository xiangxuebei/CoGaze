from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


PHONE_DEFAULT_DEVICE = {
    "device_model": "phone_default",
    "screen_width_cm": 7.3,
    "screen_height_cm": 15.8,
    "screen_diag_cm": 17.4049,
    "screen_width_px": 1080.0,
    "screen_height_px": 2340.0,
    "device_class": "phone",
}

TABLET_DEFAULT_DEVICE = {
    "device_model": "tablet_default",
    "screen_width_cm": 31.394,
    "screen_height_cm": 19.604,
    "screen_diag_cm": 37.0121,
    "screen_width_px": 2960.0,
    "screen_height_px": 1848.0,
    "device_class": "tablet",
}

VIEWING_DISTANCE_CM = {"phone": 40.0, "tablet": 50.0, "unknown": 45.0}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def flatten_features(prefix: str, feat: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}{k}": v for k, v in feat.items()}


def estimate_sample_rate(time_ms: np.ndarray) -> float:
    if len(time_ms) < 2:
        return float("nan")
    dt = np.diff(time_ms)
    dt = dt[dt > 0]
    return float(1000.0 / np.median(dt)) if len(dt) else float("nan")


def median_dt_s(time_s: np.ndarray) -> float:
    if len(time_s) < 2:
        return float("nan")
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if len(dt) else float("nan")


def summarize_list(values: Iterable[float]) -> Tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanmedian(arr)), float(np.nanpercentile(arr, 95))


def cm_to_deg(cm: np.ndarray, viewing_dist_cm: float) -> np.ndarray:
    return np.degrees(2.0 * np.arctan(np.asarray(cm) / (2.0 * viewing_dist_cm)))


def compute_velocities(coords: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    if len(coords) < 2:
        return np.full((len(coords),), np.nan)
    dt = np.diff(time_s)
    dt = np.concatenate([[np.nan], dt])
    dt[dt == 0] = np.nan
    dxy = np.diff(coords, axis=0)
    dxy = np.vstack([np.full((1, coords.shape[1]), np.nan), dxy])
    speed = np.linalg.norm(dxy, axis=1) / dt
    speed[0] = np.nan
    return speed


def convex_hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    if len(hull) < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def lin_drift(time_s: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(time_s) & np.isfinite(values)
    if mask.sum() < 2:
        return float("nan")
    t = time_s[mask]
    v = values[mask]
    if np.allclose(t, t[0]):
        return 0.0
    coef = np.polyfit(t, v, 1)
    return float(coef[0])


def robust_valid_mask(df: pd.DataFrame) -> np.ndarray:
    return (
        (pd.to_numeric(df.get("valid", 0), errors="coerce").fillna(0).astype(int).values == 1)
        & np.isfinite(pd.to_numeric(df.get("x_cam"), errors="coerce").values)
        & np.isfinite(pd.to_numeric(df.get("y_cam"), errors="coerce").values)
        & np.isfinite(pd.to_numeric(df.get("time_ms"), errors="coerce").values)
    )


def compute_gaps(time_ms: np.ndarray, valid_mask: np.ndarray) -> List[float]:
    gaps = []
    start_idx = None
    for idx, ok in enumerate(valid_mask):
        if not ok and start_idx is None:
            start_idx = idx
        elif ok and start_idx is not None:
            gaps.append((time_ms[idx] - time_ms[start_idx]) / 1000.0)
            start_idx = None
    if start_idx is not None and len(time_ms):
        gaps.append((time_ms[-1] - time_ms[start_idx]) / 1000.0)
    return gaps


def compute_blink_stats(time_ms: np.ndarray, ok_mask: np.ndarray) -> Dict[str, float]:
    gaps = compute_gaps(time_ms, ok_mask)
    if not gaps:
        return {
            "blink_count": 0.0,
            "blink_rate_hz": 0.0,
            "blink_mean_dur_s": 0.0,
            "tracking_loss_total_s": 0.0,
            "tracking_loss_count": 0.0,
        }
    gaps = np.asarray(gaps, dtype=float)
    blink = gaps[(gaps >= 0.05) & (gaps <= 0.8)]
    loss = gaps[gaps > 0.8]
    dur_s = (time_ms[-1] - time_ms[0]) / 1000.0 if len(time_ms) > 1 else 0.0
    return {
        "blink_count": float(blink.size),
        "blink_rate_hz": float(blink.size) / max(dur_s, 1e-3),
        "blink_mean_dur_s": float(np.mean(blink)) if blink.size else 0.0,
        "tracking_loss_total_s": float(np.sum(loss)) if loss.size else 0.0,
        "tracking_loss_count": float(loss.size),
    }


def spatial_entropy(coords_norm: np.ndarray, grid: int = 10) -> Tuple[float, float]:
    if len(coords_norm) < 5:
        return float("nan"), float("nan")
    x = np.clip(coords_norm[:, 0], -0.75, 0.75)
    y = np.clip(coords_norm[:, 1], -0.75, 0.75)
    bins = np.linspace(-0.75, 0.75, grid + 1)
    xi = np.clip(np.digitize(x, bins) - 1, 0, grid - 1)
    yi = np.clip(np.digitize(y, bins) - 1, 0, grid - 1)
    cell = yi * grid + xi

    counts = np.bincount(cell, minlength=grid * grid).astype(float)
    p = counts / max(counts.sum(), 1.0)
    p = p[p > 0]
    occ_ent = -float(np.sum(p * np.log(p)))
    occ_ent = occ_ent / math.log(grid * grid)

    trans = cell[1:] * (grid * grid) + cell[:-1]
    tcounts = np.bincount(trans, minlength=(grid * grid) ** 2).astype(float)
    tp = tcounts / max(tcounts.sum(), 1.0)
    tp = tp[tp > 0]
    tr_ent = -float(np.sum(tp * np.log(tp)))
    tr_ent = tr_ent / math.log((grid * grid) ** 2)
    return occ_ent, tr_ent


def parse_device_info(info: Optional[dict]) -> Dict[str, object]:
    def to_float(val):
        try:
            return float(val)
        except Exception:
            return float("nan")

    device = (info or {}).get("device_info", {}) or {}
    screen_cm = device.get("screen_size_cm", {}) or {}
    screen_px = device.get("screen_resolution", {}) or {}

    w_cm = to_float(screen_cm.get("width"))
    h_cm = to_float(screen_cm.get("height"))
    diag_cm = math.hypot(w_cm, h_cm) if np.isfinite(w_cm) and np.isfinite(h_cm) else float("nan")
    device_class = "unknown"
    if np.isfinite(diag_cm):
        device_class = "phone" if diag_cm < 20 else "tablet"
    return {
        "device_model": device.get("model"),
        "screen_width_cm": w_cm,
        "screen_height_cm": h_cm,
        "screen_diag_cm": diag_cm,
        "screen_width_px": to_float(screen_px.get("width")),
        "screen_height_px": to_float(screen_px.get("height")),
        "device_class": device_class,
    }


def default_device_for_subject(subject: str) -> Optional[Dict[str, object]]:
    code = subject.upper()
    if code.startswith(("E00", "E01", "E02", "E03")):
        return dict(PHONE_DEFAULT_DEVICE)
    if code.startswith(("E04", "E05", "E06")):
        return dict(TABLET_DEFAULT_DEVICE)
    return None


def dynamic_hit_radius_cm(device_meta: Dict[str, object], base_min: float = 1.5) -> float:
    diag = float(device_meta.get("screen_diag_cm", float("nan")))
    if np.isfinite(diag) and diag > 0:
        return max(base_min, 0.05 * diag)
    return base_min


def maybe_scale_time_to_ms(times: np.ndarray, gaze_time_ms: np.ndarray) -> Tuple[np.ndarray, float]:
    if times.size == 0 or gaze_time_ms.size == 0:
        return times, 1.0
    tmax = float(np.nanmax(times))
    gmax = float(np.nanmax(gaze_time_ms))
    if np.isfinite(tmax) and np.isfinite(gmax) and (tmax < 1e3) and (gmax > 1e4):
        return times * 1000.0, 1000.0
    return times, 1.0
