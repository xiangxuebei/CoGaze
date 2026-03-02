from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from cogaze.features.common import (
    VIEWING_DISTANCE_CM,
    cm_to_deg,
    compute_blink_stats,
    compute_velocities,
    convex_hull_area,
    estimate_sample_rate,
    lin_drift,
    median_dt_s,
    robust_valid_mask,
    spatial_entropy,
    summarize_list,
)


def detect_ivt_events(
    coords_deg: np.ndarray,
    time_s: np.ndarray,
    vel_deg_s: np.ndarray,
    sacc_thr: float = 30.0,
    fix_thr: float = 10.0,
    min_fix_dur: float = 0.10,
    min_sacc_dur: float = 0.02,
    merge_gap: float = 0.04,
) -> Tuple[List[dict], List[dict]]:
    n = len(vel_deg_s)
    if n == 0:
        return [], []

    v = np.asarray(vel_deg_s, dtype=float)
    v[~np.isfinite(v)] = -np.inf
    is_sacc = v >= sacc_thr
    is_fix = (v >= 0) & (v <= fix_thr)

    def segments(mask: np.ndarray):
        segs = []
        i = 0
        while i < n:
            if mask[i]:
                s = i
                while i + 1 < n and mask[i + 1]:
                    i += 1
                segs.append((s, i))
            i += 1
        return segs

    def merge(segs):
        if not segs:
            return []
        out = [segs[0]]
        for s, e in segs[1:]:
            ps, pe = out[-1]
            gap = time_s[s] - time_s[pe]
            if np.isfinite(gap) and gap <= merge_gap:
                out[-1] = (ps, e)
            else:
                out.append((s, e))
        return out

    sacc_segs = merge(segments(is_sacc))
    fix_segs = merge(segments(is_fix))

    saccades = []
    for s, e in sacc_segs:
        dur = float(time_s[e] - time_s[s])
        if dur >= min_sacc_dur:
            saccades.append({
                "start_idx": s,
                "end_idx": e,
                "duration_s": dur,
                "amplitude_deg": float(np.linalg.norm(coords_deg[e] - coords_deg[s])),
                "peak_speed_deg_s": float(np.nanmax(vel_deg_s[s:e + 1])),
            })

    fixations = []
    for s, e in fix_segs:
        dur = float(time_s[e] - time_s[s])
        if dur >= min_fix_dur:
            seg = coords_deg[s:e + 1]
            disp = float(np.linalg.norm(np.nanmax(seg, axis=0) - np.nanmin(seg, axis=0)))
            fixations.append({
                "start_idx": s,
                "end_idx": e,
                "duration_s": dur,
                "dispersion_deg": disp,
            })

    return saccades, fixations


def summarize_gaze_record(gaze_df: pd.DataFrame, device_meta: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if gaze_df.empty:
        return out

    gaze_df = gaze_df.copy()
    gaze_df["time_ms"] = pd.to_numeric(gaze_df["time_ms"], errors="coerce")
    gaze_df["x_cam"] = pd.to_numeric(gaze_df["x_cam"], errors="coerce")
    gaze_df["y_cam"] = pd.to_numeric(gaze_df["y_cam"], errors="coerce")
    gaze_df["valid"] = pd.to_numeric(gaze_df.get("valid", 1), errors="coerce").fillna(0).astype(int)

    time_ms = gaze_df["time_ms"].values.astype(float)
    time_s = time_ms / 1000.0
    valid_mask = robust_valid_mask(gaze_df)

    out["n_samples"] = float(len(gaze_df))
    out["duration_s"] = float((time_ms[-1] - time_ms[0]) / 1000.0) if len(gaze_df) > 1 else 0.0
    out["sample_rate_hz"] = estimate_sample_rate(time_ms)
    out["valid_ratio"] = float(np.mean(gaze_df["valid"].values == 1))
    out["valid_ratio_strict"] = float(np.mean(valid_mask))
    out.update(compute_blink_stats(time_ms, valid_mask))

    if valid_mask.sum() < 5:
        return out

    v = gaze_df.loc[valid_mask]
    coords_cm = v[["x_cam", "y_cam"]].values.astype(float)
    v_time_s = v["time_ms"].values.astype(float) / 1000.0

    out["mean_x_cm"] = float(np.nanmean(coords_cm[:, 0]))
    out["mean_y_cm"] = float(np.nanmean(coords_cm[:, 1]))
    out["std_x_cm"] = float(np.nanstd(coords_cm[:, 0]))
    out["std_y_cm"] = float(np.nanstd(coords_cm[:, 1]))

    bbox_w = float(np.nanmax(coords_cm[:, 0]) - np.nanmin(coords_cm[:, 0]))
    bbox_h = float(np.nanmax(coords_cm[:, 1]) - np.nanmin(coords_cm[:, 1]))
    out["bbox_width_cm"] = bbox_w
    out["bbox_height_cm"] = bbox_h
    out["bbox_area_cm2"] = bbox_w * bbox_h
    finite = np.isfinite(coords_cm).all(axis=1)
    out["convex_hull_area_cm2"] = convex_hull_area(coords_cm[finite]) if finite.sum() >= 3 else 0.0

    speed_cm_s = compute_velocities(coords_cm, v_time_s)
    path_length = float(np.nansum(np.linalg.norm(np.diff(coords_cm, axis=0), axis=1)))
    out["path_length_cm"] = path_length
    out["speed_mean_cm_s"], out["speed_median_cm_s"], out["speed_p95_cm_s"] = summarize_list(speed_cm_s)
    out["speed_max_cm_s"] = float(np.nanmax(speed_cm_s)) if np.isfinite(np.nanmax(speed_cm_s)) else float("nan")

    center = np.nanmedian(coords_cm, axis=0)
    diag_cm = float(device_meta.get("screen_diag_cm", float("nan")))
    coords_norm = (coords_cm - center) / diag_cm if np.isfinite(diag_cm) and diag_cm > 0 else coords_cm - center
    occ_ent, tr_ent = spatial_entropy(coords_norm)
    out["grid_entropy"] = occ_ent
    out["transition_entropy"] = tr_ent

    device_class = str(device_meta.get("device_class", "unknown"))
    coords_deg = cm_to_deg(coords_cm - center, VIEWING_DISTANCE_CM.get(device_class, VIEWING_DISTANCE_CM["unknown"]))
    vel_deg_s = compute_velocities(coords_deg, v_time_s)
    out["speed_mean_deg_s"], out["speed_median_deg_s"], out["speed_p95_deg_s"] = summarize_list(vel_deg_s)

    merge_gap = float(np.clip(median_dt_s(v_time_s) * 0.6, 0.02, 0.10)) if np.isfinite(median_dt_s(v_time_s)) else 0.06
    saccades, fixations = detect_ivt_events(coords_deg, v_time_s, vel_deg_s, merge_gap=merge_gap)
    out["saccade_count"] = float(len(saccades))
    out["fixation_count"] = float(len(fixations))

    if saccades:
        out["saccade_amp_mean_deg"], out["saccade_amp_median_deg"], out["saccade_amp_p95_deg"] = summarize_list([s["amplitude_deg"] for s in saccades])
        out["saccade_peak_mean_deg_s"], out["saccade_peak_median_deg_s"], out["saccade_peak_p95_deg_s"] = summarize_list([s["peak_speed_deg_s"] for s in saccades])
        out["saccade_dur_mean_s"], out["saccade_dur_median_s"], out["saccade_dur_p95_s"] = summarize_list([s["duration_s"] for s in saccades])
    else:
        out["saccade_amp_mean_deg"] = float("nan")
        out["saccade_peak_mean_deg_s"] = float("nan")
        out["saccade_dur_mean_s"] = float("nan")

    if fixations:
        out["fix_dur_mean_s"], out["fix_dur_median_s"], out["fix_dur_p95_s"] = summarize_list([f["duration_s"] for f in fixations])
        out["fix_disp_mean_deg"], out["fix_disp_median_deg"], out["fix_disp_p95_deg"] = summarize_list([f["dispersion_deg"] for f in fixations])
        out["fix_time_ratio"] = float(np.sum([f["duration_s"] for f in fixations])) / max(out["duration_s"], 1e-3)
    else:
        out["fix_dur_mean_s"] = float("nan")
        out["fix_disp_mean_deg"] = float("nan")
        out["fix_time_ratio"] = 0.0

    out["drift_x_cm_per_s"] = lin_drift(v_time_s, coords_cm[:, 0])
    out["drift_y_cm_per_s"] = lin_drift(v_time_s, coords_cm[:, 1])

    if np.isfinite(diag_cm) and diag_cm > 0:
        out["norm_path_length"] = path_length / diag_cm
        out["norm_bbox_area"] = out["bbox_area_cm2"] / (diag_cm ** 2)
        out["norm_convex_hull_area"] = out["convex_hull_area_cm2"] / (diag_cm ** 2)
    else:
        out["norm_path_length"] = float("nan")
        out["norm_bbox_area"] = float("nan")
        out["norm_convex_hull_area"] = float("nan")

    return out
