from __future__ import annotations

from typing import Dict, List, Optional

import math
import numpy as np
import pandas as pd

from cogaze.features.common import dynamic_hit_radius_cm, maybe_scale_time_to_ms, robust_valid_mask


def classify_dotinfo(dotinfo: List[dict]) -> str:
    if not dotinfo:
        return "none"
    first = dotinfo[0]
    if "duration" in first or "half_second_time" in first:
        return "events"
    return "timeline"


def _first_saccade_latency_s(pre_xy: np.ndarray, post_xy: np.ndarray, post_t_ms: np.ndarray, t0_ms: float, min_move_cm: float) -> float:
    if pre_xy.size == 0 or post_xy.size == 0:
        return float("nan")
    ref = np.nanmedian(pre_xy, axis=0)
    disp = np.linalg.norm(post_xy - ref, axis=1)
    hit = np.where(np.isfinite(disp) & (disp >= min_move_cm))[0]
    if hit.size == 0:
        return float("nan")
    return float((post_t_ms[int(hit[0])] - t0_ms) / 1000.0)


def _dwell_ratio(dist_cm: np.ndarray, hit_r_cm: float) -> tuple[float, float]:
    finite = np.isfinite(dist_cm)
    if not np.any(finite):
        return float("nan"), float("nan")
    in_roi = np.mean(dist_cm[finite] <= hit_r_cm)
    return float(in_roi), float(1.0 - in_roi)


def _hemi_ratio(values_x: np.ndarray, center_x: float = 0.0) -> tuple[float, float]:
    finite = np.isfinite(values_x)
    if not np.any(finite):
        return float("nan"), float("nan")
    left = np.mean(values_x[finite] < center_x)
    return float(left), float(1.0 - left)


def _nanmean_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def align_event_targets(
    gaze_df: pd.DataFrame,
    dotinfo: List[dict],
    device_meta: Dict[str, object],
) -> Dict[str, float]:
    if not dotinfo or gaze_df.empty:
        return {"mode": "events", "event_count": float(len(dotinfo))}

    g = gaze_df.copy()
    g["time_ms"] = pd.to_numeric(g["time_ms"], errors="coerce")
    g["x_cam"] = pd.to_numeric(g["x_cam"], errors="coerce")
    g["y_cam"] = pd.to_numeric(g["y_cam"], errors="coerce")
    g = g.loc[robust_valid_mask(g)]
    if g.empty:
        return {"mode": "events", "event_count": float(len(dotinfo))}

    hit_r = dynamic_hit_radius_cm(device_meta)
    gaze_time = g["time_ms"].values.astype(float)

    starts = np.array([d.get("half_second_time", d.get("time_offset", 0)) for d in dotinfo], dtype=float)
    ends = np.array([d.get("disappear_time", np.nan) for d in dotinfo], dtype=float)
    starts, _ = maybe_scale_time_to_ms(starts, gaze_time)
    ends, _ = maybe_scale_time_to_ms(ends, gaze_time)
    if np.isnan(ends).all():
        durs = np.array([d.get("duration", np.nan) for d in dotinfo], dtype=float)
        durs, _ = maybe_scale_time_to_ms(durs, gaze_time)
        ends = starts + np.nan_to_num(durs, nan=0.0)

    latency = []
    steady_err = []
    gain = []
    miss = []
    anti_wrong_dir = []
    anti_correction = []
    first_sacc_latency = []
    dwell_in_roi = []
    dwell_out_roi = []
    target_side_ratio = []
    hemi_left_ratio = []
    hemi_right_ratio = []
    prev_tx = None
    prev_ty = None

    for i, event in enumerate(dotinfo):
        t0 = float(starts[i])
        t1 = float(ends[i]) if np.isfinite(ends[i]) else t0 + 1000.0
        tx = float(event.get("x_cam", np.nan))
        ty = float(event.get("y_cam", np.nan))
        if not (np.isfinite(t0) and np.isfinite(tx) and np.isfinite(ty)):
            continue

        pre = g[(g["time_ms"] >= t0 - 500) & (g["time_ms"] < t0)]
        post = g[(g["time_ms"] >= t0) & (g["time_ms"] <= min(t0 + 1200, t1 + 200))]
        steady = g[(g["time_ms"] >= t0 + 400) & (g["time_ms"] <= min(t1, t0 + 2000))]

        first_sacc_latency.append(
            _first_saccade_latency_s(
                pre[["x_cam", "y_cam"]].values,
                post[["x_cam", "y_cam"]].values,
                post["time_ms"].values.astype(float),
                t0,
                min_move_cm=max(0.4 * hit_r, 0.8),
            )
        )

        if not post.empty:
            dx = post["x_cam"].values - tx
            dy = post["y_cam"].values - ty
            dist = np.sqrt(dx * dx + dy * dy)

            in_roi, out_roi = _dwell_ratio(dist, hit_r)
            dwell_in_roi.append(in_roi)
            dwell_out_roi.append(out_roi)
            l_ratio, r_ratio = _hemi_ratio(post["x_cam"].values, center_x=0.0)
            hemi_left_ratio.append(l_ratio)
            hemi_right_ratio.append(r_ratio)
            if abs(tx) > 1e-6:
                same_side = np.mean(np.sign(post["x_cam"].values) == np.sign(tx))
                target_side_ratio.append(float(same_side))

            within = dist <= hit_r
            if np.any(within):
                idx = int(np.argmax(within))
                latency.append(float((post["time_ms"].values[idx] - t0) / 1000.0))
                miss.append(0)
            else:
                miss.append(1)
        else:
            miss.append(1)

        if not steady.empty:
            dx = steady["x_cam"].values - tx
            dy = steady["y_cam"].values - ty
            dist = np.sqrt(dx * dx + dy * dy)
            steady_err.append(float(np.nanmean(dist)))

        if prev_tx is not None and not pre.empty and not steady.empty:
            step = float(math.hypot(tx - prev_tx, ty - prev_ty))
            if step > 1e-6:
                pre_center = np.nanmedian(pre[["x_cam", "y_cam"]].values, axis=0)
                post_center = np.nanmedian(steady[["x_cam", "y_cam"]].values, axis=0)
                obs = float(np.linalg.norm(post_center - pre_center))
                gain.append(obs / step)

                target_dx = tx - prev_tx
                observed_dx = post_center[0] - pre_center[0]
                if abs(target_dx) > 1e-6:
                    anti_wrong_dir.append(float(np.sign(observed_dx) != np.sign(target_dx)))
                    # first correction time from wrong to correct direction
                    if not post.empty:
                        disp_x = post["x_cam"].values - pre_center[0]
                        wrong = np.sign(disp_x) != np.sign(target_dx)
                        correct = np.sign(disp_x) == np.sign(target_dx)
                        if np.any(wrong) and np.any(correct):
                            first_correct = np.argmax(correct)
                            anti_correction.append(float((post["time_ms"].values[first_correct] - t0) / 1000.0))

        prev_tx, prev_ty = tx, ty

    out = {
        "mode": "events",
        "event_count": float(len(dotinfo)),
        "miss_ratio": float(np.mean(miss)) if miss else float("nan"),
        "latency_mean_s": float(np.mean(latency)) if latency else float("nan"),
        "latency_median_s": float(np.median(latency)) if latency else float("nan"),
        "first_saccade_latency_mean_s": _nanmean_or_nan(first_sacc_latency),
        "steady_error_mean_cm": float(np.mean(steady_err)) if steady_err else float("nan"),
        "steady_error_median_cm": float(np.median(steady_err)) if steady_err else float("nan"),
        "gain_mean": float(np.mean(gain)) if gain else float("nan"),
        "gain_median": float(np.median(gain)) if gain else float("nan"),
        "anti_wrong_dir_ratio": float(np.mean(anti_wrong_dir)) if anti_wrong_dir else float("nan"),
        "direction_error_rate": float(np.mean(anti_wrong_dir)) if anti_wrong_dir else float("nan"),
        "anti_correction_mean_s": float(np.mean(anti_correction)) if anti_correction else float("nan"),
        "dwell_in_roi_ratio": _nanmean_or_nan(dwell_in_roi),
        "dwell_out_roi_ratio": _nanmean_or_nan(dwell_out_roi),
        "target_side_ratio": _nanmean_or_nan(target_side_ratio),
        "left_hemifield_ratio": _nanmean_or_nan(hemi_left_ratio),
        "right_hemifield_ratio": _nanmean_or_nan(hemi_right_ratio),
    }
    return out


def align_timeline_targets(
    gaze_df: pd.DataFrame,
    dotinfo: List[dict],
    device_meta: Dict[str, object],
) -> Dict[str, float]:
    dot_df = pd.DataFrame(dotinfo)
    if dot_df.empty or "time_offset" not in dot_df:
        return {"mode": "timeline", "aligned_samples": 0.0}

    g = gaze_df.copy()
    g["time_ms"] = pd.to_numeric(g["time_ms"], errors="coerce")
    g["x_cam"] = pd.to_numeric(g["x_cam"], errors="coerce")
    g["y_cam"] = pd.to_numeric(g["y_cam"], errors="coerce")
    g = g.loc[robust_valid_mask(g)]
    if g.empty:
        return {"mode": "timeline", "aligned_samples": 0.0}

    gaze_time = g["time_ms"].values.astype(float)

    dot_df = dot_df.rename(columns={"time_offset": "time_ms"}).copy()
    dot_df["time_ms"] = pd.to_numeric(dot_df["time_ms"], errors="coerce")
    dot_df["x_cam"] = pd.to_numeric(dot_df["x_cam"], errors="coerce")
    dot_df["y_cam"] = pd.to_numeric(dot_df["y_cam"], errors="coerce")
    dot_df = dot_df.dropna(subset=["time_ms", "x_cam", "y_cam"])
    if dot_df.empty:
        return {"mode": "timeline", "aligned_samples": 0.0}

    times, _ = maybe_scale_time_to_ms(dot_df["time_ms"].values.astype(float), gaze_time)
    dot_df["time_ms"] = times

    merged = pd.merge_asof(
        g.sort_values("time_ms"),
        dot_df[["time_ms", "x_cam", "y_cam"]].sort_values("time_ms"),
        on="time_ms",
        direction="nearest",
        tolerance=120,
        suffixes=("_g", "_t"),
    )
    merged = merged.dropna(subset=["x_cam_g", "y_cam_g", "x_cam_t", "y_cam_t"])
    if merged.empty:
        return {"mode": "timeline", "aligned_samples": 0.0}

    dx = merged["x_cam_g"].values - merged["x_cam_t"].values
    dy = merged["y_cam_g"].values - merged["y_cam_t"].values
    dist = np.sqrt(dx * dx + dy * dy)
    hit_r = dynamic_hit_radius_cm(device_meta)
    in_roi, out_roi = _dwell_ratio(dist, hit_r)
    l_ratio, r_ratio = _hemi_ratio(merged["x_cam_g"].values, center_x=0.0)
    target_side = np.mean(np.sign(merged["x_cam_g"].values) == np.sign(merged["x_cam_t"].values))

    return {
        "mode": "timeline",
        "aligned_samples": float(len(merged)),
        "mean_error_cm": float(np.mean(dist)),
        "median_error_cm": float(np.median(dist)),
        "p95_error_cm": float(np.percentile(dist, 95)),
        "rmse_error_cm": float(np.sqrt(np.mean(dist * dist))),
        "hit_ratio": float(np.mean(dist <= hit_r)),
        "dwell_in_roi_ratio": in_roi,
        "dwell_out_roi_ratio": out_roi,
        "left_hemifield_ratio": l_ratio,
        "right_hemifield_ratio": r_ratio,
        "target_side_ratio": float(target_side) if np.isfinite(target_side) else float("nan"),
    }


def compute_task_aligned_metrics(
    gaze_df: pd.DataFrame,
    dotinfo: Optional[List[dict]],
    device_meta: Dict[str, object],
) -> Dict[str, float]:
    if not dotinfo:
        return {"mode": "none"}
    dtype = classify_dotinfo(dotinfo)
    if dtype == "events":
        return align_event_targets(gaze_df, dotinfo, device_meta)
    if dtype == "timeline":
        return align_timeline_targets(gaze_df, dotinfo, device_meta)
    return {"mode": "none"}
