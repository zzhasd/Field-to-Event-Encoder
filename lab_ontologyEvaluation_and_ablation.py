"""Ontology evaluation and ablation runner for the no-leak Field-to-Event Encoder v8.1.2.

This script intentionally does not define the core encoder.  It imports and
instantiates UniversalFieldToEventEncoder from University_Field_to_Event_Encoder.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from University_Field_to_Event_Encoder import (
    GRID_SIZE, DEFAULT_OBS_RATIO, CORE_VERSION, FIELD_REGISTRY, FIELDS,
    EffectSpec, TestCase, StressConfig, build_current_fields,
    make_obstacle_grid, nearest_free, random_free_cell,
    UniversalFieldToEventEncoder,
)

EVAL_VERSION = "v8.1.2"

# ============================================================
# Test cases, stress grid, ablations, and evaluator
# ============================================================
def shape_radius(shape: str) -> float:
    return {"point_like": 1.2, "compact_blob": 2.0, "blob": 3.4, "oval": 2.3, "elongated_strip": 2.3}[shape]

def make_combo_cases() -> List[TestCase]:
    directions = ["high", "low"]
    shapes = ["point_like", "compact_blob", "blob", "oval", "elongated_strip"]
    area_trends = ["expanding", "shrinking", "stable"]
    intensity_trends = ["strengthening", "weakening", "stable"]
    fields = list(FIELDS)
    cases: List[TestCase] = []
    idx = 0
    for pol in directions:
        for shape in shapes:
            for at in area_trends:
                for it in intensity_trends:
                    idx += 1
                    field_key = fields[(idx - 1) % len(fields)]
                    angle = (idx * 0.47) % math.pi
                    center = (15.0 + 4.0 * math.sin(idx * 0.8), 15.0 + 4.0 * math.cos(idx * 0.65))
                    eff = EffectSpec(
                        effect_id=f"E{idx:03d}", field_key=field_key, polarity=pol, shape=shape,
                        area_trend=at, intensity_trend=it, center=center, start=25, end=150,
                        amplitude_sigma=3.5 if shape != "point_like" else 4.2,
                        radius=shape_radius(shape), angle=angle, axis_ratio=2.4,
                    )
                    cases.append(TestCase(f"combo_{idx:03d}_{pol}_{shape}_{at}_{it}", [eff]))
    cases.append(TestCase("normal_no_incident", []))
    return cases

def make_stress_configs(which: str = "core") -> List[StressConfig]:
    all_cfgs = [
        StressConfig("baseline"),
        StressConfig("noise_mid", noise_sigma=0.20),
        StressConfig("noise_high", noise_sigma=0.40),
        StressConfig("drift_slow", drift_sigma_per_100=0.35),
        StressConfig("drift_fast", drift_sigma_per_100=0.85),
        StressConfig("occlusion_10", occlusion_rate=0.10),
        StressConfig("occlusion_30", occlusion_rate=0.30),
        StressConfig("weak_anomaly", anomaly_scale=0.70),
        StressConfig("strong_anomaly", anomaly_scale=1.30),
        StressConfig("moving_slow", moving_px_total=4.0),
        StressConfig("moving_fast", moving_px_total=8.0),
    ]
    if which == "baseline":
        return [all_cfgs[0]]
    if which == "core":
        names = {"baseline", "noise_high", "drift_fast", "occlusion_30", "weak_anomaly", "moving_fast"}
        return [c for c in all_cfgs if c.name in names]
    if which == "all":
        return all_cfgs
    raise ValueError(f"unknown stress set: {which}")

def make_encoder_variants(which: str = "all") -> Dict[str, Dict[str, bool]]:
    variants = {
        "full": {},
        "ablate_no_bg_ema": {"use_background_ema": False},
        "ablate_no_cumulative_score": {"use_cumulative_score": False},
        "ablate_no_hysteresis": {"use_hysteresis": False},
        "ablate_no_tracking": {"use_tracking": False},
        "ablate_no_morphology_debounce": {"use_morphology_debounce": False},
    }
    if which == "full":
        return {"full": variants["full"]}
    if which == "all":
        return variants
    selected: Dict[str, Dict[str, bool]] = {}
    for name in [x.strip() for x in which.split(",") if x.strip()]:
        if name not in variants:
            raise ValueError(f"unknown variant {name}; choices={list(variants)}")
        selected[name] = variants[name]
    return selected

def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = int((mask_a & mask_b).sum())
    union = int((mask_a | mask_b).sum())
    return inter / union if union else 0.0

def centroid_error(c0: Tuple[float, float], c1: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(c0, dtype=float) - np.array(c1, dtype=float)))

def safe_mean(vals: List[Any]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.mean(xs)) if xs else None

def serializable_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in ev.items() if k != "mask"}
    return out


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS for progress and ETA reporting."""
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

class ProgressMeter:
    """Small dependency-free progress bar with ETA.

    The experiment runner can involve tens of thousands of seed/stress/variant/case
    jobs.  This class reports completed jobs, percent, elapsed time, ETA, processing
    rate, and the most recently completed configuration.  It writes to stderr so CSV
    and JSON outputs remain clean.
    """
    def __init__(self, total: int, enabled: bool = True, width: int = 32, interval_sec: float = 2.0):
        self.total = max(1, int(total))
        self.enabled = bool(enabled)
        self.width = max(8, int(width))
        self.interval_sec = max(0.1, float(interval_sec))
        self.start = time.perf_counter()
        self.last_print = 0.0
        self.is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())

    @staticmethod
    def _trim(text: str, max_len: int = 44) -> str:
        text = str(text)
        return text if len(text) <= max_len else text[:max_len - 3] + "..."

    def update(self, completed: int, status: str = "") -> None:
        if not self.enabled:
            return
        completed = max(0, min(int(completed), self.total))
        now = time.perf_counter()
        if completed not in {0, 1, self.total} and (now - self.last_print) < self.interval_sec:
            return
        self.last_print = now
        elapsed = now - self.start
        frac = completed / self.total
        filled = int(round(self.width * frac))
        bar = "#" * filled + "-" * (self.width - filled)
        rate = completed / elapsed if elapsed > 1e-9 else 0.0
        eta = (self.total - completed) / rate if rate > 1e-9 else float("nan")
        line = (
            f"[{bar}] {completed}/{self.total} "
            f"({frac * 100:6.2f}%) elapsed={format_duration(elapsed)} "
            f"ETA={format_duration(eta)} rate={rate:.2f} jobs/s"
        )
        if status:
            line += f" | {self._trim(status)}"
        if self.is_tty:
            sys.stderr.write("\r" + line)
            if completed >= self.total:
                sys.stderr.write("\n")
            sys.stderr.flush()
        else:
            print(line, file=sys.stderr, flush=True)

def clone_case_with_stress(case: TestCase, grid: np.ndarray, rng: np.random.Generator, stress: StressConfig) -> TestCase:
    effects: List[EffectSpec] = []
    for e in case.effects:
        eff = EffectSpec(**asdict(e))
        eff.center = tuple(map(float, nearest_free(grid, eff.center)))
        if rng.random() < 0.25:
            eff.center = tuple(map(float, random_free_cell(grid, rng, margin=4)))
        if stress.moving_px_total > 0:
            angle = float(rng.uniform(0, 2 * math.pi))
            eff.velocity = (stress.moving_px_total * math.sin(angle), stress.moving_px_total * math.cos(angle))
        effects.append(eff)
    return TestCase(case.case_id, effects)

def evaluate_case(
    testcase: TestCase,
    episode: int,
    steps: int,
    obs_ratio: float,
    seed: int,
    stress: StressConfig,
    variant_name: str,
    variant_kwargs: Dict[str, bool],
    all_fields: bool = False,
    warmup: int = 40,
    match_iou_threshold: float = 0.10,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed + 1234)
    grid = make_obstacle_grid(GRID_SIZE, obs_ratio, seed=seed)
    tc = clone_case_with_stress(testcase, grid, rng, stress)
    encoder = UniversalFieldToEventEncoder(grid, FIELD_REGISTRY, **variant_kwargs)
    field_keys = tuple(FIELDS) if all_fields or not tc.effects else tuple(sorted({e.field_key for e in tc.effects}))
    effect_state: Dict[str, Dict[str, Any]] = {
        e.effect_id: {
            "first_detect_t": None,
            "last_event": None,
            "best_event": None,
            "ious": [],
            "centroid_errors": [],
        }
        for e in tc.effects
    }
    tp = normal_fp = extra_event_fp = fn = 0
    normal_frame_count = 0
    effect_frame_count = 0
    frame_count = 0
    t0 = time.perf_counter()

    for t in range(steps):
        frame_count += 1
        frame_rng = np.random.default_rng(seed * 1000003 + t)
        current, gt_records = build_current_fields(grid, tc, t, field_keys, stress=stress, rng=frame_rng)
        events = encoder.update(t, current)
        active_gt = gt_records
        matched_event_ids = set()

        for gt in active_gt:
            candidates = [
                (idx, ev, iou(gt["mask"], ev["mask"]))
                for idx, ev in enumerate(events)
                if ev["field_key"] == gt["field_key"] and ev["polarity"] == gt["polarity"]
            ]
            if candidates:
                idx, best_ev, best_iou = max(candidates, key=lambda x: x[2])
            else:
                idx, best_ev, best_iou = -1, None, 0.0
            if best_ev is not None and best_iou >= match_iou_threshold:
                tp += 1
                matched_event_ids.add(idx)
                st = effect_state[gt["effect_id"]]
                if st["first_detect_t"] is None:
                    st["first_detect_t"] = t
                c_err = centroid_error(gt["centroid"], best_ev["centroid"])
                st["ious"].append(float(best_iou))
                st["centroid_errors"].append(float(c_err))
                st["last_event"] = serializable_event(best_ev)
                if st["best_event"] is None or best_iou > st["best_event"]["iou"]:
                    st["best_event"] = {"iou": float(best_iou), "event": serializable_event(best_ev), "centroid_error": float(c_err)}
            else:
                fn += 1

        if t > warmup:
            if active_gt:
                effect_frame_count += 1
            else:
                normal_frame_count += 1
            for idx, ev in enumerate(events):
                if idx not in matched_event_ids:
                    if active_gt:
                        extra_event_fp += 1
                    else:
                        normal_fp += 1

    fp = normal_fp + extra_event_fp
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    rows: List[Dict[str, Any]] = []
    case_common = {
        "variant": variant_name,
        "stress": stress.name,
        "episode": int(episode),
        "seed": int(seed),
        "case_id": tc.case_id,
        "steps": int(steps),
        "obs_ratio": float(obs_ratio),
        "noise_sigma": stress.noise_sigma,
        "drift_sigma_per_100": stress.drift_sigma_per_100,
        "occlusion_rate": stress.occlusion_rate,
        "anomaly_scale": stress.anomaly_scale,
        "moving_px_total": stress.moving_px_total,
        "tp": int(tp),
        "fp": int(fp),
        "normal_fp": int(normal_fp),
        "extra_event_fp": int(extra_event_fp),
        "fn": int(fn),
        "normal_frame_count": int(normal_frame_count),
        "effect_frame_count": int(effect_frame_count),
        "frame_count": int(frame_count),
        "elapsed_ms": round(float(elapsed_ms), 3),
        "latency_ms_per_frame": round(float(elapsed_ms / max(1, frame_count)), 6),
    }
    if not tc.effects:
        rows.append({
            **case_common,
            "effect_id": "__normal__",
            "field_key": None,
            "polarity": None,
            "expected_shape": None,
            "predicted_shape": None,
            "expected_area_trend": None,
            "predicted_area_trend": None,
            "expected_intensity_trend": None,
            "predicted_intensity_trend": None,
            "detected": 0,
            "best_iou": None,
            "mean_iou": None,
            "mean_centroid_error": None,
            "shape_match": None,
            "area_trend_match": None,
            "intensity_trend_match": None,
            "trend_match": None,
            "latency_steps": None,
            "predicted_event": None,
        })
        return rows

    for eff in tc.effects:
        st = effect_state[eff.effect_id]
        last_ev = st["last_event"]
        best = st["best_event"]
        tmp = (last_ev or {}).get("temporal_summary", {}) if last_ev else {}
        shape_match = int(last_ev.get("morphology") == eff.shape) if last_ev else 0
        area_match = int(tmp.get("area_trend") == eff.area_trend) if last_ev else 0
        int_match = int(tmp.get("intensity_trend") == eff.intensity_trend) if last_ev else 0
        rows.append({
            **case_common,
            "effect_id": eff.effect_id,
            "field_key": eff.field_key,
            "polarity": eff.polarity,
            "expected_shape": eff.shape,
            "predicted_shape": last_ev.get("morphology") if last_ev else None,
            "expected_area_trend": eff.area_trend,
            "predicted_area_trend": tmp.get("area_trend") if last_ev else None,
            "expected_intensity_trend": eff.intensity_trend,
            "predicted_intensity_trend": tmp.get("intensity_trend") if last_ev else None,
            "detected": int(st["first_detect_t"] is not None),
            "best_iou": round(float(best["iou"]), 6) if best else None,
            "mean_iou": round(float(np.mean(st["ious"])), 6) if st["ious"] else None,
            "mean_centroid_error": round(float(np.mean(st["centroid_errors"])), 6) if st["centroid_errors"] else None,
            "shape_match": shape_match,
            "area_trend_match": area_match,
            "intensity_trend_match": int_match,
            "trend_match": round(float((area_match + int_match) / 2.0), 3),
            "latency_steps": int(st["first_detect_t"] - eff.start) if st["first_detect_t"] is not None else None,
            "predicted_event": last_ev,
        })
    return rows

def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def aggregate_rows(rows: List[Dict[str, Any]], group_keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in group_keys), []).append(r)
    out = []
    for key, rs in groups.items():
        eff_rs = [r for r in rs if r.get("effect_id") != "__normal__"]
        tp_s = int(sum(int(r.get("tp", 0)) for r in rs))
        normal_fp_s = int(sum(int(r.get("normal_fp", 0)) for r in rs))
        extra_fp_s = int(sum(int(r.get("extra_event_fp", 0)) for r in rs))
        fp_s = normal_fp_s + extra_fp_s
        fn_s = int(sum(int(r.get("fn", 0)) for r in rs))
        precision, recall, f1 = precision_recall_f1(tp_s, fp_s, fn_s)
        normal_frames = int(sum(int(r.get("normal_frame_count", 0)) for r in rs))
        effect_frames = int(sum(int(r.get("effect_frame_count", 0)) for r in rs))
        row = {k: v for k, v in zip(group_keys, key)}
        row.update({
            "n_rows": len(rs),
            "n_effect_rows": len(eff_rs),
            "tp": tp_s,
            "fp": fp_s,
            "normal_fp": normal_fp_s,
            "extra_event_fp": extra_fp_s,
            "fn": fn_s,
            "detection_precision": round(float(precision), 6) if precision is not None else None,
            "detection_recall": round(float(recall), 6) if recall is not None else None,
            "detection_f1": round(float(f1), 6) if f1 is not None else None,
            "normal_fp_rate_per_frame": round(float(normal_fp_s / max(1, normal_frames)), 6),
            "normal_fp_per_100_frames": round(float(100.0 * normal_fp_s / max(1, normal_frames)), 6),
            "extra_event_fp_rate_per_effect_frame": round(float(extra_fp_s / max(1, effect_frames)), 6),
            "extra_event_fp_per_100_effect_frames": round(float(100.0 * extra_fp_s / max(1, effect_frames)), 6),
            "false_positive_rate_per_counted_frame": round(float(fp_s / max(1, normal_frames + effect_frames)), 6),
            "false_positive_per_100_counted_frames": round(float(100.0 * fp_s / max(1, normal_frames + effect_frames)), 6),
            # Backward-compatible alias; no longer divides effect-case FP by normal frames.
            "false_positive_per_100_frames": round(float(100.0 * fp_s / max(1, normal_frames + effect_frames)), 6),
            "normal_frame_count": normal_frames,
            "effect_frame_count": effect_frames,
            "detection_rate": round(float(np.mean([r.get("detected", 0) for r in eff_rs])), 6) if eff_rs else None,
            "mean_iou": round(float(np.mean([r["mean_iou"] for r in eff_rs if r.get("mean_iou") is not None])), 6) if any(r.get("mean_iou") is not None for r in eff_rs) else None,
            "best_iou": round(float(np.mean([r["best_iou"] for r in eff_rs if r.get("best_iou") is not None])), 6) if any(r.get("best_iou") is not None for r in eff_rs) else None,
            "centroid_error": round(float(np.mean([r["mean_centroid_error"] for r in eff_rs if r.get("mean_centroid_error") is not None])), 6) if any(r.get("mean_centroid_error") is not None for r in eff_rs) else None,
            "shape_accuracy": round(float(np.mean([r["shape_match"] for r in eff_rs if r.get("shape_match") is not None])), 6) if eff_rs else None,
            "area_trend_accuracy": round(float(np.mean([r["area_trend_match"] for r in eff_rs if r.get("area_trend_match") is not None])), 6) if eff_rs else None,
            "intensity_trend_accuracy": round(float(np.mean([r["intensity_trend_match"] for r in eff_rs if r.get("intensity_trend_match") is not None])), 6) if eff_rs else None,
            "trend_accuracy": round(float(np.mean([r["trend_match"] for r in eff_rs if r.get("trend_match") is not None])), 6) if eff_rs else None,
            "latency_steps": round(float(np.mean([r["latency_steps"] for r in eff_rs if r.get("latency_steps") is not None])), 6) if any(r.get("latency_steps") is not None for r in eff_rs) else None,
            "latency_ms_per_frame": round(float(np.mean([r.get("latency_ms_per_frame", 0.0) for r in rs])), 6) if rs else None,
        })
        out.append(row)
    out.sort(key=lambda x: tuple(str(x.get(k)) for k in group_keys))
    return out

def compute_ablation_deltas(agg_variant_stress: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_stress: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in agg_variant_stress:
        by_stress.setdefault(r["stress"], {})[r["variant"]] = r
    metrics = ["detection_f1", "mean_iou", "centroid_error", "shape_accuracy", "trend_accuracy", "normal_fp_per_100_frames", "extra_event_fp_per_100_effect_frames", "normal_fp_per_100_frames", "extra_event_fp_per_100_effect_frames", "false_positive_per_100_frames", "latency_steps"]
    out = []
    for stress, variants in sorted(by_stress.items()):
        full = variants.get("full")
        if not full:
            continue
        for name, r in sorted(variants.items()):
            if name == "full":
                continue
            row = {"stress": stress, "variant": name}
            for m in metrics:
                fv, av = full.get(m), r.get(m)
                if fv is None or av is None:
                    row[f"drop_vs_full__{m}"] = None
                elif m in {"centroid_error", "normal_fp_per_100_frames", "extra_event_fp_per_100_effect_frames", "normal_fp_per_100_frames", "extra_event_fp_per_100_effect_frames", "false_positive_per_100_frames", "latency_steps"}:
                    # Positive means the ablation is worse than full for error/cost metrics.
                    row[f"drop_vs_full__{m}"] = round(float(av - fv), 6)
                else:
                    # Positive means the full model is better than the ablation for score metrics.
                    row[f"drop_vs_full__{m}"] = round(float(fv - av), 6)
            out.append(row)
    return out

def case_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return aggregate_rows([r for r in rows if r.get("effect_id") != "__normal__"], ("variant", "stress", "case_id"))

def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    flat_rows = []
    for r in rows:
        rr = {k: v for k, v in r.items() if k != "predicted_event"}
        flat_rows.append(rr)
    keys = sorted({k for r in flat_rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat_rows)

def save_outputs(out_dir: str, rows: List[Dict[str, Any]], run_config: Dict[str, Any]) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    agg_variant_stress = aggregate_rows(rows, ("variant", "stress"))
    agg_variant = aggregate_rows(rows, ("variant",))
    agg_stress = aggregate_rows(rows, ("stress",))
    by_case = case_summary(rows)
    deltas = compute_ablation_deltas(agg_variant_stress)
    full_overall = [r for r in agg_variant if r.get("variant") == "full"]
    overall = full_overall[0] if full_overall else (agg_variant[0] if agg_variant else {})

    payload = {
        "version": EVAL_VERSION,
        "run_config": run_config,
        "overall_full_variant": overall,
        "aggregate_by_variant": agg_variant,
        "aggregate_by_stress": agg_stress,
        "aggregate_by_variant_stress": agg_variant_stress,
        "ablation_deltas_vs_full": deltas,
        "rows": rows,
    }
    with open(os.path.join(out_dir, "lab_ontologyEvaluation_v8_1_2_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    save_csv(os.path.join(out_dir, "detailed_event_records.csv"), rows)
    save_csv(os.path.join(out_dir, "aggregate_by_variant.csv"), agg_variant)
    save_csv(os.path.join(out_dir, "aggregate_by_stress.csv"), agg_stress)
    save_csv(os.path.join(out_dir, "aggregate_by_variant_stress.csv"), agg_variant_stress)
    save_csv(os.path.join(out_dir, "ablation_delta_vs_full.csv"), deltas)
    save_csv(os.path.join(out_dir, "case_summary.csv"), by_case)
    return payload

def audit_no_leak() -> str:
    import inspect
    sig = inspect.signature(UniversalFieldToEventEncoder.update)
    forbidden = {"baseline", "baselines", "background", "backgrounds", "background_fields", "confidence", "gt", "gt_masks", "incident", "effect", "stress"}
    bad = [p for p in sig.parameters if p in forbidden]
    return f"encoder.update signature = {sig}; forbidden parameters = {bad}"

def maybe_apply_profile(args: argparse.Namespace) -> None:
    if args.profile == "quick":
        args.seeds = args.seeds if args.seeds != 10 else 2
        args.steps = args.steps if args.steps != 170 else 90
        args.max_cases = args.max_cases if args.max_cases is not None else 8
        args.stress_set = args.stress_set if args.stress_set != "core" else "baseline"
        args.variants = args.variants if args.variants != "all" else "full"
    elif args.profile == "smoke10":
        args.seeds = args.seeds if args.seeds != 10 else 10
        args.steps = args.steps if args.steps != 170 else 120
        args.max_cases = args.max_cases if args.max_cases is not None else 8
        args.stress_set = args.stress_set if args.stress_set != "core" else "core"
        args.variants = args.variants if args.variants != "all" else "all"
    elif args.profile == "paper":
        # Full paper-oriented default: 10 seeds, all 90 effect cases + normal, core stress set, all ablations.
        pass
    else:
        raise ValueError(args.profile)

def main() -> None:
    ap = argparse.ArgumentParser(description=f"{EVAL_VERSION} no-leak ontology evaluation and ablation experiments")
    ap.add_argument("--profile", choices=["quick", "smoke10", "paper"], default="paper")
    ap.add_argument("--seeds", type=int, default=10, help="number of random seeds / episodes; paper minimum is 10+")
    ap.add_argument("--episodes-per-case", type=int, default=None, help="backward-compatible alias for --seeds")
    ap.add_argument("--steps", type=int, default=170)
    ap.add_argument("--obs-ratio", type=float, default=DEFAULT_OBS_RATIO)
    ap.add_argument("--stress-set", choices=["baseline", "core", "all"], default="core")
    ap.add_argument("--variants", type=str, default="all", help="full, all, or comma-separated variant names")
    ap.add_argument("--max-cases", type=int, default=None, help="limit number of non-normal combo cases; None means all")
    ap.add_argument("--all-fields", action="store_true", help="feed all registered fields rather than only affected fields")
    ap.add_argument("--out-dir", type=str, default="/mnt/data/universal_encoder_v8_outputs")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--print-summary", action="store_true", help="keep backward-compatible coarse progress messages")
    ap.add_argument("--progress", choices=["on", "off"], default="on", help="show a dependency-free progress bar with ETA; default: on")
    ap.add_argument("--progress-interval", type=float, default=2.0, help="minimum seconds between progress updates")
    ap.add_argument("--progress-width", type=int, default=32, help="character width of the progress bar")
    args = ap.parse_args()
    if args.episodes_per_case is not None:
        args.seeds = args.episodes_per_case
    maybe_apply_profile(args)

    base_cases = make_combo_cases()
    effect_cases = [c for c in base_cases if c.effects]
    normal_cases = [c for c in base_cases if not c.effects]
    if args.max_cases is not None and args.max_cases < len(effect_cases):
        # Evenly spaced sampling keeps polarity/shape/trend coverage in small smoke runs,
        # instead of taking only the first shape block.
        idxs = np.linspace(0, len(effect_cases) - 1, args.max_cases).round().astype(int).tolist()
        seen = set()
        effect_cases = [effect_cases[i] for i in idxs if not (i in seen or seen.add(i))]
    cases = effect_cases + normal_cases
    stresses = make_stress_configs(args.stress_set)
    variants = make_encoder_variants(args.variants)

    all_rows: List[Dict[str, Any]] = []
    global_t0 = time.perf_counter()
    total_jobs = args.seeds * len(stresses) * len(variants) * len(cases)
    job = 0
    progress = ProgressMeter(
        total_jobs,
        enabled=(args.progress == "on"),
        width=args.progress_width,
        interval_sec=args.progress_interval,
    )
    progress.update(0, "starting")
    for epi in range(args.seeds):
        for stress_idx, stress in enumerate(stresses):
            for variant_idx, (variant_name, kwargs) in enumerate(variants.items()):
                for case_idx, case in enumerate(cases):
                    job += 1
                    seed = args.seed + 100000 * epi + 1000 * stress_idx + 100 * variant_idx + case_idx
                    rows = evaluate_case(
                        case, epi, args.steps, args.obs_ratio, seed, stress, variant_name, kwargs,
                        all_fields=args.all_fields,
                    )
                    all_rows.extend(rows)
                    status = f"seed={epi + 1}/{args.seeds} stress={stress.name} variant={variant_name} case={case.case_id}"
                    progress.update(job, status)
                    if args.print_summary and (job % max(1, total_jobs // 10) == 0 or job == total_jobs):
                        print(f"progress {job}/{total_jobs} jobs | {status}")

    run_config = {
        "version": EVAL_VERSION,
        "profile": args.profile,
        "seeds": args.seeds,
        "steps": args.steps,
        "obs_ratio": args.obs_ratio,
        "stress_set": args.stress_set,
        "stress_names": [s.name for s in stresses],
        "variants": list(variants.keys()),
        "n_cases": len(cases),
        "n_effect_cases": len(effect_cases),
        "include_normal_case": bool(normal_cases),
        "all_fields": bool(args.all_fields),
        "base_seed": args.seed,
        "progress": args.progress,
        "progress_interval": args.progress_interval,
        "progress_width": args.progress_width,
        "audit_no_leak": audit_no_leak(),
        "wall_time_sec": round(float(time.perf_counter() - global_t0), 3),
    }
    payload = save_outputs(args.out_dir, all_rows, run_config)

    print(f"\n===== Ontology Evaluation and Ablation {EVAL_VERSION} =====")
    print(f"version={EVAL_VERSION} core={CORE_VERSION} profile={args.profile} seeds={args.seeds} steps={args.steps} cases={len(cases)} stresses={len(stresses)} variants={len(variants)}")
    print("NO-LEAK CHECK: encoder.update(t, current_fields) only; no clean background/baseline, confidence, incident specs, stress config, or GT masks enter the encoder.")
    print(audit_no_leak())
    print("\nFull variant overall:")
    for k, v in payload.get("overall_full_variant", {}).items():
        if k in {"variant", "detection_f1", "mean_iou", "centroid_error", "shape_accuracy", "trend_accuracy", "normal_fp_per_100_frames", "extra_event_fp_per_100_effect_frames", "false_positive_per_100_frames", "latency_steps", "latency_ms_per_frame", "n_effect_rows"}:
            print(f"  {k}: {v}")
    print("\nSaved:")
    for name in [
        "lab_ontologyEvaluation_v8_1_2_results.json",
        "run_config.json",
        "detailed_event_records.csv",
        "aggregate_by_variant.csv",
        "aggregate_by_stress.csv",
        "aggregate_by_variant_stress.csv",
        "ablation_delta_vs_full.csv",
        "case_summary.csv",
    ]:
        print(f"  {os.path.join(args.out_dir, name)}")

if __name__ == "__main__":
    main()
