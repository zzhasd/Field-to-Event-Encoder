"""Full-encoder grid scaling/time-complexity experiment (no-leak).

The 30x30 physical domain is resolution-refined through 210x210. Obstacle
topology, backgrounds, anomaly centers/radii, and motion scale together.
Only mixed current fields enter encoder.update().
"""
from __future__ import annotations
import argparse, csv, inspect, json, math, os, time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from scipy.ndimage import label
from University_Field_to_Event_Encoder import (
    CORE_VERSION, DEFAULT_OBS_RATIO, FIELD_REGISTRY, FIELDS, EffectSpec,
    TestCase, UniversalFieldToEventEncoder, nearest_free,
)
from lab_ontologyEvaluation_and_ablation import (
    ProgressMeter, make_combo_cases, precision_recall_f1,
)

EXPERIMENT_VERSION = "v1.1.0"
REFERENCE_GRID_SIZE = 30
DEFAULT_GRID_SIZES = (30, 90, 150, 210)
FULL_ENCODER_FLAGS = (
    "use_background_ema", "use_cumulative_score", "use_hysteresis",
    "use_tracking", "use_morphology_debounce",
    "use_global_drift_compensation", "use_temporal_imputation",
    "use_velocity_tracking",
)
PRIMARY_METRICS = (
    "detection_f1", "mean_iou", "centroid_error", "shape_accuracy",
    "trend_accuracy", "fp_per_100_frames", "latency_steps",
    "latency_ms_per_frame",
)


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    return float(np.mean(xs)) if xs else None


def sample_std(values: Iterable[Optional[float]]) -> Optional[float]:
    """Sample standard deviation across independent seed/round aggregates."""
    xs = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    return float(np.std(xs, ddof=1)) if len(xs) >= 2 else 0.0 if xs else None


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter, union = int((a & b).sum()), int((a | b).sum())
    return inter / union if union else 0.0


def save_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, "w", encoding="utf-8").close()
        return
    keys = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def make_reference_grid(obs_ratio: float, seed: int) -> np.ndarray:
    """Connected 30x30 map with actual obstacle fraction equal to obs_ratio."""
    n = REFERENCE_GRID_SIZE
    target = int(round(obs_ratio * n * n))
    boundary = np.zeros((n, n), dtype=bool)
    boundary[0, :] = boundary[-1, :] = True
    boundary[:, 0] = boundary[:, -1] = True
    wall_count = int(boundary.sum())
    if target < wall_count:
        raise ValueError(
            f"obs_ratio must be >= {wall_count/(n*n):.6f} for boundary walls"
        )
    if target >= n * n:
        raise ValueError("obs_ratio leaves no free cells")
    candidates = np.argwhere(~boundary)
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        grid = boundary.astype(np.uint8)
        extra = target - wall_count
        if extra:
            pts = candidates[rng.choice(len(candidates), extra, replace=False)]
            grid[pts[:, 0], pts[:, 1]] = 1
        _, count = label(grid == 0, structure=np.ones((3, 3), dtype=np.uint8))
        if count == 1:
            return grid
    raise RuntimeError("failed to generate a connected exact-density grid")


def scale_grid(reference: np.ndarray, size: int) -> np.ndarray:
    if size < 30 or size % 30:
        raise ValueError("grid sizes must be positive multiples of 30")
    factor = size // 30
    return np.repeat(np.repeat(reference, factor, axis=0), factor, axis=1)


def physical_planes(size: int) -> Tuple[np.ndarray, np.ndarray, float]:
    scale = size / 30.0
    rr, cc = np.indices((size, size), dtype=np.float32)
    return (rr + 0.5) / scale - 0.5, (cc + 0.5) / scale - 0.5, scale


def physical_backgrounds(
    grid: np.ndarray, field_keys: Tuple[str, ...], t: int,
) -> Dict[str, np.ndarray]:
    """Original backgrounds evaluated in resolution-neutral coordinates."""
    rr, cc, _ = physical_planes(grid.shape[0])
    phase = 2 * np.pi * t / 240.0
    out: Dict[str, np.ndarray] = {}
    if "temperature" in field_keys:
        out["temperature"] = 22 + .055*cc + .035*rr + .35*np.sin(phase) + .25*np.sin((rr+cc)/8.5)
    if "humidity" in field_keys:
        out["humidity"] = 55 - .035*cc + .025*rr - .25*np.sin(phase*.9) + .25*np.cos((rr-cc)/10)
    if "pressure" in field_keys:
        out["pressure"] = 5 + .012*cc - .010*rr + .10*np.sin(phase*.7)
    if "co2" in field_keys:
        out["co2"] = 430 + 1.2*rr + .6*cc + 7*np.sin(phase*.5)
    if "air_quality" in field_keys:
        out["air_quality"] = 35 + .25*rr + .18*cc + 1.5*np.sin(phase*.6)
    for key, value in out.items():
        value = value.astype(np.float32)
        value[grid == 1] = np.nan
        out[key] = value
    return out


def scale_case(case: TestCase, grid: np.ndarray) -> TestCase:
    """Scale spatial support; preserve polarity, amplitude, and time laws."""
    scale = grid.shape[0] / 30.0
    effects: List[EffectSpec] = []
    for source in case.effects:
        data = asdict(source)
        data["center"] = tuple((float(x) + .5) * scale - .5 for x in source.center)
        data["radius"] = float(source.radius) * scale
        data["velocity"] = tuple(float(x) * scale for x in source.velocity)
        effect = EffectSpec(**data)
        effect.center = tuple(map(float, nearest_free(grid, effect.center)))
        effects.append(effect)
    return TestCase(case.case_id, effects)


def build_fields(
    grid: np.ndarray, testcase: TestCase, t: int, field_keys: Tuple[str, ...],
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """Simulator/evaluator function. Ground truth never leaves this side."""
    current = physical_backgrounds(grid, field_keys, t)
    truth: List[Dict[str, Any]] = []
    for effect in testcase.effects:
        if effect.field_key not in current:
            continue
        mask, delta = effect.mask_and_delta(grid, t)
        current[effect.field_key] = (current[effect.field_key] + delta).astype(np.float32)
        if mask.any() and effect.is_active(t):
            rr, cc = np.indices(grid.shape)
            truth.append({
                "effect_id": effect.effect_id, "field_key": effect.field_key,
                "polarity": effect.polarity, "mask": mask,
                "centroid": (float(rr[mask].mean()), float(cc[mask].mean())),
            })
    return current, truth


def assert_full_encoder(encoder: UniversalFieldToEventEncoder) -> None:
    disabled = [k for k in FULL_ENCODER_FLAGS if not getattr(encoder, k, False)]
    if disabled:
        raise AssertionError(f"full encoder modules disabled: {disabled}")


def audit_no_leak() -> str:
    sig = inspect.signature(UniversalFieldToEventEncoder.update)
    forbidden = {
        "baseline", "background", "confidence", "gt", "gt_masks",
        "incident", "effect", "stress",
    }
    bad = sorted(forbidden & set(sig.parameters))
    return f"encoder.update signature = {sig}; forbidden parameters = {bad}"


def evaluate_case(
    base_case: TestCase, grid: np.ndarray, size: int, round_no: int, seed: int,
    steps: int, metric_warmup: int, match_threshold: float, field_mode: str,
) -> Dict[str, Any]:
    scale = size / 30.0
    testcase = scale_case(base_case, grid)
    field_keys = (
        tuple(FIELDS) if field_mode == "all" or not testcase.effects
        else tuple(sorted({e.field_key for e in testcase.effects}))
    )
    encoder = UniversalFieldToEventEncoder(grid, FIELD_REGISTRY)
    assert_full_encoder(encoder)
    state = {
        e.effect_id: {"first": None, "last": None} for e in testcase.effects
    }
    tp = fp = normal_fp = extra_fp = fn = 0
    counted = normal_frames = effect_frames = 0
    iou_sum = centroid_sum = centroid30_sum = 0.0
    matches = 0
    encoder_times: List[int] = []
    wall_start = time.perf_counter_ns()

    for t in range(steps):
        current, truth = build_fields(grid, testcase, t, field_keys)
        t0 = time.perf_counter_ns()
        events = encoder.update(t, current)
        encoder_times.append(time.perf_counter_ns() - t0)
        matched_ids = set()
        for gt in truth:
            candidates = [
                (idx, ev, iou(gt["mask"], ev["mask"]))
                for idx, ev in enumerate(events)
                if ev["field_key"] == gt["field_key"]
                and ev["polarity"] == gt["polarity"]
            ]
            idx, event, overlap = (
                max(candidates, key=lambda x: x[2])
                if candidates else (-1, None, 0.0)
            )
            if event is not None and overlap >= match_threshold:
                tp += 1
                matched_ids.add(idx)
                error = float(np.linalg.norm(
                    np.asarray(gt["centroid"]) - np.asarray(event["centroid"])
                ))
                iou_sum += overlap
                centroid_sum += error
                centroid30_sum += error / scale
                matches += 1
                item = state[gt["effect_id"]]
                if item["first"] is None:
                    item["first"] = t
                item["last"] = {k: v for k, v in event.items() if k != "mask"}
            else:
                fn += 1
        if t >= metric_warmup:
            counted += 1
            if truth:
                effect_frames += 1
            else:
                normal_frames += 1
            unmatched = sum(i not in matched_ids for i in range(len(events)))
            fp += unmatched
            if truth:
                extra_fp += unmatched
            else:
                normal_fp += unmatched

    wall_ns = time.perf_counter_ns() - wall_start
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    shape_sum = trend_sum = 0.0
    latencies: List[float] = []
    predicted_shapes: List[Optional[str]] = []
    for effect in testcase.effects:
        item, event = state[effect.effect_id], state[effect.effect_id]["last"]
        predicted_shapes.append(event.get("morphology") if event else None)
        if event:
            shape_sum += float(event.get("morphology") == effect.shape)
            temporal = event.get("temporal_summary", {})
            trend_sum += (
                float(temporal.get("area_trend") == effect.area_trend)
                + float(temporal.get("intensity_trend") == effect.intensity_trend)
            ) / 2
        if item["first"] is not None:
            latencies.append(float(item["first"] - effect.start))

    effect_count = len(testcase.effects)
    encoder_ns = int(sum(encoder_times))
    times_ms = np.asarray(encoder_times, dtype=float) / 1e6
    return {
        "variant": "full", "grid_size": size, "cell_count": size*size,
        "scale_vs_30": scale, "round": round_no, "seed": seed,
        "case_id": testcase.case_id, "is_normal_case": int(not effect_count),
        "field_mode": field_mode, "field_count": len(field_keys),
        "steps": steps, "metric_warmup": metric_warmup,
        "obstacle_density": float(grid.mean()),
        "tp": tp, "fp": fp, "normal_fp": normal_fp,
        "extra_event_fp": extra_fp, "fn": fn,
        "counted_frame_count": counted, "normal_frame_count": normal_frames,
        "effect_frame_count": effect_frames, "frame_count": steps,
        "detection_precision": precision, "detection_recall": recall,
        "detection_f1": f1, "iou_sum": iou_sum,
        "centroid_error_sum": centroid_sum,
        "centroid_error_base30_sum": centroid30_sum, "match_count": matches,
        "mean_iou": iou_sum/matches if matches else None,
        "centroid_error": centroid_sum/matches if matches else None,
        "centroid_error_base30": centroid30_sum/matches if matches else None,
        "effect_count": effect_count,
        "detected_effect_count": sum(
            state[e.effect_id]["first"] is not None for e in testcase.effects
        ),
        "shape_match_sum": shape_sum, "trend_match_sum": trend_sum,
        "shape_accuracy": shape_sum/effect_count if effect_count else None,
        "trend_accuracy": trend_sum/effect_count if effect_count else None,
        "latency_steps_sum": sum(latencies),
        "latency_steps_count": len(latencies),
        "latency_steps": safe_mean(latencies),
        "fp_per_100_frames": 100*fp/max(1, counted),
        "encoder_time_ns_total": encoder_ns,
        "latency_ms_per_frame": encoder_ns/1e6/max(1, steps),
        "encoder_latency_p50_ms": float(np.percentile(times_ms, 50)),
        "encoder_latency_p95_ms": float(np.percentile(times_ms, 95)),
        "wall_time_ns_total": wall_ns,
        "wall_ms_per_frame": wall_ns/1e6/max(1, steps),
        "predicted_shapes": json.dumps(predicted_shapes),
    }


def aggregate(
    records: Sequence[Dict[str, Any]], keys: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(tuple(record[k] for k in keys), []).append(record)
    output: List[Dict[str, Any]] = []
    for group, rows in groups.items():
        def total(name: str) -> float:
            return sum(float(row.get(name, 0)) for row in rows)

        tp, fp, fn = (int(total(k)) for k in ("tp", "fp", "fn"))
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        matches, effects = int(total("match_count")), int(total("effect_count"))
        latency_n = int(total("latency_steps_count"))
        frames, counted = int(total("frame_count")), int(total("counted_frame_count"))
        encoder_ns = int(total("encoder_time_ns_total"))
        row = dict(zip(keys, group))
        row.update({
            "variant": "full", "n_case_runs": len(rows),
            "cell_count": int(rows[0]["cell_count"]),
            "scale_vs_30": rows[0]["scale_vs_30"],
            "obstacle_density": safe_mean(r["obstacle_density"] for r in rows),
            "tp": tp, "fp": fp, "normal_fp": int(total("normal_fp")),
            "extra_event_fp": int(total("extra_event_fp")), "fn": fn,
            "detection_precision": precision, "detection_recall": recall,
            "detection_f1": f1,
            "mean_iou": total("iou_sum")/matches if matches else None,
            "centroid_error": total("centroid_error_sum")/matches if matches else None,
            "centroid_error_base30": (
                total("centroid_error_base30_sum")/matches if matches else None
            ),
            "shape_accuracy": total("shape_match_sum")/effects if effects else None,
            "trend_accuracy": total("trend_match_sum")/effects if effects else None,
            "fp_per_100_frames": 100*fp/max(1, counted),
            "latency_steps": (
                total("latency_steps_sum")/latency_n if latency_n else None
            ),
            "latency_ms_per_frame": encoder_ns/1e6/max(1, frames),
            "encoder_time_sec_total": encoder_ns/1e9,
            "wall_ms_per_frame": total("wall_time_ns_total")/1e6/max(1, frames),
            "frame_count": frames, "counted_frame_count": counted,
            "effect_count": effects,
            "detected_effect_count": int(total("detected_effect_count")),
            "match_count": matches,
            "mean_case_encoder_p50_ms": safe_mean(
                r["encoder_latency_p50_ms"] for r in rows
            ),
            "mean_case_encoder_p95_ms": safe_mean(
                r["encoder_latency_p95_ms"] for r in rows
            ),
        })
        output.append(row)
    return sorted(output, key=lambda row: tuple(row[k] for k in keys))


def analyze(grid_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in grid_rows if (r.get("latency_ms_per_frame") or 0) > 0]
    result: Dict[str, Any] = {
        "model": "latency_ms_per_frame = coefficient * grid_side ** exponent",
        "expected_exponent_for_per_cell_work": 2.0,
        "primary_metrics": list(PRIMARY_METRICS),
    }
    if len(valid) >= 2:
        x = np.log([r["grid_size"] for r in valid])
        y = np.log([r["latency_ms_per_frame"] for r in valid])
        exponent, intercept = np.polyfit(x, y, 1)
        predicted = exponent*x + intercept
        ss_res, ss_tot = np.sum((y-predicted)**2), np.sum((y-y.mean())**2)
        result.update({
            "side_length_exponent": float(exponent),
            "cell_count_exponent": float(exponent/2),
            "coefficient": float(math.exp(intercept)),
            "log_space_r_squared": float(1-ss_res/ss_tot) if ss_tot else None,
            "largest_vs_smallest_latency_ratio": float(
                valid[-1]["latency_ms_per_frame"]/valid[0]["latency_ms_per_frame"]
            ),
            "largest_vs_smallest_cell_ratio": float(
                valid[-1]["cell_count"]/valid[0]["cell_count"]
            ),
        })
    trends = {}
    for metric in PRIMARY_METRICS:
        points = [
            (r["grid_size"], r.get(metric))
            for r in grid_rows if r.get(metric) is not None
        ]
        if len(points) < 2:
            trends[metric] = None
        else:
            x, y = np.asarray(points, dtype=float).T
            trends[metric] = {
                "first": float(y[0]), "last": float(y[-1]),
                "last_minus_first": float(y[-1]-y[0]),
                "linear_change_per_30_side_cells": float(
                    np.polyfit(x, y, 1)[0]*30
                ),
            }
    result["metric_trends"] = trends
    return result


def summarize_round_statistics(
    round_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return the paper table as mean +/- sample SD across random seeds."""
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in round_rows:
        grouped.setdefault(int(row["grid_size"]), []).append(row)
    output: List[Dict[str, Any]] = []
    for size, rows in sorted(grouped.items()):
        item: Dict[str, Any] = {
            "grid_size": size,
            "n_rounds": len(rows),
        }
        for metric in ("detection_f1", "mean_iou", "latency_ms_per_frame"):
            values = [row.get(metric) for row in rows]
            item[f"{metric}_mean"] = safe_mean(values)
            item[f"{metric}_sample_std"] = sample_std(values)
        output.append(item)
    return output


def markdown_summary_table(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Grid L | Detection F1 ↑ | Mean IoU ↑ | Time (ms/step) ↓ |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {grid_size} | {detection_f1_mean:.4f} ± "
            "{detection_f1_sample_std:.4f} | {mean_iou_mean:.4f} ± "
            "{mean_iou_sample_std:.4f} | {latency_ms_per_frame_mean:.2f} ± "
            "{latency_ms_per_frame_sample_std:.2f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def select_cases(
    max_cases: Optional[int], normal: bool,
) -> Tuple[List[TestCase], int]:
    all_cases = make_combo_cases()
    cases = [c for c in all_cases if c.effects]
    if max_cases is not None and max_cases < len(cases):
        ids = np.linspace(0, len(cases)-1, max_cases).round().astype(int)
        cases = [cases[i] for i in dict.fromkeys(ids.tolist())]
    normal_cases = [c for c in all_cases if not c.effects] if normal else []
    return cases + normal_cases, len(cases)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Full encoder grid time complexity")
    ap.add_argument("--profile", choices=("quick", "paper"), default="paper")
    ap.add_argument(
        "--grid-sizes", nargs="+", type=int, default=list(DEFAULT_GRID_SIZES)
    )
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--max-cases", type=int)
    ap.add_argument("--no-normal", action="store_true")
    ap.add_argument("--obs-ratio", type=float, default=DEFAULT_OBS_RATIO)
    ap.add_argument("--field-mode", choices=("all", "affected"), default="all")
    ap.add_argument("--metric-warmup", type=int, default=40)
    ap.add_argument("--match-iou-threshold", type=float, default=.10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--out-dir",
        default=os.path.join("outputs", "full_encoder_time_complexity"),
    )
    ap.add_argument("--progress", choices=("on", "off"), default="on")
    ap.add_argument("--progress-interval", type=float, default=2.0)
    ap.add_argument("--progress-width", type=int, default=32)
    args = ap.parse_args()
    args.seeds = (1 if args.profile == "quick" else 10) if args.seeds is None else args.seeds
    args.steps = (90 if args.profile == "quick" else 170) if args.steps is None else args.steps
    if args.profile == "quick" and args.max_cases is None:
        args.max_cases = 8
    if args.seeds <= 0 or args.steps <= 0:
        ap.error("--seeds and --steps must be positive")
    if not 0 < args.obs_ratio < 1:
        ap.error("--obs-ratio must be in (0, 1)")
    if not 0 <= args.metric_warmup < args.steps:
        ap.error("--metric-warmup must be smaller than --steps")
    if args.max_cases is not None and args.max_cases <= 0:
        ap.error("--max-cases must be positive")
    if any(size < 30 or size % 30 for size in args.grid_sizes):
        ap.error("all grid sizes must be positive multiples of 30")
    args.grid_sizes = list(dict.fromkeys(args.grid_sizes))
    return args


def main() -> None:
    args = parse_args()
    cases, effect_cases = select_cases(args.max_cases, not args.no_normal)
    total_jobs = len(args.grid_sizes)*args.seeds*len(cases)
    progress = ProgressMeter(
        total_jobs, args.progress == "on", args.progress_width,
        args.progress_interval,
    )
    records: List[Dict[str, Any]] = []
    started, completed = time.perf_counter(), 0
    progress.update(0, "starting")
    for round_index in range(args.seeds):
        seed = args.seed + round_index
        reference = make_reference_grid(args.obs_ratio, seed)
        for size in args.grid_sizes:
            grid = scale_grid(reference, size)
            for case in cases:
                records.append(evaluate_case(
                    case, grid, size, round_index+1, seed, args.steps,
                    args.metric_warmup, args.match_iou_threshold,
                    args.field_mode,
                ))
                completed += 1
                progress.update(
                    completed,
                    f"grid={size} round={round_index+1}/{args.seeds} case={case.case_id}",
                )

    rounds = aggregate(records, ("grid_size", "round", "seed"))
    grids = aggregate(records, ("grid_size",))
    paper_table = summarize_round_statistics(rounds)
    analysis = analyze(grids)
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "core_version": CORE_VERSION,
        "profile": args.profile,
        "grid_sizes": args.grid_sizes,
        "reference_grid_size": 30,
        "spatial_scaling": (
            "background, obstacles, anomaly center/radius/motion "
            "scale proportionally"
        ),
        "obstacle_density_requested_and_actual": args.obs_ratio,
        "seeds": args.seeds, "base_seed": args.seed, "steps": args.steps,
        "metric_warmup": args.metric_warmup,
        "match_iou_threshold": args.match_iou_threshold,
        "field_mode": args.field_mode,
        "full_encoder_flags": list(FULL_ENCODER_FLAGS),
        "n_cases": len(cases), "n_effect_cases": effect_cases,
        "include_normal_case": not args.no_normal,
        "primary_metrics": list(PRIMARY_METRICS),
        "latency_definition": (
            "perf_counter_ns around encoder.update only / frames"
        ),
        "audit_no_leak": audit_no_leak(),
        "wall_time_sec": time.perf_counter()-started,
    }
    payload = {
        "experiment_version": EXPERIMENT_VERSION,
        "run_config": config,
        "complexity_analysis": analysis,
        "aggregate_by_grid_size": grids,
        "metrics_by_round": rounds,
        "paper_table_mean_sample_std": paper_table,
        "case_round_records": records,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    for filename, value in (
        ("full_encoder_time_complexity_results.json", payload),
        ("run_config.json", config),
        ("complexity_analysis.json", analysis),
    ):
        with open(
            os.path.join(args.out_dir, filename), "w", encoding="utf-8"
        ) as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
    save_csv(
        os.path.join(args.out_dir, "aggregate_by_grid_size.csv"), grids
    )
    save_csv(os.path.join(args.out_dir, "metrics_by_round.csv"), rounds)
    save_csv(
        os.path.join(args.out_dir, "paper_table_mean_sample_std.csv"),
        paper_table,
    )
    save_csv(
        os.path.join(args.out_dir, "case_round_records.csv"), records
    )
    with open(
        os.path.join(args.out_dir, "paper_table_mean_sample_std.md"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(markdown_summary_table(paper_table))

    print(f"\n===== Full Encoder Time Complexity {EXPERIMENT_VERSION} =====")
    print(
        f"core={CORE_VERSION} grids={args.grid_sizes} rounds={args.seeds} "
        f"cases={len(cases)} steps={args.steps}"
    )
    print("NO-LEAK; latency times encoder.update only.")
    print(audit_no_leak())
    print("\nGrid-size summary:")
    for row in grids:
        metrics = " ".join(
            f"{metric}={row.get(metric)}" for metric in PRIMARY_METRICS
        )
        print(f"  grid={row['grid_size']}: {metrics}")
    print("\nMean +/- sample SD across rounds:")
    print(markdown_summary_table(paper_table), end="")
    if "side_length_exponent" in analysis:
        print(
            f"\nLatency side-length exponent: "
            f"{analysis['side_length_exponent']:.4f} "
            "(per-cell reference 2.0)"
        )
    print(f"\nSaved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

