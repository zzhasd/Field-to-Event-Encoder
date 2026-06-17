"""Lab test suite for the Universal Field-to-Event Encoder.

The reusable encoder implementation lives in University_Field_to_Event_Encoder.py.
This lab script imports that core module, builds synthetic no-leak test cases,
evaluates detection/morphology/trend metrics, and writes results to
outputs/test_suite by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from University_Field_to_Event_Encoder import (
    GRID_SIZE,
    DEFAULT_OBS_RATIO,
    FIELDS,
    FIELD_REGISTRY,
    EffectSpec,
    make_obstacle_grid,
    nearest_free,
    random_free_cell,
    normal_backgrounds,
    UniversalFieldToEventEncoder,
)

@dataclass
class TestCase:
    case_id: str
    effects: List[EffectSpec]

def build_current_fields(grid: np.ndarray, testcase: TestCase, t: int, field_keys: Tuple[str, ...]) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    backgrounds = normal_backgrounds(grid, field_keys, t)
    current = {k: v.copy() for k, v in backgrounds.items()}
    gt_records: List[Dict[str, Any]] = []
    for eff in testcase.effects:
        if eff.field_key not in current:
            continue
        mask, delta = eff.mask_and_delta(grid, t)
        current[eff.field_key] = current[eff.field_key] + delta
        if mask.any() and eff.is_active(t):
            rr, cc = np.indices(grid.shape)
            gt_records.append({
                "effect_id": eff.effect_id,
                "field_key": eff.field_key,
                "polarity": eff.polarity,
                "shape": eff.shape,
                "area_trend": eff.area_trend,
                "intensity_trend": eff.intensity_trend,
                "mask": mask,
                "area": int(mask.sum()),
                "centroid": (float(rr[mask].mean()), float(cc[mask].mean())),
            })
    return current, gt_records


# ============================================================
# Test cases and evaluator
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
                    field_key = fields[(idx-1) % len(fields)]
                    angle = (idx * 0.47) % math.pi
                    center = (15.0 + 4.0*math.sin(idx*0.8), 15.0 + 4.0*math.cos(idx*0.65))
                    eff = EffectSpec(
                        effect_id=f"E{idx:03d}", field_key=field_key, polarity=pol, shape=shape,
                        area_trend=at, intensity_trend=it, center=center, start=25, end=150,
                        amplitude_sigma=3.5 if shape != "point_like" else 4.2,
                        radius=shape_radius(shape), angle=angle, axis_ratio=2.4,
                    )
                    cases.append(TestCase(f"combo_{idx:03d}_{pol}_{shape}_{at}_{it}", [eff]))
    cases.append(TestCase("normal_no_incident", []))
    return cases

def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = int((mask_a & mask_b).sum())
    union = int((mask_a | mask_b).sum())
    return inter / union if union else 0.0

def centroid_error(c0: Tuple[float, float], c1: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(c0, dtype=float) - np.array(c1, dtype=float)))

def serializable_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in ev.items() if k != "mask"}
    return out

def evaluate_case(testcase: TestCase, episode: int, steps: int, obs_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], int]:
    grid = make_obstacle_grid(GRID_SIZE, obs_ratio, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    for eff in testcase.effects:
        eff.center = tuple(map(float, nearest_free(grid, eff.center)))
        if rng.random() < 0.25:
            eff.center = tuple(map(float, random_free_cell(grid, rng, margin=4)))
    encoder = UniversalFieldToEventEncoder(grid, FIELD_REGISTRY)
    events_by_t: Dict[int, List[Dict[str, Any]]] = {}
    gt_by_eff: Dict[str, Dict[str, Any]] = {}
    normal_fp = 0
    field_keys = tuple(sorted({e.field_key for e in testcase.effects} or {"temperature", "humidity"}))
    for t in range(steps):
        current, gt_records = build_current_fields(grid, testcase, t, field_keys)
        events = encoder.update(t, current)
        events_by_t[t] = events
        if not testcase.effects and t > 40:
            normal_fp += len(events)
        for gt in gt_records:
            gt_by_eff[gt["effect_id"]] = {**gt, "t": t}
    rows: List[Dict[str, Any]] = []
    for eff in testcase.effects:
        gt = gt_by_eff.get(eff.effect_id)
        if gt is None:
            rows.append({"case_id": testcase.case_id, "episode": episode, "effect_id": eff.effect_id, "detected": 0, "pass_basic": 0})
            continue
        cand = [ev for ev in events_by_t.get(gt["t"], []) if ev["field_key"] == eff.field_key and ev["polarity"] == eff.polarity]
        best_ev = None
        best_iou = -1.0
        for ev in cand:
            val = iou(gt["mask"], ev["mask"])
            if val > best_iou:
                best_iou = val
                best_ev = ev
        if best_ev is None:
            rows.append({"case_id": testcase.case_id, "episode": episode, "effect_id": eff.effect_id, "detected": 0, "pass_basic": 0})
            continue
        c_err = centroid_error(gt["centroid"], best_ev["centroid"])
        tmp = best_ev.get("temporal_summary", {})
        shape_match = int(best_ev["morphology"] == eff.shape)
        area_match = int(tmp.get("area_trend") == eff.area_trend)
        int_match = int(tmp.get("intensity_trend") == eff.intensity_trend)
        pass_basic = int(best_iou >= 0.45 and c_err <= 4.0)
        rows.append({
            "case_id": testcase.case_id,
            "episode": episode,
            "effect_id": eff.effect_id,
            "field_key": eff.field_key,
            "polarity": eff.polarity,
            "expected_shape": eff.shape,
            "predicted_shape": best_ev["morphology"],
            "expected_area_trend": eff.area_trend,
            "predicted_area_trend": tmp.get("area_trend"),
            "expected_intensity_trend": eff.intensity_trend,
            "predicted_intensity_trend": tmp.get("intensity_trend"),
            "detected": 1,
            "pass_basic": pass_basic,
            "iou": round(float(best_iou), 6),
            "centroid_error": round(float(c_err), 6),
            "shape_match": shape_match,
            "area_trend_match": area_match,
            "intensity_trend_match": int_match,
            "predicted_event": serializable_event(best_ev),
        })
    return rows, normal_fp

def summarize(rows: List[Dict[str, Any]], normal_fp_total: int) -> Dict[str, Any]:
    valid = [r for r in rows if "detected" in r]
    detected = [r for r in valid if r.get("detected") == 1]
    def mean(key: str, data: List[Dict[str, Any]]) -> Optional[float]:
        vals = [float(r[key]) for r in data if key in r and r[key] is not None]
        return float(np.mean(vals)) if vals else None
    return {
        "normal_false_positive_total": int(normal_fp_total),
        "n_effect_records": len(valid),
        "detection_rate": mean("detected", valid),
        "pass_basic_rate": mean("pass_basic", valid),
        "mean_iou": mean("iou", detected),
        "median_iou": float(np.median([float(r["iou"]) for r in detected])) if detected else None,
        "mean_centroid_error": mean("centroid_error", detected),
        "shape_match_rate": mean("shape_match", detected),
        "area_trend_match_rate": mean("area_trend_match", detected),
        "intensity_trend_match_rate": mean("intensity_trend_match", detected),
    }

def case_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["case_id"], []).append(r)
    out = []
    for cid, rs in by.items():
        det = [r for r in rs if r.get("detected") == 1]
        def avg(k):
            vals = [float(r[k]) for r in det if k in r]
            return float(np.mean(vals)) if vals else None
        out.append({
            "case_id": cid,
            "n": len(rs),
            "pass": float(np.mean([r.get("pass_basic", 0) for r in rs])) if rs else None,
            "mean_iou": avg("iou"),
            "centroid": avg("centroid_error"),
            "shape": avg("shape_match"),
            "area_trend": avg("area_trend_match"),
            "int_trend": avg("intensity_trend_match"),
        })
    out.sort(key=lambda r: (999 if r["mean_iou"] is None else r["mean_iou"]))
    return out

def save_outputs(out_dir: str, rows: List[Dict[str, Any]], summary: Dict[str, Any], by_case: List[Dict[str, Any]]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "universal_encoder_test_results.json")
    csv_path = os.path.join(out_dir, "universal_encoder_test_results.csv")
    sum_path = os.path.join(out_dir, "universal_encoder_case_summary.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"overall": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    flat_rows = []
    for r in rows:
        rr = {k: v for k, v in r.items() if k != "predicted_event"}
        flat_rows.append(rr)
    keys = sorted({k for r in flat_rows for k in r.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(flat_rows)
    keys2 = sorted({k for r in by_case for k in r.keys()})
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys2)
        w.writeheader(); w.writerows(by_case)

def audit_no_leak() -> str:
    import inspect
    sig = inspect.signature(UniversalFieldToEventEncoder.update)
    forbidden = {"baseline", "baselines", "background", "backgrounds", "background_fields", "confidence", "gt", "gt_masks", "incident", "effect"}
    bad = [p for p in sig.parameters if p in forbidden]
    return f"encoder.update signature = {sig}; forbidden parameters = {bad}"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-per-case", type=int, default=1)
    ap.add_argument("--steps", type=int, default=170)
    ap.add_argument("--obs-ratio", type=float, default=DEFAULT_OBS_RATIO)
    ap.add_argument("--out-dir", type=str, default="outputs/test_suite")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--print-failures", action="store_true")
    args = ap.parse_args()

    cases = make_combo_cases()
    all_rows: List[Dict[str, Any]] = []
    normal_fp_total = 0
    for epi in range(args.episodes_per_case):
        for i, case in enumerate(cases):
            effects = [EffectSpec(**asdict(e)) for e in case.effects]
            tc = TestCase(case.case_id, effects)
            rows, fp = evaluate_case(tc, epi, args.steps, args.obs_ratio, args.seed + 10000*epi + i)
            all_rows.extend(rows)
            normal_fp_total += fp
    overall = summarize(all_rows, normal_fp_total)
    by_case = case_summary(all_rows)
    save_outputs(args.out_dir, all_rows, overall, by_case)

    print("\n===== Universal Encoder Lab Test Suite =====")
    print(f"cases={len(cases)}, episodes_per_case={args.episodes_per_case}, steps={args.steps}, obs_ratio={args.obs_ratio}")
    print("NO-LEAK CHECK: encoder.update(t, current_fields) only; no clean background/baseline, confidence, incident specs, or GT masks enter the encoder.")
    print(audit_no_leak())
    print("\nOverall:")
    for k, v in overall.items():
        print(f"  {k}: {v}")
    print("\nWorst cases by mean IoU/pass rate:")
    for r in by_case[:8]:
        print(f"  {r['case_id']}: pass={r['pass']}, mean_iou={r['mean_iou']}, centroid={r['centroid']}, shape={r['shape']}, area_trend={r['area_trend']}, int_trend={r['int_trend']}")
    if args.print_failures:
        fails = [r for r in all_rows if r.get("pass_basic") == 0 or r.get("shape_match") == 0 or r.get("area_trend_match") == 0 or r.get("intensity_trend_match") == 0]
        print(f"\nWeak/failure records: {len(fails)}")
        for r in fails[:40]:
            print(f"  {r.get('case_id')} {r.get('effect_id')}: det={r.get('detected')} pass={r.get('pass_basic')} iou={r.get('iou')} shape {r.get('expected_shape')}->{r.get('predicted_shape')} area {r.get('expected_area_trend')}->{r.get('predicted_area_trend')} int {r.get('expected_intensity_trend')}->{r.get('predicted_intensity_trend')}")
    print("\nSaved:")
    print(f"  {os.path.join(args.out_dir, 'universal_encoder_test_results.json')}")
    print(f"  {os.path.join(args.out_dir, 'universal_encoder_test_results.csv')}")
    print(f"  {os.path.join(args.out_dir, 'universal_encoder_case_summary.csv')}")

if __name__ == "__main__":
    main()