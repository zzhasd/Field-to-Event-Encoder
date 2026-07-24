"""Comprehensive comparison experiments for Field-to-Event (F2E) + LLM systems.

Version: v8.8.0 parameter-free adaptive evidence graph with audit-only protocol checks

This script is intentionally an experiment/evaluation layer only.  It imports the
core no-leak F2E encoder from University_Field_to_Event_Encoder.py and never
passes clean backgrounds, GT masks, injected labels, or accident labels into the
encoder.

Layer 2 comprehensive comparison
--------------------------------
This entry point runs only the seven high-level diagnosis methods: F2E graph +
LLM, F2E + LLM, threshold + LLM, raw matrix summary + LLM, threshold + rules,
F2E + rules, and raw field image + VLM. Low-level detectors are retained only
as representation generators; no separate low-level experiment is executed.

API safety
----------
The script reads DASHSCOPE_API_KEY from the environment.  It never stores or
prints the key.  API calls are optional: --api-mode offline/auto/api.

Token accounting
----------------
DashScope/OpenAI-compatible Qwen responses are measured from the official
``usage`` object whenever it is present.  Optional tokenizer/heuristic fallbacks
are recorded with explicit source flags so estimated counts are never mixed with
API-measured counts silently.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import heapq
import io
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import warnings
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label

from University_Field_to_Event_Encoder import (
    CORE_VERSION,
    DEFAULT_OBS_RATIO,
    FIELD_REGISTRY,
    FIELDS,
    GRID_SIZE,
    EffectSpec,
    StressConfig,
    TestCase,
    UniversalFieldToEventEncoder,
    build_current_fields,
    make_obstacle_grid,
    nearest_free,
)

COMPARISON_VERSION = "v8.8.0-parameter-free-adaptive-evidence-graph-audit-only-parallel"

ACCIDENT_TYPES = [
    "fire",
    "electrical_overheat",
    "water_leak",
    "steam_leak",
    "co2_accumulation",
    "dust_pollution",
    "composite_anomaly",
    "low_snr_anomaly",
    "needs_review_unknown",
    "normal",
]

# ============================================================
# Basic utilities
# ============================================================
def fmt_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

class ProgressMeter:
    def __init__(self, total: int, enabled: bool = True, width: int = 32, interval_sec: float = 2.0) -> None:
        self.total = max(1, int(total))
        self.enabled = enabled
        self.width = max(8, int(width))
        self.interval_sec = max(0.1, float(interval_sec))
        self.start = time.perf_counter()
        self.last_print = 0.0
        self.is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())

    def update(self, done: int, status: str = "") -> None:
        if not self.enabled:
            return
        done = max(0, min(int(done), self.total))
        now = time.perf_counter()
        if done not in {0, 1, self.total} and (now - self.last_print) < self.interval_sec:
            return
        self.last_print = now
        frac = done / self.total
        filled = int(round(self.width * frac))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.start
        rate = done / elapsed if elapsed > 1e-9 else 0.0
        eta = (self.total - done) / rate if rate > 1e-9 else float("nan")
        line = f"[{bar}] {done}/{self.total} ({100*frac:6.2f}%) elapsed={fmt_duration(elapsed)} ETA={fmt_duration(eta)} rate={rate:.2f} jobs/s"
        if status:
            line += " | " + (status[:72] + "..." if len(status) > 75 else status)
        if self.is_tty:
            sys.stderr.write("\r" + line)
            if done >= self.total:
                sys.stderr.write("\n")
            sys.stderr.flush()
        else:
            print(line, file=sys.stderr, flush=True)

def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def safe_mean(values: Iterable[Any]) -> Optional[float]:
    xs = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            xs.append(fv)
    return float(np.mean(xs)) if xs else None

def metric_ms(value: Any) -> str:
    return "NA" if value is None else f"{value}ms"

def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = int((mask_a & mask_b).sum())
    union = int((mask_a | mask_b).sum())
    return inter / union if union else 0.0

def centroid_error(c0: Tuple[float, float], c1: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(c0, dtype=float) - np.array(c1, dtype=float)))

def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    p = tp / (tp + fp) if (tp + fp) > 0 else None
    r = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = None if p is None or r is None or (p + r) == 0 else 2 * p * r / (p + r)
    return p, r, f1


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Optional[float]:
    """Strict multiclass macro-F1 over labels that appear in truth or predictions.

    Older experiment branches skipped labels whose F1 was undefined, which could
    overstate macro-F1 when a method completely missed a class.  This version
    assigns F1=0 to any label with support or prediction but no true positives,
    matching the usual paper/reporting expectation for sparse multiclass tests.
    """
    scores = []
    for lab in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b != lab)
        support_or_prediction = (tp + fp + fn) > 0
        if not support_or_prediction:
            continue
        if tp == 0:
            scores.append(0.0)
            continue
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
    return float(np.mean(scores)) if scores else None


# ============================================================
# Scenario ontology
# ============================================================
def shape_radius(shape: str) -> float:
    return {"point_like": 1.2, "compact_blob": 2.0, "blob": 3.4, "oval": 2.3, "elongated_strip": 2.3}[shape]

@dataclass
class AccidentScenario:
    scenario_id: str
    accident_type: str
    testcase: TestCase
    stress: StressConfig = field(default_factory=StressConfig)
    ambiguous: bool = False
    unseen_combo: bool = False
    expected_review: bool = False
    explanation_fields: Tuple[str, ...] = ()
    target_field: Optional[str] = None
    notes: str = ""
    challenge: str = ""
    llm_advantage: str = ""
    hard_family: str = "standard"
    diagnostic_goal: str = "exact_label"
    expected_event_signatures: Tuple[str, ...] = ()
    forbidden_predictions: Tuple[str, ...] = ()
    acceptable_predictions: Tuple[str, ...] = ()

def event_signature(field_key: str, polarity: str) -> str:
    return f"{field_key}:{polarity}"

def effect(eid: str, field_key: str, polarity: str, shape: str, center: Tuple[float, float],
           area: str = "stable", intensity: str = "stable", amp: float = 3.5,
           start: int = 25, end: int = 150, angle: float = 0.3) -> EffectSpec:
    return EffectSpec(
        effect_id=eid,
        field_key=field_key,
        polarity=polarity,
        shape=shape,
        area_trend=area,
        intensity_trend=intensity,
        center=center,
        start=start,
        end=end,
        amplitude_sigma=amp,
        radius=shape_radius(shape),
        angle=angle,
        axis_ratio=2.4,
    )

def make_accident_scenarios(profile: str = "paper") -> List[AccidentScenario]:
    scenarios = [
        AccidentScenario(
            "fire_vs_overheat__fire", "fire",
            TestCase("fire_vs_overheat__fire", [
                effect("T_fire", "temperature", "high", "blob", (13, 14), "expanding", "strengthening", 4.0),
                effect("AQI_fire", "air_quality", "high", "blob", (14, 15), "expanding", "strengthening", 3.2),
                effect("CO2_fire", "co2", "high", "compact_blob", (15, 15), "expanding", "strengthening", 2.6),
                effect("H_fire", "humidity", "low", "compact_blob", (14, 14), "stable", "weakening", 2.2),
            ]),
            explanation_fields=("temperature", "air_quality", "co2", "humidity"), target_field="temperature",
            notes="Fire and electrical overheating both contain high temperature; fire has AQI/CO2/humidity evidence.",
            challenge="fire_vs_electrical_overheat",
            llm_advantage="Both cases contain high temperature, but fire requires joint AQI/CO2/humidity reasoning.",
        ),
        AccidentScenario(
            "fire_vs_overheat__electrical", "electrical_overheat",
            TestCase("fire_vs_overheat__electrical", [
                effect("T_elec", "temperature", "high", "compact_blob", (16, 15), "stable", "strengthening", 4.2),
                effect("P_elec", "pressure", "high", "point_like", (16, 15), "stable", "stable", 1.6),
            ]),
            explanation_fields=("temperature",), target_field="temperature",
            notes="Electrical overheating is a high-temperature confuser without the air-quality and humidity signature of fire.",
            challenge="fire_vs_electrical_overheat",
            llm_advantage="A model should avoid calling every high-temperature event fire when combustion-side fields are absent.",
        ),
        AccidentScenario(
            "water_vs_steam__water", "water_leak",
            TestCase("water_vs_steam__water", [
                effect("H_water", "humidity", "high", "elongated_strip", (15, 12), "expanding", "stable", 3.8, angle=1.0),
                effect("P_water", "pressure", "low", "compact_blob", (16, 13), "stable", "stable", 2.2),
            ]),
            explanation_fields=("humidity", "pressure"), target_field="humidity",
            notes="Water leak and steam leak share high humidity; steam also has temperature and pressure evidence.",
            challenge="water_leak_vs_steam_leak",
            llm_advantage="Both cases contain high humidity, but water leak lacks the high-temperature/high-pressure steam signature.",
        ),
        AccidentScenario(
            "water_vs_steam__steam", "steam_leak",
            TestCase("water_vs_steam__steam", [
                effect("H_steam", "humidity", "high", "elongated_strip", (15, 17), "expanding", "strengthening", 3.8, angle=0.7),
                effect("T_steam", "temperature", "high", "oval", (14, 17), "expanding", "stable", 2.9),
                effect("P_steam", "pressure", "high", "compact_blob", (15, 18), "stable", "strengthening", 2.1),
            ]),
            explanation_fields=("humidity", "temperature", "pressure"), target_field="humidity",
            notes="Steam leak is a humidity anomaly with simultaneous heat and pressure disturbance.",
            challenge="water_leak_vs_steam_leak",
            llm_advantage="A model should combine humidity, temperature, pressure, and trend evidence instead of using humidity alone.",
        ),
        AccidentScenario(
            "co2_vs_dust__co2", "co2_accumulation",
            TestCase("co2_vs_dust__co2", [
                effect("CO2_acc", "co2", "high", "blob", (14, 14), "expanding", "strengthening", 3.5),
                effect("AQI_co2", "air_quality", "high", "compact_blob", (14, 15), "stable", "stable", 1.5),
            ]),
            explanation_fields=("co2", "air_quality"), target_field="co2",
            notes="CO2 accumulation and dust pollution can both affect air quality; CO2 field separates them.",
            challenge="co2_accumulation_vs_dust_pollution",
            llm_advantage="Both cases can look like poor air quality, but CO2 accumulation has a CO2-specific spatial field.",
        ),
        AccidentScenario(
            "co2_vs_dust__dust", "dust_pollution",
            TestCase("co2_vs_dust__dust", [
                effect("AQI_dust", "air_quality", "high", "elongated_strip", (16, 16), "expanding", "strengthening", 3.8, angle=2.1),
            ]),
            explanation_fields=("air_quality",), target_field="air_quality",
            notes="Dust pollution is primarily a generic air-quality anomaly without a matching CO2 plume.",
            challenge="co2_accumulation_vs_dust_pollution",
            llm_advantage="A model should not overfit any air-quality anomaly to CO2 accumulation when CO2 evidence is absent.",
        ),
        AccidentScenario(
            "composite_fire_and_leak", "composite_anomaly",
            TestCase("composite_fire_and_leak", [
                effect("T_comp", "temperature", "high", "blob", (11, 12), "expanding", "strengthening", 3.6),
                effect("AQI_comp", "air_quality", "high", "blob", (12, 12), "expanding", "strengthening", 2.8),
                effect("H_comp", "humidity", "high", "elongated_strip", (20, 19), "expanding", "stable", 3.2, angle=1.3),
            ]),
            ambiguous=True, unseen_combo=False, explanation_fields=("temperature", "air_quality", "humidity"), target_field="temperature",
            notes="Composite anomaly: traditional single-rule systems often misclassify into one incident.",
            challenge="composite_anomaly",
            llm_advantage="Joint fire-like and leak-like evidence should be reported as composite rather than forced into one rule label.",
            hard_family="composite_generalization",
            diagnostic_goal="composite",
            expected_event_signatures=(
                event_signature("temperature", "high"),
                event_signature("air_quality", "high"),
                event_signature("humidity", "high"),
            ),
            forbidden_predictions=("fire", "water_leak", "steam_leak"),
        ),
        AccidentScenario(
            "low_snr_multi_field", "low_snr_anomaly",
            TestCase("low_snr_multi_field", [
                effect("T_lowsnr", "temperature", "high", "compact_blob", (14, 18), "expanding", "strengthening", 2.1),
                effect("CO2_lowsnr", "co2", "high", "compact_blob", (15, 18), "stable", "strengthening", 1.8),
                effect("AQI_lowsnr", "air_quality", "high", "compact_blob", (14, 17), "stable", "strengthening", 1.7),
            ]),
            stress=StressConfig("low_snr_noise", noise_sigma=0.35, anomaly_scale=0.65),
            ambiguous=True, expected_review=True, explanation_fields=("temperature", "co2", "air_quality"), target_field="temperature",
            notes="Low SNR: a strong system should combine weak trends and may request review.",
            challenge="low_snr_multi_field",
            llm_advantage="Weak evidence across several fields should be integrated with calibrated uncertainty instead of hard thresholding.",
            hard_family="weak_evidence_integration",
            diagnostic_goal="weak_evidence",
            expected_event_signatures=(
                event_signature("temperature", "high"),
                event_signature("co2", "high"),
                event_signature("air_quality", "high"),
            ),
            acceptable_predictions=("low_snr_anomaly", "needs_review_unknown"),
            forbidden_predictions=("normal",),
        ),
        AccidentScenario(
            "missing_low_confidence_region", "needs_review_unknown",
            TestCase("missing_low_confidence_region", [
                effect("H_missing", "humidity", "high", "compact_blob", (16, 13), "expanding", "stable", 2.5),
                effect("T_missing", "temperature", "high", "compact_blob", (16, 13), "stable", "stable", 1.9),
            ]),
            stress=StressConfig("occlusion_40", occlusion_rate=0.40, anomaly_scale=0.75),
            ambiguous=True, expected_review=True, explanation_fields=("humidity", "temperature"), target_field="humidity",
            notes="Missing/low-confidence region should produce review rather than a brittle hard label.",
            challenge="missing_or_low_confidence_region",
            llm_advantage="Missing observations should trigger review/resampling instead of forcing an overconfident diagnosis.",
            hard_family="uncertainty_review",
            diagnostic_goal="review_low_confidence",
            expected_event_signatures=(
                event_signature("humidity", "high"),
                event_signature("temperature", "high"),
            ),
            acceptable_predictions=("needs_review_unknown",),
            forbidden_predictions=("water_leak", "steam_leak", "fire"),
        ),
        AccidentScenario(
            "unseen_pressure_aqi_combo", "needs_review_unknown",
            TestCase("unseen_pressure_aqi_combo", [
                effect("P_unseen", "pressure", "high", "oval", (14, 12), "expanding", "strengthening", 2.8),
                effect("AQI_unseen", "air_quality", "high", "oval", (14, 12), "stable", "strengthening", 2.7),
                effect("H_unseen", "humidity", "low", "compact_blob", (15, 12), "stable", "stable", 2.0),
            ]),
            ambiguous=True, unseen_combo=True, expected_review=True, explanation_fields=("pressure", "air_quality", "humidity"), target_field="pressure",
            notes="Unseen combination tests generalization and calibrated review behavior.",
            challenge="unseen_field_combination",
            llm_advantage="The pattern is outside the known accident templates, so generalized reasoning should prefer review.",
            hard_family="ood_template_rejection",
            diagnostic_goal="template_rejection",
            expected_event_signatures=(
                event_signature("pressure", "high"),
                event_signature("air_quality", "high"),
                event_signature("humidity", "low"),
            ),
            acceptable_predictions=("needs_review_unknown",),
            forbidden_predictions=("dust_pollution", "steam_leak", "co2_accumulation"),
        ),
        AccidentScenario(
            "unseen_temp_pressure_drop_combo", "needs_review_unknown",
            TestCase("unseen_temp_pressure_drop_combo", [
                effect("T_unseen_pd", "temperature", "high", "compact_blob", (12, 17), "stable", "strengthening", 3.1),
                effect("P_unseen_pd", "pressure", "low", "compact_blob", (12, 17), "stable", "strengthening", 2.7),
            ]),
            ambiguous=True, unseen_combo=True, expected_review=True, explanation_fields=("temperature", "pressure"), target_field="pressure",
            notes="High temperature coupled with pressure drop is outside the known fire/electrical/steam templates.",
            challenge="unseen_field_combination",
            llm_advantage="The model should reject the tempting electrical-overheat label because the pressure-drop evidence is contradictory.",
            hard_family="ood_template_rejection",
            diagnostic_goal="template_rejection",
            expected_event_signatures=(
                event_signature("temperature", "high"),
                event_signature("pressure", "low"),
            ),
            acceptable_predictions=("needs_review_unknown",),
            forbidden_predictions=("electrical_overheat", "fire", "steam_leak"),
        ),
        AccidentScenario(
            "unseen_co2_pressure_rise_combo", "needs_review_unknown",
            TestCase("unseen_co2_pressure_rise_combo", [
                effect("CO2_unseen_pr", "co2", "high", "oval", (17, 12), "expanding", "strengthening", 3.0),
                effect("P_unseen_pr", "pressure", "high", "compact_blob", (17, 12), "stable", "strengthening", 2.4),
            ]),
            ambiguous=True, unseen_combo=True, expected_review=True, explanation_fields=("co2", "pressure"), target_field="co2",
            notes="CO2 rise with pressure rise is a process-upset pattern not covered by the known accident labels.",
            challenge="unseen_field_combination",
            llm_advantage="Generalized reasoning should not collapse this novel coupled pattern into CO2 accumulation without review.",
            hard_family="ood_template_rejection",
            diagnostic_goal="template_rejection",
            expected_event_signatures=(
                event_signature("co2", "high"),
                event_signature("pressure", "high"),
            ),
            acceptable_predictions=("needs_review_unknown",),
            forbidden_predictions=("co2_accumulation", "steam_leak"),
        ),
        AccidentScenario(
            "low_confidence_fire_like_region", "needs_review_unknown",
            TestCase("low_confidence_fire_like_region", [
                effect("T_lc_fire", "temperature", "high", "blob", (15, 15), "expanding", "strengthening", 3.4),
                effect("AQI_lc_fire", "air_quality", "high", "blob", (15, 16), "expanding", "strengthening", 2.7),
                effect("CO2_lc_fire", "co2", "high", "compact_blob", (16, 15), "stable", "strengthening", 2.2),
            ]),
            stress=StressConfig("occlusion_35_fire_like", occlusion_rate=0.35, anomaly_scale=0.85),
            ambiguous=True, expected_review=True, explanation_fields=("temperature", "air_quality", "co2"), target_field="temperature",
            notes="Fire-like evidence is present but heavy missingness should force calibrated review rather than a brittle hard fire label.",
            challenge="missing_or_low_confidence_region",
            llm_advantage="The LLM should integrate event evidence with observation quality and request review instead of over-calling fire.",
            hard_family="uncertainty_review",
            diagnostic_goal="review_low_confidence",
            expected_event_signatures=(
                event_signature("temperature", "high"),
                event_signature("air_quality", "high"),
                event_signature("co2", "high"),
            ),
            acceptable_predictions=("needs_review_unknown",),
            forbidden_predictions=("fire",),
        ),
        AccidentScenario(
            "normal_no_incident", "normal",
            TestCase("normal_no_incident", []),
            explanation_fields=(), target_field=None,
            notes="Normal control case for false-positive and low-confidence filtering.",
            challenge="normal_control",
            llm_advantage="A model should preserve normal when no coherent multi-field evidence is present.",
        ),
    ]
    if profile == "quick":
        # Smoke test only: do not use this subset for paper conclusions.
        keep = {"fire_vs_overheat__fire", "water_vs_steam__steam", "low_snr_multi_field", "normal_no_incident"}
        return [s for s in scenarios if s.scenario_id in keep]
    if profile == "hard":
        # Hard set designed to expose LLM-level advantages: template
        # disambiguation, composite reasoning, calibrated review, weak evidence,
        # and OOD template rejection.  It intentionally contains more
        # ambiguous/review/OOD cases than the balanced paper profile.
        keep = {
            "fire_vs_overheat__electrical",
            "water_vs_steam__water",
            "co2_vs_dust__dust",
            "composite_fire_and_leak",
            "low_snr_multi_field",
            "missing_low_confidence_region",
            "unseen_pressure_aqi_combo",
            "unseen_temp_pressure_drop_combo",
            "unseen_co2_pressure_rise_combo",
            "low_confidence_fire_like_region",
            "normal_no_incident",
        }
        return [s for s in scenarios if s.scenario_id in keep]
    return scenarios

# ============================================================
# Low-level detectors
# ============================================================
def estimate_spatial_bg(x: np.ndarray, free_mask: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    valid = free_mask & np.isfinite(x)
    if valid.sum() == 0:
        out = np.zeros_like(x, dtype=np.float32)
        out[~free_mask] = np.nan
        return out
    med = float(np.nanmedian(x[valid]))
    filled = np.where(valid, x, med).astype(np.float32)
    weight = valid.astype(np.float32)
    num = gaussian_filter(filled * weight, sigma=sigma, mode="nearest")
    den = gaussian_filter(weight, sigma=sigma, mode="nearest") + 1e-6
    bg = (num / den).astype(np.float32)
    resid = x - bg
    bg = bg + float(np.nanmedian(resid[valid]))
    bg[~free_mask] = np.nan
    return bg

def classify_simple(mask: np.ndarray) -> str:
    pts = np.argwhere(mask)
    area = len(pts)
    if area <= 5:
        return "point_like"
    rmin, cmin = pts.min(axis=0)
    rmax, cmax = pts.max(axis=0)
    h = int(rmax - rmin + 1)
    w = int(cmax - cmin + 1)
    ratio = max(h, w) / max(1.0, min(h, w))
    if ratio >= 3.0:
        return "elongated_strip"
    if ratio >= 1.7:
        return "oval"
    if area <= 28:
        return "compact_blob"
    return "blob"

def extract_components(field_key: str, polarity: str, score: np.ndarray, mask: np.ndarray, t: int, min_area: int = 3) -> List[Dict[str, Any]]:
    lab, n = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    rr, cc = np.indices(mask.shape)
    out: List[Dict[str, Any]] = []
    sem = FIELD_REGISTRY[field_key]
    for idx in range(1, n + 1):
        comp = lab == idx
        area = int(comp.sum())
        if area < min_area:
            continue
        weights = score[comp] + 1e-6
        centroid = (float(np.average(rr[comp], weights=weights)), float(np.average(cc[comp], weights=weights)))
        morph = classify_simple(comp)
        out.append({
            "track_id": None,
            "t": int(t),
            "field_key": field_key,
            "polarity": polarity,
            "mask": comp,
            "centroid": (round(centroid[0], 3), round(centroid[1], 3)),
            "area": area,
            "morphology": morph,
            "z_core_mean": round(float(np.nanmean(score[comp])), 3),
            "score_sum": round(float(np.nansum(score[comp])), 3),
            "priority": round(float(np.nansum(score[comp])), 3),
            "physical_tag": {"label": sem.high_label if polarity == "high" else sem.low_label},
            # Single-frame connected components do not contain measured trends.
            "temporal_summary": {"area_trend": None, "intensity_trend": None},
        })
    return out

class DetectorBase:
    name = "base"
    def update(self, t: int, current_fields: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        raise NotImplementedError

class ThresholdMatrixObserver(DetectorBase):
    """No-op observer used when only matrix-threshold methods are requested."""
    name = "threshold_matrix_observer"
    def update(self, t: int, current_fields: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        return []

class CUSUMEWMADetector(DetectorBase):
    name = "cusum_ewma"
    def __init__(self, grid: np.ndarray, alpha: float = 0.92, cusum_alpha: float = 0.82, z_threshold: float = 1.4, cusum_threshold: float = 1.0) -> None:
        self.grid = grid.astype(np.uint8)
        self.free_mask = self.grid == 0
        self.alpha = alpha
        self.cusum_alpha = cusum_alpha
        self.z_threshold = z_threshold
        self.cusum_threshold = cusum_threshold
        self.bg: Dict[str, np.ndarray] = {}
        self.cusum: Dict[Tuple[str, str], np.ndarray] = {}
    def update(self, t: int, current_fields: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for k, x in current_fields.items():
            x = np.asarray(x, dtype=np.float32)
            spatial_bg = estimate_spatial_bg(x, self.free_mask)
            if k not in self.bg:
                self.bg[k] = spatial_bg.copy()
            bg = self.bg[k]
            z = (x - bg) / (FIELD_REGISTRY[k].sigma + 1e-6)
            z[~self.free_mask] = np.nan
            event_union = np.zeros_like(self.free_mask, dtype=bool)
            for pol in ("high", "low"):
                score = np.maximum(z, 0.0) if pol == "high" else np.maximum(-z, 0.0)
                score[~np.isfinite(score)] = 0.0
                ck = (k, pol)
                if ck not in self.cusum:
                    self.cusum[ck] = np.zeros_like(score, dtype=np.float32)
                c = self.cusum[ck]
                c[:] = self.cusum_alpha * c + (1 - self.cusum_alpha) * score
                mask = self.free_mask & (score >= self.z_threshold) & (c >= self.cusum_threshold)
                event_union |= mask
                events.extend(extract_components(k, pol, score, mask, t, FIELD_REGISTRY[k].min_area))
            valid = self.free_mask & np.isfinite(x) & (~event_union)
            bg = bg.copy()
            bg[valid] = self.alpha * bg[valid] + (1 - self.alpha) * x[valid]
            bg[~self.free_mask] = np.nan
            self.bg[k] = bg.astype(np.float32)
        return sorted(events, key=lambda e: e["priority"], reverse=True)

class VisionHeatmapDetector(DetectorBase):
    name = "vision_heatmap"
    def __init__(self, grid: np.ndarray, heat_sigma: float = 1.1, threshold: float = 1.20) -> None:
        self.grid = grid.astype(np.uint8)
        self.free_mask = self.grid == 0
        self.heat_sigma = heat_sigma
        self.threshold = threshold
    def update(self, t: int, current_fields: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for k, x in current_fields.items():
            x = np.asarray(x, dtype=np.float32)
            bg = estimate_spatial_bg(x, self.free_mask, sigma=4.5)
            z = (x - bg) / (FIELD_REGISTRY[k].sigma + 1e-6)
            z[~self.free_mask] = np.nan
            for pol in ("high", "low"):
                score = np.maximum(z, 0.0) if pol == "high" else np.maximum(-z, 0.0)
                score[~np.isfinite(score)] = 0.0
                heat = gaussian_filter(score, sigma=self.heat_sigma, mode="nearest")
                core = self.free_mask & (heat >= self.threshold)
                support = self.free_mask & (heat >= 0.55 * self.threshold)
                lab, n = label(support, structure=np.ones((3, 3), dtype=np.uint8))
                mask = np.zeros_like(core, dtype=bool)
                for idx in range(1, n + 1):
                    comp = lab == idx
                    if (comp & core).any():
                        mask |= comp
                events.extend(extract_components(k, pol, score, mask, t, FIELD_REGISTRY[k].min_area))
        return sorted(events, key=lambda e: e["priority"], reverse=True)

class F2EDetector(DetectorBase):
    name = "f2e_encoder"
    def __init__(self, grid: np.ndarray) -> None:
        self.encoder = UniversalFieldToEventEncoder(grid, FIELD_REGISTRY)
    def update(self, t: int, current_fields: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        return self.encoder.update(t, current_fields)

def detector_factory(name: str, grid: np.ndarray) -> DetectorBase:
    if name == "threshold_matrix_observer":
        return ThresholdMatrixObserver()
    if name == "cusum_ewma":
        return CUSUMEWMADetector(grid)
    if name == "vision_heatmap":
        return VisionHeatmapDetector(grid)
    if name == "f2e_encoder":
        return F2EDetector(grid)
    raise ValueError(name)

# ============================================================
# Case preparation and event collection
# ============================================================
def clone_scenario_for_grid(scenario: AccidentScenario, grid: np.ndarray) -> AccidentScenario:
    effects = []
    for eff0 in scenario.testcase.effects:
        e = EffectSpec(**asdict(eff0))
        e.center = tuple(map(float, nearest_free(grid, e.center)))
        effects.append(e)
    return AccidentScenario(
        scenario_id=scenario.scenario_id,
        accident_type=scenario.accident_type,
        testcase=TestCase(scenario.testcase.case_id, effects),
        stress=scenario.stress,
        ambiguous=scenario.ambiguous,
        unseen_combo=scenario.unseen_combo,
        expected_review=scenario.expected_review,
        explanation_fields=scenario.explanation_fields,
        target_field=scenario.target_field,
        notes=scenario.notes,
        challenge=scenario.challenge,
        llm_advantage=scenario.llm_advantage,
    )

def field_keys_for_scenario(s: AccidentScenario, all_fields: bool = True) -> Tuple[str, ...]:
    if all_fields or not s.testcase.effects:
        return tuple(FIELDS)
    return tuple(sorted({e.field_key for e in s.testcase.effects}))

def serializable_event(e: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in e.items() if k != "mask"}
    if "centroid" in out and isinstance(out["centroid"], tuple):
        out["centroid"] = list(out["centroid"])
    return out

def summarize_events(events: List[Dict[str, Any]], max_events: int = 10,
                     history_frames: Optional[int] = None) -> List[Dict[str, Any]]:
    out = []
    for ev in sorted(events, key=lambda x: float(x.get("priority", x.get("score_sum", 0.0))), reverse=True)[:max_events]:
        tmp = ev.get("temporal_summary") or {}
        centroid = ev.get("centroid")
        if isinstance(centroid, tuple):
            centroid = list(centroid)
        item = {
            "track_id": ev.get("track_id"),
            "field_key": ev.get("field_key"),
            "polarity": ev.get("polarity"),
            "morphology": ev.get("morphology"),
            "centroid": centroid,
            "area": ev.get("area"),
            "z_core_mean": ev.get("z_core_mean"),
            "score_sum": ev.get("score_sum", ev.get("priority")),
            "duration_steps": tmp.get("duration_steps"),
            "area_trend": tmp.get("area_trend"),
            "intensity_trend": tmp.get("intensity_trend"),
        }
        duration = tmp.get("duration_steps")
        if duration is not None and history_frames is not None and int(history_frames) > 0:
            item["track_coverage_fraction"] = round(float(duration) / float(history_frames), 4)
        for key in ("area_start", "area_current", "soft_area_start", "soft_area_current",
                    "intensity_start", "intensity_current"):
            if tmp.get(key) is not None:
                item[key] = tmp.get(key)
        out.append(item)
    return out

def diagnosis_query_t(_scenario: AccidentScenario, steps: int, interval_steps: int = 50) -> int:
    """Return one fixed scheduled diagnosis frame for every scenario.

    The scenario argument is retained for call-site compatibility but is never
    inspected: evaluation time must not depend on injected-effect timing.
    """
    last_t = max(0, int(steps) - 1)
    interval = max(1, int(interval_steps))
    scheduled = [step_1based - 1 for step_1based in range(interval, int(steps) + 1, interval)]
    return int(scheduled[-1] if scheduled else last_t)

def snapshot_events_at(trace: RunTrace, t: int) -> List[Dict[str, Any]]:
    for snap in trace.event_snapshots:
        if int(snap.get("t", -1)) == int(t):
            events = snap.get("events", [])
            return events if isinstance(events, list) else []
    if trace.event_snapshots:
        nearest = min(trace.event_snapshots, key=lambda s: abs(int(s.get("t", 0)) - int(t)))
        events = nearest.get("events", [])
        return events if isinstance(events, list) else []
    return []

def current_field_summary(fields_by_t: List[Dict[str, np.ndarray]], free_mask: np.ndarray) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if not fields_by_t:
        return summary
    keys = fields_by_t[-1].keys()
    for k in keys:
        sem = FIELD_REGISTRY[k]
        stack = np.array([f[k] for f in fields_by_t if k in f], dtype=np.float32)
        last = stack[-1]
        valid = free_mask & np.isfinite(last)
        vals = last[valid]
        if vals.size == 0:
            continue
        first = stack[0]
        valid_first = free_mask & np.isfinite(first)
        delta_med = float(np.nanmedian(last[valid]) - np.nanmedian(first[valid_first])) if valid_first.any() else 0.0
        bg = estimate_spatial_bg(last, free_mask)
        z = (last - bg) / (sem.sigma + 1e-6)
        zvals = z[valid]
        summary[k] = {
            "mean": round(float(np.nanmean(vals)), 4),
            "min": round(float(np.nanmin(vals)), 4),
            "max": round(float(np.nanmax(vals)), 4),
            "std": round(float(np.nanstd(vals)), 4),
            "median_delta_from_first": round(delta_med, 4),
            "max_abs_spatial_z": round(float(np.nanmax(np.abs(zvals))), 4) if zvals.size else None,
            "high_z_cells": int(np.nansum(z > 1.65)),
            "low_z_cells": int(np.nansum(z < -1.65)),
        }
    return summary



def observation_quality(fields_by_t: List[Dict[str, np.ndarray]], free_mask: np.ndarray,
                        recent_frames: int = 20) -> Dict[str, Any]:
    """Observation-quality metadata derived only from observed fields.

    This is evaluator-side input for LLM/rules baselines, not encoder input.  It
    merges the v8.2.4 stable missing-fraction fields with the v8.2.4-branch
    richer recent-noise summary.  No clean backgrounds, GT masks, injected
    accident labels, or stress IDs are used.
    """
    if not fields_by_t:
        return {
            "n_observed_frames": 0,
            "last_frame_missing_fraction": 1.0,
            "mean_missing_fraction_over_window": None,
            "field_missing_fraction": {},
            "low_confidence": True,
            "max_missing_fraction": 1.0,
            "max_recent_diff_noise_sigma": None,
            "max_recent_temporal_std_sigma": None,
            "high_missing_fields": [],
            "high_noise_fields": [],
            "per_field": {},
        }

    last = fields_by_t[-1]
    recent = fields_by_t[-max(2, int(recent_frames)):]
    total_free = max(1, int(np.sum(free_mask)))

    field_missing: Dict[str, float] = {}
    per_field: Dict[str, Dict[str, Any]] = {}
    last_missing_values: List[float] = []

    for k in FIELDS:
        if k not in last:
            continue
        sem = FIELD_REGISTRY[k]
        arr_last = np.asarray(last[k], dtype=np.float32)
        missing = float(np.sum(free_mask & ~np.isfinite(arr_last)) / total_free)
        field_missing[k] = round(float(missing), 6)
        last_missing_values.append(missing)

        diff_noise = None
        temporal_std = None
        stack_items = [np.asarray(f[k], dtype=np.float32) for f in recent if k in f]
        if len(stack_items) >= 2:
            stack = np.array(stack_items, dtype=np.float32)
            diffs = np.diff(stack, axis=0)[:, free_mask]
            diff_noise = float(np.nanstd(diffs) / (sem.sigma + 1e-6))
            temporal_std = float(np.nanmedian(np.nanstd(stack[:, free_mask], axis=0)) / (sem.sigma + 1e-6))
        per_field[k] = {
            "missing_fraction": round(float(missing), 4),
            "recent_diff_noise_sigma": None if diff_noise is None else round(float(diff_noise), 4),
            "recent_temporal_std_sigma": None if temporal_std is None else round(float(temporal_std), 4),
        }

    temporal_missing: List[float] = []
    for frame in fields_by_t:
        vals = []
        for arr in frame.values():
            arr_np = np.asarray(arr, dtype=np.float32)
            vals.append(1.0 - float(np.sum(free_mask & np.isfinite(arr_np))) / total_free)
        if vals:
            temporal_missing.append(float(np.mean(vals)))

    diff_vals = [v["recent_diff_noise_sigma"] for v in per_field.values() if v["recent_diff_noise_sigma"] is not None]
    std_vals = [v["recent_temporal_std_sigma"] for v in per_field.values() if v["recent_temporal_std_sigma"] is not None]
    max_missing = float(max([v["missing_fraction"] for v in per_field.values()] or [1.0]))
    last_missing = float(np.mean(last_missing_values)) if last_missing_values else 1.0

    return {
        "n_observed_frames": int(len(fields_by_t)),
        "last_frame_missing_fraction": round(float(last_missing), 6),
        "mean_missing_fraction_over_window": round(float(np.mean(temporal_missing)), 6) if temporal_missing else None,
        "field_missing_fraction": field_missing,
        "low_confidence": bool(last_missing >= 0.20 or max_missing >= 0.20),
        "max_missing_fraction": round(float(max_missing), 4),
        "max_recent_diff_noise_sigma": round(float(max(diff_vals)), 4) if diff_vals else None,
        "max_recent_temporal_std_sigma": round(float(max(std_vals)), 4) if std_vals else None,
        "high_missing_fields": [k for k, v in per_field.items() if float(v["missing_fraction"]) >= 0.20],
        "high_noise_fields": [k for k, v in per_field.items() if (v["recent_diff_noise_sigma"] is not None and float(v["recent_diff_noise_sigma"]) >= 0.20)],
        "per_field": per_field,
    }


def build_threshold_alarm_matrices(fields_by_t: List[Dict[str, np.ndarray]], grid: np.ndarray,
                                   calibration_frames: int = 20,
                                   control_limit_sigma: float = 3.0,
                                   field_alarm_fraction: float = 0.005) -> Dict[str, Any]:
    """Build a conventional fixed-limit alarm matrix without event extraction.

    Each free grid cell is treated as an individual measurement channel.  The
    first ``calibration_frames`` observed frames form the Phase-I reference;
    their per-cell mean and sample standard deviation define fixed Phase-II
    limits.  FIELD_REGISTRY noise sigma is a scale floor, preventing spuriously
    narrow limits when calibration samples happen to be nearly constant.

    The result deliberately contains no connected components, morphology,
    centroids, tracks, temporal trends, or event tokens.  It preserves the full
    current alarm matrix for every physical field.
    """
    if not fields_by_t:
        return {
            "policy": {
                "type": "per_cell_fixed_shewhart_limits",
                "phase_i_frames": 0,
                "control_limit_sigma": float(control_limit_sigma),
                "field_alarm_fraction": float(field_alarm_fraction),
            },
            "fields": {},
            "total_high_alarm_cells": 0,
            "total_low_alarm_cells": 0,
        }

    free_mask = np.asarray(grid) == 0
    phase_i_n = min(max(1, int(calibration_frames)), len(fields_by_t))
    fields_out: Dict[str, Any] = {}
    total_high = 0
    total_low = 0

    for field_key in FIELDS:
        available = [np.asarray(frame[field_key], dtype=np.float64)
                     for frame in fields_by_t if field_key in frame]
        if not available:
            continue
        calibration = np.stack(available[:phase_i_n], axis=0)
        current = available[-1]
        valid_cal = np.isfinite(calibration) & free_mask[None, :, :]
        sample_count = np.sum(valid_cal, axis=0)
        sample_sum = np.sum(np.where(valid_cal, calibration, 0.0), axis=0)

        pooled_values = calibration[valid_cal]
        fallback_center = float(np.median(pooled_values)) if pooled_values.size else 0.0
        center = np.full(current.shape, fallback_center, dtype=np.float64)
        np.divide(sample_sum, sample_count, out=center, where=sample_count > 0)

        squared_error = np.where(valid_cal, (calibration - center[None, :, :]) ** 2, 0.0)
        variance = np.zeros(current.shape, dtype=np.float64)
        np.divide(
            np.sum(squared_error, axis=0),
            np.maximum(sample_count - 1, 1),
            out=variance,
            where=sample_count > 1,
        )
        scale_floor = float(FIELD_REGISTRY[field_key].sigma)
        scale = np.maximum(np.sqrt(np.maximum(variance, 0.0)), scale_floor)
        lower = center - float(control_limit_sigma) * scale
        upper = center + float(control_limit_sigma) * scale

        valid_current = free_mask & np.isfinite(current)
        high_alarm = valid_current & (current > upper)
        low_alarm = valid_current & (current < lower)
        high_count = int(np.sum(high_alarm))
        low_count = int(np.sum(low_alarm))
        valid_count = int(np.sum(valid_current))
        missing_count = int(np.sum(free_mask & ~np.isfinite(current)))
        activation_count = max(3, int(math.ceil(float(field_alarm_fraction) * max(1, valid_count))))

        symbols = np.full(current.shape, "#", dtype="<U1")
        symbols[free_mask] = "?"
        symbols[valid_current] = "."
        symbols[high_alarm] = "H"
        symbols[low_alarm] = "L"

        fields_out[field_key] = {
            "matrix_rows": ["".join(row.tolist()) for row in symbols],
            "high_alarm_count": high_count,
            "low_alarm_count": low_count,
            "valid_cell_count": valid_count,
            "missing_cell_count": missing_count,
            "high_alarm_fraction": round(float(high_count / max(1, valid_count)), 6),
            "low_alarm_fraction": round(float(low_count / max(1, valid_count)), 6),
            "field_alarm_activation_count": activation_count,
            "high_alarm_active": bool(high_count >= activation_count),
            "low_alarm_active": bool(low_count >= activation_count),
            "median_center": round(float(np.median(center[free_mask])), 6) if np.any(free_mask) else None,
            "median_lower_control_limit": round(float(np.median(lower[free_mask])), 6) if np.any(free_mask) else None,
            "median_upper_control_limit": round(float(np.median(upper[free_mask])), 6) if np.any(free_mask) else None,
        }
        total_high += high_count
        total_low += low_count

    return {
        "policy": {
            "type": "per_cell_fixed_shewhart_limits",
            "phase_i_source": "first_observed_frames",
            "phase_i_frames": int(phase_i_n),
            "center_estimator": "per_cell_mean",
            "scale_estimator": "per_cell_sample_std_with_registered_noise_floor",
            "control_limit_sigma": float(control_limit_sigma),
            "phase_ii_rule": "current_value_outside_fixed_lower_or_upper_control_limit",
            "field_alarm_rule": "same-polarity_alarm_cell_fraction",
            "field_alarm_fraction": float(field_alarm_fraction),
            "minimum_field_alarm_cells": 3,
            "matrix_legend": {"H": "above_upper_limit", "L": "below_lower_limit", ".": "within_limits", "?": "missing", "#": "obstacle"},
        },
        "fields": fields_out,
        "total_high_alarm_cells": int(total_high),
        "total_low_alarm_cells": int(total_low),
    }



def blind_artifact_stem(seed: int, scenario_index: int, method_tag: str, mode: str) -> str:
    """Anonymized artifact stem for model-visible VLM images.

    Do not include scenario_id, accident_type, challenge names, or other labels
    in image filenames/titles.  The image bytes are what the VLM receives; local
    filenames are normally not sent, but keeping them anonymous prevents
    accidental leakage through prompts, debugging, or exported logs.
    """
    raw = f"{int(seed)}:{int(scenario_index)}:{method_tag}:{mode}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:10]
    return f"blind_case_{scenario_index:03d}_seed_{int(seed)}_{digest}_{mode}"


def blind_vlm_title(mode: str, frame_indices: Optional[Sequence[int]] = None) -> str:
    """Neutral image title rendered into VLM contact sheets.

    It intentionally excludes scenario_id and accident_type.  Frame indices are
    harmless timing metadata and are useful for interpreting temporal trends.
    """
    if frame_indices:
        return f"Temporal multi-field observation | mode={mode} | frames={list(frame_indices)}"
    return f"Temporal multi-field observation | mode={mode}"


def render_fields_image(fields: Dict[str, np.ndarray], out_path: str, title: str = "fields") -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    keys = list(fields.keys())
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        im = ax.imshow(fields[k], interpolation="nearest")
        ax.set_title(k)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path

def image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return "data:image/png;base64," + data

def render_fields_contact_sheet(fields_by_t: List[Dict[str, np.ndarray]], out_path: str,
                                title: str = "temporal_fields", mode: str = "sampled",
                                num_frames: int = 8,
                                grid: Optional[np.ndarray] = None) -> Tuple[Optional[str], List[int]]:
    """Render a temporal contact sheet for the raw_field_image+VLM baseline.

    Rows are physical fields; columns are time points.  This is deliberately a
    VLM baseline input, not an input to the F2E encoder.  It gives the VLM a fair
    chance to observe appearance, expansion/shrinking, and trend over time.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None, []
    if not fields_by_t:
        return None, []
    n_total = len(fields_by_t)
    if mode == "last":
        idxs = [n_total - 1]
    elif mode == "all":
        idxs = list(range(n_total))
        # Avoid unreadable monster images if a user accidentally requests all
        # frames for a long run.  The run_config still records the requested mode.
        if len(idxs) > 32:
            idxs = [int(round(x)) for x in np.linspace(0, n_total - 1, 32)]
    else:
        n = max(2, min(int(num_frames), n_total))
        idxs = [int(round(x)) for x in np.linspace(0, n_total - 1, n)]
    # Preserve order and remove duplicates caused by short episodes.
    idxs = list(dict.fromkeys(idxs))
    keys = list(fields_by_t[idxs[-1]].keys())
    n_rows, n_cols = len(keys), len(idxs)
    fig_w = max(3.2, 2.15 * n_cols)
    fig_h = max(3.0, 2.05 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    for r, k in enumerate(keys):
        # Use a consistent color range per field across time so VLM sees trends.
        vals = []
        for idx in idxs:
            arr = fields_by_t[idx].get(k)
            if arr is not None:
                vv = arr[np.isfinite(arr)]
                if vv.size:
                    vals.append(vv)
        if vals:
            vv = np.concatenate(vals)
            vmin, vmax = np.nanpercentile(vv, [2, 98])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-9:
                vmin, vmax = None, None
        else:
            vmin, vmax = None, None
        for c, idx in enumerate(idxs):
            ax = axes[r][c]
            arr = np.asarray(fields_by_t[idx].get(k), dtype=np.float32)
            shown = np.ma.masked_invalid(arr)
            im = ax.imshow(shown, interpolation="nearest", vmin=vmin, vmax=vmax)
            if grid is not None and np.asarray(grid).shape == arr.shape:
                grid_arr = np.asarray(grid)
                overlay = np.zeros(arr.shape + (4,), dtype=np.float32)
                overlay[grid_arr != 0] = (0.12, 0.12, 0.12, 1.0)
                overlay[(grid_arr == 0) & (~np.isfinite(arr))] = (1.0, 0.0, 1.0, 1.0)
                ax.imshow(overlay, interpolation="nearest")
            if r == 0:
                ax.set_title(f"t={idx}")
            if c == 0:
                ax.set_ylabel(k)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        title + f" | VLM contact sheet mode={mode}, frames={idxs} | "
        "obstacle=dark gray, missing free-space observation=magenta"
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=135)
    plt.close(fig)
    return out_path, idxs

@dataclass
class RunTrace:
    scenario: AccidentScenario
    seed: int
    detector_name: str
    events_last: List[Dict[str, Any]]
    events_all: List[Dict[str, Any]]
    event_snapshots: List[Dict[str, Any]]
    fields_window: List[Dict[str, np.ndarray]]
    fields_last: Dict[str, np.ndarray]
    grid: np.ndarray
    gt_last: List[Dict[str, Any]]
    low_metrics: Dict[str, Any]
    latency_ms: float

def collect_run_trace(scenario: AccidentScenario, seed: int, steps: int, detector_name: str,
                      obs_ratio: float, all_fields: bool = True,
                      capture_fields: bool = True,
                      snapshot_times: Optional[Sequence[int]] = None,
                      retain_snapshot_masks: bool = False) -> RunTrace:
    grid = make_obstacle_grid(GRID_SIZE, obs_ratio, seed=seed)
    sc = clone_scenario_for_grid(scenario, grid)
    detector = detector_factory(detector_name, grid)
    field_keys = field_keys_for_scenario(sc, all_fields=all_fields)
    events_last: List[Dict[str, Any]] = []
    events_all: List[Dict[str, Any]] = []
    event_snapshots: List[Dict[str, Any]] = []
    fields_window: List[Dict[str, np.ndarray]] = []
    fields_last: Dict[str, np.ndarray] = {}
    gt_last: List[Dict[str, Any]] = []
    detector_elapsed = 0.0
    requested_snapshots = None if snapshot_times is None else {int(t) for t in snapshot_times}
    t0 = time.perf_counter()
    for t in range(steps):
        rng = np.random.default_rng(seed * 1000003 + t)
        current, gt_records = build_current_fields(grid, sc.testcase, t, field_keys, stress=sc.stress, rng=rng)
        det_t0 = time.perf_counter()
        events = detector.update(t, current)
        detector_elapsed += time.perf_counter() - det_t0
        fields_last = current
        gt_last = gt_records
        # Threshold inputs do not need copied field histories. F2E captures them
        # for matrix/VLM inputs without changing detector execution.
        if capture_fields:
            fields_window.append({k: v.copy() for k, v in current.items()})
        events_last = events
        if requested_snapshots is None or t in requested_snapshots:
            if retain_snapshot_masks:
                snapshot_events = []
                for ev in events[:10]:
                    copied = dict(ev)
                    if isinstance(ev.get("mask"), np.ndarray):
                        copied["mask"] = ev["mask"].copy()
                    snapshot_events.append(copied)
            else:
                snapshot_events = [serializable_event(ev) for ev in events[:10]]
            event_snapshots.append({"t": int(t), "events": snapshot_events})
    elapsed = (time.perf_counter() - t0) * 1000
    low_metrics = {
        "latency_ms_per_frame": round(float(elapsed / max(1, steps)), 6),
        "detector_total_latency_ms": round(float(detector_elapsed * 1000.0), 6),
        "detector_latency_ms_per_frame": round(float(detector_elapsed * 1000.0 / max(1, steps)), 6),
    }
    return RunTrace(sc, seed, detector_name, events_last, events_all, event_snapshots, fields_window, fields_last, grid, gt_last, low_metrics, elapsed)

# ============================================================
# Rule-based diagnosis and decisions
# ============================================================
def event_presence(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    pres: Dict[str, Dict[str, float]] = {k: {"high": 0.0, "low": 0.0} for k in FIELDS}
    for ev in events:
        k, p = ev.get("field_key"), ev.get("polarity")
        if k in pres and p in pres[k]:
            pres[k][p] = max(pres[k][p], float(ev.get("z_core_mean", 0.0) or ev.get("priority", 0.0) or 0.0))
    return pres

def _cluster_spread(events: List[Dict[str, Any]]) -> float:
    pts = []
    for ev in events:
        c = ev.get("centroid")
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            try:
                pts.append((float(c[0]), float(c[1])))
            except Exception:
                pass
    if len(pts) < 2:
        return 0.0
    arr = np.array(pts, dtype=float)
    return float(np.max(np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)))

def _event_strength(events: List[Dict[str, Any]], field_key: str, polarity: str = "high") -> float:
    vals = []
    for ev in events:
        if ev.get("field_key") == field_key and ev.get("polarity") == polarity:
            vals.append(float(ev.get("z_core_mean", ev.get("score_sum", ev.get("priority", 0.0))) or 0.0))
    return max(vals) if vals else 0.0

def diagnose_by_rules(events: List[Dict[str, Any]], source_name: str = "events", observation_quality: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generic rule baseline, intentionally not an oracle.

    The rule system uses only low-level events and optional observation-quality
    metadata derived from the observed fields.  It deliberately avoids template
    branches that memorize the benchmark scenario IDs.  Ambiguous multi-field or
    missing-data cases are sent to review unless a simple canonical pattern is
    clearly present.
    """
    observation_quality = observation_quality or {}
    pres = event_presence(events)
    high = lambda k: pres.get(k, {}).get("high", 0.0) > 0
    low = lambda k: pres.get(k, {}).get("low", 0.0) > 0
    strength = lambda k, p="high": _event_strength(events, k, p)
    n_fields = sum(1 for k in FIELDS if high(k) or low(k))
    event_count = len(events)
    missing_rate = float(observation_quality.get("last_frame_missing_fraction", 0.0) or 0.0)
    spread = _cluster_spread(events)

    label = "normal"
    conf = 0.58
    review = False
    reasons: List[str] = []

    if event_count == 0:
        # No event tokens means no actionable local evidence. Missing or weak
        # observation quality triggers review; otherwise the generic rule keeps
        # the normal label.
        if missing_rate >= 0.20:
            label, conf, review = "needs_review_unknown", 0.45, True
            reasons.append("no event tokens and substantial missing observations")
        else:
            label, conf, review = "normal", 0.72, False
            reasons.append("no coherent event tokens")
    else:
        fire_like = high("temperature") and (high("air_quality") or high("co2"))
        leak_like = high("humidity")
        steam_like = high("humidity") and high("temperature") and high("pressure")
        water_like = high("humidity") and not high("temperature")
        co2_like = high("co2") and strength("co2") >= max(0.8, 0.85 * strength("air_quality"))
        dust_like = high("air_quality") and not high("co2")
        thermal_only = high("temperature") and not high("air_quality") and not high("co2") and not high("humidity")

        # Conservative composite criterion: distinct physical sub-patterns and
        # enough spatial separation.  Ordinary coupled multi-field signatures
        # such as steam or fire should not be called composite.
        if fire_like and leak_like and spread >= 5.0 and event_count >= 3:
            label, conf, review = "composite_anomaly", 0.68, True
            reasons.append("spatially separated fire-like and leak-like event groups")
        elif steam_like:
            label, conf = "steam_leak", 0.76
            reasons.append("humidity, temperature, and pressure co-occur")
        elif fire_like and not leak_like:
            label, conf = "fire", 0.76
            reasons.append("temperature co-occurs with combustion-side AQI/CO2 evidence")
        elif water_like:
            label, conf = "water_leak", 0.72
            reasons.append("humidity anomaly without thermal event")
        elif thermal_only:
            label, conf = "electrical_overheat", 0.70
            reasons.append("isolated temperature event without combustion-side fields")
        elif co2_like:
            label, conf = "co2_accumulation", 0.70
            reasons.append("CO2-specific event is present")
        elif dust_like:
            label, conf = "dust_pollution", 0.70
            reasons.append("air-quality event without CO2 event")
        elif n_fields >= 2:
            label, conf, review = "needs_review_unknown", 0.50, True
            reasons.append("multi-field pattern is outside simple generic rule templates")
        else:
            label, conf, review = "needs_review_unknown", 0.48, True
            reasons.append("single weak/local event is insufficient for a hard accident label")

        # Observation-quality and weak-evidence gates.  These are derived from
        # the input representation, not from ground truth labels.
        weak_events = [ev for ev in events if float(ev.get("z_core_mean", 0.0) or 0.0) < 1.15]
        if missing_rate >= 0.25:
            review = True
            conf = min(conf, 0.60)
            reasons.append("missing/low-confidence observations require review")
        if weak_events and event_count <= 2:
            review = True
            conf = min(conf, 0.58)
            reasons.append("weak event evidence requires confirmation")

    target = None
    if events:
        target_ev = max(events, key=lambda e: float(e.get("priority", e.get("score_sum", 0.0)) or 0.0))
        target = {"field_key": target_ev.get("field_key"), "centroid": target_ev.get("centroid")}
    return {
        "accident_type": label,
        "confidence": round(float(conf), 3),
        "review_needed": bool(review),
        "abnormal_confirmed": bool(label not in {"normal", "needs_review_unknown"} and conf >= 0.55 and not (review and conf < 0.62)),
        "resample_target": target,
        "explanation": f"Generic rule diagnosis from {source_name}: " + "; ".join(reasons),
        "evidence_fields": [k for k in FIELDS if high(k) or low(k)],
    }


def diagnose_by_threshold_alarms(threshold_matrices: Dict[str, Any],
                                 observation_quality: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Rule diagnosis from field-level fixed-limit alarms only.

    This baseline never groups cells spatially and never consumes F2E-style
    event attributes.  It reads only whether each complete physical matrix has
    enough same-polarity out-of-limit cells to activate its predeclared alarm.
    """
    observation_quality = observation_quality or {}
    fields = threshold_matrices.get("fields", {}) if isinstance(threshold_matrices, dict) else {}
    active_high = {k for k in FIELDS if bool((fields.get(k) or {}).get("high_alarm_active"))}
    active_low = {k for k in FIELDS if bool((fields.get(k) or {}).get("low_alarm_active"))}
    active_signals = [(k, "high") for k in FIELDS if k in active_high]
    active_signals.extend((k, "low") for k in FIELDS if k in active_low)

    last_missing = float(observation_quality.get("last_frame_missing_fraction", 0.0) or 0.0)
    max_missing = float(observation_quality.get("max_missing_fraction", 0.0) or 0.0)
    low_confidence = bool(observation_quality.get("low_confidence", False))
    reasons: List[str] = []
    label = "normal"
    confidence = 0.72
    review = False
    template_status = "known_template"

    if low_confidence or last_missing >= 0.30 or max_missing >= 0.30:
        label, confidence, review = "needs_review_unknown", 0.42, True
        template_status = "insufficient_data"
        reasons.append("matrix observations are too incomplete for a reliable fixed-threshold decision")
    elif not active_signals:
        reasons.append("no physical-field matrix activated its fixed-limit alarm")
    elif active_low:
        label, confidence, review = "needs_review_unknown", 0.48, True
        template_status = "novel_combination"
        reasons.append("one or more low-polarity field alarms are outside the simple high-alarm templates")
    elif active_high == {"humidity", "temperature", "pressure"}:
        label, confidence = "steam_leak", 0.74
        reasons.append("humidity, temperature, and pressure matrices have high alarms")
    elif "temperature" in active_high and bool(active_high & {"air_quality", "co2"}) and not bool(active_high & {"humidity", "pressure"}):
        label, confidence = "fire", 0.74
        reasons.append("temperature and combustion-side physical matrices have high alarms")
    elif active_high == {"humidity"}:
        label, confidence = "water_leak", 0.70
        reasons.append("only the humidity matrix has a high alarm")
    elif active_high == {"temperature"}:
        label, confidence = "electrical_overheat", 0.68
        reasons.append("only the temperature matrix has a high alarm")
    elif active_high == {"co2"}:
        label, confidence = "co2_accumulation", 0.68
        reasons.append("only the CO2 matrix has a high alarm")
    elif active_high == {"air_quality"}:
        label, confidence = "dust_pollution", 0.68
        reasons.append("only the air-quality matrix has a high alarm")
    else:
        label, confidence, review = "needs_review_unknown", 0.50, True
        template_status = "novel_combination"
        reasons.append("the activated field-alarm combination does not match a simple threshold rule")

    evidence_role = "supports_selected" if label not in {"normal", "needs_review_unknown"} else "unexplained"
    evidence_items = [
        {"field_key": field_key, "polarity": polarity, "role": evidence_role}
        for field_key, polarity in active_signals
    ]
    return {
        "accident_type": label,
        "confidence": round(float(confidence), 3),
        "review_needed": bool(review),
        "abnormal_confirmed": bool(label not in {"normal", "needs_review_unknown"}),
        "resample_target": None,
        "evidence_fields": [f"{field_key}:{polarity}" for field_key, polarity in active_signals],
        "evidence_events": evidence_items,
        "unexplained_evidence": [f"{field_key}:{polarity}" for field_key, polarity in active_signals] if label == "needs_review_unknown" else [],
        "template_status": template_status,
        "explanation": "Fixed threshold-matrix rule diagnosis: " + "; ".join(reasons),
    }

# ============================================================
# DashScope OpenAI-compatible client
# ============================================================
@dataclass
class ModelCallResult:
    ok: bool
    content: str
    parsed: Optional[Dict[str, Any]]
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model: Optional[str] = None
    error: Optional[str] = None
    from_cache: bool = False
    usage: Dict[str, Any] = field(default_factory=dict)
    token_count_source: str = "none"
    token_count_is_estimate: bool = False
    api_usage_available: bool = False
    prompt_text_tokens: Optional[int] = None
    prompt_image_tokens: Optional[int] = None
    prompt_video_tokens: Optional[int] = None
    prompt_audio_tokens: Optional[int] = None
    prompt_cached_tokens: Optional[int] = None
    completion_text_tokens: Optional[int] = None
    completion_reasoning_tokens: Optional[int] = None
    completion_audio_tokens: Optional[int] = None

def usage_value(usage: Dict[str, Any], *names: str) -> Optional[int]:
    for name in names:
        val = usage.get(name)
        if val is None:
            continue
        try:
            return int(val)
        except Exception:
            continue
    return None

def usage_detail_value(usage: Dict[str, Any], detail_name: str, *names: str) -> Optional[int]:
    detail = usage.get(detail_name)
    if not isinstance(detail, dict):
        return None
    return usage_value(detail, *names)

def usage_triplet(usage: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    prompt = usage_value(usage, "prompt_tokens", "input_tokens", "input_token_count")
    completion = usage_value(usage, "completion_tokens", "output_tokens", "output_token_count")
    total = usage_value(usage, "total_tokens", "total_token_count")
    if total is None and (prompt is not None or completion is not None):
        total = int(prompt or 0) + int(completion or 0)
    return prompt, completion, total

def usage_token_details(usage: Dict[str, Any]) -> Dict[str, Optional[int]]:
    return {
        "prompt_text_tokens": usage_detail_value(usage, "prompt_tokens_details", "text_tokens"),
        "prompt_image_tokens": usage_detail_value(usage, "prompt_tokens_details", "image_tokens"),
        "prompt_video_tokens": usage_detail_value(usage, "prompt_tokens_details", "video_tokens"),
        "prompt_audio_tokens": usage_detail_value(usage, "prompt_tokens_details", "audio_tokens"),
        "prompt_cached_tokens": usage_detail_value(usage, "prompt_tokens_details", "cached_tokens"),
        "completion_text_tokens": usage_detail_value(usage, "completion_tokens_details", "text_tokens"),
        "completion_reasoning_tokens": usage_detail_value(usage, "completion_tokens_details", "reasoning_tokens"),
        "completion_audio_tokens": usage_detail_value(usage, "completion_tokens_details", "audio_tokens"),
    }

def contains_visual_input(messages: Any) -> bool:
    if isinstance(messages, dict):
        if messages.get("type") in {"image_url", "video"}:
            return True
        return any(contains_visual_input(v) for v in messages.values())
    if isinstance(messages, list):
        return any(contains_visual_input(v) for v in messages)
    return False

def strip_visual_data_urls(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "url" and isinstance(v, str) and v.startswith("data:image/"):
                out[k] = "<image_data_url_omitted>"
            else:
                out[k] = strip_visual_data_urls(v)
        return out
    if isinstance(obj, list):
        return [strip_visual_data_urls(v) for v in obj]
    return obj

def text_messages_for_tokenizer(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, dict) and item.get("type") in {"image_url", "video"}:
                    parts.append("<visual_input>")
            text = "\n".join(p for p in parts if p)
        else:
            text = json.dumps(content, ensure_ascii=False)
        out.append({"role": role, "content": text})
    return out

@dataclass
class TokenFallbackCounter:
    mode: str = "none"
    tokenizer_model: Optional[str] = None
    local_files_only: bool = True
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _load_attempted: bool = field(default=False, init=False, repr=False)
    _load_error: Optional[str] = field(default=None, init=False, repr=False)

    def _default_tokenizer_model(self) -> str:
        return "Qwen/Qwen3-30B-A3B-Instruct-2507"

    def _load_tokenizer(self) -> Any:
        if self._load_attempted:
            return self._tokenizer
        self._load_attempted = True
        model_name = self.tokenizer_model or self._default_tokenizer_model()
        self.tokenizer_model = model_name
        try:
            from transformers import AutoTokenizer  # type: ignore
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
            )
        except Exception as e:
            self._load_error = repr(e)
            self._tokenizer = None
        return self._tokenizer

    def status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "tokenizer_model": self.tokenizer_model,
            "local_files_only": self.local_files_only,
            "load_attempted": self._load_attempted,
            "load_error": self._load_error,
        }

    def count_prompt(self, messages: List[Dict[str, Any]]) -> Tuple[Optional[int], str]:
        if self.mode == "none":
            return None, "none"
        if self.mode == "heuristic":
            payload = strip_visual_data_urls(messages)
            suffix = "_excludes_image_tokens" if contains_visual_input(messages) else ""
            return token_estimate_from_payload(payload), "char_heuristic_estimate" + suffix
        if self.mode == "hf_tokenizer":
            if contains_visual_input(messages):
                return None, "hf_tokenizer_unavailable_for_visual_input"
            tok = self._load_tokenizer()
            if tok is None:
                return None, "hf_tokenizer_unavailable"
            text_messages = text_messages_for_tokenizer(messages)
            try:
                ids = tok.apply_chat_template(text_messages, tokenize=True, add_generation_prompt=True)
                return int(len(ids)), f"hf_tokenizer_estimate:{self.tokenizer_model}"
            except Exception:
                try:
                    text = "\n".join(f"{m['role']}: {m['content']}" for m in text_messages)
                    ids = tok.encode(text)
                    return int(len(ids)), f"hf_tokenizer_encode_estimate:{self.tokenizer_model}"
                except Exception as e:
                    self._load_error = repr(e)
                    return None, "hf_tokenizer_count_failed"
        return None, "none"

    def count_completion(self, payload: Any) -> Tuple[Optional[int], str]:
        if self.mode == "none":
            return None, "none"
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        if self.mode == "heuristic":
            return token_estimate_from_payload(text), "char_heuristic_estimate"
        if self.mode == "hf_tokenizer":
            tok = self._load_tokenizer()
            if tok is None:
                return None, "hf_tokenizer_unavailable"
            try:
                ids = tok.encode(text)
                return int(len(ids)), f"hf_tokenizer_estimate:{self.tokenizer_model}"
            except Exception as e:
                self._load_error = repr(e)
                return None, "hf_tokenizer_count_failed"
        return None, "none"

def apply_token_fallback(res: ModelCallResult, messages: List[Dict[str, Any]], completion_payload: Any,
                         token_counter: TokenFallbackCounter) -> ModelCallResult:
    if res.prompt_tokens is not None and res.completion_tokens is not None and res.total_tokens is not None:
        return res
    prompt_est, prompt_source = token_counter.count_prompt(messages)
    completion_est, completion_source = token_counter.count_completion(completion_payload)
    filled = False
    if res.prompt_tokens is None and prompt_est is not None:
        res.prompt_tokens = prompt_est
        filled = True
    if res.completion_tokens is None and completion_est is not None:
        res.completion_tokens = completion_est
        filled = True
    if res.total_tokens is None and (res.prompt_tokens is not None or res.completion_tokens is not None):
        res.total_tokens = int(res.prompt_tokens or 0) + int(res.completion_tokens or 0)
        filled = True
    if filled:
        res.token_count_is_estimate = True
        source_parts = []
        if prompt_est is not None:
            source_parts.append(f"prompt={prompt_source}")
        if completion_est is not None:
            source_parts.append(f"completion={completion_source}")
        if res.api_usage_available:
            res.token_count_source = "api_usage_partial+" + ",".join(source_parts)
        else:
            res.token_count_source = ",".join(source_parts) or "estimate"
    return res

class DashScopeClient:
    def __init__(self, api_key: Optional[str], base_url: str, cache_path: Optional[str] = None,
                 enable_thinking: bool = False, timeout: float = 60.0,
                 max_retries: int = 2, retry_backoff_sec: float = 1.0,
                 proxy_mode: str = "bypass") -> None:
        if proxy_mode not in {"bypass", "environment"}:
            raise ValueError(f"Unsupported DashScope proxy mode: {proxy_mode}")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.enable_thinking = enable_thinking
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_sec = max(0.0, float(retry_backoff_sec))
        self.proxy_mode = proxy_mode
        # urllib normally honors HTTP(S)_PROXY/ALL_PROXY and the Windows system
        # proxy.  DashScope is a domestic endpoint, so the default bypass mode
        # installs an explicit empty ProxyHandler.  This prevents an application-
        # level VPN proxy from capturing the request.  A system-wide TUN/virtual
        # network adapter can still route direct sockets outside Python's control.
        proxy_handler = (
            urllib.request.ProxyHandler({})
            if proxy_mode == "bypass"
            else urllib.request.ProxyHandler()
        )
        self._url_opener = urllib.request.build_opener(proxy_handler)
        self.cache_path = cache_path
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._cache_dirty = False
        self._cache_lock = threading.Lock()
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _cache_key(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _save_cache(self) -> None:
        with self._cache_lock:
            if not self.cache_path or not self._cache_dirty:
                return
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            tmp = self.cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.cache_path)
            self._cache_dirty = False

    def flush_cache(self) -> None:
        self._save_cache()

    def _call_once(self, model: str, messages: List[Dict[str, Any]], max_tokens: int = 512, temperature: float = 0.0) -> ModelCallResult:
        if not self.api_key:
            return ModelCallResult(False, "", None, 0.0, error="DASHSCOPE_API_KEY is not set")
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.enable_thinking:
            # DashScope exposes this as extra_body={"enable_thinking": True}
            # through the OpenAI SDK; in raw HTTP it is sent as a top-level field.
            payload["enable_thinking"] = True
        key = self._cache_key(payload)
        with self._cache_lock:
            c = self.cache.get(key)
        if c is not None:
            return ModelCallResult(
                True,
                c.get("content", ""),
                c.get("parsed"),
                0.0,
                c.get("prompt_tokens"),
                c.get("completion_tokens"),
                c.get("total_tokens"),
                c.get("model", model),
                from_cache=True,
                usage=c.get("usage", {}) or {},
                token_count_source=c.get("token_count_source", "cache_api_usage" if c.get("api_usage_available") else "cache_missing_usage"),
                token_count_is_estimate=bool(c.get("token_count_is_estimate", False)),
                api_usage_available=bool(c.get("api_usage_available", False)),
                prompt_text_tokens=c.get("prompt_text_tokens"),
                prompt_image_tokens=c.get("prompt_image_tokens"),
                prompt_video_tokens=c.get("prompt_video_tokens"),
                prompt_audio_tokens=c.get("prompt_audio_tokens"),
                prompt_cached_tokens=c.get("prompt_cached_tokens"),
                completion_text_tokens=c.get("completion_text_tokens"),
                completion_reasoning_tokens=c.get("completion_reasoning_tokens"),
                completion_audio_tokens=c.get("completion_audio_tokens"),
            )
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with self._url_opener.open(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
            latency = (time.perf_counter() - t0) * 1000
            data = json.loads(body)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {}) or {}
            parsed = parse_json_object(content)
            prompt_tokens, completion_tokens, total_tokens = usage_triplet(usage)
            api_usage_available = any(v is not None for v in (prompt_tokens, completion_tokens, total_tokens))
            detail_tokens = usage_token_details(usage)
            if api_usage_available:
                token_count_source = "api_usage"
            elif usage:
                token_count_source = "api_usage_without_standard_counts"
            else:
                token_count_source = "missing_api_usage"
            if parsed is not None:
                cache_val = {
                    "content": content,
                    "parsed": parsed,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "model": data.get("model", model),
                    "usage": usage,
                    "token_count_source": token_count_source,
                    "token_count_is_estimate": False,
                    "api_usage_available": api_usage_available,
                    **detail_tokens,
                }
                with self._cache_lock:
                    self.cache[key] = cache_val
                    self._cache_dirty = True
            return ModelCallResult(
                True,
                content,
                parsed,
                latency,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                data.get("model", model),
                usage=usage,
                token_count_source=token_count_source,
                token_count_is_estimate=False,
                api_usage_available=api_usage_available,
                **detail_tokens,
            )
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")[:1000]
            return ModelCallResult(False, "", None, (time.perf_counter() - t0) * 1000, error=f"HTTP {e.code}: {err}")
        except Exception as e:
            return ModelCallResult(False, "", None, (time.perf_counter() - t0) * 1000, error=repr(e))

    def call(self, model: str, messages: List[Dict[str, Any]], max_tokens: int = 512,
             temperature: float = 0.0) -> ModelCallResult:
        total_latency_ms = 0.0
        last = ModelCallResult(False, "", None, 0.0, error="API call not attempted", model=model)
        errors: List[str] = []
        for attempt in range(self.max_retries + 1):
            last = self._call_once(model, messages, max_tokens=max_tokens, temperature=temperature)
            total_latency_ms += float(last.latency_ms)
            if last.ok and last.parsed is not None:
                last.latency_ms = total_latency_ms
                return last
            errors.append(last.error or "response was not a valid JSON object")
            if attempt < self.max_retries and self.retry_backoff_sec > 0:
                time.sleep(self.retry_backoff_sec * (2 ** attempt))
        last.latency_ms = total_latency_ms
        last.ok = False
        last.parsed = None
        last.error = f"API failed after {self.max_retries + 1} attempt(s): " + " | ".join(errors)
        return last

def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ============================================================
# F2E-aware diagnostic adapters (no-leak, observation-derived)
# ============================================================
def _centroid_or_none(ev: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    c = ev.get("centroid")
    if isinstance(c, (list, tuple)) and len(c) >= 2:
        try:
            return (float(c[0]), float(c[1]))
        except Exception:
            return None
    return None

def event_signatures_from_events(events: List[Dict[str, Any]]) -> List[str]:
    sigs = []
    for ev in events:
        f = str(ev.get("field_key", "")).lower()
        p = str(ev.get("polarity", "")).lower()
        if f in FIELDS and p in {"high", "low"}:
            sigs.append(event_signature(f, p))
    return sorted(set(sigs))

def snap_centroid_to_free(grid: np.ndarray, centroid: Sequence[float],
                          event_mask: Optional[np.ndarray] = None) -> Tuple[int, int]:
    grid_arr = np.asarray(grid)
    if event_mask is not None and np.asarray(event_mask).shape == grid_arr.shape and np.any(event_mask):
        cells = np.argwhere(np.asarray(event_mask, dtype=bool) & (grid_arr == 0))
    else:
        cells = np.argwhere(grid_arr == 0)
    if len(cells) == 0:
        raise ValueError("No free cell available")
    point = np.asarray(centroid[:2], dtype=float)
    d2 = np.sum((cells.astype(float) - point[None, :]) ** 2, axis=1)
    cell = cells[int(np.argmin(d2))]
    return int(cell[0]), int(cell[1])


class ObstacleAwareGeodesic:
    CARDINAL_COST = 1.0
    DIAGONAL_COST = math.sqrt(2.0)

    def __init__(self, grid: np.ndarray, connectivity: int = 8) -> None:
        if int(connectivity) not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8")
        self.grid = np.asarray(grid, dtype=np.uint8)
        self.free = self.grid == 0
        self.connectivity = int(connectivity)
        self.cache: Dict[Tuple[int, int], np.ndarray] = {}

    def _distance_map(self, source: Tuple[int, int]) -> np.ndarray:
        if source in self.cache:
            return self.cache[source]
        rows, cols = self.grid.shape
        dist = np.full((rows, cols), np.inf, dtype=np.float32)
        sr, sc = source
        if not (0 <= sr < rows and 0 <= sc < cols) or not self.free[sr, sc]:
            self.cache[source] = dist
            return dist
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
        if self.connectivity == 8:
            moves.extend([
                (-1, -1, self.DIAGONAL_COST), (-1, 1, self.DIAGONAL_COST),
                (1, -1, self.DIAGONAL_COST), (1, 1, self.DIAGONAL_COST),
            ])
        dist[sr, sc] = 0.0
        queue: List[Tuple[float, int, int]] = [(0.0, sr, sc)]
        while queue:
            current, r, c = heapq.heappop(queue)
            if current > float(dist[r, c]) + 1e-9:
                continue
            for dr, dc, cost in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or not self.free[nr, nc]:
                    continue
                if dr != 0 and dc != 0 and (not self.free[r + dr, c] or not self.free[r, c + dc]):
                    continue
                candidate = current + cost
                if candidate + 1e-9 < float(dist[nr, nc]):
                    dist[nr, nc] = candidate
                    heapq.heappush(queue, (candidate, nr, nc))
        self.cache[source] = dist
        return dist

    def distance(self, source: Tuple[int, int], target: Tuple[int, int]) -> Optional[float]:
        value = float(self._distance_map(source)[target])
        return None if not np.isfinite(value) else value


def _causal_baseline_frame_count(history_frames: int) -> int:
    """Choose a fixed-policy early-history baseline without a tuning knob."""
    if history_frames <= 0:
        return 0
    return int(min(history_frames, max(8, min(32, math.ceil(history_frames / 5.0)))))


def _temporal_baseline_from_observations(fields_by_t: List[Dict[str, np.ndarray]],
                                          field_key: str,
                                          free_mask: np.ndarray,
                                          baseline_frames: int) -> np.ndarray:
    """Observation-only temporal baseline for weak-candidate mining.

    This is not a clean simulator background and does not use GT masks/labels.
    It simply summarizes the earliest observed frames available to the diagnosis
    adapter, analogous to a warm-up normal reference in a streaming deployment.
    """
    n0 = max(1, min(int(baseline_frames), len(fields_by_t)))
    stack_items = []
    for frame in fields_by_t[:n0]:
        if field_key in frame:
            stack_items.append(np.asarray(frame[field_key], dtype=np.float32))
    if not stack_items:
        out = np.zeros_like(free_mask, dtype=np.float32)
        out[~free_mask] = np.nan
        return out
    stack = np.array(stack_items, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        base = np.nanmedian(stack, axis=0).astype(np.float32)
    valid = free_mask & np.isfinite(base)
    fill = float(np.nanmedian(base[valid])) if valid.any() else 0.0
    base[free_mask & (~np.isfinite(base))] = fill
    base[~free_mask] = np.nan
    return base.astype(np.float32)


def _robust_bias_correct_z(z: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    valid = free_mask & np.isfinite(z)
    if int(valid.sum()) < 25:
        return z.astype(np.float32)
    vals = z[valid]
    cutoff = float(np.nanpercentile(np.abs(vals), 75.0))
    stable = vals[np.abs(vals) <= max(0.35, cutoff)]
    if stable.size < 12:
        stable = vals
    bias = float(np.nanmedian(stable))
    if abs(bias) < 0.05:
        return z.astype(np.float32)
    bias = float(np.clip(bias, -0.85, 0.85))
    out = z - bias
    out[~free_mask] = np.nan
    return out.astype(np.float32)


def _nearest_confirmed_event(candidate_centroid: List[float], confirmed_events: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    if not isinstance(candidate_centroid, list) or len(candidate_centroid) < 2:
        return None
    c0 = np.array(candidate_centroid[:2], dtype=float)
    best = None
    best_dist = float("inf")
    for ev in confirmed_events or []:
        cent = ev.get("centroid")
        if not isinstance(cent, (list, tuple)) or len(cent) < 2:
            continue
        try:
            dist = float(np.linalg.norm(c0 - np.array(cent[:2], dtype=float)))
        except Exception:
            continue
        if dist < best_dist:
            best_dist = dist
            best = {
                "event_signature": event_signature(str(ev.get("field_key")), str(ev.get("polarity"))),
                "field_key": ev.get("field_key"),
                "polarity": ev.get("polarity"),
                "distance": round(float(dist), 3),
            }
    return best


def derive_weak_candidates_from_fields(fields_by_t: List[Dict[str, np.ndarray]], free_mask: np.ndarray,
                                       confirmed_events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Mine recent-window cumulative sub-threshold residuals.

    The output contains only observation-derived candidate measurements. It does
    not decide whether an anomaly is weak, compound, novel, known, or unknown.
    No simulator-only clean background, GT mask, scenario ID, accident label, or
    injected-effect specification is used.
    """
    if not fields_by_t:
        return {
            "candidate_events": [],
            "coherent_pairs": [],
            "confirmed_event_signatures": [],
            "recent_frames": 0,
            "baseline_frames": 0,
            "residual_detector_version": "adaptive_history_window_v3_label_agnostic",
            "note": "no observed frames",
        }

    history_frames = len(fields_by_t)
    baseline_frames = _causal_baseline_frame_count(history_frames)
    recent_frames = max(12, int(math.ceil(history_frames / 3.0)))
    recent = fields_by_t[-recent_frames:]
    # These are fixed statistical policy constants in normalized z units, not
    # experiment-facing hyperparameters. Window lengths are history-adaptive.
    weak_floor_z = 0.55
    weak_peak_z = 0.95
    cumulative_threshold = 0.24
    min_persistence = max(0.10, 3.0 / max(1, len(recent)))
    min_weak_area = 3
    rr, cc = np.indices(free_mask.shape)
    candidates: List[Dict[str, Any]] = []

    confirmed_sigs = sorted(event_signatures_from_events(confirmed_events or []))

    for field_key in FIELDS:
        if field_key not in recent[-1]:
            continue
        sem = FIELD_REGISTRY[field_key]
        baseline = _temporal_baseline_from_observations(fields_by_t, field_key, free_mask, baseline_frames=baseline_frames)
        scores_by_pol: Dict[str, List[np.ndarray]] = {"high": [], "low": []}

        for frame in recent:
            arr = np.asarray(frame[field_key], dtype=np.float32)
            z = (arr - baseline) / (sem.sigma + 1e-6)
            z[~free_mask] = np.nan
            z = _robust_bias_correct_z(z, free_mask)
            high = np.maximum(z, 0.0).astype(np.float32)
            low = np.maximum(-z, 0.0).astype(np.float32)
            high[~np.isfinite(high)] = 0.0
            low[~np.isfinite(low)] = 0.0
            scores_by_pol["high"].append(high)
            scores_by_pol["low"].append(low)

        for polarity, score_list in scores_by_pol.items():
            if not score_list:
                continue
            stack = np.stack(score_list, axis=0).astype(np.float32)
            peak_map = np.nanmax(stack, axis=0)
            mean_map = np.nanmean(stack, axis=0)
            persistence_map = np.nanmean(stack >= float(weak_floor_z), axis=0)
            # Cumulative weak evidence emphasizes persistent sub-threshold
            # deviations instead of requiring the last frame to cross a weak z.
            normalized = np.clip((stack - float(weak_floor_z)) / max(1e-6, float(weak_peak_z - weak_floor_z)), 0.0, 1.75)
            cumulative_map = np.nanmean(normalized, axis=0)
            evidence_map = (0.55 * cumulative_map + 0.25 * persistence_map + 0.20 * np.clip(peak_map / max(weak_peak_z, 1e-6), 0.0, 1.75)).astype(np.float32)
            evidence_map[~free_mask] = 0.0
            candidate_mask = free_mask & (peak_map >= float(weak_peak_z)) & (persistence_map >= float(min_persistence)) & (cumulative_map >= float(cumulative_threshold))
            # A nearby confirmed observation supplies a generic spatial anchor,
            # so a slightly weaker provisional measurement may be retained for
            # later graph-level corroboration. This rule is independent of field
            # semantics and downstream diagnosis labels.
            if confirmed_events:
                confirmed_mask_hint = np.zeros_like(candidate_mask, dtype=bool)
                for ev in confirmed_events:
                    cent = ev.get("centroid")
                    if not isinstance(cent, (list, tuple)) or len(cent) < 2:
                        continue
                    try:
                        d = np.sqrt((rr - float(cent[0])) ** 2 + (cc - float(cent[1])) ** 2)
                    except Exception:
                        continue
                    confirmed_mask_hint |= d <= 4.5
                candidate_mask |= free_mask & confirmed_mask_hint & (peak_map >= 0.85) & (persistence_map >= 0.08) & (cumulative_map >= max(0.18, float(cumulative_threshold) * 0.75))

            lab, n = label(candidate_mask, structure=np.ones((3, 3), dtype=np.uint8))
            if n <= 0:
                continue
            comps = []
            for idx in range(1, n + 1):
                comp = lab == idx
                area = int(comp.sum())
                if area < int(min_weak_area):
                    continue
                comp_score = float(np.nansum(evidence_map[comp]))
                comps.append((comp_score, comp))
            if not comps:
                continue
            comps.sort(key=lambda x: x[0], reverse=True)
            # Keep up to two weak regions per field/polarity to preserve
            # possible multi-region evidence without bloating the prompt.
            for comp_score, comp in comps[:2]:
                vals = evidence_map[comp]
                weights = vals + 1e-6
                centroid = [round(float(np.average(rr[comp], weights=weights)), 3),
                            round(float(np.average(cc[comp], weights=weights)), 3)]
                # Temporal trend from the spatial component mean score.
                series = np.array([float(np.nanmean(s[comp])) for s in stack], dtype=float)
                xs = np.arange(len(series), dtype=float)
                slope = 0.0
                if len(xs) >= 3 and np.isfinite(series).any():
                    denom = float(np.sum((xs - xs.mean()) ** 2))
                    if denom > 1e-9:
                        slope = float(np.sum((xs - xs.mean()) * (series - series.mean())) / denom)
                nearest_confirmed = _nearest_confirmed_event(centroid, confirmed_events)
                co_located = bool(nearest_confirmed and float(nearest_confirmed.get("distance", 999.0)) <= 4.5)
                candidates.append({
                    "field_key": field_key,
                    "polarity": polarity,
                    "candidate_type": "weak_candidate_event",
                    "event_signature": event_signature(field_key, polarity),
                    "centroid": centroid,
                    "area": int(comp.sum()),
                    "recent_window_cumulative_score": round(float(np.nanmean(cumulative_map[comp])), 3),
                    "temporal_persistence": round(float(np.nanmean(persistence_map[comp])), 3),
                    "peak_recent_z": round(float(np.nanmax(peak_map[comp])), 3),
                    "mean_recent_z": round(float(np.nanmean(mean_map[comp])), 3),
                    "evidence_score": round(float(np.nanmean(vals)), 3),
                    "trend": "strengthening" if slope > 0.004 else ("weakening" if slope < -0.004 else "stable"),
                    "co_located_with_confirmed_event": co_located,
                    "nearest_confirmed_event": nearest_confirmed,
                    "score_sum": round(float(comp_score), 3),
                })

    # Sort and deduplicate lightly. Confirmed events may already contain a field;
    # keep the weak candidate because it can still provide window statistics, but
    # the LLM is told it is not a confirmed event.
    candidates.sort(key=lambda c: float(c.get("evidence_score", 0.0) or 0.0), reverse=True)
    candidates = candidates[:12]

    coherent_pairs = []
    evidence_nodes = candidates + [
        {"event_signature": s, "field_key": s.split(":", 1)[0], "polarity": s.split(":", 1)[1], "centroid": None, "source": "confirmed_event"}
        for s in confirmed_sigs
    ]
    for i, a in enumerate(evidence_nodes):
        ca = a.get("centroid")
        for b in evidence_nodes[i + 1:]:
            if a.get("polarity") != b.get("polarity"):
                continue
            # Confirmed events without centroids still contribute field-set
            # evidence, but not spatial-pair evidence.
            cb = b.get("centroid")
            if not isinstance(ca, list) or not isinstance(cb, list):
                continue
            dist = float(np.linalg.norm(np.array(ca, dtype=float) - np.array(cb, dtype=float)))
            if dist <= 4.5:
                coherent_pairs.append({"a": a.get("event_signature"), "b": b.get("event_signature"), "distance": round(dist, 3)})

    return {
        "candidate_events": candidates,
        "coherent_pairs": coherent_pairs,
        "confirmed_event_signatures": confirmed_sigs,
        "recent_frames": int(len(recent)),
        "baseline_frames": int(baseline_frames),
        "residual_detector_version": "adaptive_history_window_v3_label_agnostic",
        "note": "Sub-threshold residual candidates are observation-derived measurements, not confirmed events or diagnosis flags.",
    }

def neutralize_subthreshold_evidence(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Expose raw sub-threshold measurements without a diagnosis-like hint.

    The model-visible view contains only candidate measurements and pairwise
    spatial relations; it contains no diagnosis-like boolean or template field set.
    """
    candidates: List[Dict[str, Any]] = []
    for candidate in raw.get("candidate_events") or []:
        candidates.append({
            "field_key": candidate.get("field_key"),
            "polarity": candidate.get("polarity"),
            "candidate_type": "subthreshold_residual",
            "event_signature": candidate.get("event_signature"),
            "centroid": candidate.get("centroid"),
            "area": candidate.get("area"),
            "recent_window_cumulative_score": candidate.get("recent_window_cumulative_score"),
            "temporal_persistence": candidate.get("temporal_persistence"),
            "peak_recent_z": candidate.get("peak_recent_z"),
            "mean_recent_z": candidate.get("mean_recent_z"),
            "trend": candidate.get("trend"),
            "co_located_with_confirmed_event": candidate.get("co_located_with_confirmed_event"),
            "nearest_confirmed_event": candidate.get("nearest_confirmed_event"),
        })

    persistences = [float(c.get("temporal_persistence")) for c in candidates if c.get("temporal_persistence") is not None]
    peaks = [float(c.get("peak_recent_z")) for c in candidates if c.get("peak_recent_z") is not None]
    return {
        "candidate_events": candidates,
        "pairwise_spatial_relations": list(raw.get("coherent_pairs") or []),
        "summary": {
            "candidate_count": int(len(candidates)),
            "distinct_candidate_fields": sorted({str(c.get("field_key")) for c in candidates if c.get("field_key")}),
            "co_located_candidate_count": int(sum(1 for c in candidates if c.get("co_located_with_confirmed_event"))),
            "mean_temporal_persistence": round(float(np.mean(persistences)), 3) if persistences else 0.0,
            "max_peak_recent_z": round(float(max(peaks)), 3) if peaks else 0.0,
            "recent_frames": int(raw.get("recent_frames", 0) or 0),
            "baseline_frames": int(raw.get("baseline_frames", 0) or 0),
        },
        "feature_policy": "raw_subthreshold_measurements_v1",
        "note": (
            "Candidates are observation-derived residual measurements, not confirmed events "
            "and not a low-SNR diagnosis flag."
        ),
    }


def _node_support_radius(node: Dict[str, Any]) -> float:
    """Equivalent circular support radius derived directly from observed area."""
    try:
        area = max(1.0, float(node.get("area") or 1.0))
    except Exception:
        area = 1.0
    return float(math.sqrt(area / math.pi))


def _adaptive_link_limit(node_a: Dict[str, Any], node_b: Dict[str, Any]) -> float:
    """Link overlapping measurement supports without a global distance knob."""
    return float(max(1.0, _node_support_radius(node_a) + _node_support_radius(node_b)))


def build_adaptive_evidence_graph(confirmed_events: List[Dict[str, Any]],
                                  residual_evidence: Dict[str, Any],
                                  grid: np.ndarray,
                                  history_frames: int) -> Dict[str, Any]:
    """Build the final compact graph directly from observed evidence.

    The transformation is label agnostic: it uses only node measurements,
    confirmation tier, temporal support, and obstacle-aware distance. Residual
    nodes duplicate neither confirmed field/polarity signatures nor diagnosis
    decisions. Spatial grouping uses overlap of observation-derived support
    radii, so there is no experiment-facing distance or clustering parameter.
    """
    confirmed_nodes: List[Dict[str, Any]] = []
    for event in confirmed_events:
        centroid = _centroid_or_none(event)
        if centroid is None:
            continue
        summary = summarize_events([event], max_events=1, history_frames=history_frames)[0]
        anchor = snap_centroid_to_free(grid, centroid, event.get("mask"))
        confirmed_nodes.append({
            "node_id": f"E{len(confirmed_nodes) + 1}",
            **summary,
            "centroid": [round(float(x), 3) for x in centroid],
            "geodesic_anchor": [int(anchor[0]), int(anchor[1])],
        })
    confirmed_signatures = {
        event_signature(str(node.get("field_key")), str(node.get("polarity")))
        for node in confirmed_nodes
        if node.get("field_key") in FIELDS and node.get("polarity") in {"high", "low"}
    }
    candidates = [
        dict(candidate)
        for candidate in (residual_evidence.get("candidate_events") or [])
        if candidate.get("event_signature") not in confirmed_signatures
    ]
    geodesic = ObstacleAwareGeodesic(grid, connectivity=8)
    candidate_anchors: List[Optional[Tuple[int, int]]] = []
    for candidate in candidates:
        centroid = candidate.get("centroid")
        if isinstance(centroid, list) and len(centroid) >= 2:
            candidate_anchors.append(snap_centroid_to_free(grid, [float(centroid[0]), float(centroid[1])]))
        else:
            candidate_anchors.append(None)
    corroborated_candidates: set = set()
    for i, candidate_a in enumerate(candidates):
        anchor_a = candidate_anchors[i]
        if anchor_a is None:
            continue
        for j, candidate_b in enumerate(candidates[i + 1:], start=i + 1):
            anchor_b = candidate_anchors[j]
            if anchor_b is None or candidate_a.get("field_key") == candidate_b.get("field_key"):
                continue
            distance = geodesic.distance(anchor_a, anchor_b)
            if distance is not None and float(distance) <= _adaptive_link_limit(candidate_a, candidate_b):
                corroborated_candidates.update((i, j))
    # A sub-threshold node is exposed only when another field independently
    # corroborates it nearby. This suppresses isolated noise without consulting
    # field semantics, incident templates, scenario metadata, or labels.
    candidates = [candidate for index, candidate in enumerate(candidates) if index in corroborated_candidates]

    visible_nodes: List[Dict[str, Any]] = []
    internal_nodes: List[Dict[str, Any]] = []
    for node in confirmed_nodes:
        centroid = node.get("centroid")
        if not isinstance(centroid, list) or len(centroid) < 2:
            continue
        centroid_pair = [float(centroid[0]), float(centroid[1])]
        anchor_data = node.get("geodesic_anchor")
        anchor = (
            (int(anchor_data[0]), int(anchor_data[1]))
            if isinstance(anchor_data, list) and len(anchor_data) >= 2
            else snap_centroid_to_free(grid, centroid_pair)
        )
        visible = {
            "node_id": node.get("node_id"),
            "tier": "confirmed_track",
            "field_key": node.get("field_key"),
            "polarity": node.get("polarity"),
            "centroid": [round(x, 3) for x in centroid_pair],
            "area": node.get("area"),
            "strength_z": node.get("z_core_mean"),
            "temporal_support": node.get("track_coverage_fraction"),
            "duration_steps": node.get("duration_steps"),
            "area_trend": node.get("area_trend"),
            "intensity_trend": node.get("intensity_trend"),
        }
        visible_nodes.append(visible)
        internal_nodes.append({**visible, "anchor": anchor})

    residual_node_count = 0
    for index, candidate in enumerate(candidates, start=1):
        centroid = candidate.get("centroid")
        if not isinstance(centroid, list) or len(centroid) < 2:
            continue
        centroid_pair = [float(centroid[0]), float(centroid[1])]
        anchor = snap_centroid_to_free(grid, centroid_pair)
        node_id = f"R{index}"
        visible = {
            "node_id": node_id,
            "tier": "residual_observation",
            "field_key": candidate.get("field_key"),
            "polarity": candidate.get("polarity"),
            "centroid": [round(x, 3) for x in centroid_pair],
            "area": candidate.get("area"),
            "mean_recent_z": candidate.get("mean_recent_z"),
            "peak_recent_z": candidate.get("peak_recent_z"),
            "temporal_support": candidate.get("temporal_persistence"),
            "cumulative_score": candidate.get("recent_window_cumulative_score"),
            "trend": candidate.get("trend"),
        }
        residual_node_count += 1
        visible_nodes.append(visible)
        internal_nodes.append({**visible, "anchor": anchor})

    pair_distances: Dict[Tuple[int, int], Optional[float]] = {}
    for i, node_a in enumerate(internal_nodes):
        for j, node_b in enumerate(internal_nodes[i + 1:], start=i + 1):
            distance = geodesic.distance(node_a["anchor"], node_b["anchor"])
            pair_distances[(i, j)] = distance

    parent = list(range(len(internal_nodes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    sparse_edges: List[Dict[str, Any]] = []
    for (i, j), distance in pair_distances.items():
        node_a, node_b = internal_nodes[i], internal_nodes[j]
        link_limit = _adaptive_link_limit(node_a, node_b)
        if distance is None or float(distance) > link_limit:
            continue
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i
        sparse_edges.append({
            "source": node_a.get("node_id"),
            "target": node_b.get("node_id"),
            "geodesic_distance": round(float(distance), 3),
            "adaptive_link_limit": round(float(link_limit), 3),
            "cross_field": node_a.get("field_key") != node_b.get("field_key"),
        })

    members_by_root: Dict[int, List[int]] = {}
    for index in range(len(internal_nodes)):
        members_by_root.setdefault(find(index), []).append(index)
    ordered_groups = sorted(members_by_root.values(), key=lambda members: min(members))
    node_to_group: Dict[int, str] = {}
    evidence_groups: List[Dict[str, Any]] = []
    for number, members in enumerate(ordered_groups, start=1):
        group_id = f"G{number}"
        for index in members:
            node_to_group[index] = group_id
        group_nodes = [internal_nodes[index] for index in members]
        finite_internal = [
            float(pair_distances[(min(i, j), max(i, j))])
            for offset, i in enumerate(members)
            for j in members[offset + 1:]
            if pair_distances.get((min(i, j), max(i, j))) is not None
        ]
        supports = [float(node.get("temporal_support")) for node in group_nodes if node.get("temporal_support") is not None]
        points = np.asarray([node["centroid"] for node in group_nodes], dtype=float)
        field_polarities = sorted({f"{node.get('field_key')}:{node.get('polarity')}" for node in group_nodes})
        confirmed_count = sum(node.get("tier") == "confirmed_track" for node in group_nodes)
        residual_count = sum(node.get("tier") == "residual_observation" for node in group_nodes)
        amplitude_regime = "confirmed_only" if residual_count == 0 else ("residual_only" if confirmed_count == 0 else "mixed")
        evidence_groups.append({
            "group_id": group_id,
            "node_ids": [node.get("node_id") for node in group_nodes],
            "field_polarities": field_polarities,
            "distinct_field_count": len({node.get("field_key") for node in group_nodes}),
            "confirmed_node_count": confirmed_count,
            "residual_node_count": residual_count,
            "amplitude_regime": amplitude_regime,
            "centroid": [round(float(x), 3) for x in np.mean(points, axis=0)],
            "max_internal_distance": round(max(finite_internal), 3) if finite_internal else 0.0,
            "mean_temporal_support": round(float(np.mean(supports)), 3) if supports else None,
            "min_temporal_support": round(float(np.min(supports)), 3) if supports else None,
        })

    inter_group_relations: List[Dict[str, Any]] = []
    for group_index, members_a in enumerate(ordered_groups):
        for members_b in ordered_groups[group_index + 1:]:
            distances = [
                pair_distances.get((min(i, j), max(i, j)))
                for i in members_a for j in members_b
            ]
            finite = [float(distance) for distance in distances if distance is not None]
            inter_group_relations.append({
                "source": node_to_group[members_a[0]],
                "target": node_to_group[members_b[0]],
                "minimum_geodesic_distance": round(min(finite), 3) if finite else None,
                "path_exists": bool(finite),
            })

    summary = dict(residual_evidence.get("summary") or {})
    total_confirmed = len(visible_nodes) - residual_node_count
    graph_amplitude_regime = (
        "none" if not visible_nodes else
        ("confirmed_only" if residual_node_count == 0 else ("residual_only" if total_confirmed == 0 else "mixed"))
    )
    graph = {
        "nodes": visible_nodes,
        "proximity_edges": sparse_edges,
        "evidence_groups": evidence_groups,
        "inter_group_relations": inter_group_relations,
        "observation_summary": {
            "node_count": len(visible_nodes),
            "confirmed_node_count": total_confirmed,
            "residual_node_count": residual_node_count,
            "amplitude_regime": graph_amplitude_regime,
            "distinct_fields": sorted({str(node.get("field_key")) for node in visible_nodes if node.get("field_key") in FIELDS}),
            "group_count": len(evidence_groups),
            "residual_recent_frames": summary.get("recent_frames"),
            "residual_baseline_frames": summary.get("baseline_frames"),
        },
        "graph_metadata": {
            "distance_metric": "obstacle_aware_geodesic",
            "connectivity": 8,
            "grouping": "adaptive_support_radius_overlap_components",
            "link_scale_source": "sum_of_observation_equivalent_radii",
            "distance_unit": "grid_cell",
            "tier_policy": "confirmed_tracks_plus_nonduplicating_residual_observations",
        },
        "feature_policy": "label_agnostic_parameter_free_adaptive_evidence_graph_v4",
    }
    return graph


def offline_llm_proxy(payload: Dict[str, Any], mode: str) -> Dict[str, Any]:
    # Deterministic proxy for dry-run pipeline validation. It deliberately uses
    # only the same public input representation that would be sent to the model.
    events = payload.get("events") or []
    field_summary = payload.get("field_summary") or {}
    guessed = diagnose_by_rules(events, source_name=f"offline_{mode}", observation_quality=payload.get("observation_quality")) if events else diagnose_from_matrix_summary(field_summary)
    guessed["explanation"] = f"Offline proxy ({mode}); replace with --api-mode api for real LLM/VLM measurement. " + guessed.get("explanation", "")
    return guessed

def diagnose_from_matrix_summary(field_summary: Dict[str, Any]) -> Dict[str, Any]:
    events = []
    for k, s in field_summary.items():
        if s.get("high_z_cells", 0) > 2:
            events.append({"field_key": k, "polarity": "high", "z_core_mean": s.get("max_abs_spatial_z", 0), "priority": s.get("max_abs_spatial_z", 0), "centroid": None})
        if s.get("low_z_cells", 0) > 2:
            events.append({"field_key": k, "polarity": "low", "z_core_mean": s.get("max_abs_spatial_z", 0), "priority": s.get("max_abs_spatial_z", 0), "centroid": None})
    return diagnose_by_rules(events, source_name="raw_matrix_summary", observation_quality=field_summary.get("observation_quality") if isinstance(field_summary, dict) else None)

COMMON_TASK_PROMPT = (
    "You are an industrial multi-physics accident diagnosis module. Return only one valid JSON object without markdown. "
    f"Allowed accident_type labels: {', '.join(ACCIDENT_TYPES)}. "
    "Use this common incident ontology for every representation: fire is material full-amplitude high temperature with combustion-side air-quality and/or CO2 evidence; "
    "electrical_overheat is primarily material full-amplitude isolated high temperature without combustion-side or humidity support; "
    "water_leak is material full-amplitude humidity-dominant evidence without the full thermal-pressure steam pattern; "
    "steam_leak is material full-amplitude high humidity with high temperature and high pressure; "
    "co2_accumulation requires material full-amplitude CO2-specific evidence; dust_pollution is material full-amplitude air-quality evidence without CO2 support; "
    "composite_anomaly requires two or more spatially distinct evidence groups supporting different incident patterns; "
    "low_snr_anomaly is a cross-field abnormal pattern supported only by mutually corroborating, temporally persistent sub-threshold measurements; "
    "it is distinct from the same field combination supported by confirmed tracks; "
    "needs_review_unknown means observation quality is insufficient, evidence is contradictory, or no known template adequately explains the material evidence; "
    "normal means no material abnormal evidence under adequate observation quality. "
    "Use only the supplied observation at the fixed diagnosis time. Never assume scenario identifiers, testcase names, injected effects, expected labels, "
    "acceptable or forbidden predictions, hidden ground truth, clean backgrounds, or test-set-specific combinations. "
    "Use this representation-neutral evidence-accounting protocol before choosing a label: (1) assess the reliability and completeness of the supplied "
    "observation; (2) inventory every material observation-derived evidence unit that the supplied representation actually contains; (3) test whether one "
    "known physical process parsimoniously explains all such evidence; (4) mark reliable evidence that is not explained by that process as "
    "supports_alternative or unexplained instead of inventing an unsupported secondary cause. Do not assume evidence that the representation does not provide. "
    "For any representation that distinguishes evidence tiers, corroboration across fields, temporal persistence, and spatial organization can make a set of "
    "individually sub-threshold measurements material, whereas one isolated or transient sub-threshold measurement is insufficient by itself. "
    "known_template is valid only when no material evidence remains unexplained. A single defining field can support an incident hypothesis when the ontology is "
    "field-dominant, but composite_anomaly still requires spatially distinct groups that support different incident patterns. Spatial separation alone is not "
    "sufficient for composite_anomaly when the separated measurements are complementary parts of one known coupled pattern. If any material evidence is assigned "
    "supports_alternative or unexplained, known_template is prohibited: select composite_anomaly only when spatially distinct evidence supports a second incident "
    "pattern, otherwise select needs_review_unknown with review_needed=true. "
    "Apply the same review policy to every representation: unreliable or materially missing observations require review; when supplied observation_quality has "
    "low_confidence=true, last_frame_missing_fraction>=0.30, or max_missing_fraction>=0.30, prefer needs_review_unknown, review_needed=true, "
    "abnormal_confirmed=false, template_status=insufficient_data, and confidence<=0.45. Explicit quality or noise measurements are reliability metadata rather "
    "than confirmed incidents. Absence of material evidence favors normal only when the supplied representation has adequate coverage and reliability; absence "
    "alone is not conclusive when extraction, visibility, or measurement quality is degraded. "
    "Confidence must reflect only supplied evidence and must be lower for missing, contradictory, or weak evidence. "
    "Return JSON keys in exactly this evidence-first order so the decision follows the audit: {"
    "\"evidence_fields\":[str],\"evidence_events\":[{\"field_key\":str,\"polarity\":\"high\"|\"low\","
    "\"role\":\"supports_selected\"|\"supports_alternative\"|\"unexplained\"}],\"unexplained_evidence\":[str],"
    "\"template_status\":\"known_template\"|\"composite\"|\"novel_combination\"|\"insufficient_data\",\"accident_type\":str,"
    "\"confidence\":float,\"review_needed\":bool,\"abnormal_confirmed\":bool,"
    "\"resample_target\":{\"field_key\":str|null,\"centroid\":[row,col]|null}|null,\"explanation\":str}. "
    "The shared output field evidence_events is a modality-neutral container for material field/polarity evidence items directly supported by the supplied "
    "representation; do not invent attributes that the input does not provide. Explain what supports the decision and explicitly identify unexplained or "
    "alternative evidence."
)

REPRESENTATION_ADAPTERS: Dict[str, str] = {
    "threshold_alarm_matrix": (
        "Representation adapter: threshold_alarm_matrix contains one full fixed-limit alarm matrix per physical field; it is not an event representation. "
        "Each matrix row uses H=above the fixed upper control limit, L=below the fixed lower control limit, .=within limits, ?=missing, and #=obstacle. "
        "The Phase-I calibration policy and field-level alarm counts are supplied explicitly. Do not infer connected components, centroids, morphology, tracks, "
        "duration, or temporal trends. The output evidence_events container may record only field/polarity alarm evidence directly visible in these matrices, "
        "without event attributes. No active field alarm means the predeclared matrix-level alarm count was not reached at the diagnosis frame; it does not prove "
        "that every historical measurement was anomaly-free."
    ),
    "f2e_events": (
        "Representation adapter: events are active, confirmed F2E tracks at the diagnosis time, not an archive of every event from every processed frame. "
        "track_id identifies a track; duration_steps and track_coverage_fraction describe its observed persistence; area/intensity start, current, and trend "
        "values come from track history; centroid uses one shared grid coordinate system across fields. Each supplied track maps directly to one evidence_events "
        "item. An empty list means no active track crossed F2E confirmation "
        "at this time, not that raw fields were proven anomaly-free throughout history. high_noise_fields and temporal-noise statistics, when supplied, describe "
        "track-extraction reliability rather than incident evidence; widespread noise makes an empty active-track list weaker negative evidence."
    ),
    "f2e_events_graph": (
        "Representation adapter: the observation is a compact unified F2E evidence graph. Each node is either a confirmed_track or a residual_observation; "
        "the latter is a causal sub-threshold measurement and is not equivalent to a confirmed track. temporal_support is track coverage for confirmed nodes and "
        "recent-window persistence for residual nodes. proximity_edges and evidence_groups use fixed obstacle-aware geodesic distance and an adaptive link limit "
        "equal to the sum of the two observation-derived support radii; there is no experiment-supplied merge or clustering parameter. "
        "They summarize measurement organization, not incident semantics or expected decisions. A residual field/polarity already represented by a confirmed node "
        "is omitted to prevent double counting. amplitude_regime is a direct tier-count summary, not a diagnosis. Use every reliable node in the evidence ledger. "
        "Persistent cross-field corroboration can make residual observations material; an isolated or transient residual observation should be down-weighted. "
        "A residual_only group can establish a real abnormal pattern, but it is not full-amplitude evidence and therefore cannot satisfy a full-amplitude incident "
        "template. Its amplitude tier must remain part of the classification. Treat nodes connected into one evidence group as spatially associated, "
        "while using inter_group_relations to assess genuinely separated activity. Multiple groups do not automatically imply a composite incident because separated "
        "measurements may still be complementary parts of one physical process. Apply the shared ontology independently to each reliable evidence group before the "
        "global decision; field-dominant ontology entries remain field-dominant at group level. If separate groups support different incident patterns, account for "
        "both rather than discarding a smaller group. Conversely, proximity does not make one field semantically explain another. Do not invent a causal coupling "
        "that is absent from the supplied ontology merely to force known_template. If a supplied field/polarity is not explained by the selected ontology, retain it "
        "as supports_alternative or unexplained. An empty graph means neither a "
        "confirmed track nor an admitted residual observation was available at diagnosis time; interpret that absence together with observation_quality."
    ),
    "raw_matrix_summary": (
        "Representation adapter: field_summary contains statistics rather than full matrices. mean, min, max, and std are current-frame spatial statistics; "
        "median_delta_from_first is the median change from the first observed frame; max_abs_spatial_z is the largest absolute current spatial z-score; "
        "high_z_cells and low_z_cells count current cells beyond the positive and negative z thresholds. An evidence_events item may be reported only when these "
        "supplied statistics directly support its field and polarity; location or morphology is unavailable."
    ),
    "raw_field_image": (
        "Representation adapter: the image is a temporal contact sheet. Rows are physical fields and columns are sampled time points. All panels in one field "
        "use one consistent color scale. Obstacles are dark gray and missing free-space observations are magenta. Field names and time indices are printed on the "
        "image. An evidence_events item may be reported only for a field/polarity pattern visibly supported by the supplied panels; numeric event attributes are unavailable."
    ),
}


def _shared_diagnosis_system(input_kind: str) -> str:
    adapter = REPRESENTATION_ADAPTERS.get(input_kind)
    if adapter is None:
        raise ValueError(f"Unknown representation adapter: {input_kind}")
    return COMMON_TASK_PROMPT + "\n\n" + adapter


def build_diagnosis_prompt(input_kind: str, scenario_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    user = {
        "input_kind": input_kind,
        "task": "Diagnose the accident type and propose review/resampling action from the provided no-leak observation representation.",
        "payload": scenario_payload,
    }
    return [
        {"role": "system", "content": _shared_diagnosis_system(input_kind)},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def build_vlm_messages(image_data_url: str, scenario_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = json.dumps({
        "input_kind": "raw_field_image",
        "task": "Diagnose the accident type and propose review/resampling action from the provided no-leak observation representation.",
        "payload": scenario_payload,
    }, ensure_ascii=False)
    return [
        {"role": "system", "content": _shared_diagnosis_system("raw_field_image")},
        {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]},
    ]


def save_prompt_templates(out_dir: str) -> None:
    prompt_dir = os.path.join(out_dir, "prompts")
    os.makedirs(prompt_dir, exist_ok=True)
    files = {
        "f2e_graph_llm.txt": "f2e_events_graph",
        "f2e_llm.txt": "f2e_events",
        "threshold_llm.txt": "threshold_alarm_matrix",
        "matrix_llm.txt": "raw_matrix_summary",
        "raw_image_vlm.txt": "raw_field_image",
    }
    for filename, input_kind in files.items():
        with open(os.path.join(prompt_dir, filename), "w", encoding="utf-8") as handle:
            handle.write(_shared_diagnosis_system(input_kind) + "\n\n")
            handle.write("USER MESSAGE TEMPLATE\n")
            handle.write(json.dumps({
                "input_kind": input_kind,
                "task": "Diagnose the accident type and propose review/resampling action from the provided no-leak observation representation.",
                "payload": "<runtime observation payload>",
            }, ensure_ascii=False, indent=2) + "\n")


# ============================================================
# Layer 2 diagnosis
# ============================================================
def method_display_name(method: str) -> str:
    names = {
        "f2e_events_llm": "F2E + LLM",
        "threshold_matrix_llm": "Threshold + LLM",
        "raw_matrix_summary_llm": "Raw Matrix Summary + LLM",
        "threshold_matrix_rules": "Threshold + Rules",
        "f2e_events_rules": "F2E + Rules",
        "raw_field_image_vlm": "Raw Field Image + VLM",
    }
    if method == "f2e_events_graph_llm":
        return "F2E Parameter-free Adaptive Evidence Graph + LLM"
    return names.get(method, method)


def normalize_diag(diag: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(diag, dict):
        return {"accident_type": "needs_review_unknown", "raw_prediction": None, "confidence": 0.0, "review_needed": True, "abnormal_confirmed": False, "resample_target": None, "evidence_fields": [], "evidence_events": [], "unexplained_evidence": [], "template_status": "invalid_output", "protocol_conflict": True, "protocol_conflict_reasons": ["invalid_or_missing_model_output"], "protocol_consistency_repair": False, "explanation": "invalid or missing model output"}
    raw_label = str(diag.get("accident_type", "needs_review_unknown"))
    label = raw_label
    if label not in ACCIDENT_TYPES:
        label = "needs_review_unknown"
    try:
        conf = float(diag.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    evidence_events = diag.get("evidence_events", []) if isinstance(diag.get("evidence_events", []), list) else []
    unexplained_evidence = diag.get("unexplained_evidence", []) if isinstance(diag.get("unexplained_evidence", []), list) else []
    template_status = str(diag.get("template_status", ""))
    review_needed = bool(diag.get("review_needed", label == "needs_review_unknown"))
    abnormal_confirmed = bool(diag.get("abnormal_confirmed", label not in {"normal", "needs_review_unknown"} and conf >= 0.55))
    non_selected_roles = any(
        isinstance(item, dict) and item.get("role") in {"supports_alternative", "unexplained"}
        for item in evidence_events
    )
    known_incident_labels = {
        "fire", "electrical_overheat", "water_leak", "steam_leak",
        "co2_accumulation", "dust_pollution",
    }
    conflict_reasons: List[str] = []
    if raw_label not in ACCIDENT_TYPES:
        conflict_reasons.append("invalid_accident_type")
    if label in known_incident_labels and unexplained_evidence:
        conflict_reasons.append("known_label_with_unexplained_evidence")
    if label in known_incident_labels and non_selected_roles:
        conflict_reasons.append("known_label_with_non_selected_evidence_role")
    if label in known_incident_labels and template_status in {"novel_combination", "insufficient_data"}:
        conflict_reasons.append("known_label_with_non_known_template_status")
    if label in known_incident_labels and review_needed:
        conflict_reasons.append("known_label_with_review_needed")
    if label == "needs_review_unknown" and not review_needed:
        conflict_reasons.append("review_label_without_review_needed")
    if label == "normal" and abnormal_confirmed:
        conflict_reasons.append("normal_label_with_abnormal_confirmed")
    return {
        "accident_type": label,
        "raw_prediction": raw_label,
        "confidence": max(0.0, min(1.0, conf)),
        "review_needed": review_needed,
        "abnormal_confirmed": abnormal_confirmed,
        "resample_target": diag.get("resample_target"),
        "evidence_fields": diag.get("evidence_fields", []) if isinstance(diag.get("evidence_fields", []), list) else [],
        "evidence_events": evidence_events,
        "unexplained_evidence": unexplained_evidence,
        "template_status": template_status,
        "protocol_conflict": bool(conflict_reasons),
        "protocol_conflict_reasons": conflict_reasons,
        # Kept as a stable legacy column: v8.6.2 never rewrites valid labels.
        "protocol_consistency_repair": False,
        "explanation": str(diag.get("explanation", "")),
    }

def explanation_score(diag: Dict[str, Any], scenario: AccidentScenario) -> float:
    req = list(scenario.explanation_fields)
    if not req:
        return 1.0 if diag.get("accident_type") == "normal" else 0.0
    text = (diag.get("explanation", "") + " " + " ".join(map(str, diag.get("evidence_fields", [])))).lower()
    hits = sum(1 for f in req if f.lower() in text)
    return hits / max(1, len(req))

def f1_from_sets(expected: set, reported: set) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not expected and not reported:
        return 1.0, 1.0, 1.0
    if not expected:
        return None, 0.0 if reported else 1.0, None
    tp = len(expected & reported)
    precision = tp / len(reported) if reported else 0.0
    recall = tp / len(expected)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def scenario_expected_signatures(scenario: AccidentScenario) -> set:
    if scenario.expected_event_signatures:
        return set(scenario.expected_event_signatures)
    return {event_signature(e.field_key, e.polarity) for e in scenario.testcase.effects}

def signature_field(sig: str) -> str:
    return sig.split(":", 1)[0]

def _text_mentions_field(text: str, field_key: str) -> bool:
    aliases = {
        "temperature": ("temperature", "thermal", "heat", "hotspot", "hot"),
        "humidity": ("humidity", "moisture", "wet", "dry"),
        "pressure": ("pressure", "pressure_rise", "pressure_drop"),
        "co2": ("co2", "carbon dioxide"),
        "air_quality": ("air_quality", "air quality", "aqi", "dust", "particulate"),
    }
    low = text.lower()
    return any(alias in low for alias in aliases.get(field_key, (field_key,)))

def reported_evidence_fields(diag: Dict[str, Any]) -> set:
    fields = set()
    for item in diag.get("evidence_fields", []):
        text = str(item).lower()
        for f in FIELDS:
            if f.lower() == text or _text_mentions_field(text, f):
                fields.add(f)
    for item in diag.get("evidence_events", []):
        if isinstance(item, dict):
            f = str(item.get("field_key", "")).lower()
            if f in FIELDS:
                fields.add(f)
    text = (diag.get("explanation", "") or "")
    for f in FIELDS:
        if _text_mentions_field(text, f):
            fields.add(f)
    return fields

def reported_event_signatures(diag: Dict[str, Any]) -> set:
    signatures = set()
    for item in diag.get("evidence_events", []):
        if not isinstance(item, dict):
            continue
        field_key = str(item.get("field_key", "")).lower()
        polarity = str(item.get("polarity", "")).lower()
        if field_key in FIELDS and polarity in {"high", "low"}:
            signatures.add(event_signature(field_key, polarity))
    for item in diag.get("evidence_fields", []):
        text = str(item).lower().replace(" ", "")
        if ":" in text:
            field_key, polarity = text.split(":", 1)
            if field_key in FIELDS and polarity in {"high", "low"}:
                signatures.add(event_signature(field_key, polarity))
    return signatures

def evidence_alignment_scores(diag: Dict[str, Any], scenario: AccidentScenario) -> Dict[str, Any]:
    expected_signatures = scenario_expected_signatures(scenario)
    expected_fields = {signature_field(sig) for sig in expected_signatures}
    reported_fields = reported_evidence_fields(diag)
    reported_signatures = reported_event_signatures(diag)
    fp, fr, ff1 = f1_from_sets(expected_fields, reported_fields)
    sp, sr, sf1 = f1_from_sets(expected_signatures, reported_signatures)

    if not expected_signatures:
        partial_signature = 1.0 if not reported_signatures else 0.0
    else:
        partial_scores = []
        for sig in expected_signatures:
            if sig in reported_signatures:
                partial_scores.append(1.0)
            elif signature_field(sig) in reported_fields:
                partial_scores.append(0.5)
            else:
                partial_scores.append(0.0)
        partial_signature = float(np.mean(partial_scores)) if partial_scores else 1.0

    return {
        "expected_evidence_fields": sorted(expected_fields),
        "expected_event_signatures": sorted(expected_signatures),
        "reported_evidence_fields": sorted(reported_fields),
        "reported_event_signatures": sorted(reported_signatures),
        "evidence_field_precision": fp,
        "evidence_field_recall": fr,
        "evidence_field_f1": ff1,
        "evidence_signature_precision": sp,
        "evidence_signature_recall": sr,
        "evidence_signature_f1": sf1,
        "evidence_signature_alignment": partial_signature,
    }

def behavior_goal_score(diag: Dict[str, Any], scenario: AccidentScenario) -> float:
    pred = str(diag.get("accident_type", "needs_review_unknown"))
    review = bool(diag.get("review_needed"))
    acceptable = set(scenario.acceptable_predictions or (scenario.accident_type,))
    goal = scenario.diagnostic_goal

    if goal in {"template_rejection", "review_low_confidence"}:
        if pred == "needs_review_unknown" and review:
            return 1.0
        if pred in acceptable and review:
            return 0.85
        if review:
            return 0.45
        return 0.0
    if goal == "weak_evidence":
        if pred == "low_snr_anomaly":
            return 1.0
        if pred == "needs_review_unknown" and review:
            return 0.70
        if review:
            return 0.40
        return 0.0
    if goal == "composite":
        if pred == "composite_anomaly":
            return 1.0
        if pred == "needs_review_unknown" and review:
            return 0.55
        return 0.0
    return 1.0 if pred in acceptable else 0.0

def review_calibration_score(diag: Dict[str, Any], scenario: AccidentScenario) -> float:
    review = bool(diag.get("review_needed"))
    pred = str(diag.get("accident_type", ""))
    if scenario.expected_review:
        return 1.0 if review else 0.0
    if pred == scenario.accident_type and not review:
        return 1.0
    if pred == scenario.accident_type and review:
        return 0.80
    return 0.50 if review else 0.75

def resample_target_score(diag: Dict[str, Any], scenario: AccidentScenario, fields: set) -> Optional[float]:
    if not scenario.expected_review or not scenario.target_field:
        return None
    target = diag.get("resample_target")
    if isinstance(target, dict) and str(target.get("field_key", "")).lower() == scenario.target_field:
        return 1.0
    if scenario.target_field in fields:
        return 0.50
    return 0.0

def forbidden_prediction_score(diag: Dict[str, Any], scenario: AccidentScenario) -> float:
    pred = str(diag.get("accident_type", ""))
    return 0.0 if pred in set(scenario.forbidden_predictions) else 1.0

def reasoning_alignment_scores(diag: Dict[str, Any], scenario: AccidentScenario) -> Dict[str, Any]:
    ev = evidence_alignment_scores(diag, scenario)
    behavior = behavior_goal_score(diag, scenario)
    review = review_calibration_score(diag, scenario)
    resample = resample_target_score(diag, scenario, set(ev["reported_evidence_fields"]))
    forbidden = forbidden_prediction_score(diag, scenario)
    resample_component = 1.0 if resample is None else resample
    alignment = (
        0.35 * behavior
        + 0.25 * float(ev["evidence_signature_alignment"])
        + 0.20 * review
        + 0.10 * resample_component
        + 0.10 * forbidden
    )
    template_rejection_success = (
        scenario.diagnostic_goal == "template_rejection"
        and diag.get("accident_type") == "needs_review_unknown"
        and bool(diag.get("review_needed"))
    )
    return {
        **ev,
        "behavior_goal_score": behavior,
        "review_calibration_score": review,
        "resample_target_match": resample,
        "forbidden_prediction_score": forbidden,
        "reasoning_alignment_score": alignment,
        "template_rejection_success": int(template_rejection_success),
    }

def token_estimate_from_payload(payload: Any) -> int:
    # Coarse proxy used only when explicitly requested through --token-fallback.
    return max(1, int(len(json.dumps(payload, ensure_ascii=False)) / 4))

def call_or_offline(client: DashScopeClient, api_mode: str, model: str, messages: List[Dict[str, Any]],
                    offline_payload: Dict[str, Any], mode: str, token_counter: TokenFallbackCounter,
                    max_tokens: int = 512) -> ModelCallResult:
    use_api = api_mode == "api" or (api_mode == "auto" and bool(client.api_key))
    if not use_api:
        t0 = time.perf_counter()
        parsed = offline_llm_proxy(offline_payload, mode)
        res = ModelCallResult(
            True,
            json.dumps(parsed, ensure_ascii=False),
            parsed,
            (time.perf_counter() - t0) * 1000.0,
            model="offline_proxy",
            token_count_source="offline_no_api_usage",
        )
        return apply_token_fallback(res, messages, parsed, token_counter)
    res = client.call(model, messages, max_tokens=max_tokens, temperature=0.0)
    if res.ok and res.parsed:
        return apply_token_fallback(res, messages, res.parsed, token_counter)
    # A failed formal API call is an invalid model call, never a rule prediction.
    return apply_token_fallback(res, messages, res.content or "", token_counter)

def build_method_inputs(trace_f2e: RunTrace, image_dir: str,
                        vlm_frame_mode: str = "sampled", vlm_num_frames: int = 8,
                        llm_interval_steps: int = 50, scenario_index: int = 0,
                        render_raw_image: bool = True,
                        threshold_only: bool = False) -> Dict[str, Dict[str, Any]]:
    diag_t = diagnosis_query_t(trace_f2e.scenario, len(trace_f2e.fields_window), llm_interval_steps)
    history_frames = int(diag_t) + 1
    f2e_events_raw = [] if threshold_only else snapshot_events_at(trace_f2e, diag_t)
    f2e_events = summarize_events(f2e_events_raw, history_frames=history_frames)
    fields_for_diagnosis = trace_f2e.fields_window[:diag_t + 1] or trace_f2e.fields_window
    threshold_t0 = time.perf_counter()
    threshold_alarm_matrix = build_threshold_alarm_matrices(fields_for_diagnosis, trace_f2e.grid)
    threshold_latency_ms = (time.perf_counter() - threshold_t0) * 1000.0
    matrix_t0 = time.perf_counter()
    matrix_summary = {} if threshold_only else current_field_summary(fields_for_diagnosis, trace_f2e.grid == 0)
    obs_quality = observation_quality(fields_for_diagnosis, trace_f2e.grid == 0)
    if not threshold_only:
        matrix_summary["observation_quality"] = obs_quality
    matrix_latency_ms = (time.perf_counter() - matrix_t0) * 1000.0
    graph_t0 = time.perf_counter()
    if threshold_only:
        f2e_graph_features = {}
    else:
        residual_raw = derive_weak_candidates_from_fields(
            fields_for_diagnosis,
            trace_f2e.grid == 0,
            confirmed_events=f2e_events_raw,
        )
        residual_evidence = neutralize_subthreshold_evidence(residual_raw)
        f2e_graph_features = build_adaptive_evidence_graph(
            f2e_events_raw,
            residual_evidence,
            grid=trace_f2e.grid,
            history_frames=history_frames,
        )
    graph_latency_ms = (time.perf_counter() - graph_t0) * 1000.0
    img_path: Optional[str] = None
    frame_indices: List[int] = []
    image_render_latency_ms = 0.0
    if render_raw_image:
        image_t0 = time.perf_counter()
        blind_stem = blind_artifact_stem(trace_f2e.seed, scenario_index, "vlm", f"{vlm_frame_mode}{vlm_num_frames}")
        img_path, frame_indices = render_fields_contact_sheet(
            fields_for_diagnosis,
            os.path.join(image_dir, f"{blind_stem}.png"),
            blind_vlm_title(vlm_frame_mode),
            mode=vlm_frame_mode,
            num_frames=vlm_num_frames,
            grid=trace_f2e.grid,
        )
        image_render_latency_ms = (time.perf_counter() - image_t0) * 1000.0
    prep_latency = {
        "threshold_alarm_matrix": float(threshold_latency_ms),
        "f2e_events": 0.0 if threshold_only else float(trace_f2e.low_metrics.get("detector_total_latency_ms") or 0.0),
        "raw_matrix_summary": float(matrix_latency_ms),
        "f2e_graph": 0.0 if threshold_only else float(trace_f2e.low_metrics.get("detector_total_latency_ms") or 0.0) + float(graph_latency_ms),
        "raw_field_image": float(image_render_latency_ms),
    }
    return {
        "diagnosis_t": {"t": int(diag_t), "llm_interval_steps": int(llm_interval_steps)},
        "observation_quality": obs_quality,
        "threshold_alarm_matrix": {"alarm_matrix": threshold_alarm_matrix, "observation_quality": obs_quality, "representation_latency_ms": prep_latency["threshold_alarm_matrix"], "diagnosis_t": int(diag_t)},
        "f2e_events": {"events": f2e_events, "observation_quality": obs_quality, "graph_features": f2e_graph_features, "representation_latency_ms": prep_latency["f2e_events"], "graph_representation_latency_ms": prep_latency["f2e_graph"], "graph_construction_latency_ms": graph_latency_ms, "diagnosis_t": int(diag_t)},
        "raw_matrix_summary": {"field_summary": matrix_summary, "observation_quality": obs_quality, "representation_latency_ms": prep_latency["raw_matrix_summary"]},
        "raw_field_image": {
            "image_path": img_path,
            "image_filename_policy": "blind_no_scenario_id_no_accident_type",
            "diagnosis_t": int(diag_t),
            "vlm_frame_mode": vlm_frame_mode,
            "vlm_frame_indices": frame_indices,
            "representation_latency_ms": prep_latency["raw_field_image"],
            "image_render_latency_ms": image_render_latency_ms,
        },
    }


MODEL_LAYER2_METHODS = frozenset({
    "threshold_matrix_llm",
    "f2e_events_llm",
    "f2e_events_graph_llm",
    "raw_matrix_summary_llm",
    "raw_field_image_vlm",
})


def execute_layer2_model_method(
    method: str,
    method_inputs: Dict[str, Dict[str, Any]],
    diag_t: int,
    f2e_history_frames: int,
    client: DashScopeClient,
    api_mode: str,
    llm_model: str,
    vlm_model: str,
    token_counter: TokenFallbackCounter,
) -> Tuple[float, ModelCallResult, Dict[str, Any], float]:
    """Execute one independent model method without changing its experiment input.

    The returned wall time belongs to this request itself.  It is measured
    inside the worker so queueing/result-collection time is never attributed to
    another method when several methods run concurrently.
    """
    method_t0 = time.perf_counter()
    if method == "threshold_matrix_llm":
        representation_latency_ms = float(method_inputs["threshold_alarm_matrix"].get("representation_latency_ms") or 0.0)
        payload = {
            "threshold_alarm_matrix": method_inputs["threshold_alarm_matrix"]["alarm_matrix"],
            "observation_representation": "per_cell_fixed_control_limit_alarm_matrices",
            "diagnosis_t": diag_t,
            "observation_quality": method_inputs["threshold_alarm_matrix"].get("observation_quality"),
        }
        messages = build_diagnosis_prompt("threshold_alarm_matrix", payload)
        call_res = call_or_offline(client, api_mode, llm_model, messages, payload, method, token_counter)
    elif method == "f2e_events_llm":
        representation_latency_ms = float(method_inputs["f2e_events"].get("representation_latency_ms") or 0.0)
        payload = {
            "events": method_inputs["f2e_events"]["events"],
            "observation_representation": "F2E_structured_event_tokens",
            "diagnosis_t": diag_t,
            "history_frames_processed": f2e_history_frames,
            "observation_quality": method_inputs["f2e_events"].get("observation_quality"),
        }
        messages = build_diagnosis_prompt("f2e_events", payload)
        call_res = call_or_offline(client, api_mode, llm_model, messages, payload, method, token_counter)
    elif method == "f2e_events_graph_llm":
        representation_latency_ms = float(method_inputs["f2e_events"].get("graph_representation_latency_ms") or 0.0)
        payload = {
            "observation_representation": "F2E_obstacle_aware_event_graph",
            "diagnosis_t": diag_t,
            "history_frames_processed": f2e_history_frames,
            "observation_quality": method_inputs["f2e_events"].get("observation_quality"),
            "graph_features": method_inputs["f2e_events"].get("graph_features"),
        }
        messages = build_diagnosis_prompt("f2e_events_graph", payload)
        call_res = call_or_offline(client, api_mode, llm_model, messages, payload, method, token_counter, max_tokens=768)
    elif method == "raw_matrix_summary_llm":
        representation_latency_ms = float(method_inputs["raw_matrix_summary"].get("representation_latency_ms") or 0.0)
        payload = {
            "field_summary": method_inputs["raw_matrix_summary"]["field_summary"],
            "observation_representation": "raw_matrix_statistical_summary",
            "diagnosis_t": diag_t,
            "history_frames_observed": f2e_history_frames,
            "observation_quality": (method_inputs["raw_matrix_summary"].get("field_summary") or {}).get("observation_quality"),
        }
        messages = build_diagnosis_prompt("raw_matrix_summary", payload)
        call_res = call_or_offline(client, api_mode, llm_model, messages, payload, method, token_counter)
    elif method == "raw_field_image_vlm":
        representation_latency_ms = float(method_inputs["raw_field_image"].get("representation_latency_ms") or 0.0)
        img_path = method_inputs["raw_field_image"].get("image_path")
        payload = {
            "observation_representation": "raw_field_image_temporal_contact_sheet",
            "diagnosis_t": diag_t,
            "history_frames_observed": f2e_history_frames,
            "vlm_frame_mode": method_inputs["raw_field_image"].get("vlm_frame_mode"),
            "vlm_frame_indices": method_inputs["raw_field_image"].get("vlm_frame_indices"),
        }
        if img_path and os.path.exists(img_path):
            messages = build_vlm_messages(image_to_data_url(img_path), payload)
            call_res = call_or_offline(client, api_mode, vlm_model, messages, payload, method, token_counter, max_tokens=512)
        elif api_mode == "offline" or (api_mode == "auto" and not client.api_key):
            parsed = offline_llm_proxy(payload, "raw_field_image_vlm_no_matplotlib")
            call_res = ModelCallResult(
                True,
                json.dumps(parsed),
                parsed,
                0.0,
                model="offline_proxy",
                token_count_source="offline_no_image_usage",
            )
            call_res = apply_token_fallback(
                call_res,
                build_diagnosis_prompt("raw_field_image", payload),
                parsed,
                token_counter,
            )
        else:
            call_res = ModelCallResult(False, "", None, 0.0, model=vlm_model, error="VLM image rendering failed")
    else:
        raise ValueError(f"Not a model-based Layer 2 method: {method}")

    diag = normalize_diag(call_res.parsed)
    diagnosis_wall_ms = (time.perf_counter() - method_t0) * 1000.0
    return representation_latency_ms, call_res, diag, diagnosis_wall_ms

def run_layer2(scenarios: List[AccidentScenario], seeds: int, steps: int, obs_ratio: float, base_seed: int,
               out_dir: str, api_mode: str, client: DashScopeClient, llm_model: str, vlm_model: str,
               progress: ProgressMeter, start_job: int, vlm_frame_mode: str = "sampled",
               vlm_num_frames: int = 8, llm_interval_steps: int = 50,
               token_counter: Optional[TokenFallbackCounter] = None,
               methods: Optional[List[str]] = None,
               api_workers: int = 1) -> Tuple[List[Dict[str, Any]], int]:
    diag_rows: List[Dict[str, Any]] = []
    job = start_job
    default_methods = default_layer2_methods()
    methods = methods or default_methods
    threshold_only = bool(methods) and set(methods).issubset({"threshold_matrix_llm", "threshold_matrix_rules"})
    image_dir = os.path.join(out_dir, "field_images")
    token_counter = token_counter or TokenFallbackCounter()
    for epi in range(seeds):
        for sc_idx, sc in enumerate(scenarios):
            seed = base_seed + 100000 * epi + 1000 * sc_idx
            fixed_diag_t = diagnosis_query_t(sc, steps, llm_interval_steps)
            trace_f2e = collect_run_trace(
                sc, seed, steps, "threshold_matrix_observer" if threshold_only else "f2e_encoder", obs_ratio, all_fields=True,
                capture_fields=True, snapshot_times=(fixed_diag_t,), retain_snapshot_masks=not threshold_only,
            )
            need_raw_image = "raw_field_image_vlm" in methods
            method_inputs = build_method_inputs(
                trace_f2e, image_dir, vlm_frame_mode, vlm_num_frames,
                llm_interval_steps, scenario_index=sc_idx, render_raw_image=need_raw_image,
                threshold_only=threshold_only,
            )
            diag_t = int((method_inputs.get("diagnosis_t") or {}).get("t", steps - 1))
            f2e_history_frames = diag_t + 1
            parallel_model_results: Dict[str, Tuple[float, ModelCallResult, Dict[str, Any], float]] = {}
            model_methods = [method for method in methods if method in MODEL_LAYER2_METHODS]
            use_real_api = api_mode == "api" or (api_mode == "auto" and bool(client.api_key))
            if use_real_api and api_workers > 1 and len(model_methods) > 1:
                worker_count = min(max(1, int(api_workers)), len(model_methods))
                with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="layer2-api") as executor:
                    futures: Dict[Future[Tuple[float, ModelCallResult, Dict[str, Any], float]], str] = {
                        executor.submit(
                            execute_layer2_model_method,
                            method,
                            method_inputs,
                            diag_t,
                            f2e_history_frames,
                            client,
                            api_mode,
                            llm_model,
                            vlm_model,
                            token_counter,
                        ): method
                        for method in model_methods
                    }
                    completed_calls = 0
                    for future in as_completed(futures):
                        method = futures[future]
                        parallel_model_results[method] = future.result()
                        completed_calls += 1
                        progress.update(
                            job,
                            f"L2 seed={epi+1}/{seeds} scenario={sc.scenario_id} parallel API {completed_calls}/{len(model_methods)}",
                        )
            for method in methods:
                method_t0 = time.perf_counter()
                representation_latency_ms = 0.0
                call_res = ModelCallResult(True, "", {}, 0.0, None, None, None, model="rules", token_count_source="not_applicable_rules")
                diagnosis_wall_ms: Optional[float] = None
                if method == "threshold_matrix_rules":
                    representation_latency_ms = float(method_inputs["threshold_alarm_matrix"].get("representation_latency_ms") or 0.0)
                    diag = diagnose_by_threshold_alarms(
                        method_inputs["threshold_alarm_matrix"]["alarm_matrix"],
                        observation_quality=method_inputs["threshold_alarm_matrix"].get("observation_quality"),
                    )
                elif method == "f2e_events_rules":
                    representation_latency_ms = float(method_inputs["f2e_events"].get("representation_latency_ms") or 0.0)
                    diag = diagnose_by_rules(method_inputs["f2e_events"]["events"], source_name="f2e_events", observation_quality=method_inputs["f2e_events"].get("observation_quality"))
                elif method in MODEL_LAYER2_METHODS:
                    if method in parallel_model_results:
                        representation_latency_ms, call_res, diag, diagnosis_wall_ms = parallel_model_results[method]
                    else:
                        representation_latency_ms, call_res, diag, diagnosis_wall_ms = execute_layer2_model_method(
                            method,
                            method_inputs,
                            diag_t,
                            f2e_history_frames,
                            client,
                            api_mode,
                            llm_model,
                            vlm_model,
                            token_counter,
                        )
                else:
                    raise ValueError(method)
                if diagnosis_wall_ms is None:
                    diagnosis_wall_ms = (time.perf_counter() - method_t0) * 1000.0
                end_to_end_latency_ms = representation_latency_ms + diagnosis_wall_ms
                measured_api_latency_ms = None if call_res.from_cache or call_res.model in {"rules", "offline_proxy"} else call_res.latency_ms
                valid_prediction = bool(call_res.ok and (call_res.model == "rules" or call_res.parsed is not None))
                if valid_prediction:
                    exp_score = explanation_score(diag, sc)
                    reasoning_scores = reasoning_alignment_scores(diag, sc)
                    correct = int(diag.get("accident_type") == sc.accident_type)
                else:
                    diag = {
                        "accident_type": None,
                        "confidence": None,
                        "review_needed": None,
                        "abnormal_confirmed": None,
                        "resample_target": None,
                        "evidence_fields": [],
                        "evidence_events": [],
                        "unexplained_evidence": [],
                        "template_status": "invalid_model_call",
                        "explanation": call_res.error or "invalid model call",
                    }
                    exp_score = 0.0
                    reasoning_scores = {
                        **evidence_alignment_scores(diag, sc),
                        "behavior_goal_score": 0.0,
                        "review_calibration_score": 0.0,
                        "resample_target_match": None,
                        "forbidden_prediction_score": 0.0,
                        "reasoning_alignment_score": 0.0,
                        "template_rejection_success": 0,
                    }
                    correct = 0
                if method.startswith("f2e_events"):
                    input_event_count = len(method_inputs["f2e_events"]["events"])
                else:
                    input_event_count = None
                if method.startswith("threshold_matrix"):
                    threshold_payload = method_inputs["threshold_alarm_matrix"]["alarm_matrix"]
                    input_alarm_count = int(threshold_payload.get("total_high_alarm_cells", 0)) + int(threshold_payload.get("total_low_alarm_cells", 0))
                else:
                    input_alarm_count = None
                diag_rows.append({
                    "layer": "high_level_diagnosis",
                    "seed_index": epi,
                    "seed": seed,
                    "scenario_id": sc.scenario_id,
                    "method": method,
                    "method_display_name": method_display_name(method),
                    "valid_prediction": int(valid_prediction),
                    "ground_truth": sc.accident_type,
                    "prediction": diag.get("accident_type"),
                    "raw_prediction": diag.get("raw_prediction", diag.get("accident_type")),
                    "correct": correct,
                    "ambiguous": sc.ambiguous,
                    "unseen_combo": sc.unseen_combo,
                    "expected_review": sc.expected_review,
                    "hard_family": sc.hard_family,
                    "diagnostic_goal": sc.diagnostic_goal,
                    "challenge": sc.challenge,
                    "diagnosis_t": diag_t,
                    "llm_interval_steps": int(llm_interval_steps),
                    "observation_history_frames": f2e_history_frames,
                    "f2e_history_frames": None if method.startswith("threshold_matrix") else f2e_history_frames,
                    "input_event_count": input_event_count,
                    "input_alarm_count": input_alarm_count,
                    "observation_low_confidence": ((method_inputs.get("observation_quality") or {}).get("low_confidence")),
                    "last_frame_missing_fraction": ((method_inputs.get("observation_quality") or {}).get("last_frame_missing_fraction")),
                    "review_needed": diag.get("review_needed"),
                    "confidence": diag.get("confidence"),
                    "explanation_correctness": round(float(exp_score), 6),
                    "behavior_goal_score": round(float(reasoning_scores["behavior_goal_score"]), 6),
                    "evidence_field_precision": None if reasoning_scores["evidence_field_precision"] is None else round(float(reasoning_scores["evidence_field_precision"]), 6),
                    "evidence_field_recall": None if reasoning_scores["evidence_field_recall"] is None else round(float(reasoning_scores["evidence_field_recall"]), 6),
                    "evidence_field_f1": None if reasoning_scores["evidence_field_f1"] is None else round(float(reasoning_scores["evidence_field_f1"]), 6),
                    "evidence_signature_precision": None if reasoning_scores["evidence_signature_precision"] is None else round(float(reasoning_scores["evidence_signature_precision"]), 6),
                    "evidence_signature_recall": None if reasoning_scores["evidence_signature_recall"] is None else round(float(reasoning_scores["evidence_signature_recall"]), 6),
                    "evidence_signature_f1": None if reasoning_scores["evidence_signature_f1"] is None else round(float(reasoning_scores["evidence_signature_f1"]), 6),
                    "evidence_signature_alignment": round(float(reasoning_scores["evidence_signature_alignment"]), 6),
                    "review_calibration_score": round(float(reasoning_scores["review_calibration_score"]), 6),
                    "resample_target_match": None if reasoning_scores["resample_target_match"] is None else round(float(reasoning_scores["resample_target_match"]), 6),
                    "forbidden_prediction_score": round(float(reasoning_scores["forbidden_prediction_score"]), 6),
                    "reasoning_alignment_score": round(float(reasoning_scores["reasoning_alignment_score"]), 6),
                    "template_rejection_success": reasoning_scores["template_rejection_success"],
                    "expected_evidence_fields": ";".join(reasoning_scores["expected_evidence_fields"]),
                    "expected_event_signatures": ";".join(reasoning_scores["expected_event_signatures"]),
                    "reported_evidence_fields": ";".join(reasoning_scores["reported_evidence_fields"]),
                    "reported_event_signatures": ";".join(reasoning_scores["reported_event_signatures"]),
                    "unexplained_evidence": ";".join(map(str, diag.get("unexplained_evidence", []))),
                    "template_status": diag.get("template_status"),
                    "protocol_conflict": int(bool(diag.get("protocol_conflict", False))),
                    "protocol_conflict_reasons": ";".join(map(str, diag.get("protocol_conflict_reasons", []))),
                    "protocol_consistency_repair": int(bool(diag.get("protocol_consistency_repair", False))),
                    "prompt_tokens": call_res.prompt_tokens,
                    "completion_tokens": call_res.completion_tokens,
                    "total_tokens": call_res.total_tokens,
                    "token_count_source": call_res.token_count_source,
                    "token_count_is_estimate": int(call_res.token_count_is_estimate),
                    "api_usage_available": int(call_res.api_usage_available),
                    "prompt_text_tokens": call_res.prompt_text_tokens,
                    "prompt_image_tokens": call_res.prompt_image_tokens,
                    "prompt_video_tokens": call_res.prompt_video_tokens,
                    "prompt_audio_tokens": call_res.prompt_audio_tokens,
                    "prompt_cached_tokens": call_res.prompt_cached_tokens,
                    "completion_text_tokens": call_res.completion_text_tokens,
                    "completion_reasoning_tokens": call_res.completion_reasoning_tokens,
                    "completion_audio_tokens": call_res.completion_audio_tokens,
                    "api_latency_ms": round(float(call_res.latency_ms), 3),
                    "measured_api_latency_ms": None if measured_api_latency_ms is None else round(float(measured_api_latency_ms), 3),
                    "diagnosis_wall_ms": round(float(diagnosis_wall_ms), 3),
                    "representation_latency_ms": round(float(representation_latency_ms), 3),
                    "end_to_end_latency_ms": round(float(end_to_end_latency_ms), 3),
                    "model": call_res.model,
                    "from_cache": call_res.from_cache,
                    "api_ok": call_res.ok,
                    "api_error": call_res.error,
                    "api_workers": max(1, int(api_workers)),
                    "api_scheduling": "parallel_within_scenario" if use_real_api and int(api_workers) > 1 else "serial",
                    "explanation": diag.get("explanation"),
                })
                job += 1
                progress.update(job, f"L2 seed={epi+1}/{seeds} scenario={sc.scenario_id} method={method}")
            # Persist once per scenario instead of rewriting a growing JSON file
            # after every API response.
            client.flush_cache()
    return diag_rows, job

def aggregate_layer2(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["method"], []).append(r)
    out = []
    for method, rs in sorted(groups.items()):
        metric_mean = lambda key, subset=rs: safe_mean([r.get(key) for r in subset])
        y_true = [r["ground_truth"] for r in rs]
        y_pred = [r["prediction"] for r in rs]
        amb = [r for r in rs if r.get("ambiguous")]
        unseen = [r for r in rs if r.get("unseen_combo")]
        hard = [r for r in rs if r.get("ambiguous") or r.get("unseen_combo") or r.get("expected_review")]
        reasoning_hard = [r for r in rs if r.get("hard_family") not in {None, "", "standard"} or r.get("ambiguous") or r.get("expected_review")]
        ood = [r for r in rs if r.get("diagnostic_goal") == "template_rejection"]
        uncertainty_review = [r for r in rs if r.get("diagnostic_goal") == "review_low_confidence"]
        weak_evidence = [r for r in rs if r.get("diagnostic_goal") == "weak_evidence"]
        composite_goal = [r for r in rs if r.get("diagnostic_goal") == "composite"]
        composite = [r for r in rs if r.get("ground_truth") == "composite_anomaly"]
        low_snr = [r for r in rs if r.get("ground_truth") == "low_snr_anomaly"]
        expected_review = [r for r in rs if r.get("expected_review")]
        api_usage_rows = [r for r in rs if r.get("api_usage_available")]
        estimated_rows = [r for r in rs if r.get("token_count_is_estimate")]
        uncached_api_rows = [r for r in rs if r.get("measured_api_latency_ms") is not None]
        token_applicable_rows = [r for r in rs if r.get("token_count_source") != "not_applicable_rules"]
        token_measured_rows = [r for r in token_applicable_rows if r.get("total_tokens") is not None]
        missing_token_rows = [r for r in token_applicable_rows if r.get("total_tokens") is None]
        out.append({
            "method": method,
            "method_display_name": rs[0].get("method_display_name", method),
            "n": len(rs),
            "valid_prediction_n": int(sum(int(r.get("valid_prediction", 0)) for r in rs)),
            "invalid_prediction_n": int(sum(1 - int(r.get("valid_prediction", 0)) for r in rs)),
            "protocol_conflict_n": int(sum(1 if r.get("protocol_conflict") else 0 for r in rs)),
            "protocol_conflict_rate": round(float(np.mean([1 if r.get("protocol_conflict") else 0 for r in rs])), 6) if rs else None,
            "protocol_consistency_repair_n": int(sum(1 if r.get("protocol_consistency_repair") else 0 for r in rs)),
            "token_applicable_n": len(token_applicable_rows),
            "token_measured_n": len(token_measured_rows),
            "token_missing_n": len(missing_token_rows),
            "token_not_applicable_n": len(rs) - len(token_applicable_rows),
            "accident_classification_accuracy": round(float(np.mean([r.get("correct", 0) for r in rs])), 6) if rs else None,
            "macro_f1": round(float(macro_f1(y_true, y_pred, ACCIDENT_TYPES)), 6) if macro_f1(y_true, y_pred, ACCIDENT_TYPES) is not None else None,
            "ambiguous_case_accuracy": round(float(np.mean([r.get("correct", 0) for r in amb])), 6) if amb else None,
            "hard_case_accuracy": round(float(np.mean([r.get("correct", 0) for r in hard])), 6) if hard else None,
            "unseen_combination_accuracy": round(float(np.mean([r.get("correct", 0) for r in unseen])), 6) if unseen else None,
            "composite_accuracy": round(float(np.mean([r.get("correct", 0) for r in composite])), 6) if composite else None,
            "low_snr_accuracy": round(float(np.mean([r.get("correct", 0) for r in low_snr])), 6) if low_snr else None,
            "expected_review_success": round(float(np.mean([1 if r.get("review_needed") else 0 for r in expected_review])), 6) if expected_review else None,
            "explanation_correctness": round(float(safe_mean([r.get("explanation_correctness") for r in rs])), 6) if safe_mean([r.get("explanation_correctness") for r in rs]) is not None else None,
            "behavior_goal_score": round(float(metric_mean("behavior_goal_score")), 6) if metric_mean("behavior_goal_score") is not None else None,
            "reasoning_alignment_score": round(float(metric_mean("reasoning_alignment_score")), 6) if metric_mean("reasoning_alignment_score") is not None else None,
            "hard_reasoning_alignment_score": round(float(metric_mean("reasoning_alignment_score", reasoning_hard)), 6) if metric_mean("reasoning_alignment_score", reasoning_hard) is not None else None,
            "ambiguous_reasoning_alignment_score": round(float(metric_mean("reasoning_alignment_score", amb)), 6) if metric_mean("reasoning_alignment_score", amb) is not None else None,
            "unseen_reasoning_alignment_score": round(float(metric_mean("reasoning_alignment_score", unseen)), 6) if metric_mean("reasoning_alignment_score", unseen) is not None else None,
            "evidence_field_f1": round(float(metric_mean("evidence_field_f1")), 6) if metric_mean("evidence_field_f1") is not None else None,
            "evidence_signature_alignment": round(float(metric_mean("evidence_signature_alignment")), 6) if metric_mean("evidence_signature_alignment") is not None else None,
            "review_calibration_score": round(float(metric_mean("review_calibration_score")), 6) if metric_mean("review_calibration_score") is not None else None,
            "forbidden_prediction_avoidance": round(float(metric_mean("forbidden_prediction_score")), 6) if metric_mean("forbidden_prediction_score") is not None else None,
            "ood_template_rejection_n": len(ood),
            "ood_template_rejection_accuracy": round(float(np.mean([r.get("correct", 0) for r in ood])), 6) if ood else None,
            "ood_template_rejection_success": round(float(metric_mean("template_rejection_success", ood)), 6) if metric_mean("template_rejection_success", ood) is not None else None,
            "ood_reasoning_alignment_score": round(float(metric_mean("reasoning_alignment_score", ood)), 6) if metric_mean("reasoning_alignment_score", ood) is not None else None,
            "uncertainty_review_n": len(uncertainty_review),
            "uncertainty_review_success": round(float(np.mean([1 if r.get("review_needed") else 0 for r in uncertainty_review])), 6) if uncertainty_review else None,
            "uncertainty_review_accuracy": round(float(np.mean([r.get("correct", 0) for r in uncertainty_review])), 6) if uncertainty_review else None,
            "uncertainty_review_reasoning_alignment_score": round(float(metric_mean("reasoning_alignment_score", uncertainty_review)), 6) if metric_mean("reasoning_alignment_score", uncertainty_review) is not None else None,
            "weak_evidence_n": len(weak_evidence),
            "weak_evidence_goal_score": round(float(metric_mean("behavior_goal_score", weak_evidence)), 6) if metric_mean("behavior_goal_score", weak_evidence) is not None else None,
            "weak_evidence_reasoning_alignment_score": round(float(metric_mean("reasoning_alignment_score", weak_evidence)), 6) if metric_mean("reasoning_alignment_score", weak_evidence) is not None else None,
            "composite_goal_n": len(composite_goal),
            "composite_goal_score": round(float(metric_mean("behavior_goal_score", composite_goal)), 6) if metric_mean("behavior_goal_score", composite_goal) is not None else None,
            "composite_reasoning_alignment_score": round(float(metric_mean("reasoning_alignment_score", composite_goal)), 6) if metric_mean("reasoning_alignment_score", composite_goal) is not None else None,
            "mean_prompt_tokens": round(float(safe_mean([r.get("prompt_tokens") for r in token_applicable_rows])), 3) if safe_mean([r.get("prompt_tokens") for r in token_applicable_rows]) is not None else None,
            "mean_completion_tokens": round(float(safe_mean([r.get("completion_tokens") for r in token_applicable_rows])), 3) if safe_mean([r.get("completion_tokens") for r in token_applicable_rows]) is not None else None,
            "mean_total_tokens": round(float(safe_mean([r.get("total_tokens") for r in token_applicable_rows])), 3) if safe_mean([r.get("total_tokens") for r in token_applicable_rows]) is not None else None,
            "mean_total_tokens_api_only": round(float(safe_mean([r.get("total_tokens") for r in api_usage_rows])), 3) if safe_mean([r.get("total_tokens") for r in api_usage_rows]) is not None else None,
            "mean_total_tokens_estimated_only": round(float(safe_mean([r.get("total_tokens") for r in estimated_rows])), 3) if safe_mean([r.get("total_tokens") for r in estimated_rows]) is not None else None,
            "api_usage_rate": round(float(np.mean([1 if r.get("api_usage_available") else 0 for r in token_applicable_rows])), 6) if token_applicable_rows else None,
            "token_estimate_rate": round(float(np.mean([1 if r.get("token_count_is_estimate") else 0 for r in token_applicable_rows])), 6) if token_applicable_rows else None,
            "mean_prompt_text_tokens": round(float(safe_mean([r.get("prompt_text_tokens") for r in token_applicable_rows])), 3) if safe_mean([r.get("prompt_text_tokens") for r in token_applicable_rows]) is not None else None,
            "mean_prompt_image_tokens": round(float(safe_mean([r.get("prompt_image_tokens") for r in token_applicable_rows])), 3) if safe_mean([r.get("prompt_image_tokens") for r in token_applicable_rows]) is not None else None,
            "mean_prompt_cached_tokens": round(float(safe_mean([r.get("prompt_cached_tokens") for r in token_applicable_rows])), 3) if safe_mean([r.get("prompt_cached_tokens") for r in token_applicable_rows]) is not None else None,
            "mean_completion_reasoning_tokens": round(float(safe_mean([r.get("completion_reasoning_tokens") for r in token_applicable_rows])), 3) if safe_mean([r.get("completion_reasoning_tokens") for r in token_applicable_rows]) is not None else None,
            "mean_api_latency_ms": round(float(safe_mean([r.get("api_latency_ms") for r in rs])), 3) if safe_mean([r.get("api_latency_ms") for r in rs]) is not None else None,
            "mean_measured_api_latency_ms": round(float(safe_mean([r.get("measured_api_latency_ms") for r in uncached_api_rows])), 3) if safe_mean([r.get("measured_api_latency_ms") for r in uncached_api_rows]) is not None else None,
            "mean_diagnosis_wall_ms": round(float(safe_mean([r.get("diagnosis_wall_ms") for r in rs])), 3) if safe_mean([r.get("diagnosis_wall_ms") for r in rs]) is not None else None,
            "mean_representation_latency_ms": round(float(safe_mean([r.get("representation_latency_ms") for r in rs])), 3) if safe_mean([r.get("representation_latency_ms") for r in rs]) is not None else None,
            "mean_end_to_end_latency_ms": round(float(safe_mean([r.get("end_to_end_latency_ms") for r in rs])), 3) if safe_mean([r.get("end_to_end_latency_ms") for r in rs]) is not None else None,
            "cache_hit_rate": round(float(np.mean([1 if r.get("from_cache") else 0 for r in rs])), 6) if rs else None,
            "api_success_rate": round(float(np.mean([1 if r.get("api_ok") else 0 for r in rs])), 6) if rs else None,
        })
    return out

def aggregate_layer2_by_family(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        family = str(r.get("hard_family") or "standard")
        groups.setdefault((str(r.get("method")), family), []).append(r)
    out = []
    for (method, family), rs in sorted(groups.items()):
        y_true = [r["ground_truth"] for r in rs]
        y_pred = [r["prediction"] for r in rs]
        expected_review = [r for r in rs if r.get("expected_review")]
        out.append({
            "method": method,
            "method_display_name": rs[0].get("method_display_name", method),
            "hard_family": family,
            "n": len(rs),
            "accuracy": round(float(np.mean([r.get("correct", 0) for r in rs])), 6) if rs else None,
            "macro_f1": round(float(macro_f1(y_true, y_pred, ACCIDENT_TYPES)), 6) if macro_f1(y_true, y_pred, ACCIDENT_TYPES) is not None else None,
            "behavior_goal_score": round(float(safe_mean([r.get("behavior_goal_score") for r in rs])), 6) if safe_mean([r.get("behavior_goal_score") for r in rs]) is not None else None,
            "reasoning_alignment_score": round(float(safe_mean([r.get("reasoning_alignment_score") for r in rs])), 6) if safe_mean([r.get("reasoning_alignment_score") for r in rs]) is not None else None,
            "evidence_field_f1": round(float(safe_mean([r.get("evidence_field_f1") for r in rs])), 6) if safe_mean([r.get("evidence_field_f1") for r in rs]) is not None else None,
            "evidence_signature_alignment": round(float(safe_mean([r.get("evidence_signature_alignment") for r in rs])), 6) if safe_mean([r.get("evidence_signature_alignment") for r in rs]) is not None else None,
            "review_calibration_score": round(float(safe_mean([r.get("review_calibration_score") for r in rs])), 6) if safe_mean([r.get("review_calibration_score") for r in rs]) is not None else None,
            "expected_review_success": round(float(np.mean([1 if r.get("review_needed") else 0 for r in expected_review])), 6) if expected_review else None,
            "template_rejection_success": round(float(safe_mean([r.get("template_rejection_success") for r in rs])), 6) if safe_mean([r.get("template_rejection_success") for r in rs]) is not None else None,
            "forbidden_prediction_avoidance": round(float(safe_mean([r.get("forbidden_prediction_score") for r in rs])), 6) if safe_mean([r.get("forbidden_prediction_score") for r in rs]) is not None else None,
            "protocol_conflict_n": int(sum(1 if r.get("protocol_conflict") else 0 for r in rs)),
            "protocol_conflict_rate": round(float(np.mean([1 if r.get("protocol_conflict") else 0 for r in rs])), 6) if rs else None,
        })
    return out

def aggregate_llm_advantage(summary_rows: List[Dict[str, Any]], target_method: str = "f2e_events_llm") -> List[Dict[str, Any]]:
    target = next((r for r in summary_rows if r.get("method") == target_method), None)
    if not target:
        return []
    metrics = [
        "accident_classification_accuracy",
        "macro_f1",
        "behavior_goal_score",
        "reasoning_alignment_score",
        "hard_reasoning_alignment_score",
        "ambiguous_reasoning_alignment_score",
        "unseen_reasoning_alignment_score",
        "ood_template_rejection_success",
        "ood_reasoning_alignment_score",
        "uncertainty_review_success",
        "uncertainty_review_reasoning_alignment_score",
        "weak_evidence_goal_score",
        "composite_goal_score",
        "evidence_signature_alignment",
        "review_calibration_score",
        "forbidden_prediction_avoidance",
    ]
    out = []
    for baseline in summary_rows:
        baseline_method = baseline.get("method")
        if baseline_method == target_method:
            continue
        row = {"target_method": target_method, "baseline_method": baseline_method}
        for metric in metrics:
            tv = safe_mean([target.get(metric)])
            bv = safe_mean([baseline.get(metric)])
            row[f"{metric}_target"] = None if tv is None else round(float(tv), 6)
            row[f"{metric}_baseline"] = None if bv is None else round(float(bv), 6)
            row[f"{metric}_delta"] = None if tv is None or bv is None else round(float(tv - bv), 6)
        out.append(row)
    return out

def aggregate_llm_advantage_by_family(family_rows: List[Dict[str, Any]], target_method: str = "f2e_events_llm") -> List[Dict[str, Any]]:
    metrics = [
        "accuracy",
        "behavior_goal_score",
        "reasoning_alignment_score",
        "evidence_field_f1",
        "evidence_signature_alignment",
        "review_calibration_score",
        "expected_review_success",
        "template_rejection_success",
        "forbidden_prediction_avoidance",
    ]
    target_by_family = {r.get("hard_family"): r for r in family_rows if r.get("method") == target_method}
    out = []
    for baseline in family_rows:
        family = baseline.get("hard_family")
        baseline_method = baseline.get("method")
        if baseline_method == target_method:
            continue
        target = target_by_family.get(family)
        if not target:
            continue
        row = {
            "hard_family": family,
            "target_method": target_method,
            "baseline_method": baseline_method,
            "n": target.get("n"),
        }
        for metric in metrics:
            tv = safe_mean([target.get(metric)])
            bv = safe_mean([baseline.get(metric)])
            row[f"{metric}_target"] = None if tv is None else round(float(tv), 6)
            row[f"{metric}_baseline"] = None if bv is None else round(float(bv), 6)
            row[f"{metric}_delta"] = None if tv is None or bv is None else round(float(tv - bv), 6)
        out.append(row)
    return out

# ============================================================
# Main
# ============================================================
def default_layer2_methods() -> List[str]:
    return [
        "f2e_events_graph_llm",
        "f2e_events_llm",
        "threshold_matrix_llm",
        "raw_matrix_summary_llm",
        "threshold_matrix_rules",
        "f2e_events_rules",
        "raw_field_image_vlm",
    ]

def parse_layer2_methods(text: str) -> List[str]:
    allowed = default_layer2_methods()
    if not text or text.strip().lower() == "all":
        return allowed
    aliases = {
        "core": ["threshold_matrix_llm", "f2e_events_llm", "f2e_events_graph_llm"],
        "f2e": ["f2e_events_rules", "f2e_events_llm", "f2e_events_graph_llm"],
        "llm": ["threshold_matrix_llm", "raw_matrix_summary_llm", "f2e_events_llm", "f2e_events_graph_llm"],
    }
    out: List[str] = []
    for part in text.split(","):
        name = part.strip()
        if not name:
            continue
        if name in aliases:
            out.extend(aliases[name])
            continue
        if name not in allowed:
            raise ValueError(f"Unknown layer2 method: {name}. Allowed: {allowed} plus aliases core,f2e,llm,all")
        out.append(name)
    # Stable de-duplication.
    dedup: List[str] = []
    for name in out:
        if name not in dedup:
            dedup.append(name)
    return dedup

def main() -> None:
    ap = argparse.ArgumentParser(description=f"{COMPARISON_VERSION} comprehensive F2E/LLM/VLM comparison experiments")
    ap.add_argument("--profile", choices=["quick", "paper", "hard"], default="quick")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-scenarios", type=int, default=None)
    ap.add_argument("--obs-ratio", type=float, default=DEFAULT_OBS_RATIO)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", type=str, default="outputs/comprehensive_v8_5_3_threshold_matrix_parallel")
    ap.add_argument("--api-mode", choices=["offline", "auto", "api"], default="auto", help="offline: no API; auto: call API only if key exists; api: require/call API")
    ap.add_argument("--llm-model", default="qwen3.6-flash", help="model for text/event/matrix LLM comparisons")
    ap.add_argument("--vlm-model", default="qwen3-vl-flash", help="model for raw image + VLM comparison")
    ap.add_argument("--vlm-frame-mode", choices=["last", "sampled", "all"], default="sampled", help="raw_field_image+VLM input: last frame, sampled contact sheet, or all-frame contact sheet")
    ap.add_argument("--vlm-num-frames", type=int, default=8, help="number of sampled frames for --vlm-frame-mode sampled")
    ap.add_argument("--llm-interval-steps", type=int, default=50,
                    help="Layer 2 diagnosis query cadence, matching the live F2E+LLM demo. The LLM receives F2E events from this scheduled frame after F2E has processed all prior frames.")
    ap.add_argument("--layer2-methods", default="all",
                    help="Layer 2 methods to run: all, core, f2e, llm, or comma-separated names from the seven registered methods")
    ap.add_argument("--dashscope-base-url", default=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    ap.add_argument("--dashscope-proxy-mode", choices=["bypass", "environment"], default="bypass",
                    help="bypass ignores HTTP(S)/ALL_PROXY and Windows proxy settings for DashScope; environment uses configured proxies")
    ap.add_argument("--enable-thinking", action="store_true", help="enable DashScope thinking mode. Default off for fair latency/token comparison.")
    ap.add_argument("--api-timeout", type=float, default=60.0)
    ap.add_argument("--api-retries", type=int, default=2, help="additional attempts after a failed or invalid-JSON API response")
    ap.add_argument("--api-retry-backoff", type=float, default=1.0, help="initial exponential retry backoff in seconds")
    ap.add_argument("--api-workers", type=int, default=5,
                    help="parallel independent model calls per scenario; 1 reproduces serial scheduling, 5 runs the five model methods together")
    ap.add_argument("--token-fallback", choices=["none", "hf_tokenizer", "heuristic"], default="none",
                    help="Fallback token counting only when API usage is missing. Default none keeps token metrics API-measured only.")
    ap.add_argument("--tokenizer-model", default=None,
                    help="HuggingFace tokenizer id/path for --token-fallback hf_tokenizer. Defaults to a Qwen3 tokenizer surrogate.")
    ap.add_argument("--tokenizer-allow-download", action="store_true",
                    help="Allow transformers to download tokenizer files if they are not already cached locally.")
    ap.add_argument("--progress", choices=["on", "off"], default="on")
    ap.add_argument("--progress-interval", type=float, default=2.0)
    args = ap.parse_args()

    if args.api_workers < 1:
        ap.error("--api-workers must be at least 1")

    layer2_methods = parse_layer2_methods(args.layer2_methods)
    seeds = args.seeds if args.seeds is not None else (1 if args.profile == "quick" else 10)
    steps = args.steps if args.steps is not None else (90 if args.profile == "quick" else 170)
    scenarios = make_accident_scenarios(args.profile)
    if args.max_scenarios is not None:
        scenarios = scenarios[:args.max_scenarios]

    total_jobs = seeds * len(scenarios) * len(layer2_methods)
    progress = ProgressMeter(total_jobs, enabled=args.progress == "on", interval_sec=args.progress_interval)

    os.makedirs(args.out_dir, exist_ok=True)
    save_prompt_templates(args.out_dir)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if args.api_mode == "api" and not api_key:
        print("ERROR: --api-mode api requires DASHSCOPE_API_KEY in the environment.", file=sys.stderr)
        sys.exit(2)
    client = DashScopeClient(
        api_key=api_key,
        base_url=args.dashscope_base_url,
        cache_path=os.path.join(args.out_dir, "dashscope_cache.json"),
        enable_thinking=args.enable_thinking,
        timeout=args.api_timeout,
        max_retries=args.api_retries,
        retry_backoff_sec=args.api_retry_backoff,
        proxy_mode=args.dashscope_proxy_mode,
    )
    token_counter = TokenFallbackCounter(
        mode=args.token_fallback,
        tokenizer_model=args.tokenizer_model,
        local_files_only=not args.tokenizer_allow_download,
    )

    t0 = time.perf_counter()
    job = 0
    layer2_rows: List[Dict[str, Any]] = []
    progress.update(0, "starting")
    layer2_rows, job = run_layer2(scenarios, seeds, steps, args.obs_ratio, args.seed, args.out_dir,
                                  args.api_mode, client, args.llm_model, args.vlm_model, progress, job,
                                  args.vlm_frame_mode, args.vlm_num_frames, args.llm_interval_steps, token_counter,
                                  methods=layer2_methods,
                                  api_workers=args.api_workers)
    client.flush_cache()
    progress.update(total_jobs, "done")

    layer2_summary = aggregate_layer2(layer2_rows) if layer2_rows else []
    layer2_family_summary = aggregate_layer2_by_family(layer2_rows) if layer2_rows else []
    layer2_llm_advantage = aggregate_llm_advantage(layer2_summary) if layer2_summary else []
    layer2_llm_advantage_by_family = aggregate_llm_advantage_by_family(layer2_family_summary) if layer2_family_summary else []

    run_config = {
        "version": COMPARISON_VERSION,
        "core_version": CORE_VERSION,
        "profile": args.profile,
        "experiment_scope": "layer2_comprehensive_comparison_only",
        "layer2_methods": layer2_methods,
        "layer2_method_display_names": {name: method_display_name(name) for name in layer2_methods},
        "seeds": seeds,
        "steps": steps,
        "n_scenarios": len(scenarios),
        "scenario_ids": [s.scenario_id for s in scenarios],
        "obs_ratio": args.obs_ratio,
        "api_mode": args.api_mode,
        "api_key_present": bool(api_key),
        "llm_model": args.llm_model,
        "vlm_model": args.vlm_model,
        "vlm_frame_mode": args.vlm_frame_mode,
        "vlm_num_frames": args.vlm_num_frames,
        "llm_interval_steps": args.llm_interval_steps,
        "dashscope_base_url": args.dashscope_base_url,
        "dashscope_proxy_mode": args.dashscope_proxy_mode,
        "dashscope_proxy_note": (
            "bypass disables urllib environment and Windows system proxies for DashScope. "
            "A VPN implemented as a system-wide TUN or virtual network adapter may still route direct sockets."
        ),
        "enable_thinking": bool(args.enable_thinking),
        "api_retries": args.api_retries,
        "api_retry_backoff": args.api_retry_backoff,
        "api_workers": max(1, int(args.api_workers)),
        "api_scheduling": "parallel_within_scenario" if int(args.api_workers) > 1 else "serial",
        "latency_measurement_context": (
            "Per-method diagnosis_wall_ms is measured inside its own worker and api_latency_ms is the request's measured API time. "
            "With api_workers>1, absolute latency is measured under balanced within-scenario concurrent load and should not be compared "
            "directly with absolute latency from a serial run; prompts, predictions, tokens, seeds, and scoring are unchanged."
        ),
        "graph_distance_policy": "fixed_obstacle_aware_geodesic_8_neighbor_no_corner_cutting",
        "graph_grouping_policy": "adaptive_support_radius_overlap_components_no_distance_hyperparameter",
        "graph_residual_window_policy": "history_adaptive_early_fifth_baseline_recent_third_window",
        "token_fallback": args.token_fallback,
        "tokenizer_model": token_counter.tokenizer_model,
        "tokenizer_local_files_only": not args.tokenizer_allow_download,
        "token_counter_status": token_counter.status(),
        "token_accounting_note": "Primary token metrics use DashScope/OpenAI-compatible response usage. Rules have token_not_applicable_n and blank token columns; fallback counts are estimates and flagged per row.",
        "threshold_method_policy": "per_cell_phase_i_fixed_three_sigma_alarm_matrix_v1; no connected components, centroids, morphology, tracks, or event tokens",
        "threshold_only_raw_observation_fast_path": bool(layer2_methods) and set(layer2_methods).issubset({"threshold_matrix_llm", "threshold_matrix_rules"}),
        "graph_feature_policy": "label_agnostic_parameter_free_adaptive_evidence_graph_v4",
        "graph_residual_policy": "causal_history_adaptive_residuals_no_experiment_facing_hyperparameters_v2",
        "llm_prompt_policy": "shared_representation_neutral_evidence_core_plus_compact_graph_adapter_v6",
        "output_constraint_policy": "representation_neutral_evidence_ledger_conflict_audit_only_v2",
        "primary_prediction_policy": "score_normalized_raw_model_label_no_protocol_rewrite_v1",
        "f2e_event_policy": "active_tracks_with_normalized_temporal_coverage_v1",
        "vlm_input_policy": "image_only_plus_visual_format_adapter_v1",
        "offline_graph_policy": "no_simulated_graph_gain; api_mode_api_required_for_graph_llm_claims",
        "scenario_design": [
            {
                "scenario_id": s.scenario_id,
                "accident_type": s.accident_type,
                "challenge": s.challenge,
                "hard_family": s.hard_family,
                "diagnostic_goal": s.diagnostic_goal,
                "llm_advantage": s.llm_advantage,
                "expected_review": s.expected_review,
                "ambiguous": s.ambiguous,
                "unseen_combo": s.unseen_combo,
                "expected_event_signatures": sorted(scenario_expected_signatures(s)),
                "forbidden_predictions": list(s.forbidden_predictions),
                "acceptable_predictions": list(s.acceptable_predictions),
                "notes": s.notes,
            }
            for s in scenarios
        ],
        "wall_time_sec": round(float(time.perf_counter() - t0), 3),
        "note": "DASHSCOPE_API_KEY is read from environment and is never saved in outputs.",
    }
    payload = {
        "run_config": run_config,
        "layer2_diagnosis_summary": layer2_summary,
        "layer2_hard_family_summary": layer2_family_summary,
        "layer2_llm_advantage_summary": layer2_llm_advantage,
        "layer2_llm_advantage_by_family": layer2_llm_advantage_by_family,
        "layer2_rows": layer2_rows,
    }
    with open(os.path.join(args.out_dir, "comprehensive_comparison_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    save_csv(os.path.join(args.out_dir, "layer2_diagnosis_records.csv"), layer2_rows)
    save_csv(os.path.join(args.out_dir, "layer2_diagnosis_summary.csv"), layer2_summary)
    save_csv(os.path.join(args.out_dir, "layer2_hard_family_summary.csv"), layer2_family_summary)
    save_csv(os.path.join(args.out_dir, "layer2_llm_advantage_summary.csv"), layer2_llm_advantage)
    save_csv(os.path.join(args.out_dir, "layer2_llm_advantage_by_family.csv"), layer2_llm_advantage_by_family)

    print(f"\n===== Comprehensive F2E/LLM/VLM Comparison {COMPARISON_VERSION} =====")
    print(f"core={CORE_VERSION} scope=layer2_only profile={args.profile} seeds={seeds} steps={steps} scenarios={len(scenarios)}")
    print(f"api_mode={args.api_mode} api_key_present={bool(api_key)} dashscope_proxy_mode={args.dashscope_proxy_mode} api_workers={args.api_workers} scheduling={'parallel_within_scenario' if args.api_workers > 1 else 'serial'} llm_model={args.llm_model} vlm_model={args.vlm_model} vlm_frame_mode={args.vlm_frame_mode} vlm_num_frames={args.vlm_num_frames} llm_interval_steps={args.llm_interval_steps} thinking={args.enable_thinking}")
    print("graph_policy=parameter_free_adaptive_geodesic_support_overlap residual_window=history_adaptive")
    print(f"token_accounting=api_usage_primary fallback={args.token_fallback} tokenizer={token_counter.tokenizer_model}")
    print("NO-LEAK CHECK: the F2E encoder and residual graph see only causal observed fields; model-visible graph features contain measurements/topology and confirmation tiers, with no scenario IDs, labels, GT masks, injected-effect metadata, or compound/novel/low-SNR decision flags.")
    if layer2_summary:
        print("\nLayer 2 summary:")
        for r in layer2_summary:
            print(f"  {r.get('method_display_name', r['method'])}: acc={r.get('accident_classification_accuracy')} macroF1={r.get('macro_f1')} reason={r.get('reasoning_alignment_score')} hard_reason={r.get('hard_reasoning_alignment_score')} ambiguous={r.get('ambiguous_case_accuracy')}/{r.get('ambiguous_reasoning_alignment_score')} unseen={r.get('unseen_combination_accuracy')}/{r.get('unseen_reasoning_alignment_score')} ood_reject={r.get('ood_template_rejection_success')} review={r.get('uncertainty_review_success')} tokens={r.get('mean_total_tokens')}")
    print("\nSaved:")
    for name in [
        "comprehensive_comparison_results.json", "run_config.json",
        "layer2_diagnosis_records.csv", "layer2_diagnosis_summary.csv",
        "layer2_hard_family_summary.csv",
        "layer2_llm_advantage_summary.csv", "layer2_llm_advantage_by_family.csv",
    ]:
        print("  " + os.path.join(args.out_dir, name))

if __name__ == "__main__":
    main()
