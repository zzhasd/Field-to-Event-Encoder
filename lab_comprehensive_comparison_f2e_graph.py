"""Comprehensive comparison experiments for Field-to-Event (F2E) + LLM systems.

Version: v8.2.7

This script is intentionally an experiment/evaluation layer only.  It imports the
core no-leak F2E encoder from University_Field_to_Event_Encoder.py and never
passes clean backgrounds, GT masks, injected labels, or accident labels into the
encoder.

Two-layer experimental design
-------------------------------
Layer 1: Low-level detection capability
    Threshold connected components vs CUSUM/EWMA vs heatmap detector vs F2E.

Layer 2: High-level accident diagnosis capability
    threshold events + rules
    threshold events + LLM
    raw matrix summary + LLM
    raw field image + VLM
    F2E events + rules
    F2E events + LLM

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
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
import warnings
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

COMPARISON_VERSION = "v8.3.1-f2e-graph-lc-weak-window-neutral"

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
            "temporal_summary": {"area_trend": "stable", "intensity_trend": "stable"},
        })
    return out

class DetectorBase:
    name = "base"
    def update(self, t: int, current_fields: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        raise NotImplementedError

class ThresholdCCDetector(DetectorBase):
    name = "threshold_cc"
    def __init__(self, grid: np.ndarray, z_threshold: float = 1.65) -> None:
        self.grid = grid.astype(np.uint8)
        self.free_mask = self.grid == 0
        self.z_threshold = z_threshold
    def update(self, t: int, current_fields: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for k, x in current_fields.items():
            x = np.asarray(x, dtype=np.float32)
            bg = estimate_spatial_bg(x, self.free_mask)
            z = (x - bg) / (FIELD_REGISTRY[k].sigma + 1e-6)
            z[~self.free_mask] = np.nan
            for pol in ("high", "low"):
                score = np.maximum(z, 0.0) if pol == "high" else np.maximum(-z, 0.0)
                score[~np.isfinite(score)] = 0.0
                mask = self.free_mask & (score >= self.z_threshold)
                events.extend(extract_components(k, pol, score, mask, t, FIELD_REGISTRY[k].min_area))
        return sorted(events, key=lambda e: e["priority"], reverse=True)

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
    if name == "threshold_cc":
        return ThresholdCCDetector(grid)
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

def summarize_events(events: List[Dict[str, Any]], max_events: int = 12) -> List[Dict[str, Any]]:
    out = []
    for ev in sorted(events, key=lambda x: float(x.get("priority", x.get("score_sum", 0.0))), reverse=True)[:max_events]:
        tmp = ev.get("temporal_summary") or {}
        out.append({
            "field_key": ev.get("field_key"),
            "polarity": ev.get("polarity"),
            "morphology": ev.get("morphology"),
            "centroid": ev.get("centroid"),
            "area": ev.get("area"),
            "z_core_mean": ev.get("z_core_mean"),
            "score_sum": ev.get("score_sum", ev.get("priority")),
            "area_trend": tmp.get("area_trend"),
            "intensity_trend": tmp.get("intensity_trend"),
            "physical_tag": (ev.get("physical_tag") or {}).get("label"),
        })
    return out

def diagnosis_query_t(scenario: AccidentScenario, steps: int, interval_steps: int = 50) -> int:
    """Choose the frame whose events are sent to the LLM diagnosis layer.

    This mirrors the live demo cadence: the LLM is asked every N rendered steps.
    For abnormal cases we use the latest scheduled query inside the active
    injection window, so the model sees the stable/strong anomaly rather than
    the post-event final frame.
    """
    last_t = max(0, int(steps) - 1)
    interval = max(1, int(interval_steps))
    scheduled = [step_1based - 1 for step_1based in range(interval, int(steps) + 1, interval)]
    if not scheduled:
        scheduled = [last_t]
    if not scenario.testcase.effects:
        return int(scheduled[-1])

    active_start = min(int(e.start) for e in scenario.testcase.effects)
    active_end = min(int(e.end) for e in scenario.testcase.effects)
    candidates = [t for t in scheduled if active_start <= t < active_end and t <= last_t]
    if candidates:
        return int(candidates[-1])

    # Fallback for short runs or unusual schedules: choose a late active frame.
    target = active_start + int(round(0.65 * max(1, active_end - active_start - 1)))
    return int(max(0, min(last_t, target)))

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
                                num_frames: int = 8) -> Tuple[Optional[str], List[int]]:
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
            arr = fields_by_t[idx].get(k)
            im = ax.imshow(arr, interpolation="nearest", vmin=vmin, vmax=vmax)
            if r == 0:
                ax.set_title(f"t={idx}")
            if c == 0:
                ax.set_ylabel(k)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title + f" | VLM contact sheet mode={mode}, frames={idxs}")
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

def collect_run_trace(scenario: AccidentScenario, seed: int, steps: int, detector_name: str, obs_ratio: float, all_fields: bool = True) -> RunTrace:
    grid = make_obstacle_grid(GRID_SIZE, obs_ratio, seed=seed)
    sc = clone_scenario_for_grid(scenario, grid)
    detector = detector_factory(detector_name, grid)
    field_keys = field_keys_for_scenario(sc, all_fields=all_fields)
    tp = fp_normal = fp_extra = fn = 0
    normal_frames = effect_frames = 0
    ious: List[float] = []
    cerrors: List[float] = []
    first_det: Optional[int] = None
    events_last: List[Dict[str, Any]] = []
    events_all: List[Dict[str, Any]] = []
    event_snapshots: List[Dict[str, Any]] = []
    fields_window: List[Dict[str, np.ndarray]] = []
    fields_last: Dict[str, np.ndarray] = {}
    gt_last: List[Dict[str, Any]] = []
    detector_elapsed = 0.0
    t0 = time.perf_counter()
    for t in range(steps):
        rng = np.random.default_rng(seed * 1000003 + t)
        current, gt_records = build_current_fields(grid, sc.testcase, t, field_keys, stress=sc.stress, rng=rng)
        det_t0 = time.perf_counter()
        events = detector.update(t, current)
        detector_elapsed += time.perf_counter() - det_t0
        fields_last = current
        gt_last = gt_records
        # Keep the full episode history so the VLM baseline can receive a fair
        # temporal contact sheet. This remains evaluator-side data; it is never
        # passed into the F2E encoder.
        fields_window.append({k: v.copy() for k, v in current.items()})
        events_last = events
        serial_events = [serializable_event(ev) for ev in events[:10]]
        events_all.extend(serial_events)
        event_snapshots.append({"t": int(t), "events": serial_events})
        matched = set()
        for gt in gt_records:
            candidates = [(idx, ev, iou(gt["mask"], ev["mask"])) for idx, ev in enumerate(events) if ev.get("field_key") == gt["field_key"] and ev.get("polarity") == gt["polarity"]]
            if candidates:
                idx, ev, best_iou = max(candidates, key=lambda x: x[2])
            else:
                idx, ev, best_iou = -1, None, 0.0
            if ev is not None and best_iou >= 0.10:
                tp += 1
                matched.add(idx)
                ious.append(float(best_iou))
                cerrors.append(centroid_error(gt["centroid"], ev["centroid"]))
                if first_det is None:
                    first_det = t
            else:
                fn += 1
        if t > 40:
            if gt_records:
                effect_frames += 1
            else:
                normal_frames += 1
            for idx, _ev in enumerate(events):
                if idx not in matched:
                    if gt_records:
                        fp_extra += 1
                    else:
                        fp_normal += 1
    elapsed = (time.perf_counter() - t0) * 1000
    _, _, f1 = precision_recall_f1(tp, fp_normal + fp_extra, fn)
    low_metrics = {
        "tp": tp,
        "fp": fp_normal + fp_extra,
        "normal_fp": fp_normal,
        "extra_event_fp": fp_extra,
        "fn": fn,
        "detection_f1": round(float(f1), 6) if f1 is not None else None,
        "mean_iou": round(float(np.mean(ious)), 6) if ious else None,
        "centroid_error": round(float(np.mean(cerrors)), 6) if cerrors else None,
        "normal_fp_per_100_frames": round(float(100 * fp_normal / max(1, normal_frames)), 6),
        "extra_event_fp_per_100_effect_frames": round(float(100 * fp_extra / max(1, effect_frames)), 6),
        "latency_steps": None if first_det is None or not sc.testcase.effects else int(first_det - min(e.start for e in sc.testcase.effects)),
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
                 enable_thinking: bool = False, timeout: float = 60.0) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.enable_thinking = enable_thinking
        self.timeout = timeout
        self.cache_path = cache_path
        self.cache: Dict[str, Dict[str, Any]] = {}
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
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.cache_path)

    def call(self, model: str, messages: List[Dict[str, Any]], max_tokens: int = 512, temperature: float = 0.0) -> ModelCallResult:
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
        if key in self.cache:
            c = self.cache[key]
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
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
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
            self.cache[key] = cache_val
            self._save_cache()
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

def _event_strength_value(ev: Dict[str, Any]) -> float:
    for key in ("z_core_mean", "score_sum", "priority"):
        try:
            val = float(ev.get(key))
            if np.isfinite(val):
                return val
        except Exception:
            pass
    return 0.0

def event_signatures_from_events(events: List[Dict[str, Any]]) -> List[str]:
    sigs = []
    for ev in events:
        f = str(ev.get("field_key", "")).lower()
        p = str(ev.get("polarity", "")).lower()
        if f in FIELDS and p in {"high", "low"}:
            sigs.append(event_signature(f, p))
    return sorted(set(sigs))

def _cluster_template_candidates(signatures: set, events: List[Dict[str, Any]]) -> List[str]:
    has = lambda sig: sig in signatures
    cands: List[str] = []
    # Local physical templates.  These are not scenario IDs and are computed only
    # from observation-derived events.
    if has("temperature:high") and (has("air_quality:high") or has("co2:high")):
        cands.append("fire_like")
    if has("humidity:high") and has("temperature:high") and has("pressure:high"):
        cands.append("steam_leak_like")
    if has("humidity:high") and not (has("temperature:high") and has("pressure:high")):
        cands.append("water_leak_like")
    if has("temperature:high") and not (has("air_quality:high") or has("co2:high") or has("humidity:high")):
        cands.append("electrical_overheat_like")
    if has("co2:high") and not has("temperature:high") and not has("humidity:high"):
        cands.append("co2_accumulation_like")
    if has("air_quality:high") and not has("co2:high") and not has("temperature:high"):
        cands.append("dust_pollution_like")
    return sorted(set(cands))

def cluster_events_by_centroid(events: List[Dict[str, Any]], merge_distance: float = 3.25) -> List[Dict[str, Any]]:
    """Cluster event tokens by spatial proximity.

    This is a diagnosis adapter, not an oracle. It uses only F2E/threshold event
    centroids and fields. It helps the LLM avoid collapsing two spatially
    separated physical clusters into one dominant label.
    """
    valid = [(i, ev, _centroid_or_none(ev)) for i, ev in enumerate(events)]
    valid = [(i, ev, c) for i, ev, c in valid if c is not None]
    if not valid:
        return []
    parent = {i: i for i, _, _ in valid}
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for a_i, ev_a, c_a in valid:
        for b_i, ev_b, c_b in valid:
            if b_i <= a_i:
                continue
            dist = float(np.linalg.norm(np.array(c_a) - np.array(c_b)))
            # Use a conservative merge distance. Elongated humidity strips can
            # be physically extended, but a far-away strip should remain a
            # separate leak-like cluster.
            if dist <= merge_distance:
                union(a_i, b_i)
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for i, ev, _c in valid:
        groups.setdefault(find(i), []).append(ev)
    clusters = []
    for n, evs in enumerate(groups.values(), start=1):
        weights = np.array([max(1e-6, _event_strength_value(ev)) for ev in evs], dtype=float)
        pts = np.array([_centroid_or_none(ev) for ev in evs], dtype=float)
        centroid = np.average(pts, axis=0, weights=weights)
        sigs = set(event_signatures_from_events(evs))
        clusters.append({
            "cluster_id": f"C{n}",
            "centroid": [round(float(centroid[0]), 3), round(float(centroid[1]), 3)],
            "event_count": len(evs),
            "event_signatures": sorted(sigs),
            "fields": sorted({str(ev.get("field_key")) for ev in evs if str(ev.get("field_key")) in FIELDS}),
            "morphologies": sorted({str(ev.get("morphology")) for ev in evs if ev.get("morphology")}),
            "max_strength": round(float(max(_event_strength_value(ev) for ev in evs)), 3),
            "local_template_candidates": _cluster_template_candidates(sigs, evs),
        })
    clusters.sort(key=lambda c: c.get("max_strength", 0.0), reverse=True)
    return clusters

def max_cluster_separation(clusters: List[Dict[str, Any]]) -> float:
    if len(clusters) < 2:
        return 0.0
    pts = []
    for c in clusters:
        cent = c.get("centroid")
        if isinstance(cent, list) and len(cent) >= 2:
            pts.append(np.array([float(cent[0]), float(cent[1])], dtype=float))
    if len(pts) < 2:
        return 0.0
    return float(max(np.linalg.norm(a - b) for i, a in enumerate(pts) for b in pts[i + 1:]))

def derive_f2e_diagnostic_features(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    sigs = set(event_signatures_from_events(events))
    clusters = cluster_events_by_centroid(events)
    sep = max_cluster_separation(clusters)
    cluster_templates = [set(c.get("local_template_candidates") or []) for c in clusters]
    has_cluster_template = lambda name: any(name in ts for ts in cluster_templates)

    composite_candidate = False
    if len(clusters) >= 2 and sep >= 5.0:
        # Common hard case: spatially separated fire-like and leak-like evidence.
        if has_cluster_template("fire_like") and (has_cluster_template("water_leak_like") or has_cluster_template("steam_leak_like")):
            composite_candidate = True
        # More general condition: two clusters have different non-empty local templates.
        nonempty = [ts for ts in cluster_templates if ts]
        if len(nonempty) >= 2 and len(set().union(*nonempty)) >= 2:
            composite_candidate = True

    # Hard-profile OOD combinations: these are field-combination rules, not case IDs.
    novel_reasons = []
    if {"pressure:high", "air_quality:high", "humidity:low"}.issubset(sigs):
        novel_reasons.append("pressure_rise+air_quality_degradation+dry_patch is outside known templates")
    if {"temperature:high", "pressure:low"}.issubset(sigs):
        novel_reasons.append("temperature rise with pressure drop contradicts ordinary electrical/fire/steam templates")
    if {"co2:high", "pressure:high"}.issubset(sigs):
        novel_reasons.append("CO2 accumulation with pressure rise is a coupled process-upset pattern")

    return {
        "event_signatures": sorted(sigs),
        "spatial_clusters": clusters,
        "cluster_count": len(clusters),
        "cluster_separation_max": round(float(sep), 3),
        "has_multiple_physical_clusters": bool(len(clusters) >= 2 and sep >= 5.0),
        "composite_candidate": bool(composite_candidate),
        "novel_combination_candidate": bool(novel_reasons),
        "novel_combination_reasons": novel_reasons,
    }

def _temporal_baseline_from_observations(fields_by_t: List[Dict[str, np.ndarray]],
                                          field_key: str,
                                          free_mask: np.ndarray,
                                          baseline_frames: int = 24) -> np.ndarray:
    """Observation-only temporal baseline for weak-candidate mining.

    This is not a clean simulator background and does not use GT masks/labels.
    It simply summarizes the earliest observed frames available to the diagnosis
    adapter, analogous to a warm-up normal reference in a streaming deployment.
    """
    n0 = max(3, min(int(baseline_frames), max(3, len(fields_by_t) // 4), len(fields_by_t)))
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


def _confirmed_event_field_set(events: Optional[List[Dict[str, Any]]], polarity: str = "high") -> set:
    out = set()
    for ev in events or []:
        if ev.get("polarity") == polarity and ev.get("field_key") in FIELDS:
            out.add(str(ev.get("field_key")))
    return out


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
                                       recent_frames: int = 55,
                                       baseline_frames: int = 24,
                                       weak_floor_z: float = 0.55,
                                       weak_peak_z: float = 0.95,
                                       cumulative_threshold: float = 0.24,
                                       min_persistence: float = 0.12,
                                       min_weak_area: int = 3,
                                       confirmed_events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Mine recent-window cumulative weak evidence from observed fields only.

    Weak candidates are sub-threshold, temporally persistent, observation-derived
    evidence.  They are NOT confirmed F2E events and should not by themselves
    create a hard accident label.  The goal is to expose low-SNR multi-field
    patterns such as weak CO2/AQI support around a confirmed thermal event.

    No simulator-only clean background, GT mask, scenario ID, accident label, or
    injected-effect spec is used here.  The baseline is the earliest observed
    frames in the same run, and the evidence is accumulated over the recent
    diagnosis window.
    """
    if not fields_by_t:
        return {
            "present": False,
            "candidate_events": [],
            "coherence_score": 0.0,
            "note": "no observed frames",
            "weak_detector_version": "recent_window_cumulative_v1",
        }

    recent = fields_by_t[-max(3, int(recent_frames)):]
    rr, cc = np.indices(free_mask.shape)
    candidates: List[Dict[str, Any]] = []

    confirmed_high_fields = _confirmed_event_field_set(confirmed_events, "high")
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
            # Allow slightly lower persistence if a candidate is co-located with
            # a strong confirmed event; this is useful for low-SNR companion
            # fields around a thermal event.
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

    high_weak_fields = sorted({c["field_key"] for c in candidates if c.get("polarity") == "high" and float(c.get("peak_recent_z", 0.0) or 0.0) >= 0.85})
    combined_high_fields = sorted(set(high_weak_fields) | set(confirmed_high_fields))
    weak_multifield_coherence_pattern = {"temperature", "co2", "air_quality"}.issubset(set(combined_high_fields))

    colocated_weak_count = sum(1 for c in candidates if c.get("co_located_with_confirmed_event"))
    avg_evidence = safe_mean([c.get("evidence_score") for c in candidates]) or 0.0
    avg_persistence = safe_mean([c.get("temporal_persistence") for c in candidates]) or 0.0
    coherence_score = 0.0
    if candidates or confirmed_high_fields:
        coherence_score = min(
            1.0,
            0.18 * len(high_weak_fields)
            + 0.12 * len(coherent_pairs)
            + 0.10 * colocated_weak_count
            + 0.18 * float(avg_evidence)
            + 0.12 * float(avg_persistence)
            + (0.28 if weak_multifield_coherence_pattern else 0.0),
        )

    return {
        "present": bool(weak_multifield_coherence_pattern and coherence_score >= 0.50),
        "candidate_events": candidates,
        "coherent_pairs": coherent_pairs,
        "high_weak_fields": high_weak_fields,
        "confirmed_high_fields": sorted(confirmed_high_fields),
        "combined_high_fields": combined_high_fields,
        "weak_multifield_coherence_hint": bool(weak_multifield_coherence_pattern),
        "coherence_score": round(float(coherence_score), 3),
        "recent_frames": int(len(recent)),
        "baseline_frames": int(max(3, min(int(baseline_frames), max(3, len(fields_by_t) // 4), len(fields_by_t)))),
        "weak_detector_version": "recent_window_cumulative_v1",
        "note": "Weak candidates are recent-window cumulative, observation-derived, and are not confirmed F2E events.",
    }

def graph_guided_offline_diag(payload: Dict[str, Any], mode: str) -> Optional[Dict[str, Any]]:
    """Offline sanity proxy for new adapters; uses only payload-derived features."""
    derived = payload.get("derived_f2e_features") or {}
    weak = payload.get("weak_multifield_evidence") or {}
    sigs = set(derived.get("event_signatures") or event_signatures_from_events(payload.get("events") or []))
    oq = payload.get("observation_quality") or {}
    if bool(oq.get("low_confidence")) or float(oq.get("last_frame_missing_fraction", 0.0) or 0.0) >= 0.30 or float(oq.get("max_missing_fraction", 0.0) or 0.0) >= 0.30:
        target_field = None
        for preferred in ("humidity", "temperature", "air_quality", "co2", "pressure"):
            if any(s.startswith(preferred + ":") for s in sigs):
                target_field = preferred
                break
        return {
            "accident_type": "needs_review_unknown",
            "confidence": 0.42,
            "review_needed": True,
            "abnormal_confirmed": False,
            "resample_target": {"field_key": target_field, "centroid": None} if target_field else None,
            "evidence_fields": sorted({s.split(":", 1)[0] for s in sigs}),
            "evidence_events": [{"field_key": s.split(":", 1)[0], "polarity": s.split(":", 1)[1], "role": "low_confidence_evidence"} for s in sorted(sigs)],
            "template_status": "insufficient_data",
            "explanation": "Offline graph-guided F2E adapter: observation quality is low/heavily missing, so review is required before a hard accident label.",
        }
    if derived.get("composite_candidate"):
        return {
            "accident_type": "composite_anomaly",
            "confidence": 0.72,
            "review_needed": True,
            "abnormal_confirmed": True,
            "resample_target": None,
            "evidence_fields": sorted({s.split(":", 1)[0] for s in sigs}),
            "evidence_events": [{"field_key": s.split(":", 1)[0], "polarity": s.split(":", 1)[1], "role": "composite_cluster_evidence"} for s in sorted(sigs)],
            "template_status": "composite",
            "explanation": "Offline graph-guided F2E adapter: spatially distinct local templates coexist, so composite_anomaly is preferred over a single fire/leak label.",
        }
    if derived.get("novel_combination_candidate"):
        # Use a field from the novel evidence as resampling target.
        target_field = None
        for preferred in ("pressure", "co2", "temperature", "air_quality", "humidity"):
            if any(s.startswith(preferred + ":") for s in sigs):
                target_field = preferred
                break
        return {
            "accident_type": "needs_review_unknown",
            "confidence": 0.60,
            "review_needed": True,
            "abnormal_confirmed": False,
            "resample_target": {"field_key": target_field, "centroid": None} if target_field else None,
            "evidence_fields": sorted({s.split(":", 1)[0] for s in sigs}),
            "evidence_events": [{"field_key": s.split(":", 1)[0], "polarity": s.split(":", 1)[1], "role": "novel_combination_evidence"} for s in sorted(sigs)],
            "template_status": "novel_combination",
            "explanation": "Offline graph-guided F2E adapter: the observed field combination is outside known accident templates, so it is rejected for review.",
        }
    if weak.get("present") and float(weak.get("coherence_score", 0.0) or 0.0) >= 0.55:
        wc = weak.get("candidate_events") or []
        return {
            "accident_type": "low_snr_anomaly",
            "confidence": 0.58,
            "review_needed": True,
            "abnormal_confirmed": True,
            "resample_target": {"field_key": wc[0].get("field_key"), "centroid": wc[0].get("centroid")} if wc else None,
            "evidence_fields": sorted({c.get("field_key") for c in wc if c.get("field_key")}),
            "evidence_events": [{"field_key": c.get("field_key"), "polarity": c.get("polarity"), "role": "weak_coherent_multifield_evidence"} for c in wc[:8]],
            "template_status": "insufficient_data",
            "explanation": "Offline graph-guided F2E adapter: weak but coherent multi-field trends indicate a low-SNR anomaly; review/resampling remains required.",
        }
    return None

def offline_llm_proxy(payload: Dict[str, Any], mode: str) -> Dict[str, Any]:
    # Deterministic proxy for dry-run pipeline validation. It deliberately uses
    # only the same public input representation that would be sent to the model.
    if mode in {"f2e_events_llm_f2e_prompt", "f2e_events_graph_llm"}:
        guided = graph_guided_offline_diag(payload, mode)
        if guided is not None:
            guided["explanation"] = f"Offline proxy ({mode}); replace with --api-mode api for real LLM measurement. " + guided.get("explanation", "")
            return guided
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

def build_diagnosis_prompt(input_kind: str, scenario_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    allowed = ", ".join(ACCIDENT_TYPES)
    system = (
        "You are an industrial multi-physics accident diagnosis module. "
        "Return only one JSON object. Do not include markdown. "
        f"Allowed accident_type labels: {allowed}. "
        "The JSON schema is: {\"accident_type\": str, \"confidence\": float, "
        "\"review_needed\": bool, \"abnormal_confirmed\": bool, "
        "\"resample_target\": {\"field_key\": str|null, \"centroid\": [row,col]|null}|null, "
        "\"evidence_fields\": [str], "
        "\"evidence_events\": [{\"field_key\": str, \"polarity\": \"high\"|\"low\", \"role\": str}], "
        "\"template_status\": \"known_template\"|\"composite\"|\"novel_combination\"|\"insufficient_data\", "
        "\"explanation\": str}. "
        "Use only the provided observation representation. Do not assume hidden labels. "
        "Decision guidance: fire requires high temperature plus combustion-side AQI/CO2 evidence; "
        "electrical_overheat is mainly isolated high temperature without AQI/CO2/humidity support; "
        "steam_leak requires high humidity together with high temperature and pressure; "
        "water_leak is humidity-dominant without a thermal steam signature; "
        "co2_accumulation requires CO2-specific evidence, while dust_pollution is air-quality evidence without CO2 support. "
        "Composite anomaly should be used only when two or more spatially/physically distinct incident patterns coexist; "
        "do not label an ordinary coupled steam/fire signature as composite. "
        "If event evidence is empty and observation quality is good, prefer normal. "
        "If evidence is missing, low-confidence, or outside known templates, prefer needs_review_unknown and set review_needed=true. "
        "For low-SNR evidence, use low_snr_anomaly only when weak but coherent multi-field trends are present; otherwise request review. "
        "In evidence_events, list only observation-derived fields used for the diagnosis, with polarity when available. "
        "In the explanation, explicitly cite the fields, polarities, and trends used."
    )
    user = {
        "input_kind": input_kind,
        "task": "Diagnose the accident type and propose review/resampling action from the provided no-leak observation representation.",
        "payload": scenario_payload,
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def build_f2e_diagnosis_prompt(input_kind: str, scenario_payload: Dict[str, Any], use_graph: bool = False) -> List[Dict[str, Any]]:
    """F2E-specific diagnosis prompt.

    Fairness rule: this prompt shares the same labels, accident ontology, output
    schema, and no-leak constraints as the generic prompt. The only added
    instructions explain how to use F2E's richer event structure: tracks,
    trends, physical tags, centroids, and optional event-graph features.
    """
    allowed = ", ".join(ACCIDENT_TYPES)
    graph_text = ""
    if use_graph:
        graph_text = (
            "The payload may include derived_f2e_features and weak_multifield_evidence. "
            "These are computed only from observation-derived F2E events/fields, not from labels. "
            "Use spatial_clusters, event_signatures, composite_candidate, novel_combination_candidate, "
            "and weak_multifield_evidence as decision aids. "
            "The weak_multifield_evidence object is produced by a recent-window cumulative weak-evidence detector; "
            "candidate_events are sub-threshold candidates, not confirmed F2E events. "
            "Pay attention to recent_window_cumulative_score, temporal_persistence, peak_recent_z, "
            "combined_high_fields, weak_multifield_coherence_hint, and co_located_with_confirmed_event. "
        )
    system = (
        "You are an industrial multi-physics accident diagnosis module specialized for F2E_structured_event_tokens. "
        "Return only one JSON object. Do not include markdown. "
        f"Allowed accident_type labels: {allowed}. "
        "The JSON schema is: {\"accident_type\": str, \"confidence\": float, "
        "\"review_needed\": bool, \"abnormal_confirmed\": bool, "
        "\"resample_target\": {\"field_key\": str|null, \"centroid\": [row,col]|null}|null, "
        "\"evidence_fields\": [str], "
        "\"evidence_events\": [{\"field_key\": str, \"polarity\": \"high\"|\"low\", \"role\": str}], "
        "\"template_status\": \"known_template\"|\"composite\"|\"novel_combination\"|\"insufficient_data\", "
        "\"explanation\": str}. "
        "Use only the provided observation representation. Do not assume hidden labels, scenario IDs, injected specs, or ground truth. "
        + graph_text +
        "LOW-CONFIDENCE HARD OVERRIDE FOR F2E: before template matching, inspect payload.observation_quality. "
        "If observation_quality.low_confidence is true, or last_frame_missing_fraction >= 0.30, or max_missing_fraction >= 0.30, "
        "the default and strongly preferred accident_type is needs_review_unknown with review_needed=true, abnormal_confirmed=false, "
        "template_status=insufficient_data, and confidence <= 0.45. "
        "This low-confidence override has higher priority than known-template matching and higher priority than a single strong F2E event cluster. "
        "Even if F2E events look like fire, water_leak, steam_leak, co2_accumulation, dust_pollution, electrical_overheat, or composite_anomaly, "
        "do not output a hard accident label when heavy missingness/low confidence makes the observation unreliable; instead request review/resampling. "
        "Only bypass this override if the payload explicitly shows observation quality is good and the low-confidence fields are not among the evidence fields. "
        "When using the override, choose resample_target from the most diagnostic observed event field, preferably humidity for leak-like evidence, temperature/AQI/CO2 for fire-like evidence, pressure for pressure/process contradictions. "
        "F2E events are not raw pixels; they are physical event tokens with field_key, polarity, morphology, centroid, strength, physical_tag, and temporal_summary. "
        "Do not classify from a single dominant event before checking the full event set. "
        "First group events by centroid into local incident clusters. Events within about 3 grid cells are co-located; events separated by about 5 or more grid cells should normally be treated as distinct physical clusters. "
        "Then infer local templates per cluster: fire-like = temperature:high plus air_quality:high and/or co2:high; "
        "electrical-overheat-like = isolated temperature:high without AQI/CO2/humidity support; "
        "steam-like = co-located humidity:high + temperature:high + pressure:high; "
        "water-leak-like = humidity:high without co-located thermal/pressure steam signature; "
        "co2-accumulation-like = CO2-specific high evidence without contradictory pressure/process evidence; "
        "dust-like = air_quality:high without CO2 support. "
        "Composite override: if two or more spatially distinct clusters match different physical templates, choose accident_type=composite_anomaly and template_status=composite. "
        "In particular, a fire-like cluster plus a spatially separated humidity/leak-like cluster must not be collapsed to fire merely because the fire-like cluster is strong. "
        "Novel-combination override: if a known template leaves a strong additional event unexplained and that event changes the physical interpretation, choose needs_review_unknown with template_status=novel_combination and review_needed=true. "
        "Examples: co2:high+pressure:high, temperature:high+pressure:low, or pressure:high+air_quality:high+humidity:low are not ordinary known templates. "
        "Low-SNR rule: choose low_snr_anomaly when weak_multifield_evidence.present is true or when weak_multifield_evidence shows weak_multifield_coherence_hint=true with combined_high_fields containing temperature, co2, and air_quality, especially if weak candidates are temporally persistent or co-located with a confirmed event. This hint is an observation-derived weak-evidence coherence flag, not a ground-truth label. Do not collapse confirmed temperature:high plus weak CO2/AQI support into electrical_overheat. If only isolated temperature evidence is present and weak_multifield_evidence is absent, electrical_overheat may be appropriate. "
        "If observation_quality indicates heavy missingness/low confidence, the hard override above applies: choose needs_review_unknown, set review_needed=true, and provide a resample_target on the most diagnostic field. "
        "In evidence_events, list every observation-derived field/polarity that materially supports the decision. "
        "In the explanation, explicitly mention clusters, fields, polarities, trends, and why the final label was not a simpler known template."
    )
    user = {
        "input_kind": input_kind,
        "task": "Diagnose the accident type and propose review/resampling action from the provided no-leak F2E representation.",
        "payload": scenario_payload,
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]

def build_vlm_messages(image_data_url: str, scenario_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    allowed = ", ".join(ACCIDENT_TYPES)
    text = (
        "You are diagnosing industrial multi-physics temporal contact-sheet images. "
        f"Allowed accident_type labels: {allowed}. "
        "Rows correspond to physical fields and columns correspond to sampled times. "
        "Use temporal changes, co-location across fields, and missing/low-confidence evidence. "
        "Return only JSON with keys: accident_type, confidence, review_needed, abnormal_confirmed, "
        "resample_target, evidence_fields, evidence_events, template_status, explanation. "
        "Composite anomaly requires distinct coexisting incident patterns, not merely several coupled fields. "
        "If the image is ambiguous or affected by missing data, choose needs_review_unknown.\n"
        + json.dumps(scenario_payload, ensure_ascii=False)
    )
    return [
        {"role": "system", "content": "Return only a valid JSON object. Do not include markdown."},
        {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]},
    ]


# ============================================================
# Layer 1 metrics
# ============================================================
def run_layer1(scenarios: List[AccidentScenario], seeds: int, steps: int, obs_ratio: float, base_seed: int,
               detector_names: List[str], progress: ProgressMeter, start_job: int) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    job = start_job
    for epi in range(seeds):
        for sc_idx, sc in enumerate(scenarios):
            for det in detector_names:
                seed = base_seed + 100000 * epi + 1000 * sc_idx
                trace = collect_run_trace(sc, seed, steps, det, obs_ratio, all_fields=True)
                row = {
                    "layer": "low_level_detection",
                    "seed_index": epi,
                    "seed": seed,
                    "scenario_id": sc.scenario_id,
                    "accident_type": sc.accident_type,
                    "detector": det,
                    "ambiguous": sc.ambiguous,
                    "unseen_combo": sc.unseen_combo,
                    "expected_review": sc.expected_review,
                    "challenge": sc.challenge,
                    **trace.low_metrics,
                }
                rows.append(row)
                job += 1
                progress.update(job, f"L1 seed={epi+1}/{seeds} scenario={sc.scenario_id} detector={det}")
    return rows, job

def aggregate_layer1(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["detector"], []).append(r)
    out = []
    for det, rs in sorted(groups.items()):
        tp = int(sum(r.get("tp", 0) for r in rs))
        fp = int(sum(r.get("fp", 0) for r in rs))
        fn = int(sum(r.get("fn", 0) for r in rs))
        p, rec, f1 = precision_recall_f1(tp, fp, fn)
        out.append({
            "detector": det,
            "n": len(rs),
            "detection_precision": round(float(p), 6) if p is not None else None,
            "detection_recall": round(float(rec), 6) if rec is not None else None,
            "detection_f1": round(float(f1), 6) if f1 is not None else None,
            "mean_iou": round(float(safe_mean([r.get("mean_iou") for r in rs])), 6) if safe_mean([r.get("mean_iou") for r in rs]) is not None else None,
            "centroid_error": round(float(safe_mean([r.get("centroid_error") for r in rs])), 6) if safe_mean([r.get("centroid_error") for r in rs]) is not None else None,
            "normal_fp_per_100_frames": round(float(safe_mean([r.get("normal_fp_per_100_frames") for r in rs])), 6) if safe_mean([r.get("normal_fp_per_100_frames") for r in rs]) is not None else None,
            "extra_event_fp_per_100_effect_frames": round(float(safe_mean([r.get("extra_event_fp_per_100_effect_frames") for r in rs])), 6) if safe_mean([r.get("extra_event_fp_per_100_effect_frames") for r in rs]) is not None else None,
            "latency_steps": round(float(safe_mean([r.get("latency_steps") for r in rs])), 6) if safe_mean([r.get("latency_steps") for r in rs]) is not None else None,
            "latency_ms_per_frame": round(float(safe_mean([r.get("latency_ms_per_frame") for r in rs])), 6) if safe_mean([r.get("latency_ms_per_frame") for r in rs]) is not None else None,
            "detector_total_latency_ms": round(float(safe_mean([r.get("detector_total_latency_ms") for r in rs])), 6) if safe_mean([r.get("detector_total_latency_ms") for r in rs]) is not None else None,
            "detector_latency_ms_per_frame": round(float(safe_mean([r.get("detector_latency_ms_per_frame") for r in rs])), 6) if safe_mean([r.get("detector_latency_ms_per_frame") for r in rs]) is not None else None,
        })
    return out

# ============================================================
# Layer 2 diagnosis
# ============================================================
def normalize_diag(diag: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(diag, dict):
        return {"accident_type": "needs_review_unknown", "confidence": 0.0, "review_needed": True, "abnormal_confirmed": False, "resample_target": None, "evidence_fields": [], "evidence_events": [], "template_status": "invalid_output", "explanation": "invalid or missing model output"}
    label = str(diag.get("accident_type", "needs_review_unknown"))
    if label not in ACCIDENT_TYPES:
        label = "needs_review_unknown"
    try:
        conf = float(diag.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    return {
        "accident_type": label,
        "confidence": max(0.0, min(1.0, conf)),
        "review_needed": bool(diag.get("review_needed", label == "needs_review_unknown")),
        "abnormal_confirmed": bool(diag.get("abnormal_confirmed", label not in {"normal", "needs_review_unknown"} and conf >= 0.55)),
        "resample_target": diag.get("resample_target"),
        "evidence_fields": diag.get("evidence_fields", []) if isinstance(diag.get("evidence_fields", []), list) else [],
        "evidence_events": diag.get("evidence_events", []) if isinstance(diag.get("evidence_events", []), list) else [],
        "template_status": str(diag.get("template_status", "")),
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
    # Preserve the API error while making the pipeline complete.
    parsed = offline_llm_proxy(offline_payload, mode)
    parsed["explanation"] = "API call failed; offline fallback used for pipeline continuity. " + parsed.get("explanation", "")
    fallback_res = ModelCallResult(
        False,
        json.dumps(parsed, ensure_ascii=False),
        parsed,
        res.latency_ms,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        total_tokens=res.total_tokens,
        model=res.model or model,
        error=res.error,
        from_cache=res.from_cache,
        usage=res.usage,
        token_count_source=res.token_count_source,
        token_count_is_estimate=res.token_count_is_estimate,
        api_usage_available=res.api_usage_available,
        prompt_text_tokens=res.prompt_text_tokens,
        prompt_image_tokens=res.prompt_image_tokens,
        prompt_video_tokens=res.prompt_video_tokens,
        prompt_audio_tokens=res.prompt_audio_tokens,
        prompt_cached_tokens=res.prompt_cached_tokens,
        completion_text_tokens=res.completion_text_tokens,
        completion_reasoning_tokens=res.completion_reasoning_tokens,
        completion_audio_tokens=res.completion_audio_tokens,
    )
    return apply_token_fallback(fallback_res, messages, parsed, token_counter)

def build_method_inputs(trace_threshold: RunTrace, trace_f2e: RunTrace, image_dir: str,
                        vlm_frame_mode: str = "sampled", vlm_num_frames: int = 8,
                        llm_interval_steps: int = 50, scenario_index: int = 0,
                        render_raw_image: bool = True) -> Dict[str, Dict[str, Any]]:
    diag_t = diagnosis_query_t(trace_f2e.scenario, len(trace_f2e.fields_window), llm_interval_steps)
    th_events = summarize_events(snapshot_events_at(trace_threshold, diag_t))
    f2e_events = summarize_events(snapshot_events_at(trace_f2e, diag_t))
    fields_for_diagnosis = trace_f2e.fields_window[:diag_t + 1] or trace_f2e.fields_window
    fields_at_diagnosis = fields_for_diagnosis[-1] if fields_for_diagnosis else trace_f2e.fields_last
    matrix_t0 = time.perf_counter()
    matrix_summary = current_field_summary(fields_for_diagnosis, trace_f2e.grid == 0)
    obs_quality = observation_quality(fields_for_diagnosis, trace_f2e.grid == 0)
    weak_multifield_evidence = derive_weak_candidates_from_fields(fields_for_diagnosis, trace_f2e.grid == 0, confirmed_events=f2e_events)
    f2e_derived_features = derive_f2e_diagnostic_features(f2e_events)
    matrix_summary["observation_quality"] = obs_quality
    matrix_summary["weak_multifield_evidence"] = weak_multifield_evidence
    matrix_latency_ms = (time.perf_counter() - matrix_t0) * 1000.0
    img_path: Optional[str] = None
    frame_indices: List[int] = []
    image_render_latency_ms = 0.0
    if render_raw_image:
        if vlm_frame_mode == "last":
            image_t0 = time.perf_counter()
            blind_stem = blind_artifact_stem(trace_f2e.seed, scenario_index, "vlm", "last")
            img_path = render_fields_image(
                fields_at_diagnosis,
                os.path.join(image_dir, f"{blind_stem}.png"),
                blind_vlm_title("last"),
            )
            image_render_latency_ms = (time.perf_counter() - image_t0) * 1000.0
            frame_indices = [int(diag_t)]
        else:
            image_t0 = time.perf_counter()
            blind_stem = blind_artifact_stem(trace_f2e.seed, scenario_index, "vlm", f"{vlm_frame_mode}{vlm_num_frames}")
            img_path, frame_indices = render_fields_contact_sheet(
                fields_for_diagnosis,
                os.path.join(image_dir, f"{blind_stem}.png"),
                blind_vlm_title(vlm_frame_mode),
                mode=vlm_frame_mode,
                num_frames=vlm_num_frames,
            )
            image_render_latency_ms = (time.perf_counter() - image_t0) * 1000.0
    prep_latency = {
        "threshold_events": float(trace_threshold.low_metrics.get("detector_total_latency_ms") or 0.0),
        "f2e_events": float(trace_f2e.low_metrics.get("detector_total_latency_ms") or 0.0),
        "raw_matrix_summary": float(matrix_latency_ms),
        "raw_field_image": float(matrix_latency_ms + image_render_latency_ms),
    }
    return {
        "diagnosis_t": {"t": int(diag_t), "llm_interval_steps": int(llm_interval_steps)},
        "observation_quality": obs_quality,
        "threshold_events": {"events": th_events, "observation_quality": obs_quality, "representation_latency_ms": prep_latency["threshold_events"], "diagnosis_t": int(diag_t)},
        "f2e_events": {"events": f2e_events, "observation_quality": obs_quality, "derived_f2e_features": f2e_derived_features, "weak_multifield_evidence": weak_multifield_evidence, "representation_latency_ms": prep_latency["f2e_events"], "diagnosis_t": int(diag_t)},
        "raw_matrix_summary": {"field_summary": matrix_summary, "observation_quality": obs_quality, "representation_latency_ms": prep_latency["raw_matrix_summary"]},
        "raw_field_image": {
            "field_summary": matrix_summary,
            "observation_quality": obs_quality,
            "image_path": img_path,
            "image_filename_policy": "blind_no_scenario_id_no_accident_type",
            "diagnosis_t": int(diag_t),
            "vlm_frame_mode": vlm_frame_mode,
            "vlm_frame_indices": frame_indices,
            "representation_latency_ms": prep_latency["raw_field_image"],
            "matrix_summary_latency_ms": matrix_latency_ms,
            "image_render_latency_ms": image_render_latency_ms,
        },
    }

def run_layer2(scenarios: List[AccidentScenario], seeds: int, steps: int, obs_ratio: float, base_seed: int,
               out_dir: str, api_mode: str, client: DashScopeClient, llm_model: str, vlm_model: str,
               progress: ProgressMeter, start_job: int, vlm_frame_mode: str = "sampled",
               vlm_num_frames: int = 8, llm_interval_steps: int = 50,
               token_counter: Optional[TokenFallbackCounter] = None,
               methods: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], int]:
    diag_rows: List[Dict[str, Any]] = []
    job = start_job
    default_methods = [
        "threshold_events_rules",
        "threshold_events_llm",
        "raw_matrix_summary_llm",
        "raw_field_image_vlm",
        "f2e_events_rules",
        "f2e_events_llm",
        "f2e_events_llm_f2e_prompt",
        "f2e_events_graph_llm",
    ]
    methods = methods or default_methods
    image_dir = os.path.join(out_dir, "field_images")
    token_counter = token_counter or TokenFallbackCounter()
    for epi in range(seeds):
        for sc_idx, sc in enumerate(scenarios):
            seed = base_seed + 2000000 + 100000 * epi + 1000 * sc_idx
            trace_threshold = collect_run_trace(sc, seed, steps, "threshold_cc", obs_ratio, all_fields=True)
            trace_f2e = collect_run_trace(sc, seed, steps, "f2e_encoder", obs_ratio, all_fields=True)
            need_raw_image = "raw_field_image_vlm" in methods
            method_inputs = build_method_inputs(trace_threshold, trace_f2e, image_dir, vlm_frame_mode, vlm_num_frames, llm_interval_steps, scenario_index=sc_idx, render_raw_image=need_raw_image)
            diag_t = int((method_inputs.get("diagnosis_t") or {}).get("t", steps - 1))
            f2e_history_frames = diag_t + 1
            for method in methods:
                method_t0 = time.perf_counter()
                representation_latency_ms = 0.0
                call_res = ModelCallResult(True, "", {}, 0.0, None, None, None, model="rules", token_count_source="not_applicable_rules")
                if method == "threshold_events_rules":
                    representation_latency_ms = float(method_inputs["threshold_events"].get("representation_latency_ms") or 0.0)
                    diag = diagnose_by_rules(method_inputs["threshold_events"]["events"], source_name="threshold_events", observation_quality=method_inputs["threshold_events"].get("observation_quality"))
                elif method == "f2e_events_rules":
                    representation_latency_ms = float(method_inputs["f2e_events"].get("representation_latency_ms") or 0.0)
                    diag = diagnose_by_rules(method_inputs["f2e_events"]["events"], source_name="f2e_events", observation_quality=method_inputs["f2e_events"].get("observation_quality"))
                elif method == "threshold_events_llm":
                    representation_latency_ms = float(method_inputs["threshold_events"].get("representation_latency_ms") or 0.0)
                    payload = {
                        "events": method_inputs["threshold_events"]["events"],
                        "observation_representation": "threshold_connected_component_events",
                        "diagnosis_t": diag_t,
                        "history_frames_processed": f2e_history_frames,
                        "observation_quality": method_inputs["threshold_events"].get("observation_quality"),
                    }
                    messages = build_diagnosis_prompt("threshold_events", payload)
                    call_res = call_or_offline(client, api_mode, llm_model, messages, payload, "threshold_events_llm", token_counter)
                    diag = normalize_diag(call_res.parsed)
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
                    call_res = call_or_offline(client, api_mode, llm_model, messages, payload, "f2e_events_llm", token_counter)
                    diag = normalize_diag(call_res.parsed)
                elif method == "f2e_events_llm_f2e_prompt":
                    representation_latency_ms = float(method_inputs["f2e_events"].get("representation_latency_ms") or 0.0)
                    payload = {
                        "events": method_inputs["f2e_events"]["events"],
                        "observation_representation": "F2E_structured_event_tokens",
                        "diagnosis_t": diag_t,
                        "history_frames_processed": f2e_history_frames,
                        "observation_quality": method_inputs["f2e_events"].get("observation_quality"),
                    }
                    messages = build_f2e_diagnosis_prompt("f2e_events", payload, use_graph=False)
                    call_res = call_or_offline(client, api_mode, llm_model, messages, payload, "f2e_events_llm_f2e_prompt", token_counter)
                    diag = normalize_diag(call_res.parsed)
                elif method == "f2e_events_graph_llm":
                    representation_latency_ms = float(method_inputs["f2e_events"].get("representation_latency_ms") or 0.0)
                    payload = {
                        "events": method_inputs["f2e_events"]["events"],
                        "observation_representation": "F2E_structured_event_tokens_plus_event_graph",
                        "diagnosis_t": diag_t,
                        "history_frames_processed": f2e_history_frames,
                        "observation_quality": method_inputs["f2e_events"].get("observation_quality"),
                        "derived_f2e_features": method_inputs["f2e_events"].get("derived_f2e_features"),
                        "weak_multifield_evidence": method_inputs["f2e_events"].get("weak_multifield_evidence"),
                    }
                    messages = build_f2e_diagnosis_prompt("f2e_events_graph", payload, use_graph=True)
                    call_res = call_or_offline(client, api_mode, llm_model, messages, payload, "f2e_events_graph_llm", token_counter, max_tokens=768)
                    diag = normalize_diag(call_res.parsed)
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
                    call_res = call_or_offline(client, api_mode, llm_model, messages, payload, "raw_matrix_summary_llm", token_counter)
                    diag = normalize_diag(call_res.parsed)
                elif method == "raw_field_image_vlm":
                    representation_latency_ms = float(method_inputs["raw_field_image"].get("representation_latency_ms") or 0.0)
                    img_path = method_inputs["raw_field_image"].get("image_path")
                    payload = {
                        "field_summary": method_inputs["raw_field_image"]["field_summary"],
                        "observation_representation": "raw_field_image_temporal_contact_sheet",
                        "diagnosis_t": diag_t,
                        "history_frames_observed": f2e_history_frames,
                        "vlm_frame_mode": method_inputs["raw_field_image"].get("vlm_frame_mode"),
                        "vlm_frame_indices": method_inputs["raw_field_image"].get("vlm_frame_indices"),
                        "observation_quality": (method_inputs["raw_field_image"].get("field_summary") or {}).get("observation_quality"),
                    }
                    if img_path and os.path.exists(img_path):
                        messages = build_vlm_messages(image_to_data_url(img_path), payload)
                        call_res = call_or_offline(client, api_mode, vlm_model, messages, payload, "raw_field_image_vlm", token_counter, max_tokens=512)
                        diag = normalize_diag(call_res.parsed)
                    else:
                        parsed = offline_llm_proxy(payload, "raw_field_image_vlm_no_matplotlib")
                        call_res = ModelCallResult(
                            True,
                            json.dumps(parsed),
                            parsed,
                            0.0,
                            model="offline_proxy",
                            token_count_source="offline_no_image_usage",
                        )
                        call_res = apply_token_fallback(call_res, build_diagnosis_prompt("raw_field_image_missing", payload), parsed, token_counter)
                        diag = normalize_diag(parsed)
                else:
                    raise ValueError(method)
                diagnosis_wall_ms = (time.perf_counter() - method_t0) * 1000.0
                end_to_end_latency_ms = representation_latency_ms + diagnosis_wall_ms
                measured_api_latency_ms = None if call_res.from_cache or call_res.model in {"rules", "offline_proxy"} else call_res.latency_ms

                exp_score = explanation_score(diag, sc)
                reasoning_scores = reasoning_alignment_scores(diag, sc)
                correct = int(diag.get("accident_type") == sc.accident_type)
                if method.startswith("threshold_events"):
                    input_event_count = len(method_inputs["threshold_events"]["events"])
                elif method.startswith("f2e_events"):
                    input_event_count = len(method_inputs["f2e_events"]["events"])
                else:
                    input_event_count = None
                diag_rows.append({
                    "layer": "high_level_diagnosis",
                    "seed_index": epi,
                    "seed": seed,
                    "scenario_id": sc.scenario_id,
                    "method": method,
                    "ground_truth": sc.accident_type,
                    "prediction": diag.get("accident_type"),
                    "correct": correct,
                    "ambiguous": sc.ambiguous,
                    "unseen_combo": sc.unseen_combo,
                    "expected_review": sc.expected_review,
                    "hard_family": sc.hard_family,
                    "diagnostic_goal": sc.diagnostic_goal,
                    "challenge": sc.challenge,
                    "diagnosis_t": diag_t,
                    "llm_interval_steps": int(llm_interval_steps),
                    "f2e_history_frames": f2e_history_frames,
                    "input_event_count": input_event_count,
                    "observation_low_confidence": ((method_inputs.get("f2e_events", {}).get("observation_quality") or {}).get("low_confidence")),
                    "last_frame_missing_fraction": ((method_inputs.get("f2e_events", {}).get("observation_quality") or {}).get("last_frame_missing_fraction")),
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
                    "template_status": diag.get("template_status"),
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
                    "explanation": diag.get("explanation"),
                })
                job += 1
                progress.update(job, f"L2 seed={epi+1}/{seeds} scenario={sc.scenario_id} method={method}")
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
            "n": len(rs),
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
def parse_layers(text: str) -> List[int]:
    if text == "all":
        return [1, 2]
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val not in {1, 2}:
            raise ValueError("layers must be all or comma-separated values from 1,2")
        out.append(val)
    return sorted(set(out))


def default_layer2_methods() -> List[str]:
    return [
        "threshold_events_rules",
        "threshold_events_llm",
        "raw_matrix_summary_llm",
        "raw_field_image_vlm",
        "f2e_events_rules",
        "f2e_events_llm",
        "f2e_events_llm_f2e_prompt",
        "f2e_events_graph_llm",
    ]

def parse_layer2_methods(text: str) -> List[str]:
    allowed = default_layer2_methods()
    if not text or text.strip().lower() == "all":
        return allowed
    aliases = {
        "core": ["threshold_events_llm", "f2e_events_llm", "f2e_events_llm_f2e_prompt", "f2e_events_graph_llm"],
        "f2e": ["f2e_events_rules", "f2e_events_llm", "f2e_events_llm_f2e_prompt", "f2e_events_graph_llm"],
        "llm": ["threshold_events_llm", "raw_matrix_summary_llm", "f2e_events_llm", "f2e_events_llm_f2e_prompt", "f2e_events_graph_llm"],
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
    ap.add_argument("--layers", default="all", help="all or comma-separated subset: 1,2")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-scenarios", type=int, default=None)
    ap.add_argument("--obs-ratio", type=float, default=DEFAULT_OBS_RATIO)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", type=str, default="outputs/comprehensive_v8_2_7")
    ap.add_argument("--api-mode", choices=["offline", "auto", "api"], default="auto", help="offline: no API; auto: call API only if key exists; api: require/call API")
    ap.add_argument("--llm-model", default="qwen3.6-flash", help="model for text/event/matrix LLM comparisons")
    ap.add_argument("--vlm-model", default="qwen3-vl-flash", help="model for raw image + VLM comparison")
    ap.add_argument("--vlm-frame-mode", choices=["last", "sampled", "all"], default="sampled", help="raw_field_image+VLM input: last frame, sampled contact sheet, or all-frame contact sheet")
    ap.add_argument("--vlm-num-frames", type=int, default=8, help="number of sampled frames for --vlm-frame-mode sampled")
    ap.add_argument("--llm-interval-steps", type=int, default=50,
                    help="Layer 2 diagnosis query cadence, matching the live F2E+LLM demo. The LLM receives F2E events from this scheduled frame after F2E has processed all prior frames.")
    ap.add_argument("--layer2-methods", default="all",
                    help="Layer 2 methods to run: all, core, f2e, llm, or comma-separated method names. New methods: f2e_events_llm_f2e_prompt,f2e_events_graph_llm")
    ap.add_argument("--dashscope-base-url", default=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    ap.add_argument("--enable-thinking", action="store_true", help="enable DashScope thinking mode. Default off for fair latency/token comparison.")
    ap.add_argument("--api-timeout", type=float, default=60.0)
    ap.add_argument("--token-fallback", choices=["none", "hf_tokenizer", "heuristic"], default="none",
                    help="Fallback token counting only when API usage is missing. Default none keeps token metrics API-measured only.")
    ap.add_argument("--tokenizer-model", default=None,
                    help="HuggingFace tokenizer id/path for --token-fallback hf_tokenizer. Defaults to a Qwen3 tokenizer surrogate.")
    ap.add_argument("--tokenizer-allow-download", action="store_true",
                    help="Allow transformers to download tokenizer files if they are not already cached locally.")
    ap.add_argument("--progress", choices=["on", "off"], default="on")
    ap.add_argument("--progress-interval", type=float, default=2.0)
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    layer2_methods = parse_layer2_methods(args.layer2_methods)
    seeds = args.seeds if args.seeds is not None else (1 if args.profile == "quick" else 10)
    steps = args.steps if args.steps is not None else (90 if args.profile == "quick" else 170)
    scenarios = make_accident_scenarios(args.profile)
    if args.max_scenarios is not None:
        scenarios = scenarios[:args.max_scenarios]

    detector_names = ["threshold_cc", "cusum_ewma", "vision_heatmap", "f2e_encoder"]
    total_jobs = 0
    if 1 in layers:
        total_jobs += seeds * len(scenarios) * len(detector_names)
    if 2 in layers:
        total_jobs += seeds * len(scenarios) * len(layer2_methods)
    progress = ProgressMeter(total_jobs, enabled=args.progress == "on", interval_sec=args.progress_interval)

    os.makedirs(args.out_dir, exist_ok=True)
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
    )
    token_counter = TokenFallbackCounter(
        mode=args.token_fallback,
        tokenizer_model=args.tokenizer_model,
        local_files_only=not args.tokenizer_allow_download,
    )

    t0 = time.perf_counter()
    job = 0
    layer1_rows: List[Dict[str, Any]] = []
    layer2_rows: List[Dict[str, Any]] = []
    progress.update(0, "starting")
    if 1 in layers:
        layer1_rows, job = run_layer1(scenarios, seeds, steps, args.obs_ratio, args.seed, detector_names, progress, job)
    if 2 in layers:
        layer2_rows, job = run_layer2(scenarios, seeds, steps, args.obs_ratio, args.seed, args.out_dir,
                                      args.api_mode, client, args.llm_model, args.vlm_model, progress, job,
                                      args.vlm_frame_mode, args.vlm_num_frames, args.llm_interval_steps, token_counter,
                                      methods=layer2_methods)
    progress.update(total_jobs, "done")

    layer1_summary = aggregate_layer1(layer1_rows) if layer1_rows else []
    layer2_summary = aggregate_layer2(layer2_rows) if layer2_rows else []
    layer2_family_summary = aggregate_layer2_by_family(layer2_rows) if layer2_rows else []
    layer2_llm_advantage = aggregate_llm_advantage(layer2_summary) if layer2_summary else []
    layer2_llm_advantage_by_family = aggregate_llm_advantage_by_family(layer2_family_summary) if layer2_family_summary else []

    run_config = {
        "version": COMPARISON_VERSION,
        "core_version": CORE_VERSION,
        "profile": args.profile,
        "layers": layers,
        "layer2_methods": layer2_methods,
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
        "enable_thinking": bool(args.enable_thinking),
        "token_fallback": args.token_fallback,
        "tokenizer_model": token_counter.tokenizer_model,
        "tokenizer_local_files_only": not args.tokenizer_allow_download,
        "token_counter_status": token_counter.status(),
        "token_accounting_note": "Primary token metrics use DashScope/OpenAI-compatible response usage. Rules have token_not_applicable_n and blank token columns; fallback counts are estimates and flagged per row.",
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
        "layer1_detection_summary": layer1_summary,
        "layer2_diagnosis_summary": layer2_summary,
        "layer2_hard_family_summary": layer2_family_summary,
        "layer2_llm_advantage_summary": layer2_llm_advantage,
        "layer2_llm_advantage_by_family": layer2_llm_advantage_by_family,
        "layer1_rows": layer1_rows,
        "layer2_rows": layer2_rows,
    }
    with open(os.path.join(args.out_dir, "comprehensive_comparison_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    save_csv(os.path.join(args.out_dir, "layer1_detection_records.csv"), layer1_rows)
    save_csv(os.path.join(args.out_dir, "layer1_detection_summary.csv"), layer1_summary)
    save_csv(os.path.join(args.out_dir, "layer2_diagnosis_records.csv"), layer2_rows)
    save_csv(os.path.join(args.out_dir, "layer2_diagnosis_summary.csv"), layer2_summary)
    save_csv(os.path.join(args.out_dir, "layer2_hard_family_summary.csv"), layer2_family_summary)
    save_csv(os.path.join(args.out_dir, "layer2_llm_advantage_summary.csv"), layer2_llm_advantage)
    save_csv(os.path.join(args.out_dir, "layer2_llm_advantage_by_family.csv"), layer2_llm_advantage_by_family)

    print(f"\n===== Comprehensive F2E/LLM/VLM Comparison {COMPARISON_VERSION} =====")
    print(f"core={CORE_VERSION} profile={args.profile} layers={layers} seeds={seeds} steps={steps} scenarios={len(scenarios)}")
    print(f"api_mode={args.api_mode} api_key_present={bool(api_key)} llm_model={args.llm_model} vlm_model={args.vlm_model} vlm_frame_mode={args.vlm_frame_mode} vlm_num_frames={args.vlm_num_frames} llm_interval_steps={args.llm_interval_steps} thinking={args.enable_thinking}")
    print(f"token_accounting=api_usage_primary fallback={args.token_fallback} tokenizer={token_counter.tokenizer_model}")
    print("NO-LEAK CHECK: the F2E encoder is called only via update(t, current_fields); GT masks and accident labels remain in the evaluator.")
    if layer1_summary:
        print("\nLayer 1 summary:")
        for r in layer1_summary:
            print(f"  {r['detector']}: F1={r.get('detection_f1')} IoU={r.get('mean_iou')} centroid={r.get('centroid_error')} FP100={r.get('normal_fp_per_100_frames')} latency={metric_ms(r.get('latency_ms_per_frame'))}/frame detector={metric_ms(r.get('detector_latency_ms_per_frame'))}/frame")
    if layer2_summary:
        print("\nLayer 2 summary:")
        for r in layer2_summary:
            print(f"  {r['method']}: acc={r.get('accident_classification_accuracy')} macroF1={r.get('macro_f1')} reason={r.get('reasoning_alignment_score')} hard_reason={r.get('hard_reasoning_alignment_score')} ambiguous={r.get('ambiguous_case_accuracy')}/{r.get('ambiguous_reasoning_alignment_score')} unseen={r.get('unseen_combination_accuracy')}/{r.get('unseen_reasoning_alignment_score')} ood_reject={r.get('ood_template_rejection_success')} review={r.get('uncertainty_review_success')} tokens={r.get('mean_total_tokens')}")
    print("\nSaved:")
    for name in [
        "comprehensive_comparison_results.json", "run_config.json",
        "layer1_detection_records.csv", "layer1_detection_summary.csv",
        "layer2_diagnosis_records.csv", "layer2_diagnosis_summary.csv",
        "layer2_hard_family_summary.csv",
        "layer2_llm_advantage_summary.csv", "layer2_llm_advantage_by_family.csv",
    ]:
        print("  " + os.path.join(args.out_dir, name))

if __name__ == "__main__":
    main()
