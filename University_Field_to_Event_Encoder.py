"""
Universal Field-to-Event Encoder Test Suite, no-leak version v8.1.0 core.

Academic-integrity rule
-----------------------
The Field-to-Event Encoder receives ONLY:
    encoder.update(t, current_fields)
where current_fields are the mixed fields after background + injected anomaly.

The encoder does NOT receive clean background/baseline, confidence, injected
incident specs, ground-truth masks, event center/radius/type/trend.

This module is the reusable no-leak core used by lab_ontologyEvaluation_and_ablation.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy.ndimage import label, binary_dilation, binary_closing, gaussian_filter

GRID_SIZE = 30
DEFAULT_OBS_RATIO = 0.20
CORE_VERSION = "v8.1.1"

# ============================================================
# Field registry
# ============================================================
@dataclass(frozen=True)
class FieldSemantic:
    key: str
    display_name: str
    zh_name: str
    unit: str
    sigma: float
    high_label: str
    low_label: str
    high_zh: str
    low_zh: str
    z_threshold: float = 1.65
    boundary_z_threshold: float = 0.45
    cum_threshold: float = 1.05
    min_area: int = 3

FIELD_REGISTRY: Dict[str, FieldSemantic] = {
    "temperature": FieldSemantic("temperature", "Temperature", "温度", "degC", 1.50, "hotspot", "cold_spot", "高温热源", "低温冷斑", 1.65, 0.45, 1.05, 3),
    "humidity": FieldSemantic("humidity", "Humidity", "湿度", "%RH", 4.00, "wet_patch", "dry_patch", "高湿湿斑", "低湿干斑", 1.65, 0.35, 1.05, 3),
    "pressure": FieldSemantic("pressure", "Pressure Differential", "气压/压差", "Pa", 0.80, "pressure_rise", "pressure_drop", "压强升高区", "压强下降区", 1.65, 0.45, 1.05, 3),
    "co2": FieldSemantic("co2", "CO2", "二氧化碳", "ppm", 35.0, "co2_accumulation", "co2_drop", "CO2积聚区", "CO2降低区", 1.65, 0.45, 1.05, 3),
    "air_quality": FieldSemantic("air_quality", "Air Quality Index", "空气质量", "AQI", 8.0, "air_quality_degradation", "air_quality_improvement", "空气质量恶化区", "空气质量改善区", 1.65, 0.45, 1.05, 3),
}
FIELDS = tuple(FIELD_REGISTRY.keys())

SHAPE_ZH = {
    "point_like": "点状",
    "compact_blob": "小团块状",
    "blob": "团块状",
    "oval": "椭圆状",
    "elongated_strip": "条带状",
}

# ============================================================
# Map utilities
# ============================================================
def make_obstacle_grid(size: int = GRID_SIZE, obs_ratio: float = DEFAULT_OBS_RATIO, seed: int = 0) -> np.ndarray:
    """0=free, 1=obstacle. Keep the largest free-space component only."""
    rng = np.random.default_rng(seed)
    grid = (rng.random((size, size)) < obs_ratio).astype(np.uint8)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
    grid = keep_largest_free_component(grid)
    return grid

def keep_largest_free_component(grid: np.ndarray) -> np.ndarray:
    free = grid == 0
    lab, n = label(free, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8))
    if n <= 1:
        return grid.copy()
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    main = int(counts.argmax())
    cleaned = np.ones_like(grid, dtype=np.uint8)
    cleaned[lab == main] = 0
    return cleaned

def random_free_cell(grid: np.ndarray, rng: np.random.Generator, margin: int = 3) -> Tuple[int, int]:
    free = np.argwhere(grid == 0)
    if margin > 0:
        ok = (free[:, 0] >= margin) & (free[:, 0] < grid.shape[0]-margin) & (free[:, 1] >= margin) & (free[:, 1] < grid.shape[1]-margin)
        if ok.any():
            free = free[ok]
    r, c = free[int(rng.integers(0, len(free)))]
    return int(r), int(c)

def nearest_free(grid: np.ndarray, center: Tuple[float, float]) -> Tuple[int, int]:
    free = np.argwhere(grid == 0)
    d2 = (free[:, 0] - center[0]) ** 2 + (free[:, 1] - center[1]) ** 2
    r, c = free[int(d2.argmin())]
    return int(r), int(c)

# ============================================================
# Simulator-only backgrounds and injected effects
# ============================================================
def normal_backgrounds(grid: np.ndarray, field_keys: Tuple[str, ...], t: int) -> Dict[str, np.ndarray]:
    """Clean background used by simulator ONLY. Never pass this into encoder."""
    rr, cc = np.indices(grid.shape)
    out: Dict[str, np.ndarray] = {}
    phase = 2 * np.pi * t / 240.0
    if "temperature" in field_keys:
        out["temperature"] = 22.0 + 0.055*cc + 0.035*rr + 0.35*np.sin(phase) + 0.25*np.sin((rr+cc)/8.5)
    if "humidity" in field_keys:
        out["humidity"] = 55.0 - 0.035*cc + 0.025*rr - 0.25*np.sin(phase*0.9) + 0.25*np.cos((rr-cc)/10.0)
    if "pressure" in field_keys:
        out["pressure"] = 5.0 + 0.012*cc - 0.010*rr + 0.10*np.sin(phase*0.7)
    if "co2" in field_keys:
        out["co2"] = 430.0 + 1.2*rr + 0.6*cc + 7.0*np.sin(phase*0.5)
    if "air_quality" in field_keys:
        out["air_quality"] = 35.0 + 0.25*rr + 0.18*cc + 1.5*np.sin(phase*0.6)
    for k in out:
        out[k] = out[k].astype(np.float32)
        out[k][grid == 1] = np.nan
    return out

@dataclass
class EffectSpec:
    effect_id: str
    field_key: str
    polarity: str                 # high or low
    shape: str                    # point_like / compact_blob / blob / oval / elongated_strip
    area_trend: str               # expanding / shrinking / stable
    intensity_trend: str          # strengthening / weakening / stable
    center: Tuple[float, float]
    start: int = 25
    end: int = 150
    amplitude_sigma: float = 3.2
    radius: float = 3.0
    angle: float = 0.0
    axis_ratio: float = 2.2
    velocity: Tuple[float, float] = (0.0, 0.0)  # total displacement over active lifetime, in grid cells

    def progress(self, t: int) -> float:
        if t < self.start:
            return 0.0
        if t >= self.end:
            return 1.0
        return float((t - self.start) / max(1, self.end - self.start - 1))

    def is_active(self, t: int) -> bool:
        return self.start <= t < self.end

    def _radius_scale(self, p: float) -> float:
        if self.area_trend == "expanding":
            return 0.55 + 0.75*p
        if self.area_trend == "shrinking":
            return 1.30 - 0.65*p
        return 1.0

    def _amp_scale(self, p: float) -> float:
        if self.intensity_trend == "strengthening":
            return 0.65 + 0.70*p
        if self.intensity_trend == "weakening":
            return 1.35 - 0.70*p
        return 1.0

    def mask_and_delta(self, grid: np.ndarray, t: int, amplitude_multiplier: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Return GT mask and physical delta. Used by simulator/evaluator only."""
        sem = FIELD_REGISTRY[self.field_key]
        mask = np.zeros(grid.shape, dtype=bool)
        delta = np.zeros(grid.shape, dtype=np.float32)
        if not self.is_active(t):
            return mask, delta
        rr, cc = np.indices(grid.shape)
        p = self.progress(t)
        r0 = float(self.center[0]) + float(self.velocity[0]) * p
        c0 = float(self.center[1]) + float(self.velocity[1]) * p
        rscale = self._radius_scale(p)
        amp = self.amplitude_sigma * sem.sigma * self._amp_scale(p) * amplitude_multiplier
        sign = 1.0 if self.polarity == "high" else -1.0
        radius = self.radius * rscale

        if self.shape == "point_like":
            d = np.sqrt((rr-r0)**2 + (cc-c0)**2)
            field = np.exp(-(d**2) / (2*max(0.55, radius*0.55)**2))
            mask = d <= max(1.0, radius)
        elif self.shape in ("compact_blob", "blob"):
            d = np.sqrt((rr-r0)**2 + (cc-c0)**2)
            field = np.exp(-(d**2) / (2*max(0.9, radius*0.55)**2))
            mask = d <= max(1.3, radius)
        elif self.shape == "oval":
            ang = self.angle
            x = cc - c0
            y = rr - r0
            xp = x*np.cos(ang) + y*np.sin(ang)
            yp = -x*np.sin(ang) + y*np.cos(ang)
            a = radius * self.axis_ratio
            b = radius
            q = (xp/a)**2 + (yp/b)**2
            field = np.exp(-q*1.35)
            mask = q <= 1.0
        elif self.shape == "elongated_strip":
            ang = self.angle
            x = cc - c0
            y = rr - r0
            xp = x*np.cos(ang) + y*np.sin(ang)
            yp = -x*np.sin(ang) + y*np.cos(ang)
            length = max(4.0, radius * 3.8)
            width = max(0.8, radius * 0.45)
            curve = 0.22 * np.sin(xp / max(2.5, length) * np.pi)
            dist = np.abs(yp - curve * length)
            along = np.abs(xp)
            field = np.exp(-(dist**2)/(2*width**2)) * np.exp(-(along**2)/(2*(length*0.75)**2))
            mask = (dist <= width*1.35) & (along <= length)
        else:
            raise ValueError(self.shape)

        mask &= (grid == 0)
        delta = sign * amp * field.astype(np.float32)
        delta[~mask] = 0.0
        return mask, delta

@dataclass
class TestCase:
    case_id: str
    effects: List[EffectSpec]

@dataclass(frozen=True)
class StressConfig:
    """Simulator-only perturbations. These are never passed to the encoder."""
    name: str = "baseline"
    noise_sigma: float = 0.0              # Gaussian sensor noise, in units of each field's nominal sigma
    drift_sigma_per_100: float = 0.0      # smooth background drift speed per 100 frames, in sigma units
    occlusion_rate: float = 0.0           # random missing-cell ratio among free cells
    anomaly_scale: float = 1.0            # multiplicative anomaly intensity scale
    moving_px_total: float = 0.0          # total displacement of anomaly center over lifetime

def build_current_fields(
    grid: np.ndarray,
    testcase: TestCase,
    t: int,
    field_keys: Tuple[str, ...],
    stress: Optional[StressConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """Build mixed fields for the simulator.

    The returned current fields are the only arrays later given to the encoder.
    The clean backgrounds, stress parameters, injected effects, and GT masks stay
    entirely inside the simulator/evaluator.
    """
    stress = stress or StressConfig()
    rng = rng or np.random.default_rng(0)
    backgrounds = normal_backgrounds(grid, field_keys, t)
    current = {k: v.copy() for k, v in backgrounds.items()}
    gt_records: List[Dict[str, Any]] = []

    for eff in testcase.effects:
        if eff.field_key not in current:
            continue
        mask, delta = eff.mask_and_delta(grid, t, amplitude_multiplier=stress.anomaly_scale)
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

    rr, cc = np.indices(grid.shape)
    spatial_pattern = np.sin((rr + 1.7 * cc) / 11.0 + 0.03 * t).astype(np.float32)
    for k, x in current.items():
        sem = FIELD_REGISTRY[k]
        valid = (grid == 0) & np.isfinite(x)
        if stress.drift_sigma_per_100 > 0:
            drift = sem.sigma * stress.drift_sigma_per_100 * (t / 100.0)
            x[valid] = x[valid] + drift + 0.25 * drift * spatial_pattern[valid]
        if stress.noise_sigma > 0:
            x[valid] = x[valid] + rng.normal(0.0, sem.sigma * stress.noise_sigma, size=int(valid.sum())).astype(np.float32)
        if stress.occlusion_rate > 0:
            occ = valid & (rng.random(x.shape) < stress.occlusion_rate)
            x[occ] = np.nan
        current[k] = x.astype(np.float32)

    return current, gt_records

# ============================================================
# Universal Field-to-Event Encoder, NO clean baseline input
# ============================================================
@dataclass
class EventTrack:
    track_id: str
    field_key: str
    polarity: str
    last_seen: int
    history: deque

    def update(self, t: int, event: Dict[str, Any]) -> None:
        self.last_seen = t
        history_item = {
            "t": t,
            "area": event["area"],
            "soft_area": float(event.get("soft_area", event["area"])),
            "support_area": float(event.get("support_area", event["area"])),
            "weighted_radius": float(event.get("weighted_radius", 0.0)),
            "score_sum": float(event.get("score_sum", event.get("priority", 0.0))),
            "core_z": event["z_core_mean"],
            "centroid": event["centroid"],
            "morphology": event["morphology"],
        }
        if self.history:
            prev = self.history[-1]
            dt = max(1, int(t - prev["t"]))
            c0 = np.array(prev["centroid"], dtype=float)
            c1 = np.array(event["centroid"], dtype=float)
            history_item["velocity"] = tuple(((c1 - c0) / dt).tolist())
        else:
            history_item["velocity"] = (0.0, 0.0)
        self.history.append(history_item)

    def _trend(self, key: str, stable_tol_frac: float = 0.10) -> str:
        """Robust trend estimated from the whole track history.

        v8.1 change: area trend uses soft spatial statistics by default, not
        only the hard connected-component mask area.  This is important for
        point-like and compact anomalies whose hard mask often stays quantized
        while the physical support radius is expanding or shrinking.
        """
        t_vals = np.array([h["t"] for h in self.history], dtype=np.float32)
        vals = np.array([h.get(key, np.nan) for h in self.history], dtype=np.float32)
        valid = np.isfinite(vals)
        t_vals = t_vals[valid]
        vals = vals[valid]
        n = len(vals)
        if n < 6:
            return "stable"

        # Trim the most unstable early detections when enough history exists.
        if n >= 12:
            t_vals = t_vals[2:]
            vals = vals[2:]
            n = len(vals)

        t_mean = np.mean(t_vals)
        v_mean = np.mean(vals)
        denom = np.sum((t_vals - t_mean) ** 2)
        if denom < 1e-5:
            return "stable"

        slope = np.sum((t_vals - t_mean) * (vals - v_mean)) / denom
        delta = float(slope * (t_vals[-1] - t_vals[0]))
        scale = max(1e-6, float(np.nanmedian(np.abs(vals))))
        rel = delta / max(1.0, scale)

        if key in {"soft_area", "area", "support_area", "weighted_radius", "score_sum"}:
            # Small-object trend needs a lower normalized threshold; hard area
            # may be 3/4/5 cells for many frames even while soft radius changes.
            small_scale = scale <= 18.0
            if key == "weighted_radius":
                tol_abs = 0.06 if small_scale else 0.10
                tol_rel = 0.035 if small_scale else 0.055
            elif key == "score_sum":
                tol_abs = 0.25 if small_scale else 0.50
                tol_rel = 0.080 if small_scale else 0.100
            else:
                tol_abs = 0.35 if small_scale else max(1.00, 0.055 * scale)
                tol_rel = 0.045 if small_scale else 0.075
            if delta > tol_abs or rel > tol_rel:
                return "expanding"
            if delta < -tol_abs or rel < -tol_rel:
                return "shrinking"
            return "stable"

        # Intensity/core-z trend.
        tol = max(0.28, 0.085 * scale)
        if delta > tol:
            return "strengthening"
        if delta < -tol:
            return "weakening"
        return "stable"

    def temporal_summary(self) -> Dict[str, Any]:
        first = self.history[0]
        last = self.history[-1]
        # Soft-area trend is the primary area trend in v8.1.  Weighted radius
        # is used as a fallback for very small masks.
        area_trend = self._trend("soft_area", 0.06)
        if area_trend == "stable" and np.nanmedian([h.get("area", 0.0) for h in self.history]) <= 11:
            area_trend = self._trend("weighted_radius", 0.04)
        return {
            "duration_steps": int(last["t"] - first["t"] + 1),
            "area_start": int(first["area"]),
            "area_current": int(last["area"]),
            "soft_area_start": round(float(first.get("soft_area", first["area"])), 3),
            "soft_area_current": round(float(last.get("soft_area", last["area"])), 3),
            "weighted_radius_start": round(float(first.get("weighted_radius", 0.0)), 3),
            "weighted_radius_current": round(float(last.get("weighted_radius", 0.0)), 3),
            "area_trend": area_trend,
            "intensity_start": round(float(first["core_z"]), 3),
            "intensity_current": round(float(last["core_z"]), 3),
            "intensity_trend": self._trend("core_z", 0.035),
        }

class UniversalFieldToEventEncoder:
    """Stateful streaming encoder.

    Public API:
        update(t, current_fields)

    It intentionally does NOT accept clean background/baseline, confidence,
    injected-event specs, or GT masks. It estimates a smooth background from
    the current mixed field internally.
    """
    def __init__(self, grid: np.ndarray, registry: Dict[str, FieldSemantic] = FIELD_REGISTRY, rtca_alpha: float = 0.82, max_track_gap: int = 10, use_background_ema: bool = True, use_cumulative_score: bool = True, use_hysteresis: bool = True, use_tracking: bool = True, use_morphology_debounce: bool = True, use_global_drift_compensation: bool = True, use_temporal_imputation: bool = True, use_velocity_tracking: bool = True):
        self.grid = grid.astype(np.uint8)
        self.free_mask = self.grid == 0
        self.registry = registry
        self.rtca_alpha = rtca_alpha
        self.max_track_gap = max_track_gap
        self.use_background_ema = use_background_ema
        self.use_cumulative_score = use_cumulative_score
        self.use_hysteresis = use_hysteresis
        self.use_tracking = use_tracking
        self.use_morphology_debounce = use_morphology_debounce
        self.use_global_drift_compensation = use_global_drift_compensation
        self.use_temporal_imputation = use_temporal_imputation
        self.use_velocity_tracking = use_velocity_tracking
        self.cum_scores: Dict[Tuple[str, str], np.ndarray] = {}
        self.tracks: Dict[str, EventTrack] = {}
        self.next_track_id = 1
        self.latest_residuals: Dict[str, np.ndarray] = {}
        self.latest_background_estimates: Dict[str, np.ndarray] = {}
        self.latest_observed_masks: Dict[str, np.ndarray] = {}
        self.bg_models: Dict[str, np.ndarray] = {}
        self.last_imputed_fields: Dict[str, np.ndarray] = {}

    def estimate_background(self, x: np.ndarray) -> np.ndarray:
        valid = self.free_mask & np.isfinite(x)
        if valid.sum() == 0:
            return np.zeros_like(x, dtype=np.float32)
        med = float(np.nanmedian(x[valid]))
        filled = np.where(valid, x, med).astype(np.float32)
        weight = valid.astype(np.float32)
        sig = 5.0
        num = gaussian_filter(filled * weight, sigma=sig, mode="nearest")
        den = gaussian_filter(weight, sigma=sig, mode="nearest") + 1e-6
        bg = (num / den).astype(np.float32)
        resid = x - bg
        bg = bg + float(np.nanmedian(resid[valid]))
        bg[~self.free_mask] = np.nan
        return bg

    def _temporal_impute(self, field_key: str, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fill missing/occluded cells without using GT.

        The encoder only observes current fields.  When sensors are missing, we
        reuse the previous imputed field at those cells and fall back to the
        current spatial median for never-observed cells.  This keeps occluded
        anomalies from being split into many fragments.
        """
        observed = self.free_mask & np.isfinite(x)
        if (not self.use_temporal_imputation) or observed.all():
            return x.astype(np.float32), observed
        out = x.astype(np.float32).copy()
        valid_vals = observed
        med = float(np.nanmedian(x[valid_vals])) if valid_vals.any() else 0.0
        if field_key in self.last_imputed_fields:
            prev = self.last_imputed_fields[field_key]
            missing = self.free_mask & (~observed)
            out[missing] = prev[missing]
            out[missing & (~np.isfinite(out))] = med
        else:
            out[self.free_mask & (~observed)] = med
        out[~self.free_mask] = np.nan
        return out, observed

    def _robust_global_drift_correct(self, z: np.ndarray, observed: np.ndarray) -> np.ndarray:
        """Remove per-frame global residual drift without using GT.

        v8.1.1 adjustment: use a conservative robust median correction.  A
        full per-frame plane can overfit and suppress point-like anomalies on a
        30x30 grid, so the low-order term is intentionally omitted here.
        """
        if not self.use_global_drift_compensation:
            return z
        valid = self.free_mask & observed & np.isfinite(z)
        if int(valid.sum()) < 25:
            return z
        vals = z[valid]
        cutoff = float(np.nanpercentile(np.abs(vals), 75.0))
        stable_vals = vals[np.abs(vals) <= max(0.35, cutoff)]
        if stable_vals.size < 12:
            stable_vals = vals
        bias = float(np.nanmedian(stable_vals))
        # Ignore tiny numerical bias; clip extreme bias so true local events are
        # not erased by a single frame's robust estimate.
        if abs(bias) < 0.08:
            return z
        bias = float(np.clip(bias, -0.85, 0.85))
        out = z - bias
        out[~self.free_mask] = np.nan
        return out.astype(np.float32)

    def update(self, t: int, current_fields: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for field_key, x in current_fields.items():
            if field_key not in self.registry:
                continue
            sem = self.registry[field_key]
            raw_x = np.asarray(x, dtype=np.float32)
            x, observed_mask = self._temporal_impute(field_key, raw_x)
            self.latest_observed_masks[field_key] = observed_mask.copy()
            spatial_bg = self.estimate_background(x)
            if field_key not in self.bg_models:
                self.bg_models[field_key] = spatial_bg.copy()
            est_bg = self.bg_models[field_key] if self.use_background_ema else spatial_bg
            z = (x - est_bg) / (sem.sigma + 1e-6)
            z[~self.free_mask] = np.nan
            z = self._robust_global_drift_correct(z, observed_mask)
            self.latest_background_estimates[field_key] = est_bg.copy()
            self.latest_residuals[field_key] = z
            field_event_union = np.zeros_like(self.free_mask, dtype=bool)
            field_abs_score = np.zeros_like(x, dtype=np.float32)
            for polarity in ("high", "low"):
                score = np.maximum(z, 0.0) if polarity == "high" else np.maximum(-z, 0.0)
                score[~np.isfinite(score)] = 0.0
                if self.use_cumulative_score:
                    key = (field_key, polarity)
                    if key not in self.cum_scores:
                        self.cum_scores[key] = np.zeros_like(score, dtype=np.float32)
                    cum = self.cum_scores[key]
                    cum[:] = self.rtca_alpha * cum + (1 - self.rtca_alpha) * score
                    core_mask = self.free_mask & (score >= sem.z_threshold) & (cum >= sem.cum_threshold)
                    support_mask = self.free_mask & (score >= sem.boundary_z_threshold) & (cum >= sem.cum_threshold * 0.35)
                else:
                    core_mask = self.free_mask & (score >= sem.z_threshold)
                    support_mask = self.free_mask & (score >= sem.boundary_z_threshold)
                event_mask = self._hysteresis_mask(core_mask, support_mask) if self.use_hysteresis else core_mask
                # Missing cells often carve holes in a physical anomaly.  Close
                # only within the free space to merge occlusion-caused fragments.
                if self.use_temporal_imputation and event_mask.any():
                    event_mask = binary_closing(event_mask, structure=np.ones((3, 3), dtype=bool)) & self.free_mask
                field_event_union |= event_mask
                field_abs_score = np.maximum(field_abs_score, score.astype(np.float32))
                events.extend(self._extract_components(field_key, polarity, sem, z, score, core_mask, event_mask, t))
            if self.use_background_ema:
                valid = self.free_mask & observed_mask & np.isfinite(raw_x)
                normal_mask = valid & (~field_event_union) & (field_abs_score < 0.80)
                bg = self.bg_models[field_key].copy()
                # v8.1: slightly faster adaptation on confident normal cells,
                # but conservative update under uncertain/event cells.
                bg[normal_mask] = 0.975 * bg[normal_mask] + 0.025 * raw_x[normal_mask]
                uncertain_mask = valid & (~normal_mask)
                bg[uncertain_mask] = 0.996 * bg[uncertain_mask] + 0.004 * spatial_bg[uncertain_mask]
                bg[~self.free_mask] = np.nan
                self.bg_models[field_key] = bg.astype(np.float32)
            else:
                self.bg_models[field_key] = spatial_bg.astype(np.float32)
            self.last_imputed_fields[field_key] = x.astype(np.float32)
        for ev in events:
            if self.use_tracking:
                self._assign_track(ev, t)
            else:
                self._assign_single_frame_event(ev, t)
        if self.use_tracking:
            self._drop_stale_tracks(t)
        events.sort(key=lambda e: e["priority"], reverse=True)
        return events

    def _hysteresis_mask(self, core: np.ndarray, support: np.ndarray) -> np.ndarray:
        lab, n = label(support, structure=np.ones((3, 3), dtype=np.uint8))
        out = np.zeros_like(core, dtype=bool)
        for idx in range(1, n + 1):
            comp = lab == idx
            if (comp & core).any():
                out |= comp
        return out

    def _extract_components(self, field_key: str, polarity: str, sem: FieldSemantic, z: np.ndarray, score: np.ndarray, core_mask: np.ndarray, event_mask: np.ndarray, t: int) -> List[Dict[str, Any]]:
        lab, n = label(event_mask, structure=np.ones((3, 3), dtype=np.uint8))
        rr, cc = np.indices(event_mask.shape)
        out: List[Dict[str, Any]] = []
        for idx in range(1, n + 1):
            mask = lab == idx
            area = int(mask.sum())
            if area < sem.min_area:
                continue
            weights = score[mask] + 1e-6
            centroid = (float(np.average(rr[mask], weights=weights)), float(np.average(cc[mask], weights=weights)))
            core = mask & core_mask
            if core.any():
                z_core_mean = float(np.nanmean(score[core]))
            else:
                z_core_mean = float(np.nanmean(score[mask]))
            # Soft metrics are used by temporal trend estimation.
            support_score = np.clip(score[mask] / max(sem.boundary_z_threshold, 1e-6), 0.0, 1.0)
            soft_area = float(np.sum(support_score))
            score_sum = float(np.nansum(score[mask]))
            dr = rr[mask].astype(np.float32) - float(centroid[0])
            dc = cc[mask].astype(np.float32) - float(centroid[1])
            if weights.sum() > 0:
                weighted_radius = float(np.sqrt(np.average(dr * dr + dc * dc, weights=weights)))
            else:
                weighted_radius = 0.0
            morph, morph_metrics = self._classify_morphology(mask)
            near_obs_ratio = morph_metrics["near_obstacle_ratio"]
            physical_label = sem.high_label if polarity == "high" else sem.low_label
            physical_zh = sem.high_zh if polarity == "high" else sem.low_zh
            direction = "high_value_anomaly" if polarity == "high" else "low_value_anomaly"
            event = {
                "track_id": None,
                "t": int(t),
                "field_key": field_key,
                "field_zh": sem.zh_name,
                "polarity": polarity,
                "generic_event": {"direction": direction, "morphology": morph},
                "physical_tag": {"label": physical_label, "label_zh": physical_zh},
                "morphology": morph,
                "morphology_zh": SHAPE_ZH[morph],
                "morphology_metrics": morph_metrics,
                "centroid": (round(centroid[0], 3), round(centroid[1], 3)),
                "area": area,
                "core_area": int(core.sum()),
                "support_area": int(mask.sum()),
                "soft_area": round(float(soft_area), 6),
                "weighted_radius": round(float(weighted_radius), 6),
                "score_sum": round(float(score_sum), 6),
                "z_core_mean": round(z_core_mean, 3),
                "near_obstacle": bool(near_obs_ratio >= 0.25),
                "near_obstacle_ratio": round(float(near_obs_ratio), 3),
                "mask": mask,
                "priority": round(float(score_sum), 3),
            }
            out.append(event)
        return out

    def _classify_morphology(self, mask: np.ndarray) -> Tuple[str, Dict[str, float]]:
        pts = np.argwhere(mask)
        area = int(len(pts))
        if area == 0:
            return "blob", {}
        rmin, cmin = pts.min(axis=0)
        rmax, cmax = pts.max(axis=0)
        h = int(rmax - rmin + 1)
        w = int(cmax - cmin + 1)
        bbox_ratio = max(h, w) / max(1.0, min(h, w))
        if area <= 5:
            morph = "point_like"
        else:
            centered = pts.astype(np.float32) - pts.mean(axis=0, keepdims=True)
            if area >= 3:
                cov = np.cov(centered.T)
                vals = np.linalg.eigvalsh(cov)
                pca_ratio = float(np.sqrt((vals[-1] + 1e-6) / (vals[0] + 1e-6)))
            else:
                pca_ratio = 1.0
            geo_diam = float(max(h, w))
            geodesic_elongation = geo_diam / max(1.0, math.sqrt(area))
            thickness = area / max(1.0, geo_diam)
            dil_obs = binary_dilation(self.grid == 1, structure=np.ones((3, 3), dtype=bool))
            near_obs_ratio = float((mask & dil_obs).sum() / max(1, area))

            # 根据实际几何比例微调判断标准
            if (pca_ratio >= 2.6 or bbox_ratio >= 3.0 or (geodesic_elongation >= 2.6 and thickness <= 3.6) or (near_obs_ratio >= 0.35 and geodesic_elongation >= 2.3 and thickness <= 3.8)):
                morph = "elongated_strip"
            elif max(pca_ratio, bbox_ratio) >= 1.7:
                morph = "oval"
            elif area <= 11:
                morph = "point_like"
            elif area <= 28:
                morph = "compact_blob"
            else:
                morph = "blob"
            
            return morph, {
                "area": float(area), "bbox_ratio": round(float(bbox_ratio), 3),
                "pca_ratio": round(float(pca_ratio), 3), "geodesic_diameter": round(float(geo_diam), 3),
                "geodesic_elongation": round(float(geodesic_elongation), 3), "thickness_proxy": round(float(thickness), 3),
                "near_obstacle_ratio": round(float(near_obs_ratio), 3),
            }
        
        dil_obs = binary_dilation(self.grid == 1, structure=np.ones((3, 3), dtype=bool))
        near_obs_ratio = float((mask & dil_obs).sum() / max(1, area))
        return morph, {"area": float(area), "bbox_ratio": round(float(bbox_ratio), 3), "near_obstacle_ratio": round(float(near_obs_ratio), 3), "pca_ratio": 1.0, "geodesic_diameter": 0.0, "geodesic_elongation": 0.0, "thickness_proxy": float(area)}

    def _component_geodesic_diameter(self, mask: np.ndarray) -> float:
        pts = np.argwhere(mask)
        if len(pts) <= 1:
            return 0.0
        index = {tuple(p): i for i, p in enumerate(pts)}
        def farthest(start: Tuple[int, int]) -> Tuple[Tuple[int, int], int]:
            q = deque([(start, 0)])
            seen = {start}
            best = (start, 0)
            while q:
                (r, c), d = q.popleft()
                if d > best[1]:
                    best = ((r, c), d)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nb = (r+dr, c+dc)
                        if nb in index and nb not in seen:
                            seen.add(nb)
                            q.append((nb, d+1))
            return best
        a, _ = farthest(tuple(pts[0]))
        _, diam = farthest(a)
        return float(diam)

    def _assign_single_frame_event(self, event: Dict[str, Any], t: int) -> None:
        event["track_id"] = f"single_frame_{event['field_key'][:1]}_{event['polarity']}_{t}_{self.next_track_id:03d}"
        self.next_track_id += 1
        event["temporal_summary"] = {
            "duration_steps": 1,
            "area_start": int(event["area"]),
            "area_current": int(event["area"]),
            "area_trend": "stable",
            "intensity_start": round(float(event["z_core_mean"]), 3),
            "intensity_current": round(float(event["z_core_mean"]), 3),
            "intensity_trend": "stable",
        }
        event["sentence_zh"] = self._sentence(event)

    def _assign_track(self, event: Dict[str, Any], t: int) -> None:
        best_id = None
        best_score = -1.0
        for tid, tr in self.tracks.items():
            if tr.field_key != event["field_key"] or tr.polarity != event["polarity"]:
                continue
            dt = t - tr.last_seen
            if dt > self.max_track_gap:
                continue
            last = tr.history[-1]
            c_last = np.array(last["centroid"], dtype=float)
            c1 = np.array(event["centroid"], dtype=float)
            velocity = np.array(last.get("velocity", (0.0, 0.0)), dtype=float)
            c_pred = c_last + velocity * max(1, dt) if self.use_velocity_tracking else c_last
            dist_pred = float(np.linalg.norm(c_pred - c1))
            dist_last = float(np.linalg.norm(c_last - c1))
            area0 = max(1.0, float(last.get("soft_area", last.get("area", 1.0))))
            area1 = max(1.0, float(event.get("soft_area", event.get("area", 1.0))))
            area_ratio_penalty = abs(math.log(area1 / area0))
            morph_bonus = 0.15 if last.get("morphology") == event.get("morphology") else 0.0
            # Prediction improves fast-moving anomaly continuity; the fallback
            # distance keeps stationary cases robust when velocity is noisy.
            dist = min(dist_pred, dist_last + 0.75)
            score = 1.0 / (1.0 + dist) - 0.06 * area_ratio_penalty + morph_bonus
            if score > best_score:
                best_score = score
                best_id = tid

        if best_id is None or best_score < 0.10:
            best_id = f"{event['field_key'][:1].upper()}_{event['polarity']}_track_{self.next_track_id:03d}"
            self.next_track_id += 1
            self.tracks[best_id] = EventTrack(best_id, event["field_key"], event["polarity"], t, deque(maxlen=300))

        event["track_id"] = best_id
        self.tracks[best_id].update(t, event)

        history = self.tracks[best_id].history
        if self.use_morphology_debounce and len(history) > 3:
            morphs = [h["morphology"] for h in history]
            strip_cnt = morphs.count("elongated_strip")
            oval_cnt = morphs.count("oval")
            total = len(morphs)

            if strip_cnt > total * 0.3:
                refined_morph = "elongated_strip"
            elif oval_cnt > total * 0.3:
                refined_morph = "oval"
            else:
                max_soft_area = max(float(h.get("soft_area", h["area"])) for h in history)
                if max_soft_area <= 11:
                    refined_morph = "point_like"
                elif max_soft_area <= 28:
                    refined_morph = "compact_blob"
                else:
                    refined_morph = "blob"

            event["morphology"] = refined_morph
            event["morphology_zh"] = SHAPE_ZH[refined_morph]

        event["temporal_summary"] = self.tracks[best_id].temporal_summary()
        event["sentence_zh"] = self._sentence(event)

    def _drop_stale_tracks(self, t: int) -> None:
        old = [tid for tid, tr in self.tracks.items() if t - tr.last_seen > self.max_track_gap]
        for tid in old:
            self.tracks.pop(tid, None)

    def _sentence(self, event: Dict[str, Any]) -> str:
        tmp = event.get("temporal_summary", {})
        return (f"{event['track_id']}: {event['field_zh']}场在 {event['centroid']} 附近出现"
                f"{event['morphology_zh']}{'高值' if event['polarity']=='high' else '低值'}异常，"
                f"物理标签为{event['physical_tag']['label_zh']}；"
                f"面积{tmp.get('area_trend','stable')}，强度{tmp.get('intensity_trend','stable')}。")




__all__ = [
    "GRID_SIZE", "DEFAULT_OBS_RATIO", "CORE_VERSION", "FieldSemantic", "FIELD_REGISTRY", "FIELDS", "SHAPE_ZH",
    "make_obstacle_grid", "keep_largest_free_component", "random_free_cell", "nearest_free",
    "normal_backgrounds", "EffectSpec", "TestCase", "StressConfig", "build_current_fields",
    "EventTrack", "UniversalFieldToEventEncoder",
]
