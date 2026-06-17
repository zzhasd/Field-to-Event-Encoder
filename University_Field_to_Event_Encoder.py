"""Reusable core for the Universal Field-to-Event Encoder.

This module contains the no-leak field registry, simulator utilities used to
create mixed current fields, and the stateful UniversalFieldToEventEncoder.

Academic-integrity rule
-----------------------
The encoder receives ONLY:
    encoder.update(t, current_fields)
where current_fields are the mixed fields after background + injected anomaly.

The encoder does NOT receive clean background/baseline, confidence, injected
incident specs, ground-truth masks, event center/radius/type/trend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy.ndimage import label, binary_dilation, gaussian_filter

GRID_SIZE = 30
DEFAULT_OBS_RATIO = 0.20

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

    def mask_and_delta(self, grid: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return GT mask and physical delta. Used by simulator/evaluator only."""
        sem = FIELD_REGISTRY[self.field_key]
        mask = np.zeros(grid.shape, dtype=bool)
        delta = np.zeros(grid.shape, dtype=np.float32)
        if not self.is_active(t):
            return mask, delta
        rr, cc = np.indices(grid.shape)
        r0, c0 = self.center
        p = self.progress(t)
        rscale = self._radius_scale(p)
        amp = self.amplitude_sigma * sem.sigma * self._amp_scale(p)
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
        self.history.append({
            "t": t,
            "area": event["area"],
            "core_z": event["z_core_mean"],
            "centroid": event["centroid"],
            "morphology": event["morphology"],
        })

    def _trend(self, key: str, stable_tol_frac: float) -> str:
        """
        使用一元线性回归（最小二乘法）计算历史趋势，
        相比局部中位差，它对背景扰动和噪声有极强的鲁棒性。
        """
        t_vals = np.array([h["t"] for h in self.history], dtype=np.float32)
        vals = np.array([h[key] for h in self.history], dtype=np.float32)
        valid = np.isfinite(vals)
        t_vals = t_vals[valid]
        vals = vals[valid]
        n = len(vals)
        if n < 6:
            return "stable"

        # 手动计算最小二乘法斜率，避免 import 依赖和 RankWarning
        t_mean = np.mean(t_vals)
        v_mean = np.mean(vals)
        denom = np.sum((t_vals - t_mean)**2)
        if denom < 1e-5:
            return "stable"

        slope = np.sum((t_vals - t_mean) * (vals - v_mean)) / denom
        delta = float(slope * (t_vals[-1] - t_vals[0]))
        scale = max(1.0, float(np.mean(np.abs(vals))))

        if key == "area":
            # 面积通常有倍数级的变化，设立合理的物理下限阈值
            tol = max(2.0, 0.15 * scale)
            if delta > tol: return "expanding"
            if delta < -tol: return "shrinking"
            return "stable"
        else:
            # 强度(core_z)通常在 0.5~1 左右波动
            tol = max(0.35, 0.10 * scale)
            if delta > tol: return "strengthening"
            if delta < -tol: return "weakening"
            return "stable"

    def temporal_summary(self) -> Dict[str, Any]:
        first = self.history[0]
        last = self.history[-1]
        return {
            "duration_steps": int(last["t"] - first["t"] + 1),
            "area_start": int(first["area"]),
            "area_current": int(last["area"]),
            "area_trend": self._trend("area", 0.08),
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
    def __init__(self, grid: np.ndarray, registry: Dict[str, FieldSemantic] = FIELD_REGISTRY, rtca_alpha: float = 0.82, max_track_gap: int = 6):
        self.grid = grid.astype(np.uint8)
        self.free_mask = self.grid == 0
        self.registry = registry
        self.rtca_alpha = rtca_alpha
        self.max_track_gap = max_track_gap
        self.cum_scores: Dict[Tuple[str, str], np.ndarray] = {}
        self.tracks: Dict[str, EventTrack] = {}
        self.next_track_id = 1
        self.latest_residuals: Dict[str, np.ndarray] = {}
        self.latest_background_estimates: Dict[str, np.ndarray] = {}
        self.bg_models: Dict[str, np.ndarray] = {}

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

    def update(self, t: int, current_fields: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for field_key, x in current_fields.items():
            if field_key not in self.registry:
                continue
            sem = self.registry[field_key]
            x = np.asarray(x, dtype=np.float32)
            spatial_bg = self.estimate_background(x)
            if field_key not in self.bg_models:
                self.bg_models[field_key] = spatial_bg.copy()
            est_bg = self.bg_models[field_key]
            z = (x - est_bg) / (sem.sigma + 1e-6)
            z[~self.free_mask] = np.nan
            self.latest_background_estimates[field_key] = est_bg.copy()
            self.latest_residuals[field_key] = z
            field_event_union = np.zeros_like(self.free_mask, dtype=bool)
            field_abs_score = np.zeros_like(x, dtype=np.float32)
            for polarity in ("high", "low"):
                score = np.maximum(z, 0.0) if polarity == "high" else np.maximum(-z, 0.0)
                score[~np.isfinite(score)] = 0.0
                key = (field_key, polarity)
                if key not in self.cum_scores:
                    self.cum_scores[key] = np.zeros_like(score, dtype=np.float32)
                cum = self.cum_scores[key]
                cum[:] = self.rtca_alpha*cum + (1-self.rtca_alpha)*score
                core_mask = self.free_mask & (score >= sem.z_threshold) & (cum >= sem.cum_threshold)
                support_mask = self.free_mask & (score >= sem.boundary_z_threshold) & (cum >= sem.cum_threshold*0.35)
                event_mask = self._hysteresis_mask(core_mask, support_mask)
                field_event_union |= event_mask
                field_abs_score = np.maximum(field_abs_score, score.astype(np.float32))
                events.extend(self._extract_components(field_key, polarity, sem, z, score, core_mask, event_mask, t))
            valid = self.free_mask & np.isfinite(x)
            normal_mask = valid & (~field_event_union) & (field_abs_score < 0.80)
            bg = self.bg_models[field_key].copy()
            bg[normal_mask] = 0.985 * bg[normal_mask] + 0.015 * x[normal_mask]
            uncertain_mask = valid & (~normal_mask)
            bg[uncertain_mask] = 0.997 * bg[uncertain_mask] + 0.003 * spatial_bg[uncertain_mask]
            bg[~self.free_mask] = np.nan
            self.bg_models[field_key] = bg.astype(np.float32)
        for ev in events:
            self._assign_track(ev, t)
        self._drop_stale_tracks(t)
        events.sort(key=lambda e: e["priority"], reverse=True)
        return events

    def _hysteresis_mask(self, core: np.ndarray, support: np.ndarray) -> np.ndarray:
        lab, n = label(support, structure=np.ones((3, 3), dtype=np.uint8))
        out = np.zeros_like(core, dtype=bool)
        for idx in range(1, n+1):
            comp = lab == idx
            if (comp & core).any():
                out |= comp
        return out

    def _extract_components(self, field_key: str, polarity: str, sem: FieldSemantic, z: np.ndarray, score: np.ndarray, core_mask: np.ndarray, event_mask: np.ndarray, t: int) -> List[Dict[str, Any]]:
        lab, n = label(event_mask, structure=np.ones((3, 3), dtype=np.uint8))
        rr, cc = np.indices(event_mask.shape)
        out: List[Dict[str, Any]] = []
        for idx in range(1, n+1):
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
                "z_core_mean": round(z_core_mean, 3),
                "near_obstacle": bool(near_obs_ratio >= 0.25),
                "near_obstacle_ratio": round(float(near_obs_ratio), 3),
                "mask": mask,
                "priority": round(float(np.nansum(score[mask])), 3),
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

    def _assign_track(self, event: Dict[str, Any], t: int) -> None:
        best_id = None
        best_score = -1.0
        for tid, tr in self.tracks.items():
            if tr.field_key != event["field_key"] or tr.polarity != event["polarity"]:
                continue
            if t - tr.last_seen > self.max_track_gap:
                continue
            last = tr.history[-1]
            c0 = np.array(last["centroid"], dtype=float)
            c1 = np.array(event["centroid"], dtype=float)
            dist = float(np.linalg.norm(c0-c1))
            score = 1.0 / (1.0 + dist)
            if score > best_score:
                best_score = score
                best_id = tid
                
        if best_id is None or best_score < 0.12:
            best_id = f"{event['field_key'][:1].upper()}_{event['polarity']}_track_{self.next_track_id:03d}"
            self.next_track_id += 1
            # 将 maxlen 从 50 扩大到 300，保留完整事件追踪记录
            self.tracks[best_id] = EventTrack(best_id, event["field_key"], event["polarity"], t, deque(maxlen=300))
            
        event["track_id"] = best_id
        self.tracks[best_id].update(t, event)

        # -----------------------------------------------------------------
        # 轨迹形态防抖：修复“正在收缩的团块”与“稳定的小斑点”在单帧下的判定重合
        # -----------------------------------------------------------------
        history = self.tracks[best_id].history
        if len(history) > 3:
            morphs = [h["morphology"] for h in history]
            strip_cnt = morphs.count("elongated_strip")
            oval_cnt = morphs.count("oval")
            total = len(morphs)

            # 对具有明显几何拉伸的形状进行多数投票防抖
            if strip_cnt > total * 0.3:
                refined_morph = "elongated_strip"
            elif oval_cnt > total * 0.3:
                refined_morph = "oval"
            else:
                # 对圆/方正状的拓扑形态，采用生命周期最大面积作为区分不变量
                # 无论它是膨胀、收缩还是稳定，最大面积始终反映了它的基准物理尺度
                max_area = max(h["area"] for h in history)
                if max_area <= 11:
                    refined_morph = "point_like"
                elif max_area <= 28:
                    refined_morph = "compact_blob"
                else:
                    refined_morph = "blob"

            event["morphology"] = refined_morph
            event["morphology_zh"] = SHAPE_ZH[refined_morph]
        # -----------------------------------------------------------------

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
    "GRID_SIZE",
    "DEFAULT_OBS_RATIO",
    "FieldSemantic",
    "FIELD_REGISTRY",
    "FIELDS",
    "SHAPE_ZH",
    "make_obstacle_grid",
    "keep_largest_free_component",
    "random_free_cell",
    "nearest_free",
    "normal_backgrounds",
    "EffectSpec",
    "EventTrack",
    "UniversalFieldToEventEncoder",
]
