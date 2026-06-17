"""Six dynamic visualization demos for the Universal Field-to-Event Encoder.

Run examples:
    python field_event_visual_demos.py --demo fire
    python field_event_visual_demos.py --demo electrical
    python field_event_visual_demos.py --demo leak
    python field_event_visual_demos.py --demo steam
    python field_event_visual_demos.py --demo co2
    python field_event_visual_demos.py --demo dust

The encoder receives only encoder.update(t, current_fields), where current_fields
are the already mixed physical fields: normal background + injected anomaly.
Clean backgrounds, anomaly masks, ground-truth labels, event center/radius/type,
and incident specs are never passed into the encoder.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -----------------------------------------------------------------------------
# Load the shared no-leak encoder core module.
# -----------------------------------------------------------------------------

def _load_encoder_module():
    """Load University_Field_to_Event_Encoder.py next to this demo.

    ENCODER_PATH can override the path for experiments. A legacy compatibility
    lookup is kept so old notebooks can still run, but the intended source of
    the core algorithm is University_Field_to_Event_Encoder.py.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / "University_Field_to_Event_Encoder.py",
        here / "University _Field_to_Event_Encoder.py",
    ]
    env_path = os.environ.get("ENCODER_PATH")
    if env_path:
        candidates.insert(0, Path(env_path))
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("University_Field_to_Event_Encoder", str(p))
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules["University_Field_to_Event_Encoder"] = mod
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return mod
    raise FileNotFoundError(
        "Cannot find University_Field_to_Event_Encoder.py. "
        "Put it next to this demo or set ENCODER_PATH."
    )

ue = _load_encoder_module()

GRID_SIZE = ue.GRID_SIZE
OBS_RATIO = ue.DEFAULT_OBS_RATIO
STEPS = 200
FIELDS: Tuple[str, ...] = tuple(ue.FIELDS)
FIELD_REGISTRY = ue.FIELD_REGISTRY

# Fixed ranges make per-frame rendering cheap and visually comparable.
FIELD_CLIMS: Dict[str, Tuple[float, float]] = {
    "temperature": (16.0, 38.0),
    "humidity": (22.0, 92.0),
    "pressure": (0.0, 12.5),
    "co2": (380.0, 850.0),
    "air_quality": (25.0, 125.0),
}

# -----------------------------------------------------------------------------
# Chinese font handling. Matplotlib otherwise often renders Chinese as squares.
# -----------------------------------------------------------------------------

def setup_chinese_font(verbose: bool = False) -> Optional[str]:
    """Pick a CJK font when available and install it into Matplotlib rcParams."""
    common_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for p in common_paths:
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            name = font_manager.FontProperties(fname=p).get_name()
            mpl.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            mpl.rcParams["font.family"] = "sans-serif"
            mpl.rcParams["axes.unicode_minus"] = False
            if verbose:
                print(f"Using CJK font: {name} ({p})")
            return name

    # Fallback: scan installed fonts by name.
    preferred = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
        "Microsoft YaHei", "SimHei", "PingFang SC", "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei", "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            mpl.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            mpl.rcParams["font.family"] = "sans-serif"
            mpl.rcParams["axes.unicode_minus"] = False
            if verbose:
                print(f"Using CJK font: {name}")
            return name
    mpl.rcParams["axes.unicode_minus"] = False
    if verbose:
        print("WARNING: no CJK font found; Chinese may render as boxes.")
    return None

# -----------------------------------------------------------------------------
# Scenario definitions.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    key: str
    name_zh: str
    short_zh: str
    description: str
    effects: Tuple[Any, ...]  # ue.EffectSpec instances


def E(
    effect_id: str,
    field_key: str,
    polarity: str,
    shape: str,
    area_trend: str,
    intensity_trend: str,
    center: Tuple[float, float],
    start: int,
    end: int,
    amplitude_sigma: float,
    radius: float,
    angle: float = 0.0,
    axis_ratio: float = 2.2,
):
    return ue.EffectSpec(
        effect_id=effect_id,
        field_key=field_key,
        polarity=polarity,
        shape=shape,
        area_trend=area_trend,
        intensity_trend=intensity_trend,
        center=center,
        start=start,
        end=end,
        amplitude_sigma=amplitude_sigma,
        radius=radius,
        angle=angle,
        axis_ratio=axis_ratio,
    )


def make_scenarios() -> Dict[str, Scenario]:
    c = (15.0, 15.0)
    return {
        "fire": Scenario(
            key="fire",
            name_zh="着火 / 明火初期",
            short_zh="火灾初期",
            description="温度明显升高并扩散，湿度局部下降，通风压差扰动，CO2升高，空气质量恶化。",
            effects=(
                E("fire_T", "temperature", "high", "blob", "expanding", "strengthening", c, 18, 180, 4.6, 3.4, 0.2, 2.0),
                E("fire_H", "humidity", "low", "blob", "expanding", "strengthening", (15.0, 15.5), 22, 180, 2.9, 3.2, 0.0, 2.0),
                E("fire_P", "pressure", "high", "elongated_strip", "stable", "stable", (15.0, 15.0), 28, 165, 2.4, 2.4, 0.95, 2.0),
                E("fire_CO2", "co2", "high", "blob", "expanding", "strengthening", (15.5, 14.5), 30, 185, 3.8, 4.2, 0.0, 2.0),
                E("fire_AQ", "air_quality", "high", "blob", "expanding", "strengthening", (15.5, 14.0), 30, 185, 4.4, 4.4, 0.0, 2.0),
            ),
        ),
        "electrical": Scenario(
            key="electrical",
            name_zh="电气柜过热",
            short_zh="电气过热",
            description="温度在局部小范围升高，其他物理场基本不变，空气质量最多轻微恶化。",
            effects=(
                E("elec_T", "temperature", "high", "compact_blob", "stable", "strengthening", (9.0, 21.0), 18, 175, 4.8, 2.0, 0.0, 2.0),
                E("elec_AQ", "air_quality", "high", "point_like", "stable", "stable", (9.0, 21.0), 60, 165, 1.9, 1.2, 0.0, 2.0),
            ),
        ),
        "leak": Scenario(
            key="leak",
            name_zh="漏水",
            short_zh="漏水",
            description="湿度沿地面扩散并升高，温度基本不变或局部下降，若靠近管路可伴随压降。",
            effects=(
                E("leak_H", "humidity", "high", "elongated_strip", "expanding", "strengthening", (21.0, 9.0), 20, 185, 4.2, 2.5, 0.25, 2.4),
                E("leak_T", "temperature", "low", "elongated_strip", "expanding", "stable", (21.0, 9.0), 30, 180, 2.0, 2.1, 0.25, 2.4),
                E("leak_P", "pressure", "low", "compact_blob", "stable", "stable", (21.0, 9.0), 45, 170, 2.0, 1.8, 0.0, 2.0),
            ),
        ),
        "steam": Scenario(
            key="steam",
            name_zh="蒸汽泄漏",
            short_zh="蒸汽泄漏",
            description="温湿同步升高，局部压差异常，CO2基本不变，空气质量可能轻微变差。",
            effects=(
                E("steam_T", "temperature", "high", "oval", "expanding", "strengthening", (10.0, 10.0), 20, 180, 3.5, 2.5, 0.75, 2.1),
                E("steam_H", "humidity", "high", "blob", "expanding", "strengthening", (10.0, 10.0), 20, 180, 4.6, 3.4, 0.0, 2.0),
                E("steam_P", "pressure", "high", "compact_blob", "stable", "strengthening", (10.5, 10.2), 25, 160, 3.0, 2.0, 0.0, 2.0),
                E("steam_AQ", "air_quality", "high", "compact_blob", "stable", "stable", (10.0, 10.0), 55, 165, 1.8, 2.0, 0.0, 2.0),
            ),
        ),
        "co2": Scenario(
            key="co2",
            name_zh="CO2 积聚 / 通风不良",
            short_zh="CO2积聚",
            description="CO2缓慢升高并扩散，压差/通风异常，空气质量变差，温湿可能轻微变化。",
            effects=(
                E("co2_C", "co2", "high", "blob", "expanding", "strengthening", (18.0, 19.0), 10, 195, 4.2, 5.0, 0.0, 2.0),
                E("co2_P", "pressure", "low", "elongated_strip", "stable", "stable", (17.5, 18.5), 15, 195, 2.6, 2.8, 1.55, 2.6),
                E("co2_AQ", "air_quality", "high", "blob", "expanding", "strengthening", (18.0, 19.0), 30, 195, 3.1, 4.8, 0.0, 2.0),
                E("co2_T", "temperature", "high", "blob", "expanding", "stable", (18.0, 19.0), 60, 190, 1.7, 4.2, 0.0, 2.0),
                E("co2_H", "humidity", "high", "blob", "stable", "stable", (18.0, 19.0), 65, 180, 1.5, 3.6, 0.0, 2.0),
            ),
        ),
        "dust": Scenario(
            key="dust",
            name_zh="空气污染 / 粉尘泄漏",
            short_zh="粉尘泄漏",
            description="空气质量为主导异常，沿风向/通道呈条带状恶化，其他场不一定变化。",
            effects=(
                E("dust_AQ", "air_quality", "high", "elongated_strip", "expanding", "strengthening", (13.0, 8.0), 18, 188, 5.0, 2.8, 0.15, 2.8),
                E("dust_P", "pressure", "high", "elongated_strip", "stable", "stable", (13.0, 8.0), 20, 170, 2.0, 2.4, 0.15, 2.8),
                E("dust_H", "humidity", "low", "compact_blob", "stable", "stable", (13.0, 8.0), 65, 150, 1.3, 2.0, 0.0, 2.0),
            ),
        ),
    }

# -----------------------------------------------------------------------------
# Simulation: build mixed current fields only.
# -----------------------------------------------------------------------------

def _clone_effect(eff: Any, center: Optional[Tuple[float, float]] = None) -> Any:
    # EffectSpec is a dataclass in the user module, so replace() works.
    return replace(eff, center=eff.center if center is None else center)


def prepare_grid_and_effects(scenario: Scenario, seed: int, obs_ratio: float = OBS_RATIO) -> Tuple[np.ndarray, List[Any]]:
    """Generate a 30x30 obstacle map and move effect centers to free cells."""
    grid = ue.make_obstacle_grid(size=GRID_SIZE, obs_ratio=obs_ratio, seed=seed)
    effects: List[Any] = []
    rng = np.random.default_rng(seed + 9973)
    for eff in scenario.effects:
        # Keep scenario layout stable, but never inject inside obstacles.
        free_center = tuple(map(float, ue.nearest_free(grid, eff.center)))
        # If a shape would be badly clipped by obstacles, a tiny deterministic jitter
        # moves only the simulator injection center. This center is never passed to the encoder.
        if rng.random() < 0.10:
            free_center = tuple(map(float, ue.random_free_cell(grid, rng, margin=4)))
        effects.append(_clone_effect(eff, center=free_center))
    return grid, effects


def mixed_fields_for_step(grid: np.ndarray, effects: Sequence[Any], t: int) -> Dict[str, np.ndarray]:
    """Return mixed fields = normal background + injected changes.

    This function is simulator-only. The encoder receives only its returned dict.
    """
    current: Dict[str, np.ndarray] = ue.normal_backgrounds(grid, FIELDS, t)
    current = {k: v.copy() for k, v in current.items()}
    for eff in effects:
        if eff.field_key not in current:
            continue
        _mask, delta = eff.mask_and_delta(grid, t)
        current[eff.field_key] = current[eff.field_key] + delta
        current[eff.field_key][grid == 1] = np.nan
    return current


# -----------------------------------------------------------------------------
# LLM high-level judgment from encoder event list only.
# -----------------------------------------------------------------------------

CASE_LIBRARY_ZH = """
参考案例库：六类常见异常事件与五个物理场的典型变化
1. 着火 / 明火初期：温度明显升高并扩散；湿度通常下降或局部波动；气压/压差可能出现通风扰动；CO2 升高；空气质量明显恶化。典型解释：高危复合异常。
2. 电气柜过热：温度局部升高；湿度、气压/压差、CO2 基本不变；空气质量轻微变化或不变。典型解释：更像设备热故障，不一定是明火。
3. 漏水：温度基本不变或局部下降；湿度明显升高并沿地面扩散；若靠近管路可能出现压降；CO2 基本不变；空气质量基本不变。典型解释：湿度主导异常。
4. 蒸汽泄漏：温度升高；湿度明显升高；局部气压/压差异常；CO2 基本不变；空气质量可能轻微变差。典型解释：温湿同步升高。
5. CO2 积聚 / 通风不良：温度缓慢升高或基本不变；湿度轻微变化；通风/压差异常；CO2 明显升高；空气质量变差。典型解释：多为大范围慢变化。
6. 空气污染 / 粉尘泄漏：温度、湿度、气压/压差、CO2 不一定变化；空气质量明显恶化，常受风向或通道影响呈局部或条带状扩散。典型解释：AQ 主导异常。
""".strip()

LLM_SYSTEM_PROMPT_ZH = f"""
你是工业空间多物理场异常事件判读助手。
你的输入来自 field_event 编码器在当前时间步输出的事件列表。编码器只接收混合后的五个物理场，不知道仿真真值、异常类型、异常中心或 mask。

你需要根据事件列表和下面的参考案例库推理当前可能发生的异常事件。你可以从案例库中选择最匹配的一类，也可以在证据不足、混合异常或不符合案例时判断为“复合异常”“其他异常事件”或“暂未形成稳定异常”。

{CASE_LIBRARY_ZH}

判读要求：
- 主要依据编码器事件列表中的字段：物理场名称、异常方向 high/low、面积、形态、位置、强度 z_core_mean、面积趋势、强度趋势。
- 不要把参考案例当作唯一答案；参考案例只是先验知识。
- 不要输出长篇逐步思考，只输出简短依据。
- 必须输出合法 JSON object，且不要输出 Markdown、代码块或额外说明。
- JSON 字段必须严格为：
{{
  "推断的异常事件": "...",
  "简短的推断理由或者过程": "..."
}}
""".strip()


def _event_to_llm_record(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, JSON-safe event record. Never includes masks or injected truth."""
    tmp = ev.get("temporal_summary", {}) or {}
    return {
        "track_id": ev.get("track_id"),
        "物理场": ev.get("field_zh", ev.get("field_key")),
        "field_key": ev.get("field_key"),
        "异常方向": "高值" if ev.get("polarity") == "high" else "低值",
        "polarity": ev.get("polarity"),
        "物理标签": (ev.get("physical_tag") or {}).get("label_zh"),
        "形态": ev.get("morphology_zh", ev.get("morphology")),
        "morphology": ev.get("morphology"),
        "中心": ev.get("centroid"),
        "面积": ev.get("area"),
        "核心强度_z": ev.get("z_core_mean"),
        "近障碍物": ev.get("near_obstacle"),
        "面积趋势": tmp.get("area_trend"),
        "强度趋势": tmp.get("intensity_trend"),
        "持续步数": tmp.get("duration_steps"),
    }


def sanitize_events_for_llm(events: Sequence[Dict[str, Any]], max_events: int = 14) -> List[Dict[str, Any]]:
    ordered = sorted(events, key=lambda e: float(e.get("priority", 0.0)), reverse=True)
    return [_event_to_llm_record(ev) for ev in ordered[:max_events]]


def build_llm_user_prompt(t: int, scenario_hint_for_ui_only: Optional[str], events: Sequence[Dict[str, Any]]) -> str:
    """Build Chinese prompt. The scenario hint is only put in metadata when explicitly used for offline evaluation; normal demo passes None."""
    records = sanitize_events_for_llm(events)
    payload = {
        "当前时间步": int(t),
        "说明": "以下是 field_event 编码器当前输出的事件列表。请根据事件列表和案例库推断异常事件，并返回 JSON。",
        "事件列表": records,
    }
    if scenario_hint_for_ui_only:
        # This is only for optional offline debugging; normal live judgment never sends GT labels.
        payload["调试备注_不要作为判读依据"] = scenario_hint_for_ui_only
    return "请阅读下面的编码器事件列表，结合参考案例库，输出指定 JSON：\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def parse_llm_json(text: str) -> Dict[str, str]:
    """Parse model output robustly, accepting a bare JSON object or JSON inside text."""
    if not text:
        raise ValueError("empty LLM response")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM response is not a JSON object")
    event_type = str(data.get("推断的异常事件", "未知异常事件")).strip()
    reason = str(data.get("简短的推断理由或者过程", "未给出理由")).strip()
    return {"推断的异常事件": event_type, "简短的推断理由或者过程": reason}


def call_qwen_judge_sync(
    t: int,
    events: Sequence[Dict[str, Any]],
    api_key: str,
    model: str = "qwen3.6-flash",
    endpoint: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    timeout_s: float = 18.0,
    enable_thinking: bool = False,
) -> Dict[str, str]:
    """Call DashScope/OpenAI-compatible Qwen chat completions and return parsed JSON."""
    if not api_key:
        raise ValueError("missing API key; set DASHSCOPE_API_KEY or QWEN_API_KEY")
    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT_ZH},
        {"role": "user", "content": build_llm_user_prompt(t, None, events)},
    ]
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        # DashScope/Qwen JSON mode should be used with non-thinking mode unless the selected model requires otherwise.
        "enable_thinking": bool(enable_thinking),
    }
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=raw,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp_text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen HTTP {e.code}: {detail[:800]}") from e
    data = json.loads(resp_text)
    content = data["choices"][0]["message"].get("content", "")
    return parse_llm_json(content)


# -----------------------------------------------------------------------------
# Compact right-bottom text panel helpers.
# -----------------------------------------------------------------------------

# Keep the demo window smaller and keep all text inside the right-bottom panel.
DEMO_FIGSIZE: Tuple[float, float] = (11.2, 6.3)
PANEL_WRAP_WIDTH = 68          # visual width; CJK chars count as 2 columns
PANEL_MAX_LINES = 24           # hard cap; prevents the GUI from growing/jumping
LLM_REASON_MAX_LINES = 4       # long model reasons are summarized in a fixed viewport
EVENT_PANEL_LINES = 6          # visible encoder-event rows in the scrolling viewport
EVENT_SCROLL_EVERY_STEPS = 8   # lower-right event section scroll speed


def _char_display_width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in {"F", "W"} else 1


def _display_width(text: str) -> int:
    return sum(_char_display_width(ch) for ch in text)


def _trim_to_display_width(text: str, width: int) -> str:
    if width <= 0:
        return ""
    out: List[str] = []
    used = 0
    for ch in text:
        w = _char_display_width(ch)
        if used + w > width:
            break
        out.append(ch)
        used += w
    return "".join(out).rstrip()


def wrap_for_panel(
    text: str,
    width: int = PANEL_WRAP_WIDTH,
    initial_indent: str = "",
    subsequent_indent: str = "",
) -> List[str]:
    """Wrap Chinese/English text by visual width so CJK text cannot stretch the window."""
    wrapped: List[str] = []
    paragraphs = str(text).splitlines() or [""]
    for para in paragraphs:
        if para == "":
            wrapped.append("")
            continue
        cur = initial_indent
        cur_w = _display_width(cur)
        sub_w = _display_width(subsequent_indent)
        for ch in para:
            ch_w = _char_display_width(ch)
            # Break aggressively; this is deliberate for long model output without spaces.
            if cur.strip() and cur_w + ch_w > width:
                wrapped.append(cur.rstrip())
                cur = subsequent_indent + ch
                cur_w = sub_w + ch_w
            else:
                cur += ch
                cur_w += ch_w
        if cur or not wrapped:
            wrapped.append(cur.rstrip())
    return wrapped


def _limit_lines(lines: Sequence[str], max_lines: int, width: int = PANEL_WRAP_WIDTH) -> List[str]:
    if len(lines) <= max_lines:
        return list(lines)
    kept = list(lines[:max_lines])
    kept[-1] = _trim_to_display_width(kept[-1], max(1, width - 2)) + "…"
    return kept


def _wrap_and_limit_text(text: str, max_lines: int, width: int = PANEL_WRAP_WIDTH) -> str:
    return "\n".join(_limit_lines(wrap_for_panel(text, width=width), max_lines=max_lines, width=width))


@dataclass
class LLMJudgeDisplayState:
    status: str = "waiting"      # waiting / pending / done / error / disabled
    last_request_step: Optional[int] = None
    last_result_step: Optional[int] = None
    inferred_event: str = "等待第 50 步触发大模型判读"
    reason: str = ""
    error: str = ""
    raw_events_count: int = 0

    def text(self) -> str:
        if self.status == "disabled":
            return "高层判读：未配置大模型 API Key，暂不调用大模型。"
        if self.status == "pending":
            return f"高层判读：大模型判读中（第 {self.last_request_step} 步事件列表已发送）..."
        if self.status == "done":
            event_text = _wrap_and_limit_text(f"高层判读：{self.inferred_event}", max_lines=2)
            reason = self.reason or "未给出理由"
            reason_text = _wrap_and_limit_text(
                f"推断依据：{reason}",
                max_lines=LLM_REASON_MAX_LINES,
            )
            source_text = f"判读来源：Qwen，第 {self.last_result_step} 步事件列表。"
            return f"{event_text}\n{reason_text}\n{source_text}"
        if self.status == "error":
            err = _wrap_and_limit_text(f"高层判读：大模型调用失败，等待下一次触发。错误：{self.error}", max_lines=3)
            return err
        return "高层判读：等待第 50 步触发大模型判读。"


class AsyncQwenJudge:
    """Non-blocking Qwen judgment so animation rendering does not freeze."""
    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        endpoint: str,
        interval_steps: int = 50,
        timeout_s: float = 18.0,
        enable_thinking: bool = False,
    ) -> None:
        self.api_key = api_key or ""
        self.model = model
        self.endpoint = endpoint
        self.interval_steps = max(1, int(interval_steps))
        self.timeout_s = timeout_s
        self.enable_thinking = enable_thinking
        self.state = LLMJudgeDisplayState(status="waiting" if self.api_key else "disabled")
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future: Optional[concurrent.futures.Future] = None
        self.future_step: Optional[int] = None

    def maybe_submit(self, step_1based: int, events: Sequence[Dict[str, Any]]) -> None:
        self.poll()
        if not self.api_key:
            self.state.status = "disabled"
            return
        if step_1based % self.interval_steps != 0:
            return
        if self.future is not None and not self.future.done():
            return
        # Submit a compact copy; never pass masks or simulator specs.
        compact_events = sanitize_events_for_llm(events)
        self.state.status = "pending"
        self.state.last_request_step = step_1based
        self.state.raw_events_count = len(compact_events)
        self.future_step = step_1based
        self.future = self.executor.submit(
            call_qwen_judge_sync,
            step_1based,
            compact_events,
            self.api_key,
            self.model,
            self.endpoint,
            self.timeout_s,
            self.enable_thinking,
        )

    def poll(self) -> None:
        if self.future is None or not self.future.done():
            return
        try:
            result = self.future.result()
            self.state.status = "done"
            self.state.last_result_step = self.future_step
            self.state.inferred_event = result.get("推断的异常事件", "未知异常事件")
            self.state.reason = result.get("简短的推断理由或者过程", "未给出理由")
            self.state.error = ""
        except Exception as e:
            self.state.status = "error"
            self.state.error = str(e)
        finally:
            self.future = None
            self.future_step = None

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


def _event_sentence_for_ui(ev: Dict[str, Any]) -> str:
    sent = ev.get("sentence_zh")
    if sent:
        return str(sent)
    return (
        f"{ev.get('field_zh', ev.get('field_key'))} {ev.get('polarity')} "
        f"{ev.get('morphology_zh', ev.get('morphology'))} area={ev.get('area')}"
    )


def summarize_events_for_ui(
    events: Sequence[Dict[str, Any]],
    llm_state: LLMJudgeDisplayState,
    max_lines: int = EVENT_PANEL_LINES,
    scroll_step: int = 0,
) -> str:
    """Build a fixed-size text panel; encoder events auto-scroll when too many."""
    lines: List[str] = []
    lines.extend(wrap_for_panel(llm_state.text()))
    lines.append("")

    event_count = len(events)
    if not events:
        lines.append("编码器原始事件摘要：")
        lines.extend(wrap_for_panel("暂无稳定场事件。", initial_indent="• ", subsequent_indent="  "))
        return "\n".join(_limit_lines(lines, PANEL_MAX_LINES))

    event_texts = [_event_sentence_for_ui(ev) for ev in events]
    if event_count > max_lines:
        offset = (max(0, scroll_step) // EVENT_SCROLL_EVERY_STEPS) % event_count
        event_texts = event_texts[offset:] + event_texts[:offset]
        title = f"编码器原始事件摘要（{event_count} 条，自动滚动）："
    else:
        title = f"编码器原始事件摘要（{event_count} 条）："
    lines.append(title)

    for sent in event_texts[:max_lines]:
        lines.extend(wrap_for_panel(sent, initial_indent="• ", subsequent_indent="  "))

    if event_count > max_lines:
        lines.append(f"… 共 {event_count} 条；每 {EVENT_SCROLL_EVERY_STEPS} 步滚动显示下一批。")

    return "\n".join(_limit_lines(lines, PANEL_MAX_LINES))

# -----------------------------------------------------------------------------
# Rendering.
# -----------------------------------------------------------------------------

def obstacle_overlay(grid: np.ndarray) -> np.ndarray:
    rgba = np.zeros((grid.shape[0], grid.shape[1], 4), dtype=np.float32)
    rgba[grid == 1] = (0.0, 0.0, 0.0, 0.70)
    return rgba


def masked_for_render(arr: np.ndarray, grid: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where((grid == 1) | ~np.isfinite(arr), arr)


def field_title(field_key: str) -> str:
    sem = FIELD_REGISTRY[field_key]
    return f"{sem.zh_name} ({sem.unit})"


def make_figure(grid: np.ndarray, scenario: Scenario) -> Tuple[Any, Dict[str, Any], Any]:
    fig, axes = plt.subplots(2, 3, figsize=DEMO_FIGSIZE, constrained_layout=True)
    fig.canvas.manager.set_window_title(f"Field Event Demo - {scenario.name_zh}") if hasattr(fig.canvas, "manager") else None
    axes_flat = axes.ravel()
    obs = obstacle_overlay(grid)
    artists: Dict[str, Any] = {}

    # Blue-white-red: low values are blue and high values are red.
    cmap = mpl.colormaps.get_cmap("coolwarm").copy()
    cmap.set_bad((0.88, 0.88, 0.88, 1.0))

    for ax, field in zip(axes_flat[:5], FIELDS):
        dummy = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=np.float32)
        im = ax.imshow(masked_for_render(dummy, grid), interpolation="nearest", animated=False, cmap=cmap)
        im.set_clim(*FIELD_CLIMS[field])
        ax.imshow(obs, interpolation="nearest", animated=False)
        ax.set_title(field_title(field), fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="3%", pad=0.025)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label("低值 ←  → 高值", fontsize=6.5)
        artists[field] = im

    text_ax = axes_flat[5]
    text_ax.axis("off")
    text_ax.set_xlim(0.0, 1.0)
    text_ax.set_ylim(0.0, 1.0)
    text_obj = text_ax.text(
        0.02,
        0.98,
        "",
        ha="left",
        va="top",
        fontsize=7.4,
        linespacing=1.12,
        wrap=False,
        clip_on=True,
        transform=text_ax.transAxes,
    )
    # Do not let changing text participate in constrained_layout; otherwise long
    # LLM reasons can make the window/axes jump between frames.
    text_obj.set_in_layout(False)
    text_obj.set_clip_path(text_ax.patch)
    fig.suptitle(f"{scenario.name_zh}：五物理场动态注入 + 编码器读取 + Qwen高层判读", fontsize=12)
    return fig, artists, text_obj


def run_demo(
    demo_key: str,
    seed: int = 2026,
    steps: int = STEPS,
    obs_ratio: float = OBS_RATIO,
    interval_ms: int = 80,
    save_gif: Optional[str] = None,
    no_show: bool = False,
    llm_api_key: Optional[str] = None,
    llm_model: str = "qwen3.6-flash",
    llm_endpoint: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    llm_interval_steps: int = 50,
    llm_timeout_s: float = 18.0,
    llm_enable_thinking: bool = False,
    disable_llm: bool = False,
) -> None:
    setup_chinese_font(verbose=True)
    scenarios = make_scenarios()
    if demo_key not in scenarios:
        raise ValueError(f"Unknown demo: {demo_key}. Choose from {', '.join(scenarios)}")
    scenario = scenarios[demo_key]
    grid, effects = prepare_grid_and_effects(scenario, seed=seed, obs_ratio=obs_ratio)
    encoder = ue.UniversalFieldToEventEncoder(grid, FIELD_REGISTRY)

    if disable_llm:
        api_key = ""
    else:
        api_key = llm_api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY") or ""
    llm_judge = AsyncQwenJudge(
        api_key=api_key,
        model=llm_model,
        endpoint=llm_endpoint,
        interval_steps=llm_interval_steps,
        timeout_s=llm_timeout_s,
        enable_thinking=llm_enable_thinking,
    )

    fig, field_artists, text_obj = make_figure(grid, scenario)

    def update_frame(t: int):
        # IMPORTANT: current is already mixed background + anomaly. The encoder
        # gets no clean background, no anomaly channel, and no event spec.
        current = mixed_fields_for_step(grid, effects, t)
        events = encoder.update(t, current)

        step_1based = t + 1
        llm_judge.maybe_submit(step_1based, events)
        llm_judge.poll()

        changed: List[Any] = []
        for field, im in field_artists.items():
            im.set_data(masked_for_render(current[field], grid))
            changed.append(im)

        header_lines: List[str] = []
        for item in (
            f"异常注入标签：{scenario.short_zh}",
            f"时间步：{step_1based:03d}/{steps} | 网格：30×30 | 障碍密度：{obs_ratio:.2f}",
            f"注入模式：{scenario.description}",
            f"大模型：每 {llm_interval_steps} 步发送事件列表给 {llm_model}。",
        ):
            header_lines.extend(wrap_for_panel(item))
        body = summarize_events_for_ui(events, llm_judge.state, scroll_step=step_1based)
        panel_lines = header_lines + [""] + body.splitlines()
        text_obj.set_text("\n".join(_limit_lines(panel_lines, PANEL_MAX_LINES)))
        changed.append(text_obj)
        return changed

    anim = FuncAnimation(fig, update_frame, frames=steps, interval=interval_ms, blit=False, repeat=True)
    # Keep references alive for interactive backends.
    fig._field_event_anim = anim  # type: ignore[attr-defined]
    fig._qwen_judge = llm_judge  # type: ignore[attr-defined]

    try:
        if save_gif:
            anim.save(save_gif, writer=PillowWriter(fps=max(1, int(1000 / interval_ms))))
            print(f"Saved GIF: {save_gif}")
        if not no_show:
            plt.show()
        else:
            plt.close(fig)
    finally:
        llm_judge.shutdown()


def headless_check(seed: int = 2026, steps: int = 20) -> None:
    """Fast non-GUI sanity check for CI/sandbox use. Does not call the LLM."""
    scenarios = make_scenarios()
    for i, (key, scenario) in enumerate(scenarios.items()):
        grid, effects = prepare_grid_and_effects(scenario, seed=seed + i, obs_ratio=OBS_RATIO)
        assert grid.shape == (GRID_SIZE, GRID_SIZE)
        encoder = ue.UniversalFieldToEventEncoder(grid, FIELD_REGISTRY)
        last_count = 0
        llm_state = LLMJudgeDisplayState(status="disabled")
        for t in range(steps):
            current = mixed_fields_for_step(grid, effects, t)
            # No-leak API: only t and mixed current fields are passed.
            events = encoder.update(t, current)
            _ = summarize_events_for_ui(events, llm_state)
            last_count = len(events)
        print(f"{key:>10}: ok, last_event_count={last_count}, obstacles={int(grid.sum())}")


def _scenario_match(expected_key: str, predicted: str) -> bool:
    pred = (predicted or "").lower()
    aliases = {
        "fire": ["着火", "明火", "火灾", "fire"],
        "electrical": ["电气", "过热", "柜", "electrical", "overheat"],
        "leak": ["漏水", "渗水", "water", "leak"],
        "steam": ["蒸汽", "steam"],
        "co2": ["co2", "二氧化碳", "通风", "积聚"],
        "dust": ["粉尘", "空气污染", "污染", "dust", "air"],
    }
    return any(a.lower() in pred for a in aliases.get(expected_key, []))


def llm_accuracy_eval(
    api_key: str,
    model: str = "qwen3.6-flash",
    endpoint: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    eval_step: int = 150,
    seed: int = 2026,
    timeout_s: float = 25.0,
    enable_thinking: bool = False,
    out_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Call Qwen once per scenario and compute simple scenario-level accuracy."""
    scenarios = make_scenarios()
    rows: List[Dict[str, Any]] = []
    for i, (key, scenario) in enumerate(scenarios.items()):
        grid, effects = prepare_grid_and_effects(scenario, seed=seed + i, obs_ratio=OBS_RATIO)
        encoder = ue.UniversalFieldToEventEncoder(grid, FIELD_REGISTRY)
        events: List[Dict[str, Any]] = []
        for t in range(eval_step):
            current = mixed_fields_for_step(grid, effects, t)
            events = encoder.update(t, current)
        start = time.time()
        result = call_qwen_judge_sync(
            eval_step,
            events,
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            timeout_s=timeout_s,
            enable_thinking=enable_thinking,
        )
        elapsed = time.time() - start
        pred = result.get("推断的异常事件", "")
        ok = _scenario_match(key, pred)
        rows.append({
            "demo": key,
            "expected": scenario.short_zh,
            "predicted": pred,
            "reason": result.get("简短的推断理由或者过程", ""),
            "match": bool(ok),
            "event_count": len(events),
            "latency_s": round(elapsed, 3),
        })
        print(f"{key:>10}: expected={scenario.short_zh} predicted={pred} match={ok} latency={elapsed:.2f}s")
    acc = float(np.mean([r["match"] for r in rows])) if rows else 0.0
    summary = {"model": model, "eval_step": eval_step, "n": len(rows), "accuracy": acc, "rows": rows}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if out_json:
        Path(out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved LLM eval: {out_json}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Six no-leak field-event visualization demos with Qwen LLM high-level judgment")
    parser.add_argument("--demo", choices=tuple(make_scenarios().keys()), default="fire")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--obs-ratio", type=float, default=OBS_RATIO)
    parser.add_argument("--interval-ms", type=int, default=80)
    parser.add_argument("--save-gif", type=str, default=None, help="Optional output GIF path")
    parser.add_argument("--no-show", action="store_true", help="Do not open GUI; useful with --save-gif")
    parser.add_argument("--headless-check", action="store_true", help="Run a quick non-GUI smoke test without LLM calls")

    parser.add_argument("--llm-model", type=str, default="qwen3.6-flash")
    parser.add_argument("--llm-endpoint", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    parser.add_argument("--llm-api-key", type=str, default=None, help="Prefer env var DASHSCOPE_API_KEY/QWEN_API_KEY instead of this CLI option")
    parser.add_argument("--llm-interval-steps", type=int, default=50)
    parser.add_argument("--llm-timeout-s", type=float, default=18.0)
    parser.add_argument("--llm-enable-thinking", action="store_true", help="Usually keep off for JSON mode")
    parser.add_argument("--disable-llm", action="store_true", help="Disable Qwen high-level judgment")

    parser.add_argument("--llm-eval", action="store_true", help="Headless: call Qwen once per demo and report scenario-level accuracy")
    parser.add_argument("--llm-eval-step", type=int, default=150)
    parser.add_argument("--llm-eval-out", type=str, default="outputs/qwen_eval_results.json")

    args = parser.parse_args(argv)

    if args.headless_check:
        headless_check(seed=args.seed, steps=min(args.steps, 40))
        return

    if args.llm_eval:
        api_key = args.llm_api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY") or ""
        if not api_key:
            raise SystemExit("Missing API key. Set DASHSCOPE_API_KEY or QWEN_API_KEY, or pass --llm-api-key.")
        llm_accuracy_eval(
            api_key=api_key,
            model=args.llm_model,
            endpoint=args.llm_endpoint,
            eval_step=args.llm_eval_step,
            seed=args.seed,
            timeout_s=max(args.llm_timeout_s, 25.0),
            enable_thinking=args.llm_enable_thinking,
            out_json=args.llm_eval_out,
        )
        return

    run_demo(
        demo_key=args.demo,
        seed=args.seed,
        steps=args.steps,
        obs_ratio=args.obs_ratio,
        interval_ms=args.interval_ms,
        save_gif=args.save_gif,
        no_show=args.no_show,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        llm_endpoint=args.llm_endpoint,
        llm_interval_steps=args.llm_interval_steps,
        llm_timeout_s=args.llm_timeout_s,
        llm_enable_thinking=args.llm_enable_thinking,
        disable_llm=args.disable_llm,
    )


if __name__ == "__main__":
    main()
