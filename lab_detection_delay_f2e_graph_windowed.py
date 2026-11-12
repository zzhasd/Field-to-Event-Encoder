"""Windowed seven-method detection-delay experiment.

This entry point reuses the simulator, F2E encoder, adaptive evidence graph,
prompts, rules, API client, and scoring functions from
``lab_comprehensive_comparison_f2e_graph_neutral_new.py``.  It uses the same
500-step physical timeline and map/RNG replay protocol as ``pipeline_compare``,
while replacing TFRA reconstructions with complete true physical fields:

* every still-active method is queried every ``call_interval_steps``;
* F2E, Graph, F2E rules, and VLM methods use the most recent
  ``window_steps`` observed matrices and are rebuilt independently per window;
* threshold and raw-matrix methods diagnose the current time point;
* original 170-step effect onsets are scaled to the run length and remain
  active until the terminal frame, exactly as in ``pipeline_compare``;
* a method stops independently after its first exact correct prediction;
* the final step is queried even when it is not an interval multiple.

No scenario ID, ground-truth label, injected-effect metadata, or GT mask is
included in any model-visible input.  Ground truth is used only after a method
returns, to score the first-correct detection time and implement evaluation
early stopping.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import label

import lab_comprehensive_comparison_f2e_graph_neutral_new as base


EXPERIMENT_VERSION = "v8.8.0-windowed-detection-delay-v2-pipeline-timeline"

METHOD_LABELS = {
    "f2e_events_graph_llm": "F2E Graph LLM",
    "f2e_events_llm": "F2E LLM",
    "threshold_matrix_llm": "Threshold LLM",
    "raw_matrix_summary_llm": "Raw matrix LLM",
    "threshold_matrix_rules": "Threshold rules",
    "f2e_events_rules": "F2E rules",
    "raw_field_image_vlm": "Raw field image VLM",
}

WINDOW_METHODS = frozenset({
    "f2e_events_graph_llm",
    "f2e_events_llm",
    "f2e_events_rules",
    "raw_field_image_vlm",
})

TIMEPOINT_METHODS = frozenset({
    "threshold_matrix_llm",
    "raw_matrix_summary_llm",
    "threshold_matrix_rules",
})

F2E_WINDOW_METHODS = frozenset({
    "f2e_events_graph_llm",
    "f2e_events_llm",
    "f2e_events_rules",
})


def scale_scenarios_to_pipeline_timeline(
    scenarios: Sequence[base.AccidentScenario],
    steps: int,
) -> List[base.AccidentScenario]:
    """Match pipeline_compare's 170-step-to-run-length incident timeline."""
    scaled = copy.deepcopy(list(scenarios))
    for scenario in scaled:
        for injected_effect in scenario.testcase.effects:
            injected_effect.start = max(
                1,
                int(round(float(injected_effect.start) / 170.0 * int(steps))),
            )
            injected_effect.end = int(steps)
    return scaled


def make_pipeline_obstacle_grid(
    size: int,
    obstacle_density: float,
    seed: int,
) -> np.ndarray:
    """Reproduce pipeline_compare's seeded SensorGridEnv obstacle map."""
    rng = np.random.RandomState(int(seed))
    grid = np.zeros((int(size), int(size)), dtype=np.uint8)
    obstacle_count = int(grid.size * float(obstacle_density))
    obstacle_indices = rng.choice(grid.size, obstacle_count, replace=False)
    grid[np.unravel_index(obstacle_indices, grid.shape)] = 1

    free_space = 1 - grid
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    components, _ = label(free_space, structure=structure)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    grid[components != int(sizes.argmax())] = 1
    return grid


def collect_pipeline_full_field_trace(
    scenario: base.AccidentScenario,
    seed: int,
    steps: int,
    obstacle_density: float,
) -> base.RunTrace:
    """Build the same true-field replay as pipeline_compare, without TFRA."""
    started = time.perf_counter()
    grid = make_pipeline_obstacle_grid(base.GRID_SIZE, obstacle_density, seed)
    field_keys = tuple(base.FIELDS)
    rng = np.random.default_rng(int(seed) + 99173)
    fields_window: List[Dict[str, np.ndarray]] = []
    fields_last: Dict[str, np.ndarray] = {}
    gt_last: List[Dict[str, Any]] = []

    # pipeline_compare materializes physical frames with one-based frame
    # indices and a single episode RNG before any reconstruction takes place.
    for frame in range(1, int(steps) + 1):
        current, gt_records = base.build_current_fields(
            grid,
            scenario.testcase,
            frame,
            field_keys,
            scenario.stress,
            rng,
        )
        fields_last = current
        gt_last = gt_records
        fields_window.append({key: value.copy() for key, value in current.items()})

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    low_metrics = {
        "latency_ms_per_frame": round(elapsed_ms / max(1, int(steps)), 6),
        "detector_total_latency_ms": 0.0,
        "detector_latency_ms_per_frame": 0.0,
    }
    return base.RunTrace(
        scenario=scenario,
        seed=int(seed),
        detector_name="pipeline_full_true_field_replay",
        events_last=[],
        events_all=[],
        event_snapshots=[],
        fields_window=fields_window,
        fields_last=fields_last,
        grid=grid,
        gt_last=gt_last,
        low_metrics=low_metrics,
        latency_ms=elapsed_ms,
    )


def diagnosis_schedule(steps: int, interval_steps: int) -> List[int]:
    """Return one-based query steps, including the final step exactly once."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if interval_steps < 1:
        raise ValueError("call_interval_steps must be positive")
    schedule = list(range(interval_steps, steps + 1, interval_steps))
    if not schedule or schedule[-1] != steps:
        schedule.append(steps)
    return schedule


def window_bounds(query_step: int, window_steps: int) -> Tuple[int, int, int]:
    """Return one-based inclusive window start/end and frame count."""
    start = max(1, int(query_step) - int(window_steps) + 1)
    end = int(query_step)
    return start, end, end - start + 1


def reconstruct_f2e_window(
    fields_window: Sequence[Dict[str, np.ndarray]],
    grid: np.ndarray,
    global_start_t: int,
) -> Tuple[List[Dict[str, Any]], float]:
    """Run the unchanged F2E encoder from a fresh state on one sliding window."""
    detector = base.detector_factory("f2e_encoder", grid)
    events: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for offset, current in enumerate(fields_window):
        events = detector.update(global_start_t + offset, current)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return events, elapsed_ms


def render_window_contact_sheet(
    fields_window: Sequence[Dict[str, np.ndarray]],
    out_path: str,
    grid: np.ndarray,
    global_start_t: int,
    mode: str,
    num_frames: int,
) -> Tuple[Optional[str], List[int]]:
    """Render the existing VLM contact-sheet representation with global times."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None, []
    if not fields_window:
        return None, []

    n_total = len(fields_window)
    if mode == "last":
        local_indices = [n_total - 1]
    elif mode == "all":
        local_indices = list(range(n_total))
        if len(local_indices) > 32:
            local_indices = [
                int(round(value)) for value in np.linspace(0, n_total - 1, 32)
            ]
    else:
        count = max(2, min(int(num_frames), n_total))
        local_indices = [
            int(round(value)) for value in np.linspace(0, n_total - 1, count)
        ]
    local_indices = list(dict.fromkeys(local_indices))
    global_indices = [global_start_t + index for index in local_indices]

    keys = list(fields_window[local_indices[-1]].keys())
    n_rows, n_cols = len(keys), len(local_indices)
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(3.2, 2.15 * n_cols), max(3.0, 2.05 * n_rows)),
        squeeze=False,
    )
    grid_array = np.asarray(grid)
    for row, field_key in enumerate(keys):
        values = []
        for index in local_indices:
            array = fields_window[index].get(field_key)
            if array is not None:
                finite = np.asarray(array)[np.isfinite(array)]
                if finite.size:
                    values.append(finite)
        if values:
            merged = np.concatenate(values)
            vmin, vmax = np.nanpercentile(merged, [2, 98])
            if (
                not np.isfinite(vmin)
                or not np.isfinite(vmax)
                or abs(float(vmax) - float(vmin)) < 1e-9
            ):
                vmin, vmax = None, None
        else:
            vmin, vmax = None, None

        for column, (local_index, global_index) in enumerate(
            zip(local_indices, global_indices)
        ):
            axis = axes[row][column]
            array = np.asarray(fields_window[local_index][field_key], dtype=np.float32)
            axis.imshow(
                np.ma.masked_invalid(array),
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )
            overlay = np.zeros(array.shape + (4,), dtype=np.float32)
            overlay[grid_array != 0] = (0.12, 0.12, 0.12, 1.0)
            overlay[(grid_array == 0) & (~np.isfinite(array))] = (1.0, 0.0, 1.0, 1.0)
            axis.imshow(overlay, interpolation="nearest")
            if row == 0:
                axis.set_title(f"t={global_index}")
            if column == 0:
                axis.set_ylabel(field_key)
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(
        "Temporal multi-field observation"
        f" | mode={mode}, frames={global_indices}"
        " | obstacle=dark gray, missing=magenta"
    )
    figure.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    figure.savefig(out_path, dpi=135)
    plt.close(figure)
    return out_path, global_indices


def build_windowed_method_inputs(
    trace: base.RunTrace,
    query_step: int,
    window_steps: int,
    active_methods: Sequence[str],
    image_dir: str,
    scenario_index: int,
    call_index: int,
    vlm_frame_mode: str,
    vlm_num_frames: int,
) -> Dict[str, Dict[str, Any]]:
    """Build method inputs while preserving each underlying base algorithm."""
    query_t = int(query_step) - 1
    diagnosis_frame = int(query_step)
    window_start_step, _, window_frame_count = window_bounds(query_step, window_steps)
    window_start_t = window_start_step - 1
    fields_history = trace.fields_window[: query_t + 1]
    fields_window = trace.fields_window[window_start_t : query_t + 1]
    current_fields = [trace.fields_window[query_t]]
    free_mask = np.asarray(trace.grid) == 0
    active = set(active_methods)

    current_quality = base.observation_quality(current_fields, free_mask)
    window_quality = base.observation_quality(fields_window, free_mask)
    result: Dict[str, Dict[str, Any]] = {
        "diagnosis_t": {
            "t": diagnosis_frame,
            "array_index": query_t,
            "query_step": int(query_step),
            "window_start_step": int(window_start_step),
            "window_end_step": int(query_step),
            "window_frame_count": int(window_frame_count),
        },
        "observation_quality": window_quality,
    }

    if active & {"threshold_matrix_llm", "threshold_matrix_rules"}:
        started = time.perf_counter()
        # The current alarm matrix remains a time-point representation.  The
        # unchanged fixed-limit algorithm uses the causal Phase-I history only
        # to estimate its control limits.
        alarm_matrix = base.build_threshold_alarm_matrices(fields_history, trace.grid)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result["threshold_alarm_matrix"] = {
            "alarm_matrix": alarm_matrix,
            "observation_quality": current_quality,
            "representation_latency_ms": elapsed_ms,
            "diagnosis_t": diagnosis_frame,
            "input_frame_count": 1,
            "calibration_history_frames": len(fields_history),
        }

    if "raw_matrix_summary_llm" in active:
        started = time.perf_counter()
        field_summary = base.current_field_summary(current_fields, free_mask)
        field_summary["observation_quality"] = current_quality
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result["raw_matrix_summary"] = {
            "field_summary": field_summary,
            "observation_quality": current_quality,
            "representation_latency_ms": elapsed_ms,
            "input_frame_count": 1,
        }

    if active & F2E_WINDOW_METHODS:
        events_raw, f2e_elapsed_ms = reconstruct_f2e_window(
            fields_window,
            trace.grid,
            global_start_t=window_start_step,
        )
        events = base.summarize_events(
            events_raw,
            history_frames=len(fields_window),
        )
        graph_features: Dict[str, Any] = {}
        graph_elapsed_ms = 0.0
        if "f2e_events_graph_llm" in active:
            started = time.perf_counter()
            residual_raw = base.derive_weak_candidates_from_fields(
                fields_window,
                free_mask,
                confirmed_events=events_raw,
            )
            residual_evidence = base.neutralize_subthreshold_evidence(residual_raw)
            graph_features = base.build_adaptive_evidence_graph(
                events_raw,
                residual_evidence,
                grid=trace.grid,
                history_frames=len(fields_window),
            )
            graph_elapsed_ms = (time.perf_counter() - started) * 1000.0
        result["f2e_events"] = {
            "events": events,
            "observation_quality": window_quality,
            "graph_features": graph_features,
            "representation_latency_ms": f2e_elapsed_ms,
            "graph_representation_latency_ms": f2e_elapsed_ms + graph_elapsed_ms,
            "graph_construction_latency_ms": graph_elapsed_ms,
            "diagnosis_t": diagnosis_frame,
            "input_frame_count": len(fields_window),
        }

    if "raw_field_image_vlm" in active:
        started = time.perf_counter()
        filename_mode = f"{vlm_frame_mode}{vlm_num_frames}_call{call_index}"
        filename = base.blind_artifact_stem(
            trace.seed,
            scenario_index,
            "vlm_window",
            filename_mode,
        )
        image_path, frame_indices = render_window_contact_sheet(
            fields_window,
            os.path.join(image_dir, f"{filename}.png"),
            trace.grid,
            global_start_t=window_start_step,
            mode=vlm_frame_mode,
            num_frames=vlm_num_frames,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result["raw_field_image"] = {
            "image_path": image_path,
            "image_filename_policy": "blind_no_scenario_id_no_accident_type",
            "diagnosis_t": diagnosis_frame,
            "vlm_frame_mode": vlm_frame_mode,
            "vlm_frame_indices": frame_indices,
            "representation_latency_ms": elapsed_ms,
            "image_render_latency_ms": elapsed_ms,
            "input_frame_count": len(fields_window),
        }

    return result


def method_history_frames(method: str, window_frame_count: int) -> int:
    return int(window_frame_count) if method in WINDOW_METHODS else 1


def method_quality(
    method: str,
    method_inputs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if method.startswith("threshold_matrix"):
        return method_inputs["threshold_alarm_matrix"].get("observation_quality") or {}
    if method == "raw_matrix_summary_llm":
        return method_inputs["raw_matrix_summary"].get("observation_quality") or {}
    if method.startswith("f2e_events"):
        return method_inputs["f2e_events"].get("observation_quality") or {}
    return method_inputs.get("observation_quality") or {}


def execute_rule_method(
    method: str,
    method_inputs: Dict[str, Dict[str, Any]],
) -> Tuple[float, base.ModelCallResult, Dict[str, Any], float]:
    started = time.perf_counter()
    if method == "threshold_matrix_rules":
        representation_latency_ms = float(
            method_inputs["threshold_alarm_matrix"].get("representation_latency_ms")
            or 0.0
        )
        diag = base.diagnose_by_threshold_alarms(
            method_inputs["threshold_alarm_matrix"]["alarm_matrix"],
            observation_quality=method_inputs["threshold_alarm_matrix"].get(
                "observation_quality"
            ),
        )
    elif method == "f2e_events_rules":
        representation_latency_ms = float(
            method_inputs["f2e_events"].get("representation_latency_ms") or 0.0
        )
        diag = base.diagnose_by_rules(
            method_inputs["f2e_events"]["events"],
            source_name="f2e_events",
            observation_quality=method_inputs["f2e_events"].get(
                "observation_quality"
            ),
        )
    else:
        raise ValueError(f"Not a rule method: {method}")
    normalized = base.normalize_diag(diag)
    wall_ms = (time.perf_counter() - started) * 1000.0
    result = base.ModelCallResult(
        ok=True,
        content=json.dumps(diag, ensure_ascii=False),
        parsed=diag,
        latency_ms=0.0,
        model="rules",
        token_count_source="not_applicable_rules",
    )
    return representation_latency_ms, result, normalized, wall_ms


def invalid_diagnosis(call_result: base.ModelCallResult) -> Dict[str, Any]:
    return {
        "accident_type": None,
        "raw_prediction": None,
        "confidence": None,
        "review_needed": None,
        "abnormal_confirmed": None,
        "evidence_fields": [],
        "evidence_events": [],
        "unexplained_evidence": [],
        "template_status": "invalid_model_call",
        "protocol_conflict": False,
        "protocol_conflict_reasons": [],
        "protocol_consistency_repair": False,
        "explanation": call_result.error or "invalid model call",
    }


def make_call_record(
    scenario: base.AccidentScenario,
    seed_index: int,
    seed: int,
    method: str,
    call_index: int,
    query_step: int,
    window_steps: int,
    method_inputs: Dict[str, Dict[str, Any]],
    representation_latency_ms: float,
    call_result: base.ModelCallResult,
    diag: Dict[str, Any],
    diagnosis_wall_ms: float,
    api_workers: int,
    use_parallel_api: bool,
) -> Dict[str, Any]:
    valid_prediction = bool(
        call_result.ok
        and (call_result.model == "rules" or call_result.parsed is not None)
    )
    if not valid_prediction:
        diag = invalid_diagnosis(call_result)
    correct = int(
        valid_prediction and diag.get("accident_type") == scenario.accident_type
    )
    explanation = base.explanation_score(diag, scenario) if valid_prediction else 0.0
    if valid_prediction:
        reasoning = base.reasoning_alignment_scores(diag, scenario)
    else:
        reasoning = {
            **base.evidence_alignment_scores(diag, scenario),
            "behavior_goal_score": 0.0,
            "review_calibration_score": 0.0,
            "resample_target_match": None,
            "forbidden_prediction_score": 0.0,
            "reasoning_alignment_score": 0.0,
            "template_rejection_success": 0,
        }

    window_start_step, window_end_step, window_frame_count = window_bounds(
        query_step,
        window_steps,
    )
    if method in TIMEPOINT_METHODS:
        input_start_step = query_step
        input_frame_count = 1
        input_mode = "time_point"
    else:
        input_start_step = window_start_step
        input_frame_count = window_frame_count
        input_mode = "sliding_window"

    input_event_count = None
    graph_node_count = None
    graph_group_count = None
    input_alarm_count = None
    if method.startswith("f2e_events"):
        input_event_count = len(method_inputs["f2e_events"].get("events") or [])
    if method == "f2e_events_graph_llm":
        graph = method_inputs["f2e_events"].get("graph_features") or {}
        graph_node_count = len(graph.get("nodes") or [])
        graph_group_count = len(graph.get("evidence_groups") or [])
    if method.startswith("threshold_matrix"):
        alarm = method_inputs["threshold_alarm_matrix"]["alarm_matrix"]
        input_alarm_count = int(alarm.get("total_high_alarm_cells", 0)) + int(
            alarm.get("total_low_alarm_cells", 0)
        )

    quality = method_quality(method, method_inputs)
    effect_starts = [int(effect.start) for effect in scenario.testcase.effects]
    effect_ends = [int(effect.end) for effect in scenario.testcase.effects]
    first_effect_start_t = min(effect_starts) if effect_starts else None
    last_effect_end_t_exclusive = max(effect_ends) if effect_ends else None
    detection_delay_after_onset = (
        query_step - first_effect_start_t
        if correct and first_effect_start_t is not None
        else None
    )
    measured_api_latency_ms = (
        None
        if call_result.from_cache or call_result.model in {"rules", "offline_proxy"}
        else call_result.latency_ms
    )
    end_to_end_latency_ms = representation_latency_ms + diagnosis_wall_ms
    return {
        "layer": "windowed_detection_delay",
        "seed_index": int(seed_index),
        "seed": int(seed),
        "scenario_id": scenario.scenario_id,
        "method": method,
        "method_display_name": METHOD_LABELS[method],
        "call_index": int(call_index),
        "query_step": int(query_step),
        "diagnosis_t": int(query_step),
        "input_mode": input_mode,
        "input_start_step": int(input_start_step),
        "input_end_step": int(window_end_step),
        "input_frame_count": int(input_frame_count),
        "configured_window_steps": int(window_steps),
        "valid_prediction": int(valid_prediction),
        "ground_truth": scenario.accident_type,
        "prediction": diag.get("accident_type"),
        "raw_prediction": diag.get("raw_prediction", diag.get("accident_type")),
        "correct": int(correct),
        "first_correct_on_this_call": int(correct),
        "detection_delay_call_index": int(call_index) if correct else None,
        "detection_delay_query_step": int(query_step) if correct else None,
        "first_effect_start_t": first_effect_start_t,
        "last_effect_end_t_exclusive": last_effect_end_t_exclusive,
        "prediction_made_while_effect_active": (
            None
            if first_effect_start_t is None or last_effect_end_t_exclusive is None
            else int(first_effect_start_t <= query_step < last_effect_end_t_exclusive)
        ),
        "detection_delay_after_onset_steps": detection_delay_after_onset,
        "ambiguous": scenario.ambiguous,
        "unseen_combo": scenario.unseen_combo,
        "expected_review": scenario.expected_review,
        "hard_family": scenario.hard_family,
        "diagnostic_goal": scenario.diagnostic_goal,
        "challenge": scenario.challenge,
        "input_event_count": input_event_count,
        "graph_node_count": graph_node_count,
        "graph_group_count": graph_group_count,
        "input_alarm_count": input_alarm_count,
        "observation_low_confidence": quality.get("low_confidence"),
        "last_frame_missing_fraction": quality.get("last_frame_missing_fraction"),
        "review_needed": diag.get("review_needed"),
        "confidence": diag.get("confidence"),
        "explanation_correctness": round(float(explanation), 6),
        "behavior_goal_score": round(float(reasoning["behavior_goal_score"]), 6),
        "evidence_field_f1": (
            None
            if reasoning["evidence_field_f1"] is None
            else round(float(reasoning["evidence_field_f1"]), 6)
        ),
        "evidence_signature_alignment": round(
            float(reasoning["evidence_signature_alignment"]), 6
        ),
        "review_calibration_score": round(
            float(reasoning["review_calibration_score"]), 6
        ),
        "reasoning_alignment_score": round(
            float(reasoning["reasoning_alignment_score"]), 6
        ),
        "template_rejection_success": reasoning["template_rejection_success"],
        "reported_evidence_fields": ";".join(reasoning["reported_evidence_fields"]),
        "reported_event_signatures": ";".join(
            reasoning["reported_event_signatures"]
        ),
        "template_status": diag.get("template_status"),
        "protocol_conflict": int(bool(diag.get("protocol_conflict", False))),
        "protocol_conflict_reasons": ";".join(
            map(str, diag.get("protocol_conflict_reasons", []))
        ),
        "prompt_tokens": call_result.prompt_tokens,
        "completion_tokens": call_result.completion_tokens,
        "total_tokens": call_result.total_tokens,
        "token_count_source": call_result.token_count_source,
        "token_count_is_estimate": int(call_result.token_count_is_estimate),
        "api_usage_available": int(call_result.api_usage_available),
        "prompt_text_tokens": call_result.prompt_text_tokens,
        "prompt_image_tokens": call_result.prompt_image_tokens,
        "completion_reasoning_tokens": call_result.completion_reasoning_tokens,
        "api_latency_ms": round(float(call_result.latency_ms), 3),
        "measured_api_latency_ms": (
            None
            if measured_api_latency_ms is None
            else round(float(measured_api_latency_ms), 3)
        ),
        "diagnosis_wall_ms": round(float(diagnosis_wall_ms), 3),
        "representation_latency_ms": round(float(representation_latency_ms), 3),
        "end_to_end_latency_ms": round(float(end_to_end_latency_ms), 3),
        "model": call_result.model,
        "from_cache": call_result.from_cache,
        "api_ok": call_result.ok,
        "api_error": call_result.error,
        "api_workers": max(1, int(api_workers)),
        "api_scheduling": "parallel_within_query" if use_parallel_api else "serial",
        "explanation": diag.get("explanation"),
    }


def sum_numeric(rows: Sequence[Dict[str, Any]], key: str) -> float:
    total = 0.0
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except Exception:
            continue
        if np.isfinite(number):
            total += number
    return total


def make_case_summary(
    scenario: base.AccidentScenario,
    seed_index: int,
    seed: int,
    method: str,
    rows: Sequence[Dict[str, Any]],
    schedule: Sequence[int],
    steps: int,
) -> Dict[str, Any]:
    first = next((row for row in rows if int(row.get("correct", 0)) == 1), None)
    detected = first is not None
    last = rows[-1] if rows else {}
    effect_starts = [int(effect.start) for effect in scenario.testcase.effects]
    effect_ends = [int(effect.end) for effect in scenario.testcase.effects]
    onset_t = min(effect_starts) if effect_starts else None
    end_t_exclusive = max(effect_ends) if effect_ends else None
    return {
        "seed_index": int(seed_index),
        "seed": int(seed),
        "scenario_id": scenario.scenario_id,
        "ground_truth": scenario.accident_type,
        "method": method,
        "method_display_name": METHOD_LABELS[method],
        "detected": int(detected),
        "censored": int(not detected),
        "first_correct_call_index": (
            int(first["call_index"]) if first is not None else None
        ),
        "first_correct_query_step": (
            int(first["query_step"]) if first is not None else None
        ),
        "detection_delay_calls": (
            int(first["call_index"]) if first is not None else None
        ),
        "detection_delay_steps": (
            int(first["query_step"]) if first is not None else None
        ),
        "first_effect_start_t": onset_t,
        "last_effect_end_t_exclusive": end_t_exclusive,
        "first_correct_while_effect_active": (
            None
            if first is None
            or onset_t is None
            or end_t_exclusive is None
            else int(onset_t <= int(first["diagnosis_t"]) < end_t_exclusive)
        ),
        "detection_delay_after_onset_steps": (
            int(first["detection_delay_after_onset_steps"])
            if first is not None
            and first.get("detection_delay_after_onset_steps") is not None
            else None
        ),
        "calls_executed": len(rows),
        "maximum_scheduled_calls": len(schedule),
        "censoring_call_index": None if detected else len(schedule),
        "censoring_step": None if detected else int(steps),
        "final_prediction": last.get("prediction"),
        "final_valid_prediction": last.get("valid_prediction"),
        "total_tokens_until_stop": round(sum_numeric(rows, "total_tokens"), 3),
        "total_api_latency_ms_until_stop": round(
            sum_numeric(rows, "measured_api_latency_ms"), 3
        ),
        "total_end_to_end_latency_ms_until_stop": round(
            sum_numeric(rows, "end_to_end_latency_ms"), 3
        ),
        "hard_family": scenario.hard_family,
        "diagnostic_goal": scenario.diagnostic_goal,
        "ambiguous": scenario.ambiguous,
        "unseen_combo": scenario.unseen_combo,
        "expected_review": scenario.expected_review,
    }


def optional_stat(values: Sequence[Any], reducer: str) -> Optional[float]:
    numeric = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except Exception:
            continue
        if np.isfinite(number):
            numeric.append(number)
    if not numeric:
        return None
    if reducer == "mean":
        return float(np.mean(numeric))
    if reducer == "median":
        return float(np.median(numeric))
    raise ValueError(reducer)


def aggregate_detection_delay(
    case_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in case_rows:
        groups.setdefault(str(row["method"]), []).append(row)
    output = []
    for method in base.default_layer2_methods():
        rows = groups.get(method, [])
        if not rows:
            continue
        detected = [row for row in rows if int(row.get("detected", 0)) == 1]
        mean_call = optional_stat(
            [row.get("first_correct_call_index") for row in detected], "mean"
        )
        median_call = optional_stat(
            [row.get("first_correct_call_index") for row in detected], "median"
        )
        mean_step = optional_stat(
            [row.get("first_correct_query_step") for row in detected], "mean"
        )
        mean_onset_delay = optional_stat(
            [row.get("detection_delay_after_onset_steps") for row in detected],
            "mean",
        )
        output.append({
            "method": method,
            "method_display_name": METHOD_LABELS[method],
            "n_cases": len(rows),
            "detected_n": len(detected),
            "censored_n": len(rows) - len(detected),
            "detection_rate": round(len(detected) / len(rows), 6),
            "mean_detection_delay_calls_detected": (
                None if mean_call is None else round(mean_call, 6)
            ),
            "median_detection_delay_calls_detected": (
                None if median_call is None else round(median_call, 6)
            ),
            "mean_first_correct_query_step_detected": (
                None if mean_step is None else round(mean_step, 6)
            ),
            "mean_detection_delay_after_onset_steps_detected": (
                None if mean_onset_delay is None else round(mean_onset_delay, 6)
            ),
            "mean_calls_executed": round(
                float(np.mean([int(row["calls_executed"]) for row in rows])), 6
            ),
            "mean_total_tokens_until_stop": round(
                float(np.mean([float(row["total_tokens_until_stop"]) for row in rows])),
                3,
            ),
            "total_tokens": round(
                float(sum(float(row["total_tokens_until_stop"]) for row in rows)),
                3,
            ),
        })
    return output


def run_detection_delay(
    scenarios: Sequence[base.AccidentScenario],
    seeds: int,
    steps: int,
    obs_ratio: float,
    base_seed: int,
    out_dir: str,
    api_mode: str,
    client: base.DashScopeClient,
    llm_model: str,
    vlm_model: str,
    progress: base.ProgressMeter,
    vlm_frame_mode: str,
    vlm_num_frames: int,
    call_interval_steps: int,
    window_steps: int,
    token_counter: base.TokenFallbackCounter,
    methods: Sequence[str],
    api_workers: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    call_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    schedule = diagnosis_schedule(steps, call_interval_steps)
    image_dir = os.path.join(out_dir, "field_images")
    completed_cases = 0
    use_real_api = api_mode == "api" or (
        api_mode == "auto" and bool(client.api_key)
    )

    for seed_index in range(seeds):
        for scenario_index, scenario in enumerate(scenarios):
            seed = base_seed + 100000 * seed_index + 1000 * scenario_index
            # Generate the pipeline-compatible complete true-field replay.
            # This is the controlled upper-bound counterpart of the TFRA
            # reconstructed replay used by pipeline_compare.
            trace = collect_pipeline_full_field_trace(
                scenario,
                seed,
                steps,
                obstacle_density=obs_ratio,
            )
            active = set(methods)
            rows_by_method: Dict[str, List[Dict[str, Any]]] = {
                method: [] for method in methods
            }

            for call_index, query_step in enumerate(schedule, start=1):
                if not active:
                    break
                window_start_step, _, window_frame_count = window_bounds(
                    query_step,
                    window_steps,
                )
                active_in_order = [method for method in methods if method in active]
                method_inputs = build_windowed_method_inputs(
                    trace,
                    query_step=query_step,
                    window_steps=window_steps,
                    active_methods=active_in_order,
                    image_dir=image_dir,
                    scenario_index=scenario_index,
                    call_index=call_index,
                    vlm_frame_mode=vlm_frame_mode,
                    vlm_num_frames=vlm_num_frames,
                )

                model_methods = [
                    method
                    for method in active_in_order
                    if method in base.MODEL_LAYER2_METHODS
                ]
                parallel_results: Dict[
                    str,
                    Tuple[float, base.ModelCallResult, Dict[str, Any], float],
                ] = {}
                use_parallel_api = (
                    use_real_api and api_workers > 1 and len(model_methods) > 1
                )
                if use_parallel_api:
                    worker_count = min(max(1, int(api_workers)), len(model_methods))
                    with ThreadPoolExecutor(
                        max_workers=worker_count,
                        thread_name_prefix="delay-api",
                    ) as executor:
                        futures: Dict[
                            Future[
                                Tuple[
                                    float,
                                    base.ModelCallResult,
                                    Dict[str, Any],
                                    float,
                                ]
                            ],
                            str,
                        ] = {}
                        for method in model_methods:
                            history_frames = method_history_frames(
                                method,
                                window_frame_count,
                            )
                            future = executor.submit(
                                base.execute_layer2_model_method,
                                method,
                                method_inputs,
                                query_step,
                                history_frames,
                                client,
                                api_mode,
                                llm_model,
                                vlm_model,
                                token_counter,
                            )
                            futures[future] = method
                        completed_calls = 0
                        for future in as_completed(futures):
                            method = futures[future]
                            parallel_results[method] = future.result()
                            completed_calls += 1
                            progress.update(
                                completed_cases,
                                f"seed={seed_index + 1}/{seeds} "
                                f"scenario={scenario.scenario_id} call={call_index} "
                                f"API={completed_calls}/{len(model_methods)}",
                            )

                newly_detected: List[str] = []
                for method in active_in_order:
                    if method in {"threshold_matrix_rules", "f2e_events_rules"}:
                        result = execute_rule_method(method, method_inputs)
                    elif method in parallel_results:
                        result = parallel_results[method]
                    elif method in base.MODEL_LAYER2_METHODS:
                        result = base.execute_layer2_model_method(
                            method,
                            method_inputs,
                            query_step,
                            method_history_frames(method, window_frame_count),
                            client,
                            api_mode,
                            llm_model,
                            vlm_model,
                            token_counter,
                        )
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    representation_ms, call_result, diag, wall_ms = result
                    row = make_call_record(
                        scenario,
                        seed_index,
                        seed,
                        method,
                        call_index,
                        query_step,
                        window_steps,
                        method_inputs,
                        representation_ms,
                        call_result,
                        diag,
                        wall_ms,
                        api_workers,
                        use_parallel_api,
                    )
                    call_rows.append(row)
                    rows_by_method[method].append(row)
                    if int(row["correct"]) == 1:
                        newly_detected.append(method)

                for method in newly_detected:
                    active.remove(method)
                    completed_cases += 1
                    progress.update(
                        completed_cases,
                        f"detected seed={seed_index + 1}/{seeds} "
                        f"scenario={scenario.scenario_id} method={method} "
                        f"call={call_index} step={query_step}",
                    )
                client.flush_cache()

            # Unresolved methods are right-censored at the final scheduled query.
            for method in list(active):
                completed_cases += 1
                progress.update(
                    completed_cases,
                    f"censored seed={seed_index + 1}/{seeds} "
                    f"scenario={scenario.scenario_id} method={method}",
                )
            for method in methods:
                case_rows.append(
                    make_case_summary(
                        scenario,
                        seed_index,
                        seed,
                        method,
                        rows_by_method[method],
                        schedule,
                        steps,
                    )
                )

    return call_rows, case_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seven-method sliding-window detection-delay experiment using the "
            "unchanged v8.8.0 F2E/Graph/LLM algorithms."
        )
    )
    parser.add_argument("--profile", choices=["quick", "paper", "hard"], default="paper")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--obs-ratio", type=float, default=base.DEFAULT_OBS_RATIO)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--out-dir",
        default="outputs/detection_delay_v880_window100_interval75",
    )
    parser.add_argument("--call-interval-steps", type=int, default=75)
    parser.add_argument("--window-steps", type=int, default=100)
    parser.add_argument(
        "--api-mode",
        choices=["offline", "auto", "api"],
        default="auto",
    )
    parser.add_argument("--llm-model", default="qwen3.6-flash")
    parser.add_argument("--vlm-model", default="qwen3-vl-flash")
    parser.add_argument(
        "--vlm-frame-mode",
        choices=["last", "sampled", "all"],
        default="sampled",
    )
    parser.add_argument("--vlm-num-frames", type=int, default=8)
    parser.add_argument("--layer2-methods", default="all")
    parser.add_argument(
        "--dashscope-base-url",
        default=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )
    parser.add_argument(
        "--dashscope-proxy-mode",
        choices=["bypass", "environment"],
        default="bypass",
        help=(
            "bypass ignores HTTP(S)/ALL_PROXY and Windows proxy settings for "
            "DashScope; use this when a proxy/VPN interferes with the domestic API"
        ),
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--api-retry-backoff", type=float, default=1.0)
    parser.add_argument("--api-workers", type=int, default=5)
    parser.add_argument(
        "--token-fallback",
        choices=["none", "hf_tokenizer", "heuristic"],
        default="none",
    )
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--tokenizer-allow-download", action="store_true")
    parser.add_argument("--progress", choices=["on", "off"], default="on")
    parser.add_argument("--progress-interval", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds is not None and args.seeds < 1:
        raise SystemExit("--seeds must be at least 1")
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if args.call_interval_steps < 1:
        raise SystemExit("--call-interval-steps must be at least 1")
    if args.window_steps < 1:
        raise SystemExit("--window-steps must be at least 1")
    if args.api_workers < 1:
        raise SystemExit("--api-workers must be at least 1")

    methods = base.parse_layer2_methods(args.layer2_methods)
    seeds = args.seeds if args.seeds is not None else (1 if args.profile == "quick" else 10)
    scenarios = scale_scenarios_to_pipeline_timeline(
        base.make_accident_scenarios(args.profile),
        args.steps,
    )
    if args.max_scenarios is not None:
        scenarios = scenarios[: args.max_scenarios]
    schedule = diagnosis_schedule(args.steps, args.call_interval_steps)
    schedule_rows = []
    for call_index, query_step in enumerate(schedule, start=1):
        start_step, end_step, count = window_bounds(query_step, args.window_steps)
        schedule_rows.append({
            "call_index": call_index,
            "query_step": query_step,
            "window_start_step": start_step,
            "window_end_step": end_step,
            "window_frame_count": count,
        })

    os.makedirs(args.out_dir, exist_ok=True)
    base.save_prompt_templates(args.out_dir)
    base.save_csv(os.path.join(args.out_dir, "query_schedule.csv"), schedule_rows)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if args.api_mode == "api" and not api_key:
        print(
            "ERROR: --api-mode api requires DASHSCOPE_API_KEY in the environment.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    client = base.DashScopeClient(
        api_key=api_key,
        base_url=args.dashscope_base_url,
        cache_path=os.path.join(args.out_dir, "dashscope_cache.json"),
        enable_thinking=args.enable_thinking,
        timeout=args.api_timeout,
        max_retries=args.api_retries,
        retry_backoff_sec=args.api_retry_backoff,
        proxy_mode=args.dashscope_proxy_mode,
    )
    token_counter = base.TokenFallbackCounter(
        mode=args.token_fallback,
        tokenizer_model=args.tokenizer_model,
        local_files_only=not args.tokenizer_allow_download,
    )

    total_cases = seeds * len(scenarios) * len(methods)
    progress = base.ProgressMeter(
        total_cases,
        enabled=args.progress == "on",
        interval_sec=args.progress_interval,
    )
    progress.update(0, "starting")
    started = time.perf_counter()
    call_rows, case_rows = run_detection_delay(
        scenarios=scenarios,
        seeds=seeds,
        steps=args.steps,
        obs_ratio=args.obs_ratio,
        base_seed=args.seed,
        out_dir=args.out_dir,
        api_mode=args.api_mode,
        client=client,
        llm_model=args.llm_model,
        vlm_model=args.vlm_model,
        progress=progress,
        vlm_frame_mode=args.vlm_frame_mode,
        vlm_num_frames=args.vlm_num_frames,
        call_interval_steps=args.call_interval_steps,
        window_steps=args.window_steps,
        token_counter=token_counter,
        methods=methods,
        api_workers=args.api_workers,
    )
    client.flush_cache()
    progress.update(total_cases, "done")

    method_summary = aggregate_detection_delay(case_rows)
    run_config = {
        "version": EXPERIMENT_VERSION,
        "source_comparison_version": base.COMPARISON_VERSION,
        "core_version": base.CORE_VERSION,
        "profile": args.profile,
        "experiment_scope": "seven_method_windowed_detection_delay",
        "layer2_methods": methods,
        "layer2_method_display_names": {
            method: METHOD_LABELS[method] for method in methods
        },
        "base_seed": args.seed,
        "seeds": seeds,
        "seed_formula": "base_seed + 100000*seed_index + 1000*scenario_index",
        "steps": args.steps,
        "n_scenarios": len(scenarios),
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "scenario_effect_intervals": {
            scenario.scenario_id: [
                {
                    "effect_id": effect.effect_id,
                    "start_t_inclusive": int(effect.start),
                    "end_t_exclusive": int(effect.end),
                }
                for effect in scenario.testcase.effects
            ]
            for scenario in scenarios
        },
        "scenario_timeline_policy": (
            "pipeline_compare: start=max(1,round(original_start/170*steps)); "
            "end=steps"
        ),
        "scenario_metadata_policy": (
            "scaled canonical scenario is used for both physical generation and "
            "scoring/reporting; effect centers are not relocated"
        ),
        "physical_field_replay_policy": (
            "pipeline_compare true-field protocol: SensorGridEnv-compatible map; "
            "frames=1..steps; one episode RNG default_rng(seed+99173); no TFRA"
        ),
        "graph_residual_policy": (
            "retain lab residual candidates; intentionally not changed in this revision"
        ),
        "call_interval_steps": args.call_interval_steps,
        "window_steps": args.window_steps,
        "query_schedule": schedule_rows,
        "final_step_query_policy": "append_final_step_when_not_interval_multiple",
        "independent_early_stop": True,
        "early_stop_criterion": "first valid exact prediction equal to ground_truth",
        "detection_delay_primary_metric": "first_correct_call_index",
        "detection_delay_secondary_metrics": [
            "first_correct_query_step",
            "detection_delay_after_onset_steps",
        ],
        "censoring_policy": "no exact correct prediction by final scheduled query",
        "method_input_policy": {
            "f2e_events_graph_llm": "sliding_window_matrices_rebuild_f2e_and_graph",
            "f2e_events_llm": "sliding_window_matrices_rebuild_f2e",
            "threshold_matrix_llm": "current_alarm_matrix_with_causal_fixed_limit_calibration",
            "raw_matrix_summary_llm": "current_matrix_summary_only",
            "threshold_matrix_rules": "current_alarm_matrix_with_causal_fixed_limit_calibration",
            "f2e_events_rules": "sliding_window_matrices_rebuild_f2e",
            "raw_field_image_vlm": "sliding_window_temporal_contact_sheet",
        },
        "obs_ratio": args.obs_ratio,
        "api_mode": args.api_mode,
        "api_key_present": bool(api_key),
        "llm_model": args.llm_model,
        "vlm_model": args.vlm_model,
        "vlm_frame_mode": args.vlm_frame_mode,
        "vlm_num_frames": args.vlm_num_frames,
        "api_workers": args.api_workers,
        "api_scheduling": "parallel_model_calls_within_each_query",
        "dashscope_base_url": args.dashscope_base_url,
        "dashscope_proxy_mode": args.dashscope_proxy_mode,
        "dashscope_proxy_note": (
            "bypass installs an empty urllib ProxyHandler and ignores environment/"
            "Windows proxy settings; a system-wide TUN adapter cannot be bypassed "
            "inside Python"
        ),
        "enable_thinking": bool(args.enable_thinking),
        "token_fallback": args.token_fallback,
        "token_counter_status": token_counter.status(),
        "no_leak_policy": (
            "model-visible inputs contain only causal observed matrices, derived "
            "events/graph features, observation quality, and obstacle topology; "
            "ground truth is consulted only after each returned prediction"
        ),
        "wall_time_sec": round(time.perf_counter() - started, 3),
    }
    payload = {
        "run_config": run_config,
        "detection_delay_method_summary": method_summary,
        "detection_delay_case_records": case_rows,
        "detection_call_records": call_rows,
    }

    with open(
        os.path.join(args.out_dir, "detection_delay_results.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with open(
        os.path.join(args.out_dir, "run_config.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)
    base.save_csv(
        os.path.join(args.out_dir, "detection_call_records.csv"),
        call_rows,
    )
    base.save_csv(
        os.path.join(args.out_dir, "detection_delay_case_records.csv"),
        case_rows,
    )
    base.save_csv(
        os.path.join(args.out_dir, "detection_delay_method_summary.csv"),
        method_summary,
    )

    print(f"\n===== Windowed Detection Delay {EXPERIMENT_VERSION} =====")
    print(
        f"profile={args.profile} seeds={seeds} scenarios={len(scenarios)} "
        f"steps={args.steps} interval={args.call_interval_steps} "
        f"window={args.window_steps}"
    )
    print(f"schedule={schedule_rows}")
    print(
        f"api_mode={args.api_mode} proxy_mode={args.dashscope_proxy_mode} "
        f"api_workers={args.api_workers}"
    )
    print(
        "NO-LEAK CHECK: early stopping compares the returned prediction with "
        "ground truth only after inference; labels and injection metadata are "
        "never placed in method inputs."
    )
    print("\nDetection-delay summary:")
    for row in method_summary:
        print(
            f"  {row['method_display_name']}: "
            f"detected={row['detected_n']}/{row['n_cases']} "
            f"rate={row['detection_rate']} "
            f"mean_call={row['mean_detection_delay_calls_detected']} "
            f"mean_step={row['mean_first_correct_query_step_detected']}"
        )
    print("\nSaved:")
    for filename in [
        "detection_delay_results.json",
        "run_config.json",
        "query_schedule.csv",
        "detection_call_records.csv",
        "detection_delay_case_records.csv",
        "detection_delay_method_summary.csv",
    ]:
        print("  " + os.path.join(args.out_dir, filename))


if __name__ == "__main__":
    main()
