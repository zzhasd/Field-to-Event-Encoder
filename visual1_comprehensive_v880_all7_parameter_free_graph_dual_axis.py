"""Visualize the v8.8.0 seven-method, fourteen-scenario comparison.

Four diagnosis-score bars use the left axis, mean token consumption uses the
right axis, and seed-level dots plus 95% Student-t confidence intervals expose
variation across the ten repeated seeds. Rule methods report token usage as N/A.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(
    SCRIPT_DIR, "outputs", "comprehensive_v880_all_7methods_14scenarios_10seeds"
)

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

METHOD_ORDER = [
    "f2e_events_graph_llm",
    "f2e_events_llm",
    "threshold_matrix_llm",
    "raw_matrix_summary_llm",
    "threshold_matrix_rules",
    "f2e_events_rules",
    "raw_field_image_vlm",
]

METHOD_LABELS = {
    "f2e_events_graph_llm": "Adaptive graph + LLM",
    "f2e_events_llm": "F2E + LLM",
    "threshold_matrix_llm": "Threshold + LLM",
    "raw_matrix_summary_llm": "Raw matrix + LLM",
    "threshold_matrix_rules": "Threshold + rules",
    "f2e_events_rules": "F2E + rules",
    "raw_field_image_vlm": "Raw field image + VLM",
}

METRICS = [
    "accident_classification_accuracy",
    "macro_f1",
    "hard_reasoning_alignment_score",
    "ood_template_rejection_success",
]

METRIC_LABELS = {
    "accident_classification_accuracy": "Acc",
    "macro_f1": "MacroF1",
    "hard_reasoning_alignment_score": "Hard reason",
    "ood_template_rejection_success": "OOD reject",
}

SCORE_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
TOKEN_COLOR = "#6F4E9C"
CONFIDENCE_LEVEL = 0.95
EXPECTED_SEEDS = 10
EXPECTED_SCENARIOS_PER_SEED = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the v8.8.0 seven-method scores and token consumption."
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--fig-dir",
        default=None,
        help="Defaults to <out-dir>/figures_v880_all7_dual_axis.",
    )
    return parser.parse_args()


def safe_output_path(path: str) -> str:
    """Use a timestamped path if an existing output is open in another app."""
    if not os.path.exists(path):
        return path
    try:
        with open(path, "a+b"):
            pass
        return path
    except PermissionError:
        stem, ext = os.path.splitext(path)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = f"{stem}_{stamp}{ext}"
        print(f"Warning: output is locked; using {fallback}")
        return fallback


def true_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def macro_f1(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    scores: List[float] = []
    pairs = list(zip(y_true, y_pred))
    for label in ACCIDENT_TYPES:
        tp = sum(actual == label and predicted == label for actual, predicted in pairs)
        fp = sum(actual != label and predicted == label for actual, predicted in pairs)
        fn = sum(actual == label and predicted != label for actual, predicted in pairs)
        if tp + fp + fn == 0:
            continue
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else np.nan


def seed_metric(seed_rows: pd.DataFrame, metric: str) -> float:
    if metric == "accident_classification_accuracy":
        return float(pd.to_numeric(seed_rows["correct"], errors="coerce").mean())
    if metric == "macro_f1":
        return macro_f1(seed_rows["ground_truth"], seed_rows["prediction"])
    if metric == "hard_reasoning_alignment_score":
        mask = (
            seed_rows["hard_family"].fillna("standard").ne("standard")
            | true_mask(seed_rows["ambiguous"])
            | true_mask(seed_rows["expected_review"])
        )
        return float(
            pd.to_numeric(
                seed_rows.loc[mask, "reasoning_alignment_score"], errors="coerce"
            ).mean()
        )
    if metric == "ood_template_rejection_success":
        subset = seed_rows[seed_rows["diagnostic_goal"] == "template_rejection"]
        return float(
            pd.to_numeric(subset["template_rejection_success"], errors="coerce").mean()
        )
    if metric == "mean_total_tokens":
        subset = seed_rows[
            seed_rows["token_count_source"].fillna("").ne("not_applicable_rules")
        ]
        return float(pd.to_numeric(subset["total_tokens"], errors="coerce").mean())
    raise ValueError(f"Unsupported metric: {metric}")


def build_seed_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = METRICS + ["mean_total_tokens"]
    for (method, seed_index), seed_rows in records.groupby(
        ["method", "seed_index"], sort=True
    ):
        row = {
            "method": method,
            "seed_index": int(seed_index),
            "seed": int(seed_rows["seed"].iloc[0]),
            "n_scenarios": int(len(seed_rows)),
        }
        for metric in metrics:
            row[metric] = seed_metric(seed_rows, metric)
        rows.append(row)
    return pd.DataFrame(rows)


def mean_ci95(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), 0.0
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    critical = float(
        student_t.ppf((1.0 + CONFIDENCE_LEVEL) / 2.0, len(values) - 1)
    )
    return float(np.mean(values)), critical * standard_error


def darker(color: str, factor: float = 0.58) -> Tuple[float, float, float]:
    return tuple(channel * factor for channel in to_rgb(color))


def point_x_positions(center: float, count: int, width: float) -> np.ndarray:
    if count <= 1:
        return np.asarray([center])
    return center + np.linspace(-0.32 * width, 0.32 * width, count)


def shown_points(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def validate_inputs(summary: pd.DataFrame, records: pd.DataFrame) -> None:
    required_summary = {"method", "mean_total_tokens", *METRICS}
    required_records = {
        "ambiguous",
        "diagnostic_goal",
        "expected_review",
        "hard_family",
        "method",
        "reasoning_alignment_score",
        "seed_index",
        "seed",
        "ground_truth",
        "prediction",
        "correct",
        "template_rejection_success",
        "total_tokens",
        "token_count_source",
    }
    missing_summary = sorted(required_summary - set(summary.columns))
    missing_records = sorted(required_records - set(records.columns))
    if missing_summary:
        raise ValueError("Summary CSV is missing: " + ", ".join(missing_summary))
    if missing_records:
        raise ValueError("Records CSV is missing: " + ", ".join(missing_records))
    summary_methods = summary["method"].astype(str)
    available = set(summary_methods)
    expected = set(METHOD_ORDER)
    missing_methods = sorted(expected - available)
    unexpected_methods = sorted(available - expected)
    if missing_methods:
        raise ValueError("Summary data is missing methods: " + ", ".join(missing_methods))
    if unexpected_methods:
        raise ValueError(
            "Summary data contains unexpected methods: " + ", ".join(unexpected_methods)
        )
    if summary_methods.duplicated().any():
        duplicates = sorted(summary_methods[summary_methods.duplicated()].unique())
        raise ValueError("Summary data duplicates methods: " + ", ".join(duplicates))

    record_methods = set(records["method"].astype(str))
    if record_methods != expected:
        missing = sorted(expected - record_methods)
        unexpected = sorted(record_methods - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError("Record method mismatch: " + "; ".join(details))

    seed_counts = records.groupby("method")["seed_index"].nunique()
    invalid_seed_counts = seed_counts[seed_counts.ne(EXPECTED_SEEDS)]
    if not invalid_seed_counts.empty:
        details = ", ".join(
            f"{method}={count}" for method, count in invalid_seed_counts.items()
        )
        raise ValueError(
            f"Expected {EXPECTED_SEEDS} seeds per method; found {details}"
        )

    scenario_counts = records.groupby(["method", "seed_index"]).size()
    invalid_scenario_counts = scenario_counts[
        scenario_counts.ne(EXPECTED_SCENARIOS_PER_SEED)
    ]
    if not invalid_scenario_counts.empty:
        details = ", ".join(
            f"{method}/seed{seed_index}={count}"
            for (method, seed_index), count in invalid_scenario_counts.items()
        )
        raise ValueError(
            f"Expected {EXPECTED_SCENARIOS_PER_SEED} scenarios per seed; found {details}"
        )


def main() -> None:
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    fig_dir = os.path.abspath(
        args.fig_dir
        or os.path.join(out_dir, "figures_v880_all7_dual_axis")
    )
    os.makedirs(fig_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "layer2_diagnosis_summary.csv")
    records_path = os.path.join(out_dir, "layer2_diagnosis_records.csv")
    summary = pd.read_csv(summary_path)
    records = pd.read_csv(records_path)
    validate_inputs(summary, records)

    plot_df = summary.set_index("method").loc[METHOD_ORDER].reset_index()
    seed_summary = build_seed_summary(records)
    seed_summary_path = safe_output_path(
        os.path.join(fig_dir, "layer2_per_seed_summary_derived.csv")
    )
    seed_summary.to_csv(seed_summary_path, index=False)

    score_values: Dict[Tuple[str, str], np.ndarray] = {}
    score_stats: Dict[Tuple[str, str], Tuple[float, float]] = {}
    token_values: Dict[str, np.ndarray] = {}
    token_stats: Dict[str, Tuple[float, float]] = {}
    for method in METHOD_ORDER:
        method_seed_rows = seed_summary[seed_summary["method"] == method]
        for metric in METRICS:
            values = pd.to_numeric(method_seed_rows[metric], errors="coerce").dropna().to_numpy()
            score_values[(method, metric)] = values * 100.0
            mean, ci = mean_ci95(values)
            score_stats[(method, metric)] = (mean * 100.0, ci * 100.0)
        values = pd.to_numeric(
            method_seed_rows["mean_total_tokens"], errors="coerce"
        ).dropna().to_numpy()
        token_values[method] = values
        token_stats[method] = mean_ci95(values)

    figure, score_axis = plt.subplots(figsize=(19.5, 6.8))
    x = np.arange(len(METHOD_ORDER), dtype=float) * 1.35
    bar_width = 0.15
    group_count = len(METRICS) + 1

    score_handles = []
    for metric_index, (metric, color) in enumerate(zip(METRICS, SCORE_COLORS)):
        offset = (metric_index - (group_count - 1) / 2.0) * bar_width
        means = np.asarray([score_stats[(m, metric)][0] for m in METHOD_ORDER])
        cis = np.asarray([score_stats[(m, metric)][1] for m in METHOD_ORDER])
        bars = score_axis.bar(
            x + offset,
            means,
            width=bar_width,
            color=color,
            label=METRIC_LABELS[metric],
            zorder=2,
        )
        score_handles.append(bars)
        dark = darker(color)
        for center, method in zip(x + offset, METHOD_ORDER):
            values = score_values[(method, metric)]
            score_axis.scatter(
                point_x_positions(center, len(values), bar_width),
                shown_points(values),
                s=34,
                color=dark,
                edgecolors="white",
                linewidths=0.7,
                zorder=5,
            )
        score_axis.errorbar(
            x + offset,
            means,
            yerr=cis,
            fmt="none",
            ecolor=dark,
            elinewidth=1.6,
            capsize=3.8,
            capthick=1.6,
            zorder=4,
        )
        labels = score_axis.bar_label(
            bars,
            labels=[f"{value:.1f}" for value in means],
            padding=6,
            rotation=90,
            fontsize=14,
            fontweight="bold",
        )
        for label in labels:
            label.set_zorder(10)

    token_axis = score_axis.twinx()
    token_offset = (len(METRICS) - (group_count - 1) / 2.0) * bar_width
    token_means = np.asarray([token_stats[m][0] for m in METHOD_ORDER])
    token_cis = np.asarray([token_stats[m][1] for m in METHOD_ORDER])
    finite_tokens = np.isfinite(token_means)
    shown_token_means = np.where(finite_tokens, token_means, 0.0)
    token_bars = token_axis.bar(
        x + token_offset,
        shown_token_means,
        width=bar_width,
        color=TOKEN_COLOR,
        edgecolor=TOKEN_COLOR,
        alpha=0.35,
        hatch="//",
        label="Tokens",
        zorder=1,
    )
    dark_token = darker(TOKEN_COLOR)
    for center, method, valid in zip(x + token_offset, METHOD_ORDER, finite_tokens):
        if not valid:
            continue
        values = token_values[method]
        token_axis.scatter(
            point_x_positions(center, len(values), bar_width),
            values,
            s=34,
            color=dark_token,
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
        )
    token_axis.errorbar(
        (x + token_offset)[finite_tokens],
        token_means[finite_tokens],
        yerr=token_cis[finite_tokens],
        fmt="none",
        ecolor=dark_token,
        elinewidth=1.6,
        capsize=3.8,
        capthick=1.6,
        zorder=4,
    )
    token_labels = [f"{value:.0f}" if valid else "N/A" for value, valid in zip(token_means, finite_tokens)]
    labels = token_axis.bar_label(
        token_bars,
        labels=token_labels,
        padding=6,
        rotation=90,
        fontsize=14,
        fontweight="bold",
        color=TOKEN_COLOR,
    )
    for label in labels:
        label.set_zorder(10)

    score_axis.set_xticks(x)
    score_axis.set_xticklabels(
        [METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=14, fontweight="bold"
    )
    score_axis.set_ylabel("Score (%)", fontsize=20, fontweight="bold")
    score_axis.set_ylim(-3, 118)
    score_axis.grid(axis="y", linestyle="--", alpha=0.35)
    score_axis.margins(x=0.04)
    score_axis.tick_params(axis="y", labelsize=15)
    token_axis.set_ylabel("Tokens", fontsize=20, fontweight="bold", color=TOKEN_COLOR)
    token_axis.tick_params(axis="y", labelsize=15, colors=TOKEN_COLOR, width=2.0)
    token_axis.spines["right"].set_color(TOKEN_COLOR)
    token_axis.spines["right"].set_linewidth(2.0)
    for axis in (score_axis, token_axis):
        for tick in axis.get_yticklabels():
            tick.set_fontweight("bold")
    max_tokens = np.nanmax(token_means)
    if np.isfinite(max_tokens) and max_tokens > 0:
        token_axis.set_ylim(0, max_tokens * 1.18)

    handles1, labels1 = score_axis.get_legend_handles_labels()
    handles2, labels2 = token_axis.get_legend_handles_labels()
    score_axis.legend(
        handles1 + handles2,
        labels1 + labels2,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        prop={"size": 18, "weight": "bold"},
        frameon=False,
        borderaxespad=0.0,
        columnspacing=1.0,
    )
    figure.tight_layout(pad=0.15)

    png_path = safe_output_path(
        os.path.join(fig_dir, "layer2_v880_all7_scores_tokens_dual_axis.png")
    )
    pdf_path = safe_output_path(
        os.path.join(fig_dir, "layer2_v880_all7_scores_tokens_dual_axis.pdf")
    )
    figure.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(figure)

    token_table = pd.DataFrame(
        {
            "Method": [METHOD_LABELS[m] for m in METHOD_ORDER],
            "Mean tokens": [
                round(token_stats[m][0], 3) if np.isfinite(token_stats[m][0]) else "N/A"
                for m in METHOD_ORDER
            ],
            "95% CI half-width": [
                round(token_stats[m][1], 3) if np.isfinite(token_stats[m][1]) else "N/A"
                for m in METHOD_ORDER
            ],
        }
    )
    token_csv_path = safe_output_path(os.path.join(fig_dir, "method_token_consumption.csv"))
    token_table.to_csv(token_csv_path, index=False)

    print(f"Validated {len(METHOD_ORDER)} methods, {records['seed_index'].nunique()} seeds, and {len(records)} records.")
    print(f"Saved per-seed summary: {seed_summary_path}")
    print(f"Saved figure PNG: {png_path}")
    print(f"Saved figure PDF: {pdf_path}")
    print(f"Saved token table: {token_csv_path}")


if __name__ == "__main__":
    main()
