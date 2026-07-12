import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib

# This script writes PNG/PDF files only; avoid a dependency on a desktop GUI
# backend (such as Tk) when it is run from automation or a headless shell.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.stats import t as student_t

# =========================
# Config
# =========================
OUT_DIR = "outputs/f2e_graph_paper10_v831_full"
SUMMARY_CSV = os.path.join(OUT_DIR, "layer2_diagnosis_summary.csv")
RECORDS_CSV = os.path.join(OUT_DIR, "layer2_diagnosis_records.csv")
FIG_DIR = os.path.join(OUT_DIR, "figures")
DERIVED_SEED_SUMMARY_CSV = os.path.join(
    FIG_DIR, "layer2_diagnosis_per_seed_summary_derived.csv"
)
os.makedirs(FIG_DIR, exist_ok=True)

# Error bars use the 95% Student-t confidence interval of the mean across the
# 10 seed-level summaries (each seed first aggregates its 14 scenarios).
SEED_ERROR_DDOF = 1
CONFIDENCE_LEVEL = 0.95
ZERO_DOT_DISPLAY_VALUE = 0.15  # Display-only offset; source values remain 0.

# =========================
# Font config for paper figure
# =========================
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 15
BAR_LABEL_FONTSIZE = 15
LEGEND_FONTSIZE = 20
SHOW_VALUE_LABELS = True  # Seed dots already show the repeated-run values.
BAR_VALUE_LABEL_ZORDER = 1000

# Horizontal layout config
# Increase FIG_WIDTH and X_GAP_SCALE if x-axis labels still overlap.
FIG_WIDTH = 19.5
FIG_HEIGHT = 6.8
X_GAP_SCALE = 1.35


def set_tick_labels_bold(ax):
    """Make tick labels larger and bold without changing axis data or layout logic."""
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

def safe_output_path(path):
    """
    Return a writable output path.

    On Windows, PermissionError often happens when an existing CSV/PNG/PDF is
    open in Excel, WPS, Acrobat, or an image viewer. If the target file is
    locked, save to a timestamped filename instead of crashing.
    """
    if not os.path.exists(path):
        return path

    try:
        with open(path, "a+b"):
            pass
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_path = f"{base}_{timestamp}{ext}"
        print(f"Warning: output file is locked, saving to: {new_path}")
        return new_path


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


def macro_f1(y_true, y_pred, labels=ACCIDENT_TYPES):
    """Match the comparison script's strict multiclass macro-F1 definition."""
    scores = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        if tp + fp + fn == 0:
            continue
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else np.nan


def true_mask(series):
    """Parse boolean columns after pandas has loaded the record CSV."""
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def seed_metric_value(seed_rows, metric):
    """Compute one metric from the 14 scenario records belonging to one seed."""
    if metric == "accident_classification_accuracy":
        return pd.to_numeric(seed_rows["correct"], errors="coerce").mean()
    if metric == "macro_f1":
        return macro_f1(seed_rows["ground_truth"], seed_rows["prediction"])
    if metric == "hard_reasoning_alignment_score":
        hard_mask = (
            seed_rows["hard_family"].fillna("standard").ne("standard")
            | true_mask(seed_rows["ambiguous"])
            | true_mask(seed_rows["expected_review"])
        )
        return pd.to_numeric(
            seed_rows.loc[hard_mask, "reasoning_alignment_score"], errors="coerce"
        ).mean()
    if metric == "ood_template_rejection_success":
        ood_rows = seed_rows[seed_rows["diagnostic_goal"] == "template_rejection"]
        return pd.to_numeric(
            ood_rows["template_rejection_success"], errors="coerce"
        ).mean()
    if metric == "mean_total_tokens":
        token_rows = seed_rows[
            seed_rows["token_count_source"].fillna("").ne("not_applicable_rules")
        ]
        return pd.to_numeric(token_rows["total_tokens"], errors="coerce").mean()
    raise ValueError(f"Unsupported seed metric: {metric}")


def build_seed_summary(records, metrics):
    """Create a non-destructive method x seed table from the raw trial records."""
    rows = []
    for (method, seed_index), seed_rows in records.groupby(["method", "seed_index"], sort=True):
        row = {
            "method": method,
            "seed_index": int(seed_index),
            "seed": int(seed_rows["seed"].iloc[0]),
            "n_scenarios": len(seed_rows),
            "hard_reasoning_n": int((
                seed_rows["hard_family"].fillna("standard").ne("standard")
                | true_mask(seed_rows["ambiguous"])
                | true_mask(seed_rows["expected_review"])
            ).sum()),
            "ood_template_rejection_n": int(
                seed_rows["diagnostic_goal"].eq("template_rejection").sum()
            ),
        }
        for metric in metrics:
            row[metric] = seed_metric_value(seed_rows, metric)
        rows.append(row)
    return pd.DataFrame(rows)


def in_bar_positions(center, count, width):
    """Spread repeated-run points horizontally without changing their y values."""
    if count <= 1:
        return np.asarray([center])
    return center + np.linspace(-0.32 * width, 0.32 * width, count)


def mean_and_ci95(values):
    """Return mean and the two-sided 95% Student-t CI half-width."""
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(np.mean(values)), 0.0
    sd = np.std(values, ddof=SEED_ERROR_DDOF)
    standard_error = sd / np.sqrt(len(values))
    t_critical = student_t.ppf((1.0 + CONFIDENCE_LEVEL) / 2.0, len(values) - 1)
    return float(np.mean(values)), float(t_critical * standard_error)


def display_scatter_values(values):
    """Lift only zero-valued dots slightly above the baseline for visibility."""
    shown = np.asarray(values, dtype=float).copy()
    shown[np.isclose(shown, 0.0)] = ZERO_DOT_DISPLAY_VALUE
    return shown


def darker(color, factor=0.58):
    """Return a darker opaque version of a Matplotlib bar color."""
    return tuple(channel * factor for channel in to_rgb(color))


# =========================
# Load summary
# =========================
df = pd.read_csv(SUMMARY_CSV)
records = None
if os.path.exists(RECORDS_CSV):
    candidate_records = pd.read_csv(RECORDS_CSV)
    record_columns = {"method", "seed_index", "ground_truth", "prediction"}
    if record_columns.issubset(candidate_records.columns):
        records = candidate_records
    else:
        print("Warning: record CSV is missing seed-level columns; error bars are skipped.")
else:
    print("Warning: layer2_diagnosis_records.csv was not found; error bars are skipped.")

# =========================
# Methods to show, in display order
# =========================
method_order = [
    "f2e_events_graph_llm",
    "f2e_events_llm_f2e_prompt",
    "threshold_events_llm",
    "raw_matrix_summary_llm",
    "threshold_events_rules",
    "f2e_events_rules",
    "raw_field_image_vlm",
]

method_labels = {
    "f2e_events_graph_llm": "F2E graph + LLM",
    "f2e_events_llm_f2e_prompt": "F2E + LLM",
    "threshold_events_llm": "Threshold + LLM",
    "raw_matrix_summary_llm": "Raw matrix + LLM",
    "threshold_events_rules": "Threshold + rules",
    "f2e_events_rules": "F2E + rules",
    "raw_field_image_vlm": "Raw field image + VLM",
}

# =========================
# Metrics to show
# Keep only: Acc | MacroF1 | Hard reason | OOD reject
# =========================
metrics = [
    "accident_classification_accuracy",
    "macro_f1",
    "hard_reasoning_alignment_score",
    "ood_template_rejection_success",
]

metric_labels = {
    "accident_classification_accuracy": "Acc",
    "macro_f1": "MacroF1",
    "hard_reasoning_alignment_score": "Hard reason",
    "ood_template_rejection_success": "OOD reject",
}

# =========================
# Validate required columns
# =========================
required_cols = ["method"] + metrics
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(
        "Missing required columns in summary CSV: " + ", ".join(missing_cols)
    )

# Keep methods and order
plot_df = df[df["method"].isin(method_order)].copy()
plot_df["method"] = pd.Categorical(plot_df["method"], categories=method_order, ordered=True)
plot_df = plot_df.sort_values("method")

missing_methods = [m for m in method_order if m not in set(plot_df["method"].astype(str))]
if missing_methods:
    print("Warning: these methods are not found in the summary CSV and will be skipped:")
    for m in missing_methods:
        print(f"  - {m}")

# Convert scores to percentages for plotting
for m in metrics:
    plot_df[m] = plot_df[m].astype(float) * 100.0

# Derive (without modifying) a method x seed summary.  Each row comes from the
# 14 original scenario records for one seed, so the 80-row table is auditable
# against the untouched 1,120-row raw record CSV.
seed_stats = {}
seed_values = {}
seed_summary = None
if records is not None:
    seed_summary = build_seed_summary(records, metrics + ["mean_total_tokens"])
    derived_seed_path = safe_output_path(DERIVED_SEED_SUMMARY_CSV)
    seed_summary.to_csv(derived_seed_path, index=False)
    print(f"Saved non-destructive per-seed summary to: {derived_seed_path}")

    for method in plot_df["method"].astype(str):
        for metric in metrics:
            values = pd.to_numeric(
                seed_summary.loc[seed_summary["method"] == method, metric], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            seed_values[(method, metric)] = values
            mean, ci95_half_width = mean_and_ci95(values)
            seed_stats[(method, metric)] = (mean * 100.0, ci95_half_width * 100.0)
    for metric in metrics:
        seed_means = plot_df["method"].astype(str).map(
            lambda method: seed_stats[(method, metric)][0]
        )
        plot_df[metric] = seed_means.where(seed_means.notna(), plot_df[metric])
    n_seeds = records["seed_index"].nunique()
    scenarios_per_seed = [int(v) for v in sorted(seed_summary["n_scenarios"].unique())]
    print(
        f"Using mean +/- 95% CI across {n_seeds} seed-level values "
        f"({scenarios_per_seed} scenarios per seed) for score bars."
    )

# =========================
# Detect token column for the secondary y-axis
# =========================
# The CSV may use different token column names in different experiments.
# This block tries common names first; if none are found, it falls back to
# any column whose name contains "token".
token_col_candidates = [
    "tokens",
    "Tokens",
    "total_tokens",
    "avg_tokens",
    "mean_tokens",
    "median_tokens",
    "token_count",
    "avg_total_tokens",
    "mean_total_tokens",
    "total_token_count",
]

token_col = next((c for c in token_col_candidates if c in df.columns), None)
if token_col is None:
    token_like_cols = [c for c in df.columns if "token" in c.lower()]
    token_col = token_like_cols[0] if token_like_cols else None

has_token_metric = token_col is not None
if has_token_metric:
    # Use one token value per method and align it with plot_df's method order.
    token_map = df.drop_duplicates("method").set_index("method")[token_col]
    plot_df[token_col] = plot_df["method"].astype(str).map(token_map)
    plot_df[token_col] = pd.to_numeric(plot_df[token_col], errors="coerce")
    has_token_metric = plot_df[token_col].notna().any()

token_seed_stats = {}
if has_token_metric and records is not None and token_col == "mean_total_tokens":
    for method in plot_df["method"].astype(str):
        token_values_by_seed = pd.to_numeric(
            seed_summary.loc[
                seed_summary["method"] == method, "mean_total_tokens"
            ],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        seed_values[(method, "mean_total_tokens")] = token_values_by_seed
        token_seed_stats[method] = mean_and_ci95(token_values_by_seed)
    token_seed_means = plot_df["method"].astype(str).map(
        lambda method: token_seed_stats[method][0]
    )
    plot_df[token_col] = token_seed_means.where(token_seed_means.notna(), plot_df[token_col])

if token_col is None:
    print("Warning: no token column found in the summary CSV. Token curve is not added to the performance figure.")
elif not has_token_metric:
    print(f"Warning: detected token column '{token_col}', but no numeric token values are available. Token curve is not added.")
else:
    print(f"Detected token column for secondary y-axis: {token_col}")

# =========================
# Plot diagnosis metrics + token consumption with dual y-axes
# =========================
fig, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

x = np.arange(len(plot_df)) * X_GAP_SCALE

# If tokens are available, draw five bars per method:
#   4 score bars on the left y-axis + 1 token bar on the right y-axis.
# If tokens are not available, keep the original four-bar layout.
num_bar_groups = len(metrics) + (1 if has_token_metric else 0)
bar_width = 0.15 if has_token_metric else 0.17

for i, metric in enumerate(metrics):
    offset = (i - (num_bar_groups - 1) / 2) * bar_width

    bars = ax1.bar(
        x + offset,
        plot_df[metric],
        width=bar_width,
        label=metric_labels[metric],
    )

    # One high-contrast dot per seed-level summary (14 scenarios per dot),
    # plus a symmetric T-shaped 95% CI error bar.  The bar top
    # itself is the mean, so no separate mean dot obscures the repeated runs.
    if seed_stats:
        bar_positions = x + offset
        seed_means = np.asarray(
            [seed_stats[(method, metric)][0] for method in plot_df["method"].astype(str)]
        )
        seed_ci95_half_widths = np.asarray(
            [seed_stats[(method, metric)][1] for method in plot_df["method"].astype(str)]
        )
        dark_color = darker(bars[0].get_facecolor())
        for bar_position, method in zip(bar_positions, plot_df["method"].astype(str)):
            values = seed_values[(method, metric)] * 100.0
            ax1.scatter(
                in_bar_positions(bar_position, len(values), bar_width),
                display_scatter_values(values),
                s=36,
                color=dark_color,
                edgecolors="white",
                linewidths=0.75,
                alpha=0.95,
                zorder=6,
            )
        ax1.errorbar(
            bar_positions,
            seed_means,
            yerr=seed_ci95_half_widths,
            fmt="none",
            ecolor=dark_color,
            elinewidth=1.7,
            capsize=3.8,
            capthick=1.7,
            zorder=5,
        )

    # Add numeric labels at the top of bars
    if SHOW_VALUE_LABELS:
        score_labels = ax1.bar_label(
            bars,
            labels=[f"{v:.1f}" for v in plot_df[metric]],
            padding=6,
            fontsize=BAR_LABEL_FONTSIZE,
            fontweight="bold",
            rotation=90,
        )
        for label in score_labels:
            label.set_zorder(BAR_VALUE_LABEL_ZORDER)

ax1.set_xticks(x)
ax1.set_xticklabels(
    [method_labels[m] for m in plot_df["method"].astype(str)],
    rotation=0,
    ha="center",
    fontsize=TICK_LABEL_FONTSIZE,
    fontweight="bold",
)

ax1.set_ylabel("Score (%)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
ax1.set_ylim(0, 118)
ax1.margins(x=0.04)
ax1.grid(axis="y", linestyle="--", alpha=0.35)
set_tick_labels_bold(ax1)

# Secondary y-axis for token consumption.
if has_token_metric:
    ax2 = ax1.twinx()
    token_raw_values = plot_df[token_col].copy()
    token_values = pd.to_numeric(token_raw_values, errors="coerce").fillna(0).to_numpy()

    token_offset = (len(metrics) - (num_bar_groups - 1) / 2) * bar_width

    token_color = "#1f77b4"

    token_bars = ax2.bar(
        x + token_offset,
        token_values,
        width=bar_width,
        label="Tokens",
        color=token_color,
        edgecolor=token_color,
        alpha=0.35,
        hatch="//",
    )

    if token_seed_stats:
        valid_tokens = token_raw_values.notna().to_numpy()
        token_ci95_half_widths = np.asarray(
            [token_seed_stats[method][1] for method in plot_df["method"].astype(str)]
        )
        dark_token_color = darker(token_color)
        token_positions = x + token_offset
        for bar_position, method, valid in zip(
            token_positions, plot_df["method"].astype(str), valid_tokens
        ):
            if not valid:
                continue
            values = seed_values[(method, "mean_total_tokens")]
            ax2.scatter(
                in_bar_positions(bar_position, len(values), bar_width),
                display_scatter_values(values),
                s=36,
                color=dark_token_color,
                edgecolors="white",
                linewidths=0.75,
                alpha=0.95,
                zorder=6,
            )
        ax2.errorbar(
            token_positions[valid_tokens],
            token_values[valid_tokens],
            yerr=token_ci95_half_widths[valid_tokens],
            fmt="none",
            ecolor=dark_token_color,
            elinewidth=1.7,
            capsize=3.8,
            capthick=1.7,
            zorder=5,
        )

    ax2.set_ylabel(
        "Tokens",
        fontsize=AXIS_LABEL_FONTSIZE,
        fontweight="bold",
        color=token_color,
    )

    # 右侧 y 轴刻度字体：放大、加粗、改色
    ax2.tick_params(
        axis="y",
        labelsize=TICK_LABEL_FONTSIZE,
        colors=token_color,
        width=2.0,
        length=6,
    )

    for label in ax2.get_yticklabels():
        label.set_fontweight("bold")

    # 右侧 y 轴轴线：加粗、改色
    ax2.spines["right"].set_color(token_color)
    ax2.spines["right"].set_linewidth(2.0)

    # Add headroom so token labels do not touch the top border.
    token_max = np.nanmax(token_values)
    if np.isfinite(token_max) and token_max > 0:
        ax2.set_ylim(0, token_max * 1.18)

    set_tick_labels_bold(ax2)

    # Add numeric token labels.
    token_labels = []
    for v in plot_df[token_col]:
        if pd.isna(v):
            token_labels.append("None")
        elif float(v) == 0:
            token_labels.append("0")
        else:
            token_labels.append(f"{float(v):.0f}")
    if SHOW_VALUE_LABELS:
        token_value_labels = ax2.bar_label(
            token_bars,
            labels=token_labels,
            padding=6,
            fontsize=BAR_LABEL_FONTSIZE,
            fontweight="bold",
            color=token_color,
            rotation=90,
        )
        for label in token_value_labels:
            label.set_zorder(BAR_VALUE_LABEL_ZORDER)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend_handles = handles1 + handles2
    legend_labels = labels1 + labels2
    legend_ncol = 5
else:
    legend_handles, legend_labels = ax1.get_legend_handles_labels()
    legend_ncol = 4

# ax1.set_title(
#     "Layer 2 diagnosis performance and token consumption\n"
#     "Acc, MacroF1, hard reasoning, OOD rejection, and tokens across all methods."
# )

ax1.legend(
    legend_handles,
    legend_labels,
    ncol=legend_ncol,
    loc="lower center",
    prop={"size": LEGEND_FONTSIZE, "weight": "bold"},
    bbox_to_anchor=(0.5, 1.005),
    borderaxespad=0.0,
    borderpad=0.0,
    handletextpad=0.5,
    columnspacing=1.0,
    labelspacing=0.1,
    frameon=False,
)

plt.tight_layout(pad=0.15)
plt.tight_layout()

# Save performance figure
png_path = safe_output_path(os.path.join(FIG_DIR, "layer2_diagnosis_performance_all_methods_labeled_token_bar.png"))
pdf_path = safe_output_path(os.path.join(FIG_DIR, "layer2_diagnosis_performance_all_methods_labeled_token_bar.pdf"))

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.01
)

plt.savefig(
    pdf_path,
    bbox_inches="tight",
    pad_inches=0.01
)
plt.close()

print(f"Saved performance PNG to: {png_path}")
print(f"Saved performance PDF to: {pdf_path}")

# =========================
# Token consumption table
# =========================
# The CSV may use different token column names in different experiments.
# This block tries common names first; if none are found, it falls back to
# any column whose name contains "token".
token_col_candidates = [
    "tokens",
    "Tokens",
    "total_tokens",
    "avg_tokens",
    "mean_tokens",
    "median_tokens",
    "token_count",
    "avg_total_tokens",
    "mean_total_tokens",
    "total_token_count",
]

token_col = next((c for c in token_col_candidates if c in df.columns), None)
if token_col is None:
    token_like_cols = [c for c in df.columns if "token" in c.lower()]
    token_col = token_like_cols[0] if token_like_cols else None

if token_col is None:
    print("Warning: no token column found in the summary CSV. Token table is not generated.")
else:
    token_df = df[df["method"].isin(method_order)][["method", token_col]].copy()
    token_df["method"] = pd.Categorical(token_df["method"], categories=method_order, ordered=True)
    token_df = token_df.sort_values("method")
    token_df["Method"] = token_df["method"].astype(str).map(method_labels)
    token_df = token_df.rename(columns={token_col: "Tokens"})
    token_df = token_df[["Method", "Tokens"]]

    # Keep rule-based methods readable when token value is missing.
    token_df["Tokens"] = token_df["Tokens"].where(token_df["Tokens"].notna(), "N/A")

    token_csv_path = safe_output_path(os.path.join(FIG_DIR, "method_token_consumption.csv"))
    token_png_path = safe_output_path(os.path.join(FIG_DIR, "method_token_consumption_table.png"))

    token_df.to_csv(token_csv_path, index=False)

    # Save token table as an image for paper/report use.
    fig_h = max(2.4, 0.42 * len(token_df) + 0.9)
    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=token_df.values,
        colLabels=token_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.25)

    ax.set_title("Token consumption by method", pad=12)
    plt.tight_layout()
    plt.savefig(token_png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Detected token column: {token_col}")
    print(f"Saved token CSV to: {token_csv_path}")
    print(f"Saved token table PNG to: {token_png_path}")
