import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
OUT_DIR = "outputs/f2e_graph_paper10_v831_full"
SUMMARY_CSV = os.path.join(OUT_DIR, "layer2_diagnosis_summary.csv")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

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


# =========================
# Load summary
# =========================
df = pd.read_csv(SUMMARY_CSV)

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
    "threshold_events_llm": "Threshold events + LLM",
    "raw_matrix_summary_llm": "Raw matrix + LLM",
    "threshold_events_rules": "Threshold events + rules",
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

if token_col is None:
    print("Warning: no token column found in the summary CSV. Token curve is not added to the performance figure.")
elif not has_token_metric:
    print(f"Warning: detected token column '{token_col}', but no numeric token values are available. Token curve is not added.")
else:
    print(f"Detected token column for secondary y-axis: {token_col}")

# =========================
# Plot diagnosis metrics + token consumption with dual y-axes
# =========================
fig, ax1 = plt.subplots(figsize=(16.0, 6.8))

x = np.arange(len(plot_df))

# If tokens are available, draw five bars per method:
#   4 score bars on the left y-axis + 1 token bar on the right y-axis.
# If tokens are not available, keep the original four-bar layout.
num_bar_groups = len(metrics) + (1 if has_token_metric else 0)
bar_width = 0.15 if has_token_metric else 0.18

for i, metric in enumerate(metrics):
    offset = (i - (num_bar_groups - 1) / 2) * bar_width

    bars = ax1.bar(
        x + offset,
        plot_df[metric],
        width=bar_width,
        label=metric_labels[metric],
    )

    # Add numeric labels at the top of bars
    ax1.bar_label(
        bars,
        labels=[f"{v:.1f}" for v in plot_df[metric]],
        padding=2,
        fontsize=10,
        rotation=90,
    )

ax1.set_xticks(x)
ax1.set_xticklabels(
    [method_labels[m] for m in plot_df["method"].astype(str)],
    rotation=0,
    ha="center",
)

ax1.set_ylabel("Score (%)")
ax1.set_ylim(0, 118)
ax1.grid(axis="y", linestyle="--", alpha=0.35)

# Secondary y-axis for token consumption.
if has_token_metric:
    ax2 = ax1.twinx()
    token_raw_values = plot_df[token_col].copy()
    token_values = pd.to_numeric(token_raw_values, errors="coerce").fillna(0).to_numpy()

    token_offset = (len(metrics) - (num_bar_groups - 1) / 2) * bar_width

    token_bars = ax2.bar(
        x + token_offset,
        token_values,
        width=bar_width,
        label="Tokens",
        alpha=0.35,
        hatch="//",
    )

    ax2.set_ylabel("Tokens")

    # Add headroom so token labels do not touch the top border.
    token_max = np.nanmax(token_values)
    if np.isfinite(token_max) and token_max > 0:
        ax2.set_ylim(0, token_max * 1.18)

    # Add numeric token labels.
    token_labels = []
    for v in plot_df[token_col]:
        if pd.isna(v):
            token_labels.append("None")
        elif float(v) == 0:
            token_labels.append("0")
        else:
            token_labels.append(f"{float(v):.0f}")
    ax2.bar_label(
        token_bars,
        labels=token_labels,
        padding=2,
        fontsize=10,
        rotation=90,
    )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend_handles = handles1 + handles2
    legend_labels = labels1 + labels2
    legend_ncol = 5
else:
    legend_handles, legend_labels = ax1.get_legend_handles_labels()
    legend_ncol = 4

ax1.set_title(
    "Layer 2 diagnosis performance and token consumption\n"
    "Acc, MacroF1, hard reasoning, OOD rejection, and tokens across all methods."
)

ax1.legend(
    legend_handles,
    legend_labels,
    ncol=legend_ncol,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.16),
    frameon=False,
)

plt.tight_layout()

# Save performance figure
png_path = safe_output_path(os.path.join(FIG_DIR, "layer2_diagnosis_performance_all_methods_labeled_token_bar.png"))
pdf_path = safe_output_path(os.path.join(FIG_DIR, "layer2_diagnosis_performance_all_methods_labeled_token_bar.pdf"))

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
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
