import os
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

# =========================
# Load summary
# =========================
df = pd.read_csv(SUMMARY_CSV)

# Methods to show, in display order
method_order = [
    "f2e_events_graph_llm",
    "f2e_events_llm_f2e_prompt",
    "threshold_events_llm",
    "raw_matrix_summary_llm",
    "raw_field_image_vlm",
]

method_labels = {
    "f2e_events_graph_llm": "F2E graph + LLM",
    "f2e_events_llm_f2e_prompt": "F2E + LLM",
    "threshold_events_llm": "Threshold events + LLM",
    "raw_matrix_summary_llm": "Raw matrix + LLM",
    "raw_field_image_vlm": "Raw field image + VLM",
}

# Metrics to show
metrics = [
    "accident_classification_accuracy",
    "macro_f1",
    "hard_reasoning_alignment_score",
    "ood_template_rejection_success",
]

metric_labels = {
    "accident_classification_accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "hard_reasoning_alignment_score": "Hard reasoning",
    "ood_template_rejection_success": "OOD reject",
}

# Keep methods and order
plot_df = df[df["method"].isin(method_order)].copy()
plot_df["method"] = pd.Categorical(plot_df["method"], categories=method_order, ordered=True)
plot_df = plot_df.sort_values("method")

# Convert scores to percentages
for m in metrics:
    plot_df[m] = plot_df[m].astype(float) * 100.0

# =========================
# Plot
# =========================
plt.figure(figsize=(12.5, 6.3))

x = np.arange(len(plot_df))
bar_width = 0.18

for i, metric in enumerate(metrics):
    offset = (i - (len(metrics) - 1) / 2) * bar_width

    bars = plt.bar(
        x + offset,
        plot_df[metric],
        width=bar_width,
        label=metric_labels[metric],
    )

    # Add numeric labels at the top of bars
    plt.bar_label(
        bars,
        labels=[f"{v:.1f}" for v in plot_df[metric]],
        padding=2,
        fontsize=7,
        rotation=90,
    )

plt.xticks(
    x,
    [method_labels[m] for m in plot_df["method"].astype(str)],
    rotation=22,
    ha="right",
)

plt.ylabel("Score (%)")
plt.ylim(0, 118)

plt.title(
    "Layer 2 diagnosis performance\n"
    "Key diagnosis metrics across representative methods in the 10-seed paper-profile run."
)

plt.legend(
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.16),
    frameon=False,
)

plt.grid(axis="y", linestyle="--", alpha=0.35)
plt.tight_layout()

# =========================
# Save
# =========================
png_path = os.path.join(FIG_DIR, "layer2_diagnosis_performance_labeled.png")
pdf_path = os.path.join(FIG_DIR, "layer2_diagnosis_performance_labeled.pdf")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print(f"Saved PNG to: {png_path}")
print(f"Saved PDF to: {pdf_path}")