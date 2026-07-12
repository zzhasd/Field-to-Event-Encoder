import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# =========================
# 1. 读取数据
# =========================
csv_path = "outputs/paper10_core_all_v8_1_2/aggregate_by_variant.csv"
df = pd.read_csv(csv_path)

# =========================
# 2. 指标设置
# =========================
metrics = {
    "F1": "detection_f1",
    "Mean IoU": "mean_iou",
    "Shape Acc.": "shape_accuracy",
    "Trend Acc.": "trend_accuracy",
    "Latency steps ↓": "latency_steps",
}

# 正向指标：越大越好
positive_metrics = ["F1", "Mean IoU", "Shape Acc.", "Trend Acc."]

# 负向指标：越小越好
negative_metrics = ["Latency steps ↓"]

# =========================
# 3. 以 full 为基准计算相对变化
# =========================
plot_df = df[["variant"] + list(metrics.values())].copy()
plot_df = plot_df.rename(columns={v: k for k, v in metrics.items()})

full_row = plot_df.loc[plot_df["variant"] == "full"].iloc[0]

delta = plot_df.copy()

# 正向指标：variant - full，越大说明越好
for m in positive_metrics:
    delta[m] = (plot_df[m] - full_row[m]) / full_row[m] * 100

# 负向指标：full - variant，越大说明越好
for m in negative_metrics:
    delta[m] = (full_row[m] - plot_df[m]) / full_row[m] * 100

# 不显示 full，只显示消融项
delta_plot = delta[delta["variant"] != "full"].copy()

variant_name_map = {
    "ablate_no_bg_ema": "w/o \n BG EMA",
    "ablate_no_cumulative_score": "w/o  \n Cumulative \n Score",
    "ablate_no_hysteresis": "w/o \n Hysteresis",
    "ablate_no_morphology_debounce": "w/o \n Morphology \n Debounce",
    "ablate_no_tracking": "w/o \n Tracking",
}

delta_plot["variant_label"] = (
    delta_plot["variant"]
    .map(variant_name_map)
    .fillna(delta_plot["variant"])
)

heat_raw = delta_plot.set_index("variant_label")[list(metrics.keys())]

# =========================
# 4. 符号保持的对数压缩
# =========================
# signed_log(x) = sign(x) * log10(1 + abs(x))
# 原始数值仍然用于单元格标注，log 后的数值只用于颜色映射
heat_color = np.sign(heat_raw.values) * np.log10(1 + np.abs(heat_raw.values))

# =========================
# 5. 百分比显示格式
# =========================
def format_percent(value, digits=1):
    """普通指标显示方向归一化后的相对性能变化。"""
    rounded = round(float(value), digits)
    if rounded == 0:
        return f"{0:.{digits}f}%"
    return f"{rounded:+.{digits}f}%"


def format_cell(value, metric, digits=1):
    """
    保留原始相对变化计算，仅修改显示标签：
    - 普通指标仍显示方向归一化后的百分比；
    - Latency 将负值解释为 slower、正值解释为 faster，避免误读。
    """
    rounded = round(float(value), digits)

    if metric == "Latency steps ↓":
        if rounded == 0:
            return f"{0:.{digits}f}%"
        if rounded < 0:
            return f"{abs(rounded):.{digits}f}%\nslower"
        return f"{rounded:.{digits}f}%\nfaster"

    return format_percent(value, digits=digits)

# =========================
# 6. 配色：下降蓝色，上升红色
# =========================
cmap = LinearSegmentedColormap.from_list(
    "blue_white_red",
    [
        "#08306b",  # 深蓝：明显下降
        "#6baed6",  # 浅蓝：轻微下降
        "#f7f7f7",  # 接近 0
        "#fcae91",  # 浅红：轻微提升
        "#67000d",  # 深红：明显提升
    ],
    N=256,
)

max_abs_log = np.nanmax(np.abs(heat_color))
norm = TwoSlopeNorm(vmin=-max_abs_log, vcenter=0, vmax=max_abs_log)

# =========================
# 7. 绘图
# =========================
fig, ax = plt.subplots(figsize=(10.8, 4.9))

im = ax.imshow(
    heat_color,
    cmap=cmap,
    norm=norm,
    aspect="auto",
)

# 坐标轴
ax.set_xticks(np.arange(heat_raw.shape[1]))
ax.set_yticks(np.arange(heat_raw.shape[0]))
ax.set_xticklabels(heat_raw.columns, fontsize=15, fontweight="bold")
ax.set_yticklabels(heat_raw.index, fontsize=15, fontweight="bold")

# x 轴标签放在顶部
ax.tick_params(
    top=True,
    bottom=False,
    labeltop=True,
    labelbottom=False,
)

# 网格线
ax.set_xticks(np.arange(-0.5, heat_raw.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, heat_raw.shape[0], 1), minor=True)
ax.grid(
    which="minor",
    color="white",
    linestyle="-",
    linewidth=1.5,
)
ax.tick_params(which="minor", bottom=False, left=False)

# 单元格标注：显示原始百分比，0.0% 不带正负号
for i in range(heat_raw.shape[0]):
    for j in range(heat_raw.shape[1]):
        raw_value = heat_raw.iloc[i, j]
        log_value = heat_color[i, j]

        metric_name = heat_raw.columns[j]
        text = format_cell(raw_value, metric_name, digits=1)
        text_color = "white" if abs(log_value) > max_abs_log * 0.65 else "black"

        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",   # 加粗
            color=text_color,
        )

ax.set_xlabel("")
ax.set_ylabel("")

# =========================
# 8. 色条
# =========================
cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.035,
    pad=0.03,
)

# 色条刻度使用原始百分比，但位置按 log 后的坐标放置
raw_ticks = np.array(
    [-300, -100, -50, -10, 0, 10, 50, 100, 300],
    dtype=float,
)
tick_positions = np.sign(raw_ticks) * np.log10(1 + np.abs(raw_ticks))

mask = (tick_positions >= -max_abs_log) & (tick_positions <= max_abs_log)

cbar.set_ticks(tick_positions[mask])
cbar.set_ticklabels([format_percent(t, digits=0) for t in raw_ticks[mask]])

cbar.set_label(
    "Relative performance change vs. full",
    fontsize=15,
    fontweight="bold",
)

plt.tight_layout()

# =========================
# 9. 保存图片
# =========================
out_path = "ablation_relative_heatmap_log_scaled_zero_clean.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {out_path}")