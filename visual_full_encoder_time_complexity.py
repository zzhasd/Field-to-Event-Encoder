"""Visualize full-encoder scaling with latency boxes and metric error bars.

Left y-axis:
    Distribution of per-case mean encoder latency (ms/frame).
Right y-axis:
    Round-mean Detection F1, Mean IoU,
    with T-shaped +/- one sample-standard-deviation error bars.

Example:
    python visual_full_encoder_time_complexity.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


DEFAULT_INPUT_DIR = Path("outputs") / "full_encoder_time_complexity_paper"
METRICS = [
    "detection_f1",
    "mean_iou",
]
METRIC_LABELS = {
    "detection_f1": "Detection F1",
    "mean_iou": "Mean IoU",
}

# 🎨 现代化调色板
METRIC_COLORS = {
    "detection_f1": "#6366F1",  # 靛蓝 (右轴柱状图)
    "mean_iou": "#10B981",      # 翠绿 (右轴柱状图)
}

# 🔴 核心修改：箱线图全局红色系 (Tailwind Red 调色板)
BOX_FACE = "#FEE2E2"            # 浅红 (Red 100 - 箱子背景色)
BOX_EDGE = "#EF4444"            # 亮红 (Red 500 - 箱子边框、T线)
BOX_MEDIAN = "#7F1D1D"          # 深红 (Red 900 - 中间的中位数横杠)
MEAN_MARKER = "#B91C1C"         # 正红 (Red 600 - 均值点及折线)
AXIS_RED = "#B91C1C"            # 轴红 (Red 700 - 左侧坐标轴文字和线条)

TEXT_COLOR = "#000000"          # 深灰 (全局其他文字)
GRID_COLOR = "#E5E7EB"          # 浅灰 (网格线)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dual-axis visualization of full-encoder grid scaling: "
            "latency boxplots plus metric bars with T error lines."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="experiment directory containing case_round_records.csv and metrics_by_round.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="figure directory; default: <input-dir>/figures",
    )
    parser.add_argument(
        "--basename",
        default="full_encoder_scaling_dual_axis",
        help="output filename stem",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def require_columns(
    frame: pd.DataFrame, required: List[str], source: Path,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {', '.join(missing)}"
        )


def load_results(
    input_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    case_path = input_dir / "case_round_records.csv"
    round_path = input_dir / "metrics_by_round.csv"
    if not case_path.is_file():
        raise FileNotFoundError(case_path)
    if not round_path.is_file():
        raise FileNotFoundError(round_path)

    cases = pd.read_csv(case_path)
    rounds = pd.read_csv(round_path)
    require_columns(
        cases, ["grid_size", "latency_ms_per_frame"], case_path
    )
    require_columns(rounds, ["grid_size", "round", *METRICS], round_path)

    cases["grid_size"] = pd.to_numeric(cases["grid_size"], errors="raise")
    cases["latency_ms_per_frame"] = pd.to_numeric(
        cases["latency_ms_per_frame"], errors="coerce"
    )
    for metric in METRICS:
        rounds[metric] = pd.to_numeric(rounds[metric], errors="coerce")

    grid_sizes = sorted(
        set(cases["grid_size"].dropna().astype(int))
        & set(rounds["grid_size"].dropna().astype(int))
    )
    if not grid_sizes:
        raise ValueError("no common grid sizes found in case and round CSVs")
    return cases, rounds, grid_sizes


def build_summary(
    cases: pd.DataFrame,
    rounds: pd.DataFrame,
    grid_sizes: List[int],
) -> pd.DataFrame:
    rows = []
    for grid_size in grid_sizes:
        latency = cases.loc[
            cases["grid_size"] == grid_size, "latency_ms_per_frame"
        ].dropna()
        item: Dict[str, float] = {
            "grid_size": grid_size,
            "latency_n": int(latency.size),
            "latency_mean_ms": float(latency.mean()),
            "latency_median_ms": float(latency.median()),
            "latency_q1_ms": float(latency.quantile(0.25)),
            "latency_q3_ms": float(latency.quantile(0.75)),
        }
        subset = rounds.loc[rounds["grid_size"] == grid_size]
        item["round_n"] = int(subset["round"].nunique())
        for metric in METRICS:
            values = subset[metric].dropna()
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(item)
    return pd.DataFrame(rows)


def draw_figure(
    cases: pd.DataFrame,
    rounds: pd.DataFrame,
    grid_sizes: List[int],
) -> plt.Figure:
    # 🎨 全局字体与排版设置
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 30,
        "axes.labelsize": 30,        # 轴标签字号保持极大
        "axes.labelweight": "bold",  
        "legend.fontsize": 30,       
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "xtick.labelsize": 30,       
        "ytick.labelsize": 30,       
    })

    fig, latency_ax = plt.subplots(figsize=(14, 7.5))
    score_ax = latency_ax.twinx()

    # 调整图层 Z-order，让主轴(箱线图)浮在副轴(柱状图)之上
    latency_ax.set_zorder(score_ax.get_zorder() + 1)
    latency_ax.patch.set_visible(False)

    # 📐 X 轴相对坐标
    centers = np.arange(len(grid_sizes), dtype=float)
    
    # 箱体宽度缩小为原来的一半
    box_width = 0.18  
    box_positions = centers
    
    bar_width = 0.30
    bar_offsets = [-0.16, 0.16] 

    latency_groups = [
        cases.loc[
            cases["grid_size"] == size, "latency_ms_per_frame"
        ].dropna().to_numpy(dtype=float)
        for size in grid_sizes
    ]
    if any(len(group) == 0 for group in latency_groups):
        raise ValueError("at least one grid size has no latency samples")

    # 📊 1. 绘制浮在最上层的箱线图 (全面应用红色系)
    box = latency_ax.boxplot(
        latency_groups,
        positions=box_positions,
        widths=box_width,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        whis=1.5,
        meanprops={
            "marker": "o",
            "markerfacecolor": MEAN_MARKER,
            "markeredgecolor": "white",
            "markeredgewidth": 1.2,
            "markersize": 8,
            "zorder": 5,
        },
        medianprops={"color": BOX_MEDIAN, "linewidth": 2.5},  # 中位数线加粗深红
        boxprops={"facecolor": BOX_FACE, "edgecolor": BOX_EDGE, "linewidth": 1.5},
        whiskerprops={"color": BOX_EDGE, "linewidth": 1.5},   # 上下边界T线
        capprops={"color": BOX_EDGE, "linewidth": 1.5},       # 上下边界T线
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.85)

    # 📈 2. 绘制连接箱线图均值的红色折线
    latency_means = [np.mean(group) for group in latency_groups]
    latency_ax.plot(
        centers,
        latency_means,
        color=MEAN_MARKER,
        linestyle="--",
        linewidth=2.5,
        alpha=0.85,
        zorder=4,
    )

    metric_summary = rounds.groupby("grid_size")[METRICS].agg(
        ["mean", "std"]
    )
    
    # 📊 3. 绘制位于底层的柱状图
    for metric, offset in zip(METRICS, bar_offsets):
        means = np.array([
            metric_summary.loc[size, (metric, "mean")]
            for size in grid_sizes
        ], dtype=float)
        stds = np.array([
            metric_summary.loc[size, (metric, "std")]
            for size in grid_sizes
        ], dtype=float)
        positions = centers + offset
        
        bars = score_ax.bar(
            positions,
            means,
            width=bar_width,
            color=METRIC_COLORS[metric],
            edgecolor="white",
            linewidth=1.0,
            alpha=0.9,
            zorder=1,
            label=METRIC_LABELS[metric],
        )
        
        score_ax.errorbar(
            positions,
            means,
            yerr=stds,
            fmt="none",
            ecolor="#4B5563",
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
            zorder=2,
        )
        
        for rectangle, mean_val, std_val in zip(bars, means, stds):
            safe_std = std_val if not np.isnan(std_val) else 0.0
            text_y_pos = mean_val + safe_std + 0.003
            
            score_ax.text(
                rectangle.get_x() + rectangle.get_width() / 2,
                text_y_pos,
                f"{mean_val:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=20,
                fontweight="bold",
                color=TEXT_COLOR,
                zorder=3,
            )

    # 📐 坐标轴属性配置
    latency_ax.set_xticks(centers)
    latency_ax.set_xticklabels([str(size) for size in grid_sizes], fontweight="bold")
    
    latency_ax.set_xlabel("Grid Side Length", fontweight="bold", labelpad=12, color=TEXT_COLOR)
    
    # 左侧 Y轴的标签说明、刻度文字、轴线
    latency_ax.set_ylabel("Encoder Latency (ms/frame)", fontweight="bold", labelpad=12, color=AXIS_RED)
    latency_ax.tick_params(axis='y', colors=AXIS_RED)
    latency_ax.spines['left'].set_color(AXIS_RED)
    latency_ax.spines['left'].set_linewidth(1.5)
    
    # 右侧 Y 轴
    score_ax.set_ylabel("Primary Metric Score", fontweight="bold", labelpad=12, color=TEXT_COLOR)
    score_ax.set_ylim(0.8, 1.05) 

    # 🌟 核心修改：强行遍历并加粗左右两侧 Y 轴的刻度文字
    for label in latency_ax.get_yticklabels():
        label.set_fontweight("bold")
        
    for label in score_ax.get_yticklabels():
        label.set_fontweight("bold")

    
    # 隐藏顶部的无用边框
    latency_ax.spines['top'].set_visible(False)
    score_ax.spines['top'].set_visible(False)
    
    latency_ax.margins(x=0.06)

    # 🏷️ 构建精美的复合图例句柄 (更新为红色)
    legend_handles = [
        Line2D(
            [0], [0], marker="o", color=MEAN_MARKER, linestyle="--",
            markeredgecolor="white", markersize=8, label="Latency Mean",
        ),
        *[
            Patch(
                facecolor=METRIC_COLORS[metric],
                edgecolor="white",
                label=METRIC_LABELS[metric],
            )
            for metric in METRICS
        ],
    ]
    
    # 图例置于图内顶部中央，且去掉背景和边框
    latency_ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),   # 🌟 新增：绝对位置锚点！(X=0.5居中, Y=1.02控制上下)
        ncol=3,               
        frameon=False,
        columnspacing=0.3,    
        handletextpad=0.2,    
        handlelength=1.0,     
        borderaxespad=0.0,    
        prop={"weight": "bold", "size": 30} 
    )

    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cases, rounds, grid_sizes = load_results(input_dir)
    summary = build_summary(cases, rounds, grid_sizes)
    figure = draw_figure(cases, rounds, grid_sizes)

    png_path = output_dir / f"{args.basename}.png"
    pdf_path = output_dir / f"{args.basename}.pdf"
    csv_path = output_dir / f"{args.basename}_plot_summary.csv"
    
    # 使用 pad_inches=0 极限裁剪边缘，彻底消除任何白边
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0, facecolor='white')
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0, facecolor='white')
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"Grid sizes: {grid_sizes}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved plot summary: {csv_path}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()