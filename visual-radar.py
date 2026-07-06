from math import pi
import os
import matplotlib.pyplot as plt
import numpy as np

# ======================
# 全局字体设置
# ======================
plt.rcParams.update({
    'font.size': 16,
    'font.weight': 'bold',
})

metrics = {
    "Detection \n Precision": 0.959,
    "Detection \n Recall": 0.955,
    "Detection \n F1": 0.957,
    "Mean \n IoU": 0.931,
    "Shape \n Accuracy": 0.854,
    "Trend \n Accuracy": 0.850,
}

labels = list(metrics.keys())
values = list(metrics.values())

# ======================
# 雷达图角度
# ======================
N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

# 闭合雷达图
values_closed = values + values[:1]
angles_closed = angles + angles[:1]

# ======================
# 绘图
# ======================
fig = plt.figure(figsize=(10, 12))
ax = plt.subplot(111, polar=True)

# 曲线
ax.plot(
    angles_closed,
    values_closed,
    linewidth=3,
    linestyle='solid',
    marker='o',
    markersize=8
)

# 填充
ax.fill(
    angles_closed,
    values_closed,
    alpha=0.25
)

# ======================
# 分类标签：手动放置，避免遮挡
# ======================
ax.set_xticks(angles)
ax.set_xticklabels([])  # 关闭默认标签，改为手动绘制

label_radius = 1.12  # 标签距离圆心的半径，越大越靠外

for angle, label in zip(angles, labels):
    x = np.cos(angle)
    y = np.sin(angle)

    # 根据左右位置设置水平对齐
    if x > 0.5:
        ha = 'left'      # 右侧标签向右展开
    elif x < -0.5:
        ha = 'right'     # 左侧标签向左展开
    else:
        ha = 'center'

    # 根据上下位置设置垂直对齐
    if y > 0.5:
        va = 'bottom'
    elif y < -0.5:
        va = 'top'
    else:
        va = 'center'

    ax.text(
        angle,
        label_radius,
        label,
        ha=ha,
        va=va,
        fontsize=25,
        fontweight='bold',
        clip_on=False
    )

# ======================
# 数值文本
# ======================
for i in range(N):
    # 默认数值向内收缩
    r = values[i] - 0.08

    # 最右侧 Detection Precision 特别容易和标签重叠，进一步向内移动
    if i == 0:
        r = values[i] - 0.17

    ax.text(
        angles[i],
        r,
        f"{values[i]:.3f}",
        ha='center',
        va='center',
        fontsize=25,
        fontweight='bold',
        color='black',
        bbox=dict(
            facecolor='white',
            alpha=0.7,
            edgecolor='none',
            boxstyle='round,pad=0.3'
        )
    )

# ======================
# 径向刻度设置
# ======================
ax.set_rlabel_position(0)
ax.tick_params(
    axis='y',
    labelsize=14
)

# 隐藏 y 轴刻度标签
ax.set_yticklabels([])

# 网格线
ax.grid(linewidth=1.2)

# 范围
ax.set_ylim(0, 1.0)

# 标题
ax.set_title(
    "Encoder Performance",
    fontsize=35,
    fontweight='bold',
    pad=60
)

# ======================
# 布局与保存
# ======================
# 手动留白，比 tight_layout 更适合极坐标外部标签
plt.subplots_adjust(
    left=0.15,
    right=0.85,
    top=0.82,
    bottom=0.12
)

os.makedirs("outputs", exist_ok=True)

plt.savefig(
    "outputs/encoder_performance_radar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()