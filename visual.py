from math import pi
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
    "Detection Rate": 0.9889,
    "Pass Basic Rate": 0.9889,
    "Mean IoU": 0.9712,
    "Shape Match": 0.6966,
    "Area Trend": 0.8989,
    "Intensity Trend": 0.7303,
}

labels = list(metrics.keys())
values = list(metrics.values())

# 闭合雷达图
N = len(labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

values += values[:1]
angles += angles[:1]

# ======================
# 绘图
# ======================
# 稍微增大画布，为外围标签留出更充足的空间
fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

# 曲线（加入了 marker 以高亮具体的数据点）
ax.plot(
    angles,
    values,
    linewidth=3,
    linestyle='solid',
    marker='o',         # 数据点标记
    markersize=8
)

# 填充
ax.fill(
    angles,
    values,
    alpha=0.25
)

# 标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(
    labels,
    fontsize=16,
    fontweight='bold'
)

# 【关键修改 1】：增加分类标签与雷达图边缘的距离，防止与数据/数值遮挡
ax.tick_params(axis='x', pad=30)

# 【关键修改 2】：循环添加数值文本
for i in range(N):
    # 将数值位置向内侧收缩 0.08，并加入带透明度的背景框防止网格线干扰阅读
    ax.text(
        angles[i], 
        values[i] - 0.08, 
        f"{values[i]:.4f}", 
        ha='center', 
        va='center', 
        fontsize=12, 
        fontweight='bold', 
        color='black',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
    )

# 径向刻度
ax.set_rlabel_position(0)
ax.tick_params(
    axis='y',
    labelsize=14
)

# 标题
ax.set_title(
    "Encoder Performance",
    fontsize=22,
    fontweight='bold',
    pad=40
)

# 网格线稍粗
ax.grid(linewidth=1.2)

# 范围
ax.set_ylim(0, 1.0)

plt.tight_layout()

plt.savefig(
    "encoder_performance_radar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()