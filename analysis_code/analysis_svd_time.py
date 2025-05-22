import matplotlib.pyplot as plt
import numpy as np

# 示例数据 - 你可以替换成你自己的数据
# 格式: 每组数据包含两个值
# data = [
#     [1002/1024, 2005/1024],  # 第一组
#     [2713/1024, 6012/1024],  # 第二组
#     [16241/1024, 35181/1024],  # 第三组
#     [18858/1024, 40435/1024],  # 第四组
#     [32354/1024, 64856/1024],   # 第五组
#     [49423/1024,99784/1024]
# ]
data =[
    [9, 94+39],  # 第一组
    [37, 1181+129],  # 第二组
    [210, 23735+748],  # 第三组
    [253, 24211+765],  # 第四组
    [364, 13341+1209],   # 第五组
    [519, 16838+1919]
]

# 组标签
groups = ['ibmpg1t', 'ibmpg2t', 'ibmpg3t', 'ibmpg4t', 'ibmpg5t', 'ibmpg6t']

# 设置柱状图的宽度
bar_width = 0.35

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(10, 6))

# 生成x轴位置
x = np.arange(len(groups))

# 绘制两组柱状图
bar1 = ax.bar(x - bar_width/2, [item[0] for item in data], bar_width,color='#7F55B1', label='B_svd')
bar2 = ax.bar(x + bar_width/2, [item[1] for item in data], bar_width,color='#F2C078', label='SVDMOR')

# 添加标签、标题和图例
ax.set_yscale('log')
ax.set_xlabel('Circuit', fontsize=12)
ax.set_ylabel('Time use (s)', fontsize=12)
ax.set_title('Time use of B_svd and SVDMOR', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend()

# 在每个柱子上方显示数值
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bar1)
autolabel(bar2)

# 调整布局
plt.tight_layout()

# 显示图形
plt.savefig("svd_time.png")