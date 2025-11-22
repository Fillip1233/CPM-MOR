import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline DeMOR')
    parser.add_argument('--circuit', type=int, default=2, help='Circuit number')
    parser.add_argument("--port_num", type=int, default= 10000)
    parser.add_argument("--threshold", type=int, default= 0)
    
    args = parser.parse_args()
    save_path = os.path.join('./Baseline/DeMOR/{}t/'.format(args.circuit))

    RGA = np.load(os.path.join(save_path, 'RGA.npy'))
    RGA = RGA[:100, :100]  # 仅取前100x100部分进行可视化

    plt.figure(figsize=(15, 15))
    # 绘制热力图
    sns.heatmap(
        RGA.real,
        cmap="coolwarm",  # 红-蓝配色，适合正负值
        center=0.5,         # 颜色中心为 0
        vmin=0,          # 最小值 -1
        vmax=1,           # 最大值 1
        annot=False,      # 不显示数值（避免 100x100 过于密集）
        square=True,      # 保持单元格为方形
        cbar_kws={"shrink": 0.8}  # 调整颜色条大小
    )

    # 添加标题和标签
    # plt.title("RGA Matrix Visualization (100x100)", fontsize=20)
    # plt.xlabel("Input Index", fontsize=16)
    # plt.ylabel("Output Index", fontsize=16)

    # 显示图形
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'RGA.png'), dpi=300)  # 保存图像
    plt.close()