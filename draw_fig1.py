import numpy as np
import matplotlib.pyplot as plt

# 时间范围设置（仿照图片的时间范围）
time = np.linspace(0, 1, 10)

# 构造一个类似图中形状的高精度波形（高斯/凸起结构）
high = np.array([0, 10, 60, 150, 220, 270, 230, 160, 100, 80])

# 构造中精度（稍微偏离高精度）
medium = high - np.array([0, 3, 10, 20, 30, 35, 25, 20, 10, 5])

# 构造低精度（明显偏离高精度）
low = high - np.array([0, 8, 30, 60, 80, 100, 90, 70, 50, 40])

# 绘图
plt.figure(figsize=(5.5, 4))
plt.plot(time, high, label='Ground Truth', marker='*', color='brown')
plt.plot(time, medium, label='Compensated result', marker='^', linestyle='--', color='orange')
plt.plot(time, low, label='Low Fidelity', marker='o', linestyle=':', color='green')

# plt.xlabel('Time (s)')
# plt.ylabel('result')
# plt.title('Multi-Fidelity Simulation Results')
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.show()