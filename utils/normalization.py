import torch
import torch.nn as nn

class MaxMinNormalization(nn.Module):
    def __init__(self):
        super(MaxMinNormalization, self).__init__()

    def forward(self, x):
        # x: [batch_size, port_num, data_length]
        self.min_vals = x.min()  # 在第三维度计算最小值
        self.max_vals = x.max()  # 在第三维度计算最大值
        
        # 最大最小归一化
        x_normalized = (x - self.min_vals) / (self.max_vals - self.min_vals + 1e-8)  # 加一个小的常数避免除零
        
        return x_normalized
    def denormalize(self, x):
        return x * (self.max_vals - self.min_vals + 1e-8) + self.min_vals


if __name__ == '__main__':
    batch_size = 4
    port_num = 3
    data_length = 5
    x = torch.rand(batch_size, port_num, data_length)  # 假设输入数据

    normalizer = MaxMinNormalization()
    normalized_x = normalizer(x)

    print("Normalized data:")
    print(normalized_x)
