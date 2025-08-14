import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import splu

def SIPcore(G, C, E, ports):
    """
    SIPcore: 稀疏隐式投影算法，用于化简多端网络矩阵
    
    参数:
        G: 导纳矩阵 (n x n, 稀疏对称正定)
        C: 电容/电感矩阵 (n x n, 稀疏)
        E: 端口激励矩阵 (n x m)
        ports: 端口节点索引列表 (长度 m)
    
    返回:
        G_hat: 化简后的导纳矩阵 (m x m)
        C_hat: 化简后的电容矩阵 (m x m)
        E_hat: 化简后的激励矩阵 (m x m)
    """
    n = G.shape[0]
    m = len(ports)
    
    # 1. 节点重排序：将端口节点移到矩阵底部，并优化填充减少（此处简化为直接移动）
    non_ports = [i for i in range(n) if i not in ports]
    perm = non_ports + ports  # 非端口节点在前，端口节点在后
    G = G[perm, :][:, perm]   # 重排列行和列
    C = C[perm, :][:, perm]
    E = E[perm, :]
    
    # 2. 递归消元非端口节点 (i from 1 to n-m)
    for i in range(n - m):
        # 提取当前分块
        a_ii = G[i, i]
        b_i = G[i, i+1:n]
        B_i = G[i+1:n, i+1:n]
        
        # 计算 Schur 补并更新 G 和 C
        schur_G = B_i - np.outer(b_i, b_i) / a_ii
        schur_C = C[i+1:n, i+1:n] - np.outer(C[i+1:n, i], b_i) / a_ii
        
        # 更新矩阵 (仅保留右下角部分)
        G[i+1:n, i+1:n] = schur_G
        C[i+1:n, i+1:n] = schur_C
    
    # 3. 提取端口节点对应的子矩阵
    G_hat = G[-m:, -m:]
    C_hat = C[-m:, -m:]
    E_hat = E[-m:, :]
    
    return G_hat, C_hat, E_hat

# 示例测试
if __name__ == "__main__":
    # 构造一个简单的 4 节点电路 MNA 矩阵 (n=4, m=2 个端口)
    n, m = 4, 2
    G = np.array([
        [2, -1, 0, 0],
        [-1, 3, -1, 0],
        [0, -1, 2, -1],
        [0, 0, -1, 1]
    ], dtype=float)
    C = np.eye(n)  # 假设电容矩阵为单位阵
    E = np.array([[1, 0], [0, 0], [0, 1], [0, 0]])  # 端口激励
    ports = [2, 3]  # 最后两个节点为端口
    
    G_hat, C_hat, E_hat = SIPcore(G, C, E, ports)
    print("化简后的 G_hat:\n", G_hat)
    print("化简后的 C_hat:\n", C_hat)
    print("化简后的 E_hat:\n", E_hat)