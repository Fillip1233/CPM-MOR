import numpy as np
import scipy.io as spio
from scipy.linalg import svd, orth
from scipy.sparse import issparse, csr_matrix, diags
from scipy.sparse.linalg import svds, splu, LinearOperator
from scipy.sparse.linalg import aslinearoperator

def mcpack(C, G, B, L, zQ, k, g, r):
    """
    McPack 模型降阶算法 (单扩展点版本)
    
    参数:
        C, G: n x n 系统矩阵
        B, L: n x m 输入/输出矩阵
        zQ: 复数，扩展点
        k: 总Krylov子空间阶数
        g: 块矩截断阶数
        r: 秩近似参数
        
    返回:
        C_hat, G_hat: p x p 降阶系统矩阵
        B_hat, L_hat: p x m 降阶输入/输出矩阵
    """
    n = G.shape[0]
    m = B.shape[1]
    
    # 步骤1: 计算A_zQ和R_zQ
    G_zQ = G + zQ * C
    A_zQ = -np.linalg.solve(G_zQ, C)
    R_zQ = np.linalg.solve(G_zQ, B)
    
    # 步骤2: 计算前g个块矩并拼接为V_tilde
    V_blocks = []
    for i in range(g):
        V_i = L.T @ np.linalg.matrix_power(A_zQ, i) @ R_zQ
        V_blocks.append(V_i)
    V_tilde = np.hstack(V_blocks)  # 式(11)
    
    # 步骤3: 对V_tilde进行SVD并做秩r近似
    U, S, Vt = svd(V_tilde, full_matrices=False)
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vr = Vt[:r, :].T  # 式(12)
    
    # 步骤4: 构造Q矩阵 (式17)
    Q = np.zeros((m*k, r*k), dtype=complex)
    for j in range(g):
        Q[j*m:(j+1)*m, j*r:(j+1)*r] = Vr[j*m:(j+1)*m, :r]
    
    # 步骤5: 计算投影子空间X1 (式19)
    X = []
    X_prev = np.zeros((n, r))
    for i in range(k):
        if i < g:
            Qi = Q[i*m:(i+1)*m, i*r:(i+1)*r]
            X_i = A_zQ @ X_prev + R_zQ @ Qi
        else:
            X_i = A_zQ @ X_prev
        X.append(X_i)
        X_prev = X_i
    X1 = np.hstack(X[:g])  # 式(19)
    
    # 步骤6: 计算Krylov子空间X2 (Arnoldi过程)
    X2 = []
    v = X[g-1]
    for _ in range(k - g):
        v = A_zQ @ v
        v = v / np.linalg.norm(v)  # 正交化简化版
        X2.append(v)
    X2 = np.hstack(X2) if X2 else np.zeros((n, 0))
    
    # 步骤7: 合并子空间并构造投影矩阵
    V = orth(np.hstack([X1, X2]))  # 正交化
    
    # 构造降阶模型
    p = V.shape[1]
    G_hat = V.T @ G @ V
    C_hat = V.T @ C @ V
    B_hat = V.T @ B
    L_hat = V.T @ L
    
    return G_hat, C_hat, B_hat, L_hat

def mcpack_sparse(C, G, B, L, zQ, k, g, r, tol=1e-6):
    """
    McPack 算法稀疏矩阵版本
    
    参数:
        C, G: n x n 稀疏系统矩阵 (scipy.sparse格式)
        B, L: n x m 稀疏输入/输出矩阵
        zQ: 复数，扩展点
        k: 总Krylov子空间阶数
        g: 块矩截断阶数
        r: 秩近似参数
        tol: SVD截断容忍度
        
    返回:
        C_hat, G_hat: p x p 降阶系统矩阵 (密集矩阵)
        B_hat, L_hat: p x m 降阶输入/输出矩阵
    """
    # 确保输入是稀疏矩阵
    def ensure_sparse(mat):
        return mat if issparse(mat) else csr_matrix(mat)
    
    C, G = ensure_sparse(C), ensure_sparse(G)
    B, L = ensure_sparse(B), ensure_sparse(L)
    n, m = B.shape
    
    # --- 步骤1: 预计算A_zQ和R_zQ ---
    G_zQ = G + zQ * C
    # 稀疏LU分解加速求解
    solver = splu(G_zQ.tocsc())
    def solve_M(b):
        return solver.solve(b.toarray() if issparse(b) else b)
    
    # A_zQ = -G_zQ^{-1}*C (作为线性算子避免显式构造)
    def matvec_A(v):
        return -solve_M(C.dot(v))
    A_zQ = LinearOperator((n,n), matvec=matvec_A, dtype=complex)
    
    # R_zQ = G_zQ^{-1}*B
    R_zQ = solve_M(B.toarray() if issparse(B) else B)
    if issparse(B):
        R_zQ = csr_matrix(R_zQ)
    
    # --- 步骤2: 计算块矩的SVD近似 ---
    V_blocks = []
    # for i in range(g):
    #     # 计算 V_i = L^T * A_zQ^i * R_zQ (稀疏优化)
    #     v = R_zQ
    #     for _ in range(i):
    #         v = A_zQ.matvec(v.T).T  # 避免显式构造A_zQ^i
    #     V_i = L.T.dot(v)
    #     V_blocks.append(V_i.toarray() if issparse(V_i) else V_i)
    
    for i in range(g):
        tmp = A_zQ
        for _ in range(1, i):
            tmp = tmp @ A_zQ
        LT = aslinearoperator(L.T)
        R_op = aslinearoperator(R_zQ)
        V_i = LT @ tmp @ R_op   
        V_blocks.append(V_i)
    
    V_tilde = np.hstack(V_blocks)  # 式(11)

    V_blocks_dense = []

    for op in V_blocks:
        # 假设 LinearOperator 是 n x n 的
        n = op.shape[1]
        I = np.eye(n, dtype=complex)  # 单位阵
        dense_matrix = op @ I         # 得到稠密矩阵
        V_blocks_dense.append(dense_matrix)

    V_tilde = np.hstack(V_blocks_dense)
    
    # 稀疏SVD (仅计算前r个奇异值)
    U, S, Vt = svds(V_tilde, k=r, tol=tol)
    Vr = Vt[:r, :].T  # 式(12)
    
    # --- 步骤3: 构造Q矩阵 (式17) ---
    Q = np.zeros((m*k, r*k), dtype=complex)
    for j in range(g):
        block = Vr[j*m:(j+1)*m, :r]
        Q[j*m:(j+1)*m, j*r:(j+1)*r] = block
    
    # --- 步骤4-5: 递归构造投影子空间 ---
    X = []
    X_prev = np.zeros((n, r))
    for i in range(k):
        if i < g:
            Qi = Q[i*m:(i+1)*m, i*r:(i+1)*r]
            # 稀疏矩阵乘法优化
            term2 = R_zQ.dot(Qi) if issparse(R_zQ) else R_zQ @ Qi
            X_i = A_zQ.matvec(X_prev.T).T + term2
        else:
            X_i = A_zQ.matvec(X_prev.T).T
        X.append(X_i)
        X_prev = X_i
    
    X1 = np.hstack(X[:g])  # 式(19)
    
    # --- 步骤6: Arnoldi过程 (稀疏优化) ---
    X2 = []
    if k > g:
        v = X[g-1]
        for _ in range(k - g):
            v = A_zQ.matvec(v.T).T
            # 稀疏正交化
            for x in X2:
                v -= x.dot(v.T).dot(x.T)
            v /= np.linalg.norm(v)
            X2.append(v)
    X2 = np.hstack(X2) if X2 else np.zeros((n, 0))
    
    # --- 步骤7: 构造降阶模型 ---
    V = orth(np.hstack([X1, X2]))  # 正交基
    
    # 合同变换 (转换为密集小矩阵)
    G_hat = V.T.dot(G.dot(V))
    C_hat = V.T.dot(C.dot(V))
    B_hat = V.T.dot(B.toarray() if issparse(B) else B)
    L_hat = V.T.dot(L.toarray() if issparse(L) else L)
    
    return G_hat, C_hat, B_hat, L_hat

# 示例用法
if __name__ == "__main__":
    # n = 100  # 原始系统维度
    # m = 5    # 输入/输出端口数
    # k, g, r = 10, 5, 3  # 算法参数
    # zQ = 1.0 + 0j       # 扩展点
    
    # # 生成随机系统矩阵 (实际应用中应替换为真实数据)
    # C = np.random.randn(n, n)
    # G = np.random.randn(n, n)
    # B = L = np.random.randn(n, m)
    
    # # 运行McPack算法
    # G_hat, C_hat, B_hat, L_hat = mcpack(C, G, B, L, zQ, k, g, r)
    
    # print(f"降阶模型维度: {G_hat.shape}")
    data = spio.loadmat("D:\CPM项目\code\CPM-MOR\CPM-MOR\IBM_transient\ibmpg1t.mat")
    port_num = 100
    C, G, B = data['E'] * 1e-0, data['A'], data['B']
    B = B.tocsc()
    C = C.tocsc()
    G = G.tocsc()
    B = B[:, 0:0+port_num]
    # output matrix
    O = B
    f = np.array([1e2])
    m = 2
    s = 1j * 2 * np.pi * 1e2
    G_hat, C_hat, B_hat, L_hat = mcpack_sparse(C, G, B, O, s, m, m, r=20)

    pass