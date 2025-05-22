'''
2025/4/13
找出稀疏矩阵中包含超过1个非零元素的行及其列索引
'''
import numpy as np
from scipy.sparse import csr_matrix,csc_matrix,coo_matrix
import scipy.io as spio

def find_rows_with_multiple_nonz(mat):
    """
    找出稀疏矩阵中包含超过1个非零元素的行及其列索引
    
    参数:
        mat: scipy.sparse矩阵 (CSR, CSC 或 COO格式)
    
    返回:
        打印满足条件的行及其列索引
    """
    if not isinstance(mat, (csr_matrix, csc_matrix, coo_matrix)):
        mat = csr_matrix(mat)  # 转换为CSR格式
    
    # 转换为CSR格式便于行操作
    mat_csr = mat.tocsr()
    
    found = False
    
    for i in range(mat_csr.shape[0]):
        # 获取第i行的非零列索引
        cols = mat_csr.indices[mat_csr.indptr[i]:mat_csr.indptr[i+1]]
        if len(cols) > 1:
            found = True
            print(f"行 {i} 有 {len(cols)} 个非零元素，列索引: {cols}")
    
    if not found:
        print("矩阵中没有包含超过1个非零元素的行")

# 示例用法
if __name__ == "__main__":
    data = spio.loadmat("./IBM_transient/thupg1t.mat")
    C, G, B = data['E'] * 1e-0, data['A'], data['B']
    B = B.tocsc()
    mat = csr_matrix(B)
    find_rows_with_multiple_nonz(mat)
    pass