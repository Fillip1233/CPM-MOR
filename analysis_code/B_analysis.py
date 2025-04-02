'''
2025-3-1
用于分析ibmpg的B矩阵,统计电路端口数和连接电流源超过1的端口数
'''
import numpy as np
import scipy.io as spio
from scipy.sparse import csc_matrix

data = spio.loadmat("./IBM_transient/ibmpg2t.mat")

E, A, B = data['E'] * 1e-0, data['A'], data['B']

# if isinstance(B, csc_matrix):
B = B.toarray()
aa = np.nonzero(B)[0].reshape(B.shape[1], -1)[:, 0]
non_zero_counts = np.count_nonzero(B != 0, axis=1)
## 看哪些节点的连接电流源超过1
cols_with_non_zero = np.where(non_zero_counts > 1)[0] 
rows_with_more_than_non_zero = np.sum(non_zero_counts > 0)
print("电路端口数:", rows_with_more_than_non_zero)
rows_with_more_than_one_non_zero = np.sum(non_zero_counts > 1)
print("连接电流源超过1的端口数:", rows_with_more_than_one_non_zero)


# for i, count in enumerate(non_zero_counts):
#     print(f"Column {i} has {count} non-zero elements.")
