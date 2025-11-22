import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import scipy.io as spio
from scipy.sparse import eye
from scipy.sparse import identity
import time
import logging
import argparse
import os
import sys

def save_sparse_to_coo_txt(sparse_matrix, filename):
    """
    将scipy的稀疏矩阵保存为COO格式的txt文件
    格式: row col value
    """
    # 确保是coo_matrix
    coo = sparse_matrix.tocoo()

    n = sparse_matrix.shape[0]
    nnz = sparse_matrix.nnz

    with open(filename, "w") as f:
        f.write("% Matrix in COO format for MUMPS\n")
        f.write(f"{n} {n} {nnz}\n")
    # 组合成 (row, col, value)
    data = np.vstack((coo.row + 1, coo.col + 1, coo.data)).T

    # 保存到txt
    with open(filename, "a") as f:
        np.savetxt(f, data, fmt="%d %d %.18e")  # 行列索引用整数，值用浮点数
    print(f"Sparse matrix saved in COO format to {filename}")


def save_aligned_coo(G, C, prefix="matrix"):
    """
    将两个稀疏矩阵 G, C 保存为对齐后的 COO 格式文件
    - 确保 nnz、row、col 一样
    - 在另一个矩阵为 0 的位置补 0
    输出两个文件: prefix_G.txt, prefix_C.txt
    """

    # 转成 coo
    G = G.tocoo()
    C = C.tocoo()

    n = G.shape[0]
    assert G.shape == C.shape, "G and C must have the same shape"

    # 取并集非零位置
    G_idx = set(zip(G.row, G.col))
    C_idx = set(zip(C.row, C.col))
    all_idx = sorted(G_idx | C_idx)  # 并集并排序

    nnz = len(all_idx)

    # 构造 row, col, val
    rows = np.array([i for i, j in all_idx], dtype=int) + 1  # 1-based
    cols = np.array([j for i, j in all_idx], dtype=int) + 1

    # G 的值（不存在的位置补 0）
    G_dict = {(i, j): v for i, j, v in zip(G.row, G.col, G.data)}
    G_val = np.array([G_dict.get((i, j), 0.0) for i, j in all_idx])

    # C 的值（不存在的位置补 0）
    C_dict = {(i, j): v for i, j, v in zip(C.row, C.col, C.data)}
    C_val = np.array([C_dict.get((i, j), 0.0) for i, j in all_idx])

    # 保存 G
    with open(f"{prefix}_G", "w") as f:
        f.write("% Matrix G in aligned COO format for MUMPS\n")
        f.write(f"{n} {n} {nnz}\n")
        data = np.vstack((rows, cols, G_val)).T
        np.savetxt(f, data, fmt="%d %d %.18e")

    # 保存 C
    with open(f"{prefix}_C", "w") as f:
        f.write("% Matrix C in aligned COO format for MUMPS\n")
        f.write(f"{n} {n} {nnz}\n")
        data = np.vstack((rows, cols, C_val)).T
        np.savetxt(f, data, fmt="%d %d %.18e")

    print(f"Aligned sparse matrices saved to {prefix}_G.txt and {prefix}_C.txt")


def save_aligned_coo_mumps(G, C, circuit, index, prefix="matrix"):
    """
    将两个稀疏矩阵 G, C 保存为 MUMPS 可读的复数 COO 格式
    - 文件格式: row col (real,imag)
    - 实数矩阵会自动存 imag=0
    - 输出: prefix_G, prefix_C
    """
    # 转 coo
    G = G.tocoo()
    C = C.tocoo()

    n = G.shape[0]
    assert G.shape == C.shape, "G and C must have the same shape"

    # 索引并集
    G_idx = set(zip(G.row, G.col))
    C_idx = set(zip(C.row, C.col))
    all_idx = sorted(G_idx | C_idx)
    nnz = len(all_idx)

    rows = np.array([i for i, j in all_idx], dtype=int) + 1  # 1-based
    cols = np.array([j for i, j in all_idx], dtype=int) + 1

    G_dict = {(i, j): v for i, j, v in zip(G.row, G.col, G.data)}
    C_dict = {(i, j): v for i, j, v in zip(C.row, C.col, C.data)}

    G_val = np.array([G_dict.get((i, j), 0.0) for i, j in all_idx])
    C_val = np.array([C_dict.get((i, j), 0.0) for i, j in all_idx])

    def write_matrix(filename, vals):
        with open(filename, "w") as f:
            f.write("% Complex matrix in COO format for MUMPS\n")
            f.write(f"{n} {n} {nnz}\n")
            for r, c, v in zip(rows, cols, vals):
                f.write(f"{r} {c} ({np.real(v):.18e},{np.imag(v):.18e})\n")

    # 保存 G, C
    write_matrix(f"./MSIP_BDSM/SIP_matrix/{circuit}t/{prefix}_G{index}", G_val)
    write_matrix(f"./MSIP_BDSM/SIP_matrix/{circuit}t/{prefix}_C{index}", C_val)

    print(f"Saved aligned matrices to {prefix}_G{index} and {prefix}_C{index} (MUMPS format)")

def save_aligned_coo_mumps_gt(G, C, circuit, prefix="matrix"):
    """
    将两个稀疏矩阵 G, C 保存为 MUMPS 可读的复数 COO 格式
    - 文件格式: row col (real,imag)
    - 实数矩阵会自动存 imag=0
    - 输出: prefix_G, prefix_C
    """
    # 转 coo
    G = G.tocoo()
    C = C.tocoo()

    n = G.shape[0]
    assert G.shape == C.shape, "G and C must have the same shape"

    # 索引并集
    G_idx = set(zip(G.row, G.col))
    C_idx = set(zip(C.row, C.col))
    all_idx = sorted(G_idx | C_idx)
    nnz = len(all_idx)

    rows = np.array([i for i, j in all_idx], dtype=int) + 1  # 1-based
    cols = np.array([j for i, j in all_idx], dtype=int) + 1

    G_dict = {(i, j): v for i, j, v in zip(G.row, G.col, G.data)}
    C_dict = {(i, j): v for i, j, v in zip(C.row, C.col, C.data)}

    G_val = np.array([G_dict.get((i, j), 0.0) for i, j in all_idx])
    C_val = np.array([C_dict.get((i, j), 0.0) for i, j in all_idx])

    def write_matrix(filename, vals):
        with open(filename, "w") as f:
            f.write("% Complex matrix in COO format for MUMPS\n")
            f.write(f"{n} {n} {nnz}\n")
            for r, c, v in zip(rows, cols, vals):
                f.write(f"{r} {c} ({np.real(v):.18e},{np.imag(v):.18e})\n")

    # 保存 G, C
    write_matrix(f"./MSIP_BDSM/SIP_matrix/{circuit}t/{prefix}_G", G_val)
    write_matrix(f"./MSIP_BDSM/SIP_matrix/{circuit}t/{prefix}_C", C_val)

    print(f"Saved aligned matrices to {prefix}_G and {prefix}_C (MUMPS format)")


# 示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate mf_mor data for model")
    parser.add_argument("--circuit", type=int, default=2)
    args = parser.parse_args()
    save_path = os.path.join(sys.path[0], 'MSIP_BDSM/SIP_matrix/{}t/'.format(args.circuit))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(f"{save_path}/data_generate1.log")])
    logging.info(args)
    
    # 构造一个简单的稀疏矩阵
    circuit = args.circuit
    t1 = time.time()
    # data1 = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/{}t_SIP1.mat".format(circuit))
    # C1, G1, B1 = data1['C_final'] * 1e-0, data1['G_final'], data1['B_final']
    # B1 = B1.tocsc()
    # C1 = C1.tocsc()
    # G1 = G1.tocsc()
    # # G = G + 1e-10 * identity(G.shape[0], format="csc")
    # C1 = C1 + 1e-20 * identity(C1.shape[0], format="csc")

    # data2 = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/{}t_SIP2.mat".format(circuit))
    # C2, G2, B2 = data2['C_final'] * 1e-0, data2['G_final'], data2['B_final']
    # B2 = B2.tocsc()
    # C2 = C2.tocsc()
    # G2 = G2.tocsc()
    # C2 = C2 + 1e-20 * identity(C2.shape[0], format="csc")

    # f = np.array(1e9)  # array of targeting frequencies
    # s = 1j * 2 * np.pi * f
    # G1 = G1 + s * C1
    # G2 = G2 + s * C2

    # # save_sparse_to_coo_txt(C, "1tcoo_C")
    # # save_sparse_to_coo_txt(G, "1tcoo_G")
    # # save_sparse_to_coo_txt(B, "2tcoo_B")
    # # save_sparse_to_coo_txt(G + C, "1tcoo_GC")
    # save_aligned_coo_mumps(G1, C1, circuit, index=1, prefix="aligned")
    # save_aligned_coo_mumps(G2, C2, circuit, index=2, prefix="aligned")
    # t2 = time.time()
    # logging.info(f"Time taken to save matrices: {t2 - t1:.2f} seconds")

    data = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/{}t_SIP_gt.mat".format(circuit))
    C, G, B = data['C_final'] * 1e-0, data['G_final'], data['B_final']
    B = B.tocsc()
    C = C.tocsc()
    G = G.tocsc()
    # G = G + 1e-10 * identity(G.shape[0], format="csc")
    C = C + 1e-20 * identity(C.shape[0], format="csc")

    f = np.array(1e9)  # array of targeting frequencies
    s = 1j * 2 * np.pi * f
    G = G + s * C

    save_aligned_coo_mumps_gt(G, C, circuit, prefix="aligned")
    t2 = time.time()
    logging.info(f"Time taken to save matrices: {t2 - t1:.2f} seconds")
