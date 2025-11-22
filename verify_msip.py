import numpy as np
import argparse
import scipy.io as spio
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from generate_mf_mor_data import generate_u, generate_udiff, generate_uover
from utils.tdIntLinBE_new import *
from utils.calculate_metrix import calculate_metrix
import pandas as pd
import os

def coo_from_triplets(rows, cols, vals, shape=None, to='csr', dtype=np.complex128):
    """
    rows, cols, vals: 一维数组或列表（同长度）
    shape: (n_rows, n_cols)，如果不给会自动用 max 索引推断
    to: 'coo' 返回COO;'csr' 或 'csc' 会转换到对应格式并合并重复项
    dtype: 数据类型 (默认复数)
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=dtype)

    if shape is None:
        n_rows = int(rows.max()) + 1 if rows.size else 0
        n_cols = int(cols.max()) + 1 if cols.size else 0
        shape = (n_rows, n_cols)

    A = coo_matrix((vals, (rows, cols)), shape=shape, dtype=dtype)

    if to == 'coo':
        A.sum_duplicates()
        return A
    elif to == 'csr':
        return A.tocsr()
    elif to == 'csc':
        return A.tocsc()
    else:
        raise ValueError("to 只能是 'coo' / 'csr' / 'csc'")


def load_coo_txt(path, shape=None, delimiter=None, comment='#', to='csr', dtype=np.complex128):
    """
    path: 文本文件路径
    文件格式:
        第一行: "n_rows n_cols nnz"
        后续行: "row col (real,imag)"
    """
    rows = []
    cols = []
    vals = []
    detected_shape = None

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or (comment and line.startswith(comment)):
                continue

            parts = line.split(delimiter) if delimiter else line.split()

            # 第一行: 矩阵维度
            # if detected_shape is None and len(parts) == 3 and "(" not in parts[2]:
            if detected_shape is None :
                detected_shape = (int(parts[0]), int(parts[1]))
                continue

            # 行/列 (1-based 索引)
            r = int(parts[0]) - 1
            c = int(parts[1]) - 1

            # 复数部分 "(real,imag)"
            v_str = "".join(parts[2:])  # 合并可能被split的 "(a,b)"
            v_str = v_str.strip("()")
            real_str, imag_str = v_str.split(",")
            v = complex(float(real_str), float(imag_str))

            rows.append(r)
            cols.append(c)
            vals.append(v)

    final_shape = shape if shape is not None else detected_shape

    return coo_from_triplets(rows, cols, vals, shape=final_shape, to=to, dtype=dtype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vertify msip')
    parser.add_argument('--circuit', type=int, default=6, help='Circuit number')
    parser.add_argument("--port_num", type=int, default= 10000)
    args = parser.parse_args()
    data = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/{}t_SIP_gt.mat".format(args.circuit))
    port_num = args.port_num

    data_path = "/home/fillip/home/CPM-MOR/Baseline/SIP/{}t".format(args.circuit)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    C1, G1, B1 = data['C_final'] * 1e-0, data['G_final'], data['B_final']
    B1 = B1.tocsc()
    C1 = C1.tocsc()
    G1 = G1.tocsc()

    C2 = load_coo_txt(f'/home/fillip/home/My_MUMPS_5.7.1/examples/result/{args.circuit}t/SIP_C_sip',to = 'csr')
    G2 = load_coo_txt(f'/home/fillip/home/My_MUMPS_5.7.1/examples/result/{args.circuit}t/SIP_G_sip',to = 'csr')
    C2 = C2.tocsc()
    G2 = G2.tocsc()
    B2 = B1[-G2.shape[0]:,:port_num]

    B1 = B1[:,:port_num]
    f = 9

    t0 = 0
    tf = 2e-09
    dt = 1e-11
    srcType = 'pulse'
    # IS, VS = generate_udiff(args.port_num, args.circuit, seed = 399)
    IS, VS = generate_uover(args.port_num, args.circuit, seed = 99)
    x0 = np.zeros((C1.shape[0], 1))

    xAll, time1, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, C1, G1, B1, VS, IS, x0, srcType)
    y = B1.T@xAll

    xr0 = np.zeros((C2.shape[0], 1))
    xrAll, time2, dtAll, urAll = tdIntLinBE_new(t0, tf, dt, C2, G2, B2, VS, IS, xr0, srcType)
    y_mor = B2.T@xrAll

    y_save = y_mor[3000,:]
    np.save("sip_over.npy", y_save)

    recording = {'rmse':[], 'nrmse':[], 'r2':[],'mae':[], 'relative_error':[]}
    metrics = calculate_metrix(y_test = np.abs(y), y_mean_pre = np.abs(y_mor))
    recording['rmse'].append(metrics['rmse'])
    recording['nrmse'].append(metrics['nrmse'])
    recording['r2'].append(metrics['r2'])
    recording['mae'].append(metrics['mae'])
    recording['relative_error'].append(metrics['relative_error'])
    record = pd.DataFrame(recording)
    record.to_csv(data_path + '/res_sip_over.csv', index = False)

    # yy = y[0,:]
    # yy_mor = y_mor[0,:]
    # for i in range(1, port_num):
    #     yy = yy + y[i,:]
    #     yy_mor = yy_mor + y_mor[i,:]

    # plt.plot(time1, yy, color='#ED3F27', linestyle='-.', marker='*', label='GT', markevery = 35, markersize=6, linewidth=1.5)
    # plt.plot(time2, yy_mor, color='#6B3F69', linestyle='--', marker='*', label='MULtifrontal-SIP', markevery = 25, markersize=6, linewidth=1.5)
    # plt.xlabel("Time (s)", fontsize=12)
    # plt.ylabel("Response result (V)", fontsize=12)
    # plt.legend(fontsize=12)
    # plt.grid()
    # plt.title("MULtifrontal-SIP test", fontsize=14)
    # plt.tight_layout()
    # plt.savefig('MULtifrontal-SIP_{}t_{}_10000.png'.format(args.circuit,f), dpi=300)
    pass
