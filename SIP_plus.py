import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, eye, csc_matrix, identity
import scipy.io as spio
import argparse
import os
import logging
import time
from generate_mf_mor_data import generate_udiff 
from utils.tdIntLinBE_new import *
import matplotlib.pyplot as plt

USE_CPP = False
# 导入C++扩展
try:
    from sip_cpp import SIPcore3_cpp
    USE_CPP = True
    logging.info("Using C++ implementation")
except ImportError:
    USE_CPP = False
    logging.info("Using Python implementation")
    
def SIPcore3_python(G, C, E, ports, threshold=2):
    
    n = G.shape[0]
    
    t1 = time.time()

    m = len(ports) # ports + floating nodes

    mask = np.ones(n, dtype=bool)
    mask[ports] = False
    non_ports = np.nonzero(mask)[0]

    perm = np.concatenate([non_ports, ports])

    G = G[perm, :][:, perm]
    C = C[perm, :][:, perm]
    C = C + 1e-15 * eye(n, format='csc') #避免奇异
    E = E[perm, :]
    t2 = time.time()
    print("Permutation time: {}".format(t2 - t1))
    
    subG = G
    subC = C
    for i in range(0, n - m):
        
        k = n - i
        
        t3 = time.time()

        diag_vals = np.abs(subG[:-m, :-m].diagonal())
        pivot_idx = np.argmax(diag_vals)
        pivot_val = diag_vals[pivot_idx]

        print(f"Processing column {i+1}/{n-m}..., pivot value: {pivot_val}, pivot index: {pivot_idx}")
        
        if pivot_val < threshold:
            print(f"Column {i+1}, pivot value is too small: {pivot_val}")
            print("End of reduction, pivot value is too small.")
            break

        if pivot_idx != 0:
            # 交换 pivot 节点到第 0 个位置
            idx = np.arange(k)
            idx[[0, pivot_idx]] = idx[[pivot_idx, 0]]
            subG = subG[idx, :][:, idx]
            subC = subC[idx, :][:, idx]

        a_ii = subG[0, 0]
            
        b_i = subG[0, 1:k].toarray().flatten()
        
        M_sub = lil_matrix(eye(k))
        M_sub[0,1:] = -b_i / a_ii
        M_sub = csr_matrix(M_sub)
        
        subG = M_sub.T @ subG @ M_sub
        subC = M_sub.T @ subC @ M_sub
        subG = subG[1:, 1:]
        subC = subC[1:, 1:]
        t4 = time.time()
        print(f"Time taken for column {i+1}: {t4 - t3:.4f} seconds")
    
    G_hat = subG
    C_hat = subC
    E_hat = E[-G_hat.shape[0]:, :]
    print(f"Total reduction time: {time.time() - t1:.4f} seconds")
    
    return G_hat.tocsc(), C_hat.tocsc(), E_hat.tocsc()

def SIPcore3(G, C, E, ports, threshold=2):
    if USE_CPP:
        # 转换矩阵格式为Eigen兼容的格式
        G_csc = G.tocsc()
        C_csc = C.tocsc()
        E_csc = E.tocsc()
        
        # 调用C++版本
        result = SIPcore3_cpp(G_csc, C_csc, E_csc, ports, threshold)
        
        # 转换回scipy稀疏矩阵
        return (csc_matrix(result.G_hat), 
                csc_matrix(result.C_hat), 
                csc_matrix(result.E_hat))
    else:
        # 使用Python版本
        return SIPcore3_python(G, C, E, ports, threshold)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='SIP')
    parser.add_argument('--circuit', type=int, default=3, help='Circuit number')
    parser.add_argument("--port_num", type=int, default= 2000)
    parser.add_argument("--threshold", type=float, default=2)
    parser.add_argument("--use_cpp", action='store_true', help='Use C++ implementation')
    args = parser.parse_args()

    if args.use_cpp:
        USE_CPP = True
        logging.info("Forcing C++ implementation")

    save_path = os.path.join('/home/fillip/桌面/CPM-MOR/Exp_res/SIP/{}t/'.format(args.circuit))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(f"{save_path}/SIP.log")])
    logging.info(args)
    data = spio.loadmat("/home/fillip/桌面/CPM-MOR/IBM_transient/ibmpg{}t.mat".format(args.circuit))
    port_num = args.port_num
    threshold = args.threshold
    logging.info("Circuit : {}".format(args.circuit))
    C, G, B = data['E'] * 1e-0, data['A'], data['B']
    Node_size = C.shape[0]
    B = B.tocsc()
    C = C.tocsc()
    G = G.tocsc()

    B = B
    O = B
    m = 2
    s = 1j * 2 * np.pi * 1e9
    time_start = time.time()

    rows, _ = B.nonzero()
    ports = np.unique(rows)
    
    Gr, Cr, Br = SIPcore3(-G, C, B, ports=list(ports),threshold=threshold)
    Br = Br[:, :port_num]
    nr_size = Cr.shape[0]
    logging.info("Original order size: {}".format(Node_size))
    logging.info("Reduced order size: {}".format(nr_size))
    time_end = time.time()
    logging.info("SIP time: {}".format(time_end - time_start))

    t0 = 0
    tf = 1e-09
    dt = 1e-11
    srcType = 'pulse'
    VS = []
    VS = np.array(VS)
    IS, VS = generate_udiff(args.port_num, args.circuit, seed = 0)

    x0 = np.zeros((C.shape[0], 1))
    B = B[:, :port_num]
    xAll, time1, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, C, -G, B, VS, IS, x0, srcType)
    y = B.T@xAll

    xr0 = np.zeros((Cr.shape[0], 1))
    xrAll, time2, dtAll, urAll = tdIntLinBE_new(t0, tf, dt, Cr, Gr, Br, VS, IS, xr0, srcType)
    y_mor = Br.T@xrAll

    yy = y[0,:]
    yy_mor = y_mor[0,:]
    for i in range(1, port_num):
        yy = yy + y[i,:]
        yy_mor = yy_mor + y_mor[i,:]
    
    port = 1
    plt.plot(time1, yy, color='green', linestyle='-.', marker='*', label='GT', markevery = 35, markersize=6, linewidth=1.5)
    plt.plot(time2, yy_mor, color='purple', linestyle='--', marker='*', label='SIP', markevery = 25, markersize=6, linewidth=1.5)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Response result (V)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.title("SIP test", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path+'SIP_{}t_{}.png'.format(args.circuit,port_num,port), dpi=300)
    pass
    