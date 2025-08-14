import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, dok_matrix, find, eye, csc_matrix, identity
from scipy.sparse.linalg import inv,spsolve,splu
import scipy.io as spio
import argparse
import os
import logging
import time
from generate_mf_mor_data import generate_u, generate_udiff 
from utils.tdIntLinBE_new import *
import matplotlib.pyplot as plt
import scipy.sparse as sp

    
def SIPcore3(G, C, E, ports, threshold=2):
    
    n = G.shape[0]
    
    t1 = time.time()

    # floating_nodes = [i for i in range(n) if abs(G[i,i]) < 1e-12]
    # ports = list(set(floating_nodes) | set(ports))
    m = len(ports) # ports + floating nodes

    non_ports = [i for i in range(n) if i not in ports]
    perm = non_ports + ports
    G = G[perm, :][:, perm]
    C = C[perm, :][:, perm]
    C = C + 1e-15 * eye(n, format='csc') #避免奇异
    E = E[perm, :]
    t2 = time.time()
    logging.info("Permutation time: {}".format(t2 - t1))
    
    subG = G
    subC = C
    for i in range(0, n - m):
        # if i >= 20000:
        #     break
        k = n - i
        
        t3 = time.time()

        diag_vals = np.abs(subG[:-m, :-m].diagonal())
        pivot_idx = np.argmax(diag_vals)
        pivot_val = diag_vals[pivot_idx]

        logging.info(f"Processing column {i+1}/{n-m}..., pivot value: {pivot_val}, pivot index: {pivot_idx}")
        
        if pivot_val < threshold:
            logging.info(f"Column {i+1}, pivot value is too small: {pivot_val}")
            logging.info("End of reduction, pivot value is too small.")
            break

        if pivot_idx != 0:
            # 交换 pivot 节点到第 0 个位置
            idx = np.arange(k)
            idx[[0, pivot_idx]] = idx[[pivot_idx, 0]]
            subG = subG[idx, :][:, idx]
            subC = subC[idx, :][:, idx]

        a_ii = subG[0, 0]
            
        b_i = subG[0, 1:k].toarray().flatten()
        # B_i = G[i+1:n, i+1:n]
        
        M_sub = lil_matrix(eye(k))
        # M_sub[0, 0] = -np.sqrt(a_ii)
        M_sub[0,1:] = -b_i / a_ii
        M_sub = csr_matrix(M_sub)
        
        subG = M_sub.T @ subG @ M_sub
        subC = M_sub.T @ subC @ M_sub
        subG = subG[1:, 1:]
        subC = subC[1:, 1:]
        t4 = time.time()
        logging.info(f"Time taken for column {i+1}: {t4 - t3:.4f} seconds")
    
    G_hat = subG
    C_hat = subC
    E_hat = E[-G_hat.shape[0]:, :]
    logging.info(f"Total reduction time: {time.time() - t1:.4f} seconds")
    
    return G_hat.tocsc(), C_hat.tocsc(), E_hat.tocsc()

def is_singular_lu(G):
    try:
        lu = splu(G)  # 如果成功，则 G 非奇异
        return False
    except RuntimeError as e:
        if "Factor is exactly singular" in str(e):
            return True
        raise e

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='SIP')
    parser.add_argument('--circuit', type=int, default=2, help='Circuit number')
    parser.add_argument("--port_num", type=int, default= 2000)
    parser.add_argument("--threshold", type=float, default=2)
    args = parser.parse_args()
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

    # print(is_singular_lu(G))

    # B = B[:, 0:0+port_num]
    B = B
    # output matrix
    O = B
    m = 2
    s = 1j * 2 * np.pi * 1e9
    time_start = time.time()
    
    ports = []
    for j in range(B.shape[1]):
        rows, _ = B[:, j].nonzero()
        if len(rows) > 0:
            ports.append(rows[0])
    ports = np.unique(ports)
    
    Gr, Cr, Br = SIPcore3(-G, C, B, ports=list(ports),threshold=threshold)
    sp.save_npz(f'Gr_{args.circuit}.npz', Gr)
    sp.save_npz(f'Cr_{args.circuit}.npz', Cr)
    sp.save_npz(f'Br_{args.circuit}.npz', Br)
    # Gr = sp.load_npz(f'Gr_{args.circuit}.npz')
    # Cr = sp.load_npz(f'Cr_{args.circuit}.npz')
    # Br = sp.load_npz(f'Br_{args.circuit}.npz')
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
    y = O.T@xAll

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
    

    # n, m = 4, 2
    # G = np.array([
    #     [2, -1, 0, 0],
    #     [-1, 3, -1, 0],
    #     [0, -1, 2, -1],
    #     [0, 0, -1, 1]
    # ], dtype=float)
    # C = np.eye(n) 
    # E = np.array([[1, 0], [0, 0], [0, 1], [0, 0]])
    # ports = [2, 3]
    
    # G_hat, C_hat, E_hat = SIPcore(G, C, E, ports)
    # print("化简后的 G_hat:\n", G_hat)
    # print("化简后的 C_hat:\n", C_hat)
    # print("化简后的 E_hat:\n", E_hat)