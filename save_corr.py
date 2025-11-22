'''

'''
import scipy.sparse.linalg as spla
import argparse
import os
import logging
import scipy.io as spio
import numpy as np
import time
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp


def save_corr(G, C, B, savepath, topk=5):
    f = 1e9
    s = 1j * 2 * np.pi * f
    n_ports = B.shape[1]

    # Step 1: Solve (G + sC) * M0 = B
    A = G + s * C
    if sp.issparse(A):
        M0 = spla.spsolve(A, B)
    else:
        M0 = np.linalg.solve(A, B)

    # Step 2: Compute port correlation
    corr = B.conj().T @ M0
    corr_abs = np.abs(corr)

    # Step 3: 保存每个端口的top-k相关端口
    topk_idx = np.argsort(-corr_abs, axis=1)[:, :topk]
    np.save(savepath, topk_idx)

    print(f"Top-{topk} related ports saved to {savepath}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Baseline DeMOR')
    parser.add_argument('--circuit', type=int, default=6, help='Circuit number')
    parser.add_argument("--port_num", type=int, default= 10000)
    parser.add_argument("--threshold", type=int, default= 0)
    parser.add_argument("--generate", type=int, default= 1)
    parser.add_argument("--data_num", type=int, default= 1)
    parser.add_argument("--process", type=int, default= 10)
    args = parser.parse_args()
    save_path = os.path.join('./Baseline/DeMOR/{}t/'.format(args.circuit))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_path = "/home/fillip/home/CPM-MOR/Baseline/DeMOR/{}t".format(args.circuit)
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(f"{save_path}/DeMOR.log")])
    logging.info(args)
    data = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/{}t_SIP_gt.mat".format(args.circuit))
    
    port_num = args.port_num
    logging.info("Circuit : {}".format(args.circuit))
    C, G, B = data['C_final'] * 1e-0, data['G_final'], data['B_final']
    Node_size = C.shape[0]
    B = B.tocsc()
    C = C.tocsc()
    G = G.tocsc()
    B = B[:, 0:0+port_num]
    # output matrix
    O = B
    # corr_idx = np.load(os.path.join(data_path, 'corr.npy'))
    # for i in range(100):
    #     related_ports = corr_idx[i]
    #     print("Port {}: related ports {}".format(i, related_ports))
    # pass
    save_corr(G, C, B, save_path, topk=10)