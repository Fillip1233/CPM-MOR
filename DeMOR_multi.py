'''
2025-6-18 DeMOR multiprocessing
'''
import scipy.sparse.linalg as spla
import argparse
import os
import logging
import scipy.io as spio
import numpy as np
import time
import matplotlib.pyplot as plt
from generate_mf_mor_data import generate_u, generate_udiff
from utils.tdIntLinBE_new import *
from multiprocessing import Pool
import scipy as sp

def DeMOR(G, C, B, args, savepath, threshold=0.5):
    f = np.array([1e9])
    m = 2
    s = 1j * 2 * np.pi * f
    t1 = time.time()
    M0 = spla.spsolve(G+s[0]*C, B) # Step 1: Solve GM = B for M0

    H_DC = B.T @ M0  # Step 2: Compute H_DC = B^T * M0
    H_DC = H_DC.toarray()
    try:
        RGA = H_DC * np.linalg.inv(H_DC.T)  # Step 3: Compute relative gain array (RGA)
    except:
        logging.warning("Matrix inversion failed, using pseudo-inverse instead.")
        RGA = H_DC * np.linalg.pinv(H_DC.T)
    
    for i in range(RGA.shape[0]): # step 4: Normalize RGA values to [0, 1]
        for j in range(RGA.shape[1]):
            val = abs(RGA[i, j])
            if val <= 1:
                RGA[i, j] = val
            else:
                RGA[i, j] = 1 / val
    t2 = time.time() - t1
    logging.info(f"RGA computation time: {t2:.4f} seconds")
    threshold = np.min(np.diagonal(RGA.real))
    dominant_inputs = [] # Step 6-7: Based on threshold, select dominant input indices for each output
    for i in range(RGA.shape[0]):
        indices = np.where(RGA[i] >= threshold)[0]
        # 如果 0 不在 indices 里，就插入到最前面
        if 0 not in indices:
            indices = np.insert(indices, 0, 0)  # 在位置 0 插入 0
        dominant_inputs.append(indices)
    
    # simulate y 
    t0 = 0
    tf = 1e-09
    dt = 1e-11
    srcType = 'pulse'
    IS, VS = generate_udiff(args.port_num, args.circuit, seed = 0)
    x0 = np.zeros((C.shape[0], 1))

    xAll, time1, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, C, -G, B, VS, IS, x0, srcType)
    y = B.T@xAll
    M = s[0] * C + G  #s?
    pool_args = [(i, idx_list, C, G, B, x0, y, IS, VS, args, savepath, m, M)
                 for i, idx_list in enumerate(dominant_inputs)]
    with Pool(processes=4) as pool:
        pool.starmap(process_output, pool_args)
    
        
def process_output(i, idx_list, C, G, B, x0, y, IS, VS, args, savepath, m, M):
    import utils.PRIMA_demor as PRIMA
    try:
        lu = sp.sparse.linalg.splu(M)
        B_i = B[:, idx_list] if len(idx_list) > 0 else B[:, [i]]
        q = m * B_i.shape[1]
        tic = time.time()
        print("Process:{}".format(i))
        XX = PRIMA.PRIMA_sp(C, B_i, q, lu)
        Cr_i = (XX.T @ C) @ XX
        Gr_i = (XX.T @ G) @ XX
        Br_i = (XX.T @ B_i)
        nr_i = Cr_i.shape[0]
        toc = time.time() - tic
        print(f"Process{i}--Output {i+1}: Reduced order model size: {nr_i}, Time taken: {toc:.4f} seconds")

        s1 = time.time()
        t0 = 0
        tf = 1e-09
        dt = 1e-11
        srcType = 'pulse'
        xr0 = XX.T@ x0
        xrAll, time1, dtAll, urAll = tdIntLinBE_new(t0, tf, dt, Cr_i, -Gr_i, Br_i, VS, IS[idx_list,:], xr0, srcType)
        locat = np.where(idx_list == i)[0]
        y_mor = Br_i.T@xrAll

        yy = y[i,:]
        yy_mor = y_mor[locat,:]

        plt.plot(time1, yy, color='green', linestyle='-.', marker='*', label='GT', markevery = 35, markersize=6, linewidth=1.5)
        plt.plot(time1, yy_mor.reshape(-1), color='purple', linestyle='--', marker='*', label='DeMOR', markevery = 25, markersize=6, linewidth=1.5)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Response result (V)", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid()
        plt.title("DeMOR port{} test".format(i), fontsize=14)
        plt.tight_layout()
        plt.savefig(savepath+'DeMOR_{}t_port{}.png'.format(args.circuit,i), dpi=300)
        plt.close()
        s2 = time.time() - s1
        print(f"Process{i}--Simulation for output {i+1} took {s2:.4f} seconds.")
    except Exception as e:
        import traceback
        print(f"[Worker {i}] Error: {e}")
        traceback.print_exc()

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='DeMOR')
    parser.add_argument('--circuit', type=int, default=1, help='Circuit number')
    parser.add_argument("--port_num", type=int, default= 100)
    parser.add_argument("--threshold", type=int, default= 0)
    args = parser.parse_args()
    save_path = os.path.join('/home/fillip/home/CPM-MOR/Exp_res/DeMOR_multi/{}t/'.format(args.circuit))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(f"{save_path}/DeMOR.log")])
    logging.info(args)
    data = spio.loadmat("/home/fillip/桌面/CPM-MOR/IBM_transient/ibmpg{}t.mat".format(args.circuit))
    # data = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/thupg1t.mat")
    port_num = args.port_num
    logging.info("Circuit : {}".format(args.circuit))
    C, G, B = data['E'] * 1e-0, data['A'], data['B']
    Node_size = C.shape[0]
    B = B.tocsc()
    C = C.tocsc()
    G = G.tocsc()
    B = B[:, 0:0+port_num]
    # output matrix
    O = B
    DeMOR(G, C, B, args, save_path, threshold=args.threshold)