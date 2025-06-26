'''
2025-6-9 DeMOR GPT
6-16 revise
'''
import scipy.sparse.linalg as spla
import argparse
import os
import logging
import scipy.io as spio
import numpy as np
import time
import utils.PRIMA as PRIMA
import matplotlib.pyplot as plt
import seaborn as sns
from generate_mf_mor_data import generate_u, generate_udiff
from utils.tdIntLinBE_new import *
from scipy.sparse import hstack

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
    # RGA = np.load("RGA7t.npy")
    logging.info("RGA matrix computed and normalized.")
    # plt.figure(figsize=(15, 12))
    # # 绘制热力图
    # sns.heatmap(
    #     RGA.real,
    #     cmap="coolwarm",  # 红-蓝配色，适合正负值
    #     center=0.5,         # 颜色中心为 0
    #     vmin=0,          # 最小值 -1
    #     vmax=1,           # 最大值 1
    #     annot=True,      # 不显示数值（避免 100x100 过于密集）
    #     square=True,      # 保持单元格为方形
    #     cbar_kws={"shrink": 0.8}  # 调整颜色条大小
    # )

    # # 添加标题和标签
    # plt.title("RGA Matrix Visualization (100x100)", fontsize=20)
    # plt.xlabel("Input Index", fontsize=16)
    # plt.ylabel("Output Index", fontsize=16)

    # # 显示图形
    # plt.tight_layout()
    # plt.savefig('HDC_1t-1e10.png', dpi=300)  # 保存图像
    # plt.close()

    dominant_inputs = [] # Step 6-7: Based on threshold, select dominant input indices for each output
    threshold = np.min(np.diagonal(RGA.real))
    logging.info(f"Using threshold: {threshold:.4f} for dominant input selection.")
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
    y = O.T@xAll

    f = np.array([1e9])
    m = 2
    s = 1j * 2 * np.pi * f

    for i, idx_list in enumerate(dominant_inputs):
        B_i = B[:, idx_list] if len(idx_list) > 0 else B[:, [i]]
        q = m * B_i.shape[1]
        tic = time.time()
        XX = PRIMA.PRIMA_mp(C, G, B_i, s, q)
        Cr_i = (XX.T@ C)@XX
        Gr_i = (XX.T@ G)@XX
        Br_i = (XX.T@ B_i)
        nr_i = Cr_i.shape[0]
        toc = time.time() - tic
        logging.info(f"Output {i+1}: Reduced order model size: {nr_i}, Time taken: {toc:.4f} seconds")
        # save_path = os.path.join('/home/fillip/home/CPM-MOR/Exp_res/DeMOR/{}t/'.format(args.circuit))
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # spio.savemat(os.path.join(save_path, f"DeMOR_{i+1}.mat"), {
        #     'Cr': Cr_i,
        #     'Gr': Gr_i,
        #     'Br': Br_i,
        #     'XX': XX
        # })
        # break
        s1 = time.time()
        simulate(Cr_i, Gr_i, Br_i, XX, idx_list, x0, y, IS, VS, i, args, savepath)
        s2 = time.time() - s1
        logging.info(f"Simulation for output {i+1} took {s2:.4f} seconds.")
        
    

def simulate(Cr_i, Gr_i, Br_i, XX, port_idx, x0, y, IS, VS, select_port, args, save_path, t0 = 0, tf = 1e-09, dt = 1e-11):
    srcType = 'pulse'
    # IS, VS = generate_udiff(args.port_num, args.circuit, seed = 0)
    xr0 = XX.T@ x0
    xrAll, time1, dtAll, urAll = tdIntLinBE_new(t0, tf, dt, Cr_i, -Gr_i, Br_i, VS, IS[port_idx,:], xr0, srcType)
    locat = np.where(port_idx == select_port)[0]
    y_mor = Br_i.T@xrAll

    yy = y[select_port,:]
    yy_mor = y_mor[locat,:]
    # for i in range(1, port_num):
    #     yy = yy + y[i,:]
    #     yy_mor = yy_mor + y_mor[i,:]

    plt.plot(time1, yy, color='green', linestyle='-.', marker='*', label='GT', markevery = 35, markersize=6, linewidth=1.5)
    plt.plot(time1, yy_mor.reshape(-1), color='purple', linestyle='--', marker='*', label='DeMOR', markevery = 25, markersize=6, linewidth=1.5)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Response result (V)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.title("DeMOR port{} test".format(select_port), fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path+'DeMOR_{}t_port{}.png'.format(args.circuit,select_port), dpi=300)
    plt.close()
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='DeMOR')
    parser.add_argument('--circuit', type=int, default=2, help='Circuit number')
    parser.add_argument("--port_num", type=int, default= 2000)
    parser.add_argument("--threshold", type=int, default= 0)
    args = parser.parse_args()
    save_path = os.path.join('/home/fillip/home/CPM-MOR/Exp_res/DeMOR/{}t/'.format(args.circuit))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(f"{save_path}/DeMOR.log")])
    logging.info(args)
    data = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/ibmpg{}t.mat".format(args.circuit))
    # data = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/thupg1t.mat")
    # data1 = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/ibmpg1t_B1.mat")
    # data2 = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/ibmpg1t_B2.mat")
    # B1 = data1['B']
    # B2 = data2['B']
    # B1 = B1.tocsc()
    # B2 = B2.tocsc()
    # B = hstack([B1, B2], format='csc')

    port_num = args.port_num
    logging.info("Circuit : {}".format(args.circuit))
    C, G, B = data['E'] * 1e-0, data['A'], data['B']
    Node_size = C.shape[0]
    B = B.tocsc()
    C = C.tocsc()
    G = G.tocsc()
    B = B[:, 0:0+port_num]
    # B = B[:, -port_num:]
    # output matrix
    O = B
    i = 0
    # while i< port_num:
    #     print("({},{},{})".format(B.indices[i], B.indptr[i], B.data[i]))
    #     i += 1
    DeMOR(G, C, O, args, save_path, threshold=args.threshold)
    logging.info("Finish DeMOR")