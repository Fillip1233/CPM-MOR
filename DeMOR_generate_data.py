'''
2025-6-27 DeMOR data generation
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
import seaborn as sns
import utils.PRIMA_demor as PRIMA

def check_DeMOR(G, C, B, args, savepath, threshold=0.5):
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
    logging.info(f"Using threshold: {threshold}")
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
    pool_args = [(i, idx_list, C, G, B, x0, IS, VS, m, M)
                 for i, idx_list in enumerate(dominant_inputs)]
    
    y_mor_combined = np.zeros_like(y)
    with Pool(processes=4) as pool:
        results = pool.starmap(process_output, pool_args)
        for i, (port_idx, yy_mor) in enumerate(results):
            y_mor_combined[port_idx, :] = yy_mor.squeeze()
    
    # for i in range(20):
    #     plt.figure(figsize=(8, 5))
    #     plt.plot(time1, y_mor_combined[i,:], color='purple', linestyle='-.', marker='*', label='DeMOR', markevery = 35, markersize=6, linewidth=1.5)
    #     plt.plot(time1, y[i,:], color='green', linestyle='-.', marker='*', label='GroundTruth', markevery = 25, markersize=6, linewidth=1.5)
    #     plt.legend(fontsize=12)
    #     plt.title(f"ibmpg{args.circuit}t Port-{i} output response", fontsize=14)
    #     plt.xlabel("Time (s)", fontsize=12)
    #     plt.ylabel("Response result (V)", fontsize=12)
    #     plt.grid()
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(savepath, f'ibmpg{args.circuit}t_port_{i}_output_response.png'), dpi=300)
    #     plt.close()

def save_rga(G, C, B, savepath):
    f = np.array([1e9])
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
    np.save(os.path.join(savepath, 'RGA.npy'), RGA)
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
    # plt.savefig(os.path.join(savepath, 'RGA1.png'), dpi=300)  # 保存图像
    # plt.close()

def generate_DeMOR_data(G, C, B, args, savepath, seed):
    f = np.array([1e9])
    m = 2
    s = 1j * 2 * np.pi * f
    
    RGA = np.load(os.path.join(savepath, 'RGA.npy'))
    threshold = np.min(np.diagonal(RGA.real))
    logging.info(f"Using threshold: {threshold}")
    dominant_inputs = [] # Step 6-7: Based on threshold, select dominant input indices for each output
    for i in range(RGA.shape[0]):
        indices = np.where(RGA[i] >= threshold)[0]
        # 如果 0 不在 indices 里，就插入到最前面
        if 0 not in indices:
            indices = np.insert(indices, 0, 0)  # 在位置 0 插入 0
        dominant_inputs.append(indices)
    
    IS, VS = generate_udiff(args.port_num, args.circuit, seed = seed)
    x0 = np.zeros((C.shape[0], 1))

    M = s[0] * C + G  #s?
    global global_C, global_G, global_B, global_lu
    global_B = B
    global_C = C
    global_G = G
    global_lu = sp.sparse.linalg.splu(M)
    pool_args = [(i, idx_list, x0, IS, VS, m)
                 for i, idx_list in enumerate(dominant_inputs)]
    
    y_mor_combined = np.zeros_like(y)
    with Pool(processes=args.process) as pool:
        results = pool.starmap(process_output, pool_args)
        for i, (port_idx, yy_mor) in enumerate(results):
            y_mor_combined[port_idx, :] = yy_mor.squeeze()

    return  y_mor_combined

def process_output(i, idx_list, x0, IS, VS, m):
    # try:
    print("Process:{}".format(i))
    global global_G, global_C, global_B, global_lu
    # lu = sp.sparse.linalg.splu(global_M)
    B_i = global_B[:, idx_list] if len(idx_list) > 0 else global_B[:, [i]]
    q = m * B_i.shape[1]
    tic = time.time()
    
    XX = PRIMA.PRIMA_sp(global_C, B_i, q, global_lu)
    Cr_i = (XX.T @ global_C) @ XX
    Gr_i = (XX.T @ global_G) @ XX
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

    yy_mor = y_mor[locat,:]
    s2 = time.time() - s1
    print(f"Process{i}--Simulation for output {i+1} took {s2:.4f} seconds.")
    return i, yy_mor
    # except Exception as e:
    #     import traceback
    #     print(f"[Worker {i}] Error: {e}")
    #     traceback.print_exc()
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='DeMOR_data_generate')
    parser.add_argument('--circuit', type=int, default=2, help='Circuit number')
    parser.add_argument("--port_num", type=int, default= 2000)
    parser.add_argument("--threshold", type=int, default= 0)
    parser.add_argument("--generate", type=int, default= 1)
    parser.add_argument("--data_num", type=int, default= 200)
    parser.add_argument("--process", type=int, default= 10)
    args = parser.parse_args()
    save_path = os.path.join('./Exp_res/DeMOR_data/{}t/'.format(args.circuit))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(f"{save_path}/DeMOR.log")])
    logging.info(args)
    data = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/ibmpg{}t.mat".format(args.circuit))
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
    low_f = []
    high_f = []
    in_all = []
    if args.generate:
        # Generate data
        for i in range(args.data_num):
            logging.info("Generating {}th input data".format(i))
            t0 = 0
            tf = 1e-09
            dt = 1e-11
            srcType = 'pulse'
            IS, VS = generate_udiff(args.port_num, args.circuit, seed = i)
            x0 = np.zeros((C.shape[0], 1))

            t1 = time.time()
            xAll, time1, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, C, -G, B, VS, IS, x0, srcType)
            y = B.T@xAll
            t2 = time.time() - t1
            logging.info(f"Origin data generate time for {i}th input: {t2:.4f} seconds")

            t3 = time.time()
            y_low = generate_DeMOR_data(G, C, B, args, save_path, seed=i)
            t4 = time.time() - t3
            logging.info(f"DeMOR data generation time for {i}th input: {t4:.4f} seconds")

            low_f.append(y_low)
            high_f.append(y)
            in_all.append(uAll)
        low_f = np.array(low_f)
        high_f = np.array(high_f)
        time_save = np.array(time1)
        inall = np.array(in_all)
        np.save(os.path.join(save_path, 'mf_inall.npy'), inall)
        np.save(os.path.join(save_path, 'mf_time.npy'), time_save)
        np.save(os.path.join(save_path, 'mf_low_f.npy'), low_f)
        np.save(os.path.join(save_path, 'mf_high_f.npy'), high_f)
        logging.info("Data generation completed and saved.")

    else:    
        save_rga(G, C, B, save_path)
    # check_DeMOR(G, C, B, args, save_path, threshold=args.threshold)