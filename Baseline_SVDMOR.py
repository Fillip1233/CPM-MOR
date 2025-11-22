'''
2025-11-9
Baseline SVDMOR method
'''
import scipy.io as spio
import numpy as np
import utils.PRIMA as PRIMA
from datetime import datetime
from utils.tdIntLinBE_new import * 
import matplotlib.pyplot as plt
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import spsolve
from generate_mf_mor_data import generate_u, generate_udiff, generate_uover
from utils.calculate_metrix import calculate_metrix
import pandas as pd
import scipy.sparse as sp
import argparse
import os
import logging

def svdmor(G, in_mat, out_mat, threshold, load_path=None):
    if load_path == None:
        L = out_mat
        R = spsolve(G, in_mat)
        B = sp.hstack([out_mat, R])
        p = L.shape[1]
        q = R.shape[1]
        E_l = sp.vstack([
                sp.eye(p),
                sp.csr_matrix((q, p))
        ])
        E_r = sp.vstack([  
                sp.csr_matrix((p, q)), 
                sp.eye(q),              
        ])
        B = B.toarray()
        U, S, V = np.linalg.svd(B, full_matrices=False)
    else:
        p = out_mat.shape[1]
        q = in_mat.shape[1]
        E_l = sp.vstack([
                sp.eye(p),
                sp.csr_matrix((q, p))
        ])
        E_r = sp.vstack([  
                sp.csr_matrix((p, q)), 
                sp.eye(q),              
        ])
        U = np.load(load_path+r"/U.npy")
        S = np.load(load_path+r"/S.npy")
        V = np.load(load_path+r"/V.npy")
    indices = S >= threshold
    U_new = U[:, indices]               
    S_new = S[indices]                  
    V_new = V[indices, :]
    left = np.diag(S_new)@V_new@E_l
    right = np.diag(S_new)@V_new@E_r
    B_r = G @ U_new
    O_r = U_new

    return left, right, B_r, O_r


if __name__ == '__main__':
    # C = E, G = A
    parser = argparse.ArgumentParser(description='baseline SVDMOR')
    parser.add_argument('--circuit', type=int, default = 1, help='Circuit number')
    parser.add_argument("--port_num", type=int, default = 10000)
    parser.add_argument("--threshold", type=float, default = 1.1)
    args = parser.parse_args()
    save_path = os.path.join('/home/fillip/home/CPM-MOR/Baseline/SVDMOR/{}t/'.format(args.circuit))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(f"{save_path}/SVDMOR.log")])
    logging.info(args)
    data = spio.loadmat("./IBM_transient/{}t_SIP_gt.mat".format(args.circuit))
    C, G, B = data['C_final'] * 1e-0, data['G_final'], data['B_final']
    port_num = args.port_num

    data_path = "/home/fillip/home/CPM-MOR/Baseline/SVDMOR/{}t".format(args.circuit)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    B = B.tocsc()
    C = C.tocsc()
    G = G.tocsc()
    B = B[:, 0:port_num]
    # output matrix
    O = B
    N = C.shape[0]
    tic = datetime.now()
    left, right, B_r, O_r = svdmor(G, B, O, threshold = args.threshold, load_path='/home/fillip/home/CPM-MOR/Baseline/SVDMOR/{}t/'.format(args.circuit))    
    toc = datetime.now()
    ts = toc - tic
    logging.info("SVD completed. Time used: {}s".format(ts))

    t0 = 0
    tf = 2e-09
    dt = 1e-11
    srcType = 'pulse'
    # IS, VS = generate_udiff(args.port_num, args.circuit, seed = 399)
    IS, VS = generate_uover(args.port_num, args.circuit, seed = 99)
    x0 = np.zeros((N, 1))

    f = np.array([1e9])
    m = 2
    s = 1j * 2 * np.pi * f  # array of expansion points

    q = m * port_num
    tic = datetime.now()

    # XX_1 = PRIMA.PRIMA_mp(C, G, B, s, q)
    # Cr_1 = (XX_1.T @ C) @ XX_1
    # Gr_1 = (XX_1.T @ G) @ XX_1
    # Br_1 = XX_1.T @ B
    # Or_1 = XX_1.T @ O
    # nr_1 = Cr_1.shape[0]
    # toc = datetime.now()
    # tprima = toc - tic

    # print("Origin PRIMA MOR:")
    # print('PRIMA completed. Time used: ', str(tprima), 's')
    # print('The original order is', N, 'the reduced order is', nr_1)
    

    q_svd = m * B_r.shape[1]
    # perform svd mor
    tic = datetime.now()
    XX_2 = PRIMA.PRIMA_mp(C, G, B_r, s, q_svd)
    Cr_2 = (XX_2.T@ C)@XX_2
    Gr_2 = (XX_2.T@ G)@XX_2
    Br_2 = (XX_2.T@ B_r)
    Or_2 = XX_2.T@ O_r
    np.savez(save_path + '/svdmor_1t.npz', Cr_2 = Cr_2, Gr_2 = Gr_2, Br_2 = Br_2, Or_2 = Or_2, XX_2 = XX_2)
    # data = np.load("/home/fillip/home/CPM-MOR/svdmor.npz")
    # Cr_2 = data['Cr_2']
    # Gr_2 = data['Gr_2']
    # Br_2 = data['Br_2']
    # Or_2 = data['Or_2']
    # XX_2 = data['XX_2']
    nr_2 = Cr_2.shape[0]
    toc = datetime.now()
    tprima = toc - tic
    logging.info("SVD-PRIMA-MOR:")
    logging.info('MOR completed. Time used:{}'.format(tprima))
    logging.info('The original order is{}'.format(N))
    logging.info('The reduced order is{}'.format(nr_2))

    xAll, time, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, C, G, B, VS, IS, x0, srcType)
    
    y = O.T@xAll

    yy = np.zeros((y.shape[1]))
    for i in range(y.shape[0]):
        yy += np.real(y[i, :])

    # xr0 = XX_1.T@ x0
    # xrAll, time, dtAll, urAll = tdIntLinBE_new(t0, tf, dt, Cr_1, -Gr_1, Br_1, VS, IS, xr0, srcType)
    # y_mor = Or_1.T@xrAll
    # yy_mor = np.zeros((y_mor.shape[1]))
    # for i in range(y_mor.shape[0]):
    #     yy_mor += np.real(y_mor[i, :])
    
    xr1 = XX_2.T@ x0
    Br_2 = Br_2@right
    logging.info("B reduced order: {}".format(Br_2.shape[0]))
    xAll_svd, time_svd, dtAll_svd, urAll_svd = tdIntLinBE_new(t0, tf, dt, Cr_2, Gr_2, Br_2, VS, IS, xr1, srcType)
    y_svd = (Or_2@left).T@ xAll_svd
    y_save = y_svd[3000,:]
    np.save("svdmor_over_1t.npy", y_save)

    recording = {'rmse':[], 'nrmse':[], 'r2':[],'mae':[], 'relative_error':[]}
    metrics = calculate_metrix(y_test = np.abs(y), y_mean_pre = np.abs(y_svd))
    recording['rmse'].append(metrics['rmse'])
    recording['nrmse'].append(metrics['nrmse'])
    recording['r2'].append(metrics['r2'])
    recording['mae'].append(metrics['mae'])
    recording['relative_error'].append(metrics['relative_error'])
    record = pd.DataFrame(recording)
    record.to_csv(data_path + '/res_svdmor.csv', index = False)

    yy_svd = np.zeros((y_svd.shape[1]))
    for i in range(y_svd.shape[0]):
        yy_svd += np.real(y_svd[i, :])

    plt.figure(figsize=(8, 5))

    plt.plot(time, yy, color='black', linestyle='-.', marker='*', label='Origin', markevery = 35, markersize=6, linewidth=1.5)

    # plt.plot(time, yy_mor, color='blue', linestyle='-', marker='o', label='PRIMA', markersize=6,markevery = 30, linewidth=1.5)
    plt.plot(time, yy_svd, color='red', linestyle='--', marker='s', label='SVD-MOR', markersize=6, markevery = 45, linewidth=1.5)
    plt.legend(fontsize=12)
    plt.title("Method compare", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("result", fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path+"SVDMOR_{}t.png".format(args.circuit))
    plt.close()
    pass