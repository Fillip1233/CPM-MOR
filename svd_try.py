'''
2025-3-1
svd 方法的尝试
'''
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.io as spio
from scipy.sparse import csc_matrix
from datetime import datetime
import utils.PRIMA as PRIMA
from utils.tdIntLinBE_new import *
from code_try.tdIntLinBE_forsvd import *


if __name__ == '__main__':
    data = spio.loadmat("./IBM_transient/ibmpg1t.mat")

    C, G, B = data['E'] * 1e-0, data['A'], data['B']

    N = C.shape[0]
    NB = B.shape[1]

    Nb = 40 
    B = B[:, 0:Nb]
    O = B
    t0 = 0
    tf = 1e-08
    dt = 1e-11

    srcType = 'pulse'

    #     PULSE(V1 V2 TD TR TF PW PER)
    VS = []
    VS = np.array(VS)
    # main.m:32
    # 电流源的输入都是一样可能看不出啥
    is_num = int(Nb/2)
    IS1 = np.hstack(
        [np.zeros([is_num, 1]), 
            np.dot(np.ones([is_num, 1]), 0.1),
            np.dot(np.ones([is_num, 1]), tf) / 5,
            np.dot(np.ones([is_num, 1]), 1e-09),
            np.dot(np.ones([is_num, 1]), 1e-09),
            np.dot(np.ones([is_num, 1]), 5e-09),
            np.dot(np.ones([is_num, 1]), 1e-08)])
    IS2 = np.hstack([
        np.zeros([is_num, 1]),
        np.dot(np.ones([is_num, 1]), 0.2),
        np.dot(np.ones([is_num, 1]), tf) / 6,
        np.dot(np.ones([is_num, 1]), 2e-09),
        np.dot(np.ones([is_num, 1]), 2e-09),
        np.dot(np.ones([is_num, 1]), 4e-09),
        np.dot(np.ones([is_num, 1]), 1e-08)])
    IS = np.vstack([IS1, IS2])
    x0 = np.zeros((N, 1))
    # B[:,1] = 0


    xAll, time1, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, C, -G, B, VS, IS, x0, srcType)
    y = O.T@xAll
    yy = np.zeros((y.shape[1]))
    for i in range(y.shape[0]):
        yy += np.real(y[i, :])


    # simplify B using svd
    ## method 1 -> turn to dense matrix
    B_1 = csc_matrix(B, dtype=np.float64)
    t1 = time.time()
    B_dense = B_1.toarray()
    U, S, V = np.linalg.svd(B_dense,full_matrices=False)
    t2 = time.time()
    print('time for svd:', t2-t1)
    threshold = 1.1
    indices = S >= threshold
    U_new = U[:, indices]               
    S_new = S[indices]                  
    V_new = V[indices, :]
    assign_matrix = np.diag(S_new)@V_new 

    # mor with PRIMA
    f = np.array([1e2, 1e9])  # array of targeting frequencies
    m = 2  # number of moments to match per expansion point (same for all points here, though can be made different)
    s = 1j * 2 * np.pi * f  # array of expansion points
    q = m * Nb
    tic = datetime.now()

    XX = PRIMA.PRIMA_mp(C, G, B, s, q)
    Cr_1 = (XX.T @ C) @ XX
    Gr_1 = (XX.T @ G) @ XX
    Br_1 = XX.T @ B
    Or_1 = XX.T @ O
    nr = Cr_1.shape[0]
    toc = datetime.now()
    tprima = toc - tic

    print("Origin PRIMA MOR:")
    print('PRIMA completed. Time used: ', str(tprima), 's')
    print('The original order is', N, 'the reduced order is', nr)
    
    xr0 = XX.T@ x0
    xrAll, time3, dtAll, urAll = tdIntLinBE_new(t0, tf, dt, Cr_1, -Gr_1, Br_1, VS, IS, xr0, srcType)
    y_mor = Or_1.T@xrAll
    yy_mor = np.zeros((y_mor.shape[1]))
    for i in range(y_mor.shape[0]):
        yy_mor += np.real(y_mor[i, :])

    # mor-prima -> U_new
    q_new = m * U_new.shape[1]
    tic = datetime.now()
    XX_1 = PRIMA.PRIMA_mp(C, G, U_new, s, q_new)
    Cr_2 = (XX_1.T @ C) @ XX_1
    Gr_2 = (XX_1.T @ G) @ XX_1
    Br_2 = XX_1.T @ U_new
    Or_2 = XX_1.T @ U_new
    nr = Cr_2.shape[0]
    toc = datetime.now()
    tprima = toc - tic

    print("SVD-PRIMA MOR:")
    print('PRIMA completed. Time used: ', str(tprima), 's')
    print('The original order is', N, 'the reduced order is', nr)
    xr1 = XX_1.T@ x0

    B_re = Br_2@assign_matrix
    xrAll2, time2, dtAll2, urAll2 = tdIntLinBE_new(t0, tf, dt, Cr_2, -Gr_2, B_re, VS, IS, xr1, srcType)
    y_mor2 = (Or_2@assign_matrix).T@xrAll2
    yy_svd = np.zeros((y_mor2.shape[1]))
    for i in range(y_mor2.shape[0]):
        yy_svd += np.real(y_mor2[i, :])


    plt.figure(figsize=(8, 5))
    plt.plot(time2, yy, color='black', linestyle='-.', marker='*', label='Origin', markevery = 35, markersize=6, linewidth=1.5)
    plt.plot(time2, yy_mor, color='blue', linestyle='-', marker='o', label='PRIMA', markersize=6,markevery = 30, linewidth=1.5)
    plt.plot(time2, yy_svd, color='red', linestyle='--', marker='s', label='my-svd-mor', markersize=6, markevery = 45, linewidth=1.5)
    plt.legend(fontsize=12)
    plt.title("Method compare", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("result", fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()
    pass