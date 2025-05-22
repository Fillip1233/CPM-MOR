'''
2025-3-3
SVDMOR baseline
'''
import scipy.io as spio
import numpy as np
import utils.PRIMA as PRIMA
from datetime import datetime
from utils.tdIntLinBE_new import * 
import matplotlib.pyplot as plt
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp

def svdmor(G, in_mat, out_mat, threshold, load_path=None):
    
    lu = splu(G)
    # L = spsolve(lu.U.T, out_mat)
    # R = spsolve(lu.L, in_mat)
    if load_path == None:
        # L = lu.solve(out_mat.toarray(), trans='T')  # 直接求解 U.T x = out_mat
        # R = lu.solve(in_mat.toarray(), trans='N')
        # L = lu.L.T @ L
        # R = lu.U @ R
        L = spsolve(lu.U.T, out_mat)
        R = spsolve(lu.L, in_mat)
        B = sp.hstack([L, R])
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
    B_r = lu.L@ U_new
    O_r = lu.U.T@U_new

    return left, right, B_r, O_r


if __name__ == '__main__':
    # C = E, G = A
    data = spio.loadmat("./IBM_transient/ibmpg3t.mat")
    C, G, B = data['E'] * 1e-0, data['A'], data['B']
    port_num = 2000
    B = B.tocsc()
    C = C.tocsc()
    G = G.tocsc()
    B = B[:, 0:port_num]
    # output matrix
    O = B
    
    tic = datetime.now()
    left, right, B_r, O_r = svdmor(G, B, B, threshold = 2)
    toc = datetime.now()
    ts = toc - tic
    print("SVD completed. Time used: ", str(ts), 's')

    t0 = 0
    tf = 1e-09
    dt = 1e-11
    srcType = 'pulse'
    VS = []
    VS = np.array(VS)
    N = C.shape[0]
    is_num = int(port_num/2)
    IS1 = np.hstack(
        [np.zeros([is_num, 1]), 
            np.dot(np.ones([is_num, 1]), 0.01),
            np.dot(np.ones([is_num, 1]), tf) / 5,
            np.dot(np.ones([is_num, 1]), 1e-10),
            np.dot(np.ones([is_num, 1]), 1e-10),
            np.dot(np.ones([is_num, 1]), 5e-10),
            np.dot(np.ones([is_num, 1]), 1e-09)])
    IS2 = np.hstack([
        np.zeros([is_num, 1]),
        np.dot(np.ones([is_num, 1]), 0.2),
        np.dot(np.ones([is_num, 1]), tf) / 6,
        np.dot(np.ones([is_num, 1]), 2e-09),
        np.dot(np.ones([is_num, 1]), 2e-09),
        np.dot(np.ones([is_num, 1]), 4e-09),
        np.dot(np.ones([is_num, 1]), 1e-08)])
    IS = np.vstack([IS1, IS1])
    x0 = np.zeros((N, 1))

    f = np.array([1e2])
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
    nr_2 = Cr_2.shape[0]
    toc = datetime.now()
    tprima = toc - tic
    print("SVD-PRIMA-MOR:")
    print('MOR completed. Time used: ', str(tprima), 's')
    print('The original order is', N, 'the reduced order is', nr_2)

    xAll, time, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, C, -G, B, VS, IS, x0, srcType)
    
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
    xAll_svd, time_svd, dtAll_svd, urAll_svd = tdIntLinBE_new(t0, tf, dt, Cr_2, -Gr_2, Br_2, VS, IS, xr1, srcType)
    y_svd = (Or_2@left).T@ xAll_svd
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
    plt.savefig("SVDMOR.png", dpi=300)
    plt.close()
    pass