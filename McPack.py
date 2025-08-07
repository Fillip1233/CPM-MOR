import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import operator
import time
import argparse
import os
import logging
from generate_mf_mor_data import generate_u, generate_udiff 

def mcpack(C, G, B, L, zQ, k, g, r):
    """
    McPack Algorithm (Single Expansion Point)
    
    Inputs:
        C, G: nxn sparse matrices
        B, L: nxm dense matrices
        zQ: complex scalar (expansion point)
        k: number of Krylov steps
        g: number of moments for projection
        r: rank after compression
    
    Outputs:
        C_hat, G_hat, B_hat, L_hat: reduced-order model matrices (complex)
    """
    # Step 1: Define A_zQ = -(G + zQ C)^-1 C
    #         R_zQ = (G + zQ C)^-1 B
    GzC = G + zQ * C

    A_zQ = spla.spsolve(GzC, -C)
    R_zQ = spla.spsolve(GzC, B)

    # 保存稀疏矩阵
    # from scipy.sparse import save_npz, load_npz
    # save_npz("A_zQ_2000_1e9.npz", A_zQ)
    # save_npz("R_zQ_2000_1e9.npz", R_zQ)

    # 加载
    # A_zQ = load_npz("A_zQ_2000_1e9.npz")
    # R_zQ = load_npz("R_zQ_2000_1e9.npz")
    print("Finished1/5 A_zQ and R_zQ")

    # Step 2: Compute block moments V_k = L^T A_zQ^k R_zQ, for k = 0 to g-1
    n, m = B.shape
    V_blocks = []
    Ak = sp.eye(n, format='csc')  # A^0 = I
    Vk_list = []
    for i in range(g):
        if i > 0:
            Ak = A_zQ @ Ak
        Vk = L.T @ (Ak @ R_zQ)
        Vk_list.append(Vk)
        V_blocks.append(Vk.toarray() if sp.issparse(Vk) else Vk)
    V = np.hstack(V_blocks)
    print("Finished2/5 V")

    # Step 3: Truncated SVD to rank r
    U, S, Vh = np.linalg.svd(V, full_matrices=False)
    V_r = Vh[:r, :].T
    U_r = U[:, :r]
    print("Finished3/5 SVD")

    # Step 4: Approximate block moments with rank-r basis
    U_k_r_list = []
    for i in range(g):
        # Vk = L.T @ (Ak @ R_zQ)
        # Uk, Sk, Vhk = np.linalg.svd(Vk.toarray(), full_matrices=False)
        Uk_r = Vk_list[i] @ V_r[i*R_zQ.shape[1]:(i+1)*R_zQ.shape[1],:]
        # Uk_r = (L.T @ (Ak @ R_zQ)) @ V_r[i*R_zQ.shape[1]:(i+1)*R_zQ.shape[1],:]
        U_k_r_list.append(Uk_r)
    # U_k_r_list.append(U_r)

    # Step 5: Polynomial coefficients Qj = Uk_r for (s - zQ)^j
    Q_list = U_k_r_list
    print("Finished4/5 Q")

    # Step 6: Construct X_i recursively
    X_list = []
    X0 = R_zQ @ Q_list[0]
    X_list.append(X0)

    for i in range(1, g):
        Xi = A_zQ @ X_list[-1] + R_zQ @ Q_list[i]
        X_list.append(Xi)

    for i in range(g, k):
        Xi = A_zQ @ X_list[-1]
        X_list.append(Xi)

    # Step 7: Orthonormalize [X0, ..., X_{k-1}]
    V_all = np.hstack(X_list)
    V_basis, R = np.linalg.qr(V_all)
    d = np.abs(np.diag(R))
    p = d > max(d) * 1e-3
    V_basis = V_basis[:, p]

    print("Finished V_basis")

    # Step 8: Project to reduced model
    C_hat = V_basis.T @ C @ V_basis
    G_hat = V_basis.T @ G @ V_basis
    B_hat = V_basis.T @ B
    L_hat = V_basis.T @ L

    return C_hat, G_hat, B_hat, L_hat , V_basis

def mcpack_2(C, G, B, L, zQ, k, g, r, threshold=None):
    GzC = G + zQ * C
    lu = sp.linalg.splu(GzC)
    if type(B) is not np.ndarray:
        B = B.toarray()
    R_zQ = lu.solve(B)
    print("Finished R_zQ")

    V_blocks = []
    V_blocks.append(R_zQ)
    for i in range(g):
        logging.info(f"Computing V_block {i+1}")
        V1 = C @ V_blocks[i]
        V_blocks.append(lu.solve(V1))
    V = np.hstack(V_blocks)
    V_L = L.T @ V
    
    U, S, Vh = np.linalg.svd(V_L, full_matrices=False)

    if threshold is not None:
        indices = S >= threshold
        V_r = Vh[indices,:].T
    else:
        V_r = Vh[:r, :].T
        U_r = U[:, :r]
    print("Finished SVD")

    U_k_r_list = []
    X_list = []
    for i in range(g):
        Uk_r = L.T @ V_blocks[i] @ V_r[i*R_zQ.shape[1]:(i+1)*R_zQ.shape[1],:]
        U_k_r_list.append(Uk_r)
        if i == 0:
            X0 = R_zQ @ U_k_r_list[0]
            X_list.append(X0)
        else:
            CX = C @ X_list[-1]
            X_list.append(lu.solve(CX) + R_zQ @ U_k_r_list[-1])

    for i in range(g, k):
        CX = C @ X_list[-1]
        X_list.append(lu.solve(CX))
    
    print("Finished X_list")

    # Step 7: Orthonormalize [X0, ..., X_{k-1}]
    V_all = np.hstack(X_list)
    V_basis, R = np.linalg.qr(V_all)
    d = np.abs(np.diag(R))
    p = d > max(d) * 1e-3
    V_basis = V_basis[:, p]

    print("Finished V_basis")

    # Step 8: Project to reduced model
    C_hat = V_basis.T @ C @ V_basis
    G_hat = V_basis.T @ G @ V_basis
    B_hat = V_basis.T @ B
    L_hat = V_basis.T @ L

    return C_hat, G_hat, B_hat, L_hat , V_basis



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='McPack')
    parser.add_argument('--circuit', type=int, default=1, help='Circuit number')
    parser.add_argument("--port_num", type=int, default= 100)
    parser.add_argument("--threshold", type=int, default= 1)
    args = parser.parse_args()
    save_path = os.path.join('/home/fillip/home/CPM-MOR/Exp_res/McPack/{}t/'.format(args.circuit))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(f"{save_path}/McPack.log")])
    logging.info(args)
    data = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/ibmpg{}t.mat".format(args.circuit))
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
    # f = np.array([1e2])
    m = 2
    s = 1j * 2 * np.pi * 1e10
    # s = 2 * np.pi * 1e-9
    t1 = time.time()
    Cr, Gr, Br, Lr, XX = mcpack_2(C, G, B, O, s, m, m, r=50, threshold=args.threshold)
    nr_size = Cr.shape[0]
    logging.info("Original order size: {}".format(Node_size))
    logging.info("Reduced order size: {}".format(nr_size))

    t2 = time.time()
    logging.info("McPack time: {}".format(t2 - t1))
    logging.info("port_num: {}".format(Br.shape[0]))
    t0 = 0
    # main.m:24
    tf = 1e-09
    # main.m:24
    dt = 1e-11
    # main.m:24

    srcType = 'pulse'

    # if operator.eq(srcType, 'sin'):

    #     # main.m:28
    #     VS = []
    #     VS = np.array(VS)
    #     IS = np.hstack([np.ones([Nb, 1]), np.zeros([Nb, 1]), np.dot(10000000000.0, np.ones([Nb, 1])), np.zeros([Nb, 1]),
    #                     np.dot(np.ones([Nb, 1]), tf) / 10, np.dot(np.ones([Nb, 1]), 1)])
    # else:
    #     if operator.eq(srcType, 'pulse'):
    #         #     PULSE(V1 V2 TD TR TF PW PER)
    #         VS = []
    #         VS = np.array(VS)
    #         # main.m:32
    #         IS = np.hstack(
    #             [np.zeros([Nb, 1]), 
    #             np.dot(np.ones([Nb, 1]), 0.1),
    #             np.dot(np.ones([Nb, 1]), tf) / 5,
    #             np.dot(np.ones([Nb, 1]), 1e-10),
    #             np.dot(np.ones([Nb, 1]), 1e-10),
    #             np.dot(np.ones([Nb, 1]), 5e-10),
    #             np.dot(np.ones([Nb, 1]), 1e-09)])
    VS = []
    VS = np.array(VS)
    # is_num = int(port_num/2)
    # IS1 = np.hstack(
    #     [np.zeros([is_num, 1]), 
    #         np.dot(np.ones([is_num, 1]), 0.01),
    #         np.dot(np.ones([is_num, 1]), tf) / 5,
    #         np.dot(np.ones([is_num, 1]), 1e-10),
    #         np.dot(np.ones([is_num, 1]), 1e-10),
    #         np.dot(np.ones([is_num, 1]), 5e-10),
    #         np.dot(np.ones([is_num, 1]), 1e-09)])
    # IS2 = np.hstack([
    #     np.zeros([is_num, 1]),
    #     np.dot(np.ones([is_num, 1]), 0.2),
    #     np.dot(np.ones([is_num, 1]), tf) / 6,
    #     np.dot(np.ones([is_num, 1]), 2e-09),
    #     np.dot(np.ones([is_num, 1]), 2e-09),
    #     np.dot(np.ones([is_num, 1]), 4e-09),
    #     np.dot(np.ones([is_num, 1]), 1e-08)])
    # IS = np.vstack([IS1, IS1])
    IS, VS = generate_udiff(args.port_num, args.circuit, seed = 0)

    x0 = np.zeros((C.shape[0], 1))
    # B[:,1] = 0

    from utils.tdIntLinBE_new import *

    xAll, time1, dtAll, uAll = tdIntLinBE_adaptive(t0, tf, dt, C, -G, B, VS, IS, x0, srcType)
    y = O.T@xAll

    xr0 = XX.T@ x0
    # main.m:66
    xrAll, time2, dtAll, urAll = tdIntLinBE_adaptive(t0, tf, dt, Cr, -Gr, Br, VS, IS, xr0, srcType)
    # main.m:67
    xAll_mor = XX@xrAll

    y_mor = Lr.T@xrAll

    yy = y[0,:]
    yy_mor = y_mor[0,:]
    for i in range(1, port_num):
        yy = yy + y[i,:]
        yy_mor = yy_mor + y_mor[i,:]

    plt.plot(time1, yy, color='green', linestyle='-.', marker='*', label='GT', markevery = 35, markersize=6, linewidth=1.5)
    plt.plot(time2, yy_mor, color='orange', linestyle='--', marker='*', label='McPack', markevery = 25, markersize=6, linewidth=1.5)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Response result (V)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.title("McPack test", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path+'McPack_{}t_{}.png'.format(args.circuit,port_num), dpi=300)
    pass