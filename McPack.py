import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import operator
import time

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
    Ak = sp.eye(n, format='csc')
    for i in range(g):
        if i > 0:
            Ak = A_zQ @ Ak
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


if __name__ == "__main__":
    
    data = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/ibmpg2t.mat")
    port_num = 100
    Nb = 100
    C, G, B = data['E'] * 1e-0, data['A'], data['B']
    B = B.tocsc()
    C = C.tocsc()
    G = G.tocsc()
    B = B[:, 0:0+port_num]
    # output matrix
    O = B
    # f = np.array([1e2])
    m = 2
    # s = 1j * 2 * np.pi * 1e2
    s = 2 * np.pi * 1e-9
    t1 = time.time()
    Cr, Gr, Br, Lr, XX = mcpack(C, G, B, O, s, m, m, r=60)
    t2 = time.time()
    print("McPack time: ", t2 - t1)
    t0 = 0
    # main.m:24
    tf = 1e-09
    # main.m:24
    dt = 1e-11
    # main.m:24

    srcType = 'pulse'

    if operator.eq(srcType, 'sin'):

        # main.m:28
        VS = []
        VS = np.array(VS)
        IS = np.hstack([np.ones([Nb, 1]), np.zeros([Nb, 1]), np.dot(10000000000.0, np.ones([Nb, 1])), np.zeros([Nb, 1]),
                        np.dot(np.ones([Nb, 1]), tf) / 10, np.dot(np.ones([Nb, 1]), 1)])
    else:
        if operator.eq(srcType, 'pulse'):
            #     PULSE(V1 V2 TD TR TF PW PER)
            VS = []
            VS = np.array(VS)
            # main.m:32
            IS = np.hstack(
                [np.zeros([Nb, 1]), 
                np.dot(np.ones([Nb, 1]), 0.1),
                np.dot(np.ones([Nb, 1]), tf) / 5,
                np.dot(np.ones([Nb, 1]), 1e-10),
                np.dot(np.ones([Nb, 1]), 1e-10),
                np.dot(np.ones([Nb, 1]), 5e-10),
                np.dot(np.ones([Nb, 1]), 1e-09)])
    x0 = np.zeros((C.shape[0], 1))
    # B[:,1] = 0

    from utils.tdIntLinBE_new import *

    xAll, time1, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, C, -G, B, VS, IS, x0, srcType)
    y = O.T@xAll

    xr0 = XX.T@ x0
    # main.m:66
    xrAll, time1, dtAll, urAll = tdIntLinBE_new(t0, tf, dt, Cr, -Gr, Br, VS, IS, xr0, srcType)
    # main.m:67
    xAll_mor = XX@xrAll

    y_mor = Lr.T@xrAll
    plt.plot(time1, y[0, :], color='green', linestyle='-.', marker='*', label='GT', markevery = 35, markersize=6, linewidth=1.5)
    plt.plot(time1, y_mor[0, :], color='orange', linestyle='--', marker='*', label='McPack', markevery = 25, markersize=6, linewidth=1.5)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Response result (V)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.title("McPack test", fontsize=14)
    plt.tight_layout()
    plt.savefig('McPack_2t.png')
    pass