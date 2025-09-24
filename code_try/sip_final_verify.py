import numpy as np
from scipy.sparse import csc_matrix
import scipy.io as spio
from scipy.sparse import identity
import argparse
from generate_mf_mor_data import generate_u, generate_udiff 
from utils.tdIntLinBE_new import *
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SIP')
    parser.add_argument('--circuit', type=int, default=1, help='Circuit number')
    parser.add_argument("--port_num", type=int, default= 2000)
    args = parser.parse_args()

    Ap1 = np.loadtxt("SIP_res/GS_Ap.txt", dtype=np.int32)   # 列指针
    Ai1 = np.loadtxt("SIP_res/GS_Ai.txt", dtype=np.int32)   # 行索引
    Ax1 = np.loadtxt("SIP_res/GS_Ax.txt", dtype=np.float64) # 非零值

    n = len(Ap1) - 1        # 矩阵维度
    # 构造 CSC 矩阵
    Gr = csc_matrix((Ax1, Ai1, Ap1), shape=(n, n))
    print("Matrix Gr shape:", Gr.shape)
    print("Matrix Gr nnz:", Gr.nnz)

    Ap2 = np.loadtxt("SIP_res/CS_Ap.txt", dtype=np.int32)   # 列指针
    Ai2 = np.loadtxt("SIP_res/CS_Ai.txt", dtype=np.int32)   # 行索引
    Ax2 = np.loadtxt("SIP_res/CS_Ax.txt", dtype=np.float64) # 非零值

    n = len(Ap2) - 1        # 矩阵维度
    # 构造 CSC 矩阵
    Cr = csc_matrix((Ax2, Ai2, Ap2), shape=(n, n))
    print("Matrix Cr shape:", Cr.shape)
    print("Matrix Cr nnz:", Cr.nnz)

    data = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/{}t_final.mat".format(args.circuit))
    port_num = args.port_num
    C1, G1, B1 = data['C_final'] * 1e-0, data['G_final'], data['B_final']
    port_num = args.port_num
    B1 = B1.tocsc()
    C1 = C1.tocsc()
    G1 = G1.tocsc()
    G1 = G1 + 1e-10 * identity(G1.shape[0], format="csc")
    C1 = C1 + 1e-20 * identity(C1.shape[0], format="csc")

    Br = B1[-Gr.shape[0]:, :]
    Br = Br[:, :port_num]
    B1 = B1[:, :port_num]

    t0 = 0
    tf = 1e-09
    dt = 1e-11
    srcType = 'pulse'
    VS = []
    VS = np.array(VS)
    IS, VS = generate_udiff(args.port_num, args.circuit, seed = 0)

    x0 = np.zeros((Cr.shape[0], 1))
    xAll, time1, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, Cr, Gr, Br, VS, IS, x0, srcType)
    y = Br.T@xAll

    xr0 = np.zeros((C1.shape[0], 1))
    xrAll, time2, dtAll, urAll = tdIntLinBE_new(t0, tf, dt, C1, G1, B1, VS, IS, xr0, srcType)
    y_mor = B1.T@xrAll
    

    yy = y[0,:]
    yy_mor = y_mor[0,:]
    for i in range(1, port_num):
        yy = yy + y[i,:]
        yy_mor = yy_mor + y_mor[i,:]
    
    port = 1
    plt.plot(time1, yy, color='purple', linestyle='-.', marker='*', label='SIP', markevery = 35, markersize=6, linewidth=1.5)
    plt.plot(time2, yy_mor, color='green', linestyle='--', marker='*', label='GT', markevery = 25, markersize=6, linewidth=1.5)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Response result (V)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.title("SIP test", fontsize=14)
    plt.tight_layout()
    plt.savefig('SIP_{}t_{}_res.png'.format(args.circuit,port_num,port), dpi=300)
    pass