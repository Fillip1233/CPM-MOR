import argparse
import scipy.io as spio
from scipy.sparse import identity
import matplotlib.pyplot as plt
from generate_mf_mor_data import generate_u, generate_udiff
from utils.tdIntLinBE_new import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vertify preproces')
    parser.add_argument('--circuit', type=int, default=1, help='Circuit number')
    parser.add_argument("--port_num", type=int, default= 2000)
    args = parser.parse_args()
    data = spio.loadmat("/home/fillip/桌面/CPM-MOR/IBM_transient/{}t_SIP.mat".format(args.circuit))
    port_num = args.port_num
    C1, G1, B1 = data['C_final'] * 1e-0, data['G_final'], data['B_final']
    B1 = B1.tocsc()
    C1 = C1.tocsc()
    G1 = G1.tocsc()
    B1 = B1[:,:port_num]

    data = spio.loadmat("/home/fillip/桌面/CPM-MOR/IBM_transient/{}t_SIP.mat".format(args.circuit))
    C2, G2, B2 = data['C_final'] * 1e-0, data['G_final'], data['B_final']
    B2 = B2.tocsc()
    C2 = C2.tocsc()
    G2 = G2.tocsc()
    B2 = B2[:,:port_num]

    # G2 = G2 + 1e-10 * identity(G2.shape[0], format="csc")
    C2 = C2 + 1e-20 * identity(C2.shape[0], format="csc")

    t0 = 0
    tf = 1e-09
    dt = 1e-11
    srcType = 'pulse'
    IS, VS = generate_udiff(args.port_num, args.circuit, seed = 0)
    x0 = np.zeros((C1.shape[0], 1))

    xAll, time1, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, C1, G1, B1, VS, IS, x0, srcType)
    y = B1.T@xAll

    xr0 = np.zeros((C2.shape[0], 1))
    xrAll, time2, dtAll, urAll = tdIntLinBE_new(t0, tf, dt, C2, G2, B2, VS, IS, xr0, srcType)
    y_mor = B2.T@xrAll

    yy = y[0,:]
    yy_mor = y_mor[0,:]
    for i in range(1, port_num):
        yy = yy + y[i,:]
        yy_mor = yy_mor + y_mor[i,:]

    plt.plot(time1, yy, color='#ED3F27', linestyle='-.', marker='*', label='GT', markevery = 35, markersize=6, linewidth=1.5)
    plt.plot(time2, yy_mor, color='#6B3F69', linestyle='--', marker='*', label='preprocessed', markevery = 25, markersize=6, linewidth=1.5)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Response result (V)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.title("SIP-verify test", fontsize=14)
    plt.tight_layout()
    plt.savefig('SIP_{}t_verify.png'.format(args.circuit), dpi=300)
    pass
