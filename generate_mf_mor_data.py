'''
2025-3-8
generate mf_mor data for model
'''
import scipy.io as spio
import random
from utils.tdIntLinBE_new import *
import numpy as np
from SVDMOR import svdmor
import utils.PRIMA as PRIMA
import matplotlib.pyplot as plt

def generate_u(port_num, seed):
    VS = []
    VS = np.array(VS)
    vl_range = (1.748e-5, 4.139e-5)
    vh_range = (0.036, 0.121)
    # td_range = (0, 1.2e-9)
    td_range = (0, 1e-9)
    tr_range = (1e-10, 3e-10)
    tf_range = (1e-10, 3e-10) 
    pw_range = (1e-10, 3e-10)
    # tr = 1e-10
    # tf = 1e-10
    # pw = 1e-11
    # per_range = (2e-9, 3e-9)
    per = 2e-9
    random.seed(seed)
    vl = random.uniform(*vl_range)
    vh = random.uniform(*vh_range)
    td = random.uniform(*td_range)
    tr = random.uniform(*tr_range)
    # tr = 1e-10
    tf = random.uniform(*tf_range)
    # tf = 1e-10
    pw = random.uniform(*pw_range)
    # pw = 1e-11
    # per = random.uniform(*per_range)
    IS = []
    in_data = []
    in_data.append([vl, vh, td, tr, tf, pw, per])
    for _ in range(port_num):
        IS.append([vl, vh, td, tr, tf, pw, per])
    IS = np.vstack(IS)
    return IS, VS, in_data

if __name__ == '__main__':
    data = spio.loadmat("./IBM_transient/ibmpg1t.mat")
    C, G, B = data['E'] * 1e-0, data['A'], data['B']
    port_num = 100
    B = B.tocsc()
    C = C.tocsc()
    G = G.tocsc()
    B = B[:, 0:port_num]
    # output matrix
    O = B
    
    t0 = 0
    t_all = 2e-09
    dt = 1e-11
    data_length = 200
    data_num = 200

    N = C.shape[0]

    tic = datetime.now()
    left, right, B_r, O_r = svdmor(G, B, B, threshold = 2)
    toc = datetime.now()
    ts = toc - tic
    print("SVD completed. Time used: ", str(ts), 's')

    f = np.array([1e2, 1e9])
    m = 2
    s = 1j * 2 * np.pi * f
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

    low_f = []
    high_f = []
    input_data = []

    Br_2 = Br_2@right
    x0 = np.zeros((N, 1))
    xr1 = XX_2.T@ x0
    for i in range(data_num):
        print("Generating {}th input data".format(i))
        seed = i
        IS, VS, in_data = generate_u(port_num, seed)
        xAll, time, dtAll, uAll = tdIntLinBE_new(t0, t_all, dt, C, -G, B, VS, IS, x0, srcType = 'pulse')
        y = O.T@xAll
        xAll_svd, time_svd, dtAll_svd, urAll_svd = tdIntLinBE_new(t0, t_all, dt, Cr_2, -Gr_2, Br_2, VS, IS, xr1, srcType = 'pulse')
        y_svd = (Or_2@left).T@ xAll_svd
        low_f.append(y_svd)
        high_f.append(y)
        input_data.append(in_data)
    low_f = np.array(low_f)
    high_f = np.array(high_f)
    input_data = np.array(input_data)
    input_data = np.squeeze(input_data)
    np.save('train_data/mf_in.npy', input_data)
    np.save('train_data/mf_low_f.npy', low_f)
    np.save('train_data/mf_high_f.npy', high_f)

    # yy = np.zeros((y.shape[1]))
    # for i in range(y.shape[0]):
    #     yy += np.real(y[i, :])
    # yy_svd = np.zeros((y_svd.shape[1]))
    # for i in range(y_svd.shape[0]):
    #     yy_svd += np.real(y_svd[i, :])    
    # plt.figure(figsize=(8, 5))
    # plt.plot(time, yy, color='black', linestyle='-.', marker='*', label='High-fidelity-data', markevery = 35, markersize=6, linewidth=1.5)
    # plt.plot(time, yy_svd, color='red', linestyle='--', marker='s', label='Low-fidelity-data', markersize=6, markevery = 45, linewidth=1.5)
    # plt.legend(fontsize=12)
    # plt.title("MF-data", fontsize=14)
    # plt.xlabel("Time (s)", fontsize=12)
    # plt.ylabel("result", fontsize=12)
    # plt.grid(alpha=0.5)
    # plt.tight_layout()
    # plt.show()
    # plt.close()    
    pass