'''
2025-10-14
'''
import argparse
import os
import sys
import logging
import numpy as np
import scipy.io as spio
from scipy.sparse import coo_matrix, block_diag
import matplotlib.pyplot as plt
from generate_mf_mor_data import generate_u, generate_udiff
from utils.tdIntLinBE_new import *
from verify_msipbdsm import coo_from_triplets, load_coo_txt
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate mf_mor data for model")
    parser.add_argument("--port_num", type=int, default= 10000)
    parser.add_argument("--circuit", type=int, default = 1)
    parser.add_argument("--generate", type=int, default= 0)
    parser.add_argument("--block", type=int, default= 2)
    parser.add_argument("--data_num", type=int, default= 400)
    args = parser.parse_args()
    save_path = os.path.join(sys.path[0], 'MSIP_BDSM/train_data/{}t_2per'.format(args.circuit))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(f"{save_path}/data_generate.log")])
    logging.info(args)

    data = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/{}t_SIP_gt.mat".format(args.circuit))
    port_num = args.port_num
    data_num = args.data_num
    block = args.block
    C1, G1, B1 = data['C_final'] * 1e-0, data['G_final'], data['B_final']
    B1 = B1.tocsc()
    C1 = C1.tocsc()
    G1 = G1.tocsc()
    B1 = B1[:,:port_num]

    Cr = load_coo_txt(f'/home/fillip/home/My_MUMPS_5.7.1/examples/result/{args.circuit}t/SIP_C',to = 'csr')
    Gr = load_coo_txt(f'/home/fillip/home/My_MUMPS_5.7.1/examples/result/{args.circuit}t/SIP_G',to = 'csr')
    Cr = Cr.tocsc()
    Gr = Gr.tocsc()

    data1 = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/{}t_SIP1.mat".format(args.circuit))
    B2 = data1['B_final']
    B2 = B2.tocsc()

    schur_size = port_num / block
    B2 = B2[-5000:,:5000]
    data2 = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/{}t_SIP2.mat".format(args.circuit))
    B3 = data2['B_final']
    B3 = B3.tocsc()
    B3 = B3[-5000:,5000:10000]
    Br = block_diag((B2, B3), format="csc")

    f = 9
    t0 = 0
    tf = 2e-09
    dt = 1e-11
    srcType = 'pulse'

    low_f = []
    high_f = []
    in_all = []

    if args.generate == 1:
        for i in range(data_num):
            logging.info("Generating {}th input data".format(i))
            seed = i
            t1 = time.time()
            IS, VS = generate_udiff(port_num, args.circuit, seed)
            x0 = np.zeros((C1.shape[0], 1))
            xAll, time1, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, C1, G1, B1, VS, IS, x0, srcType)
            y = B1.T@xAll
            t2 = time.time()
            logging.info("High-fidelity data generated. Time used: {}s ".format(t2-t1))

            t3 = time.time()
            xr0 = np.zeros((Cr.shape[0], 1))
            xrAll, time2, dtAll, urAll = tdIntLinBE_new(t0, tf, dt, Cr, Gr, Br, VS, IS, xr0, srcType)
            y_mor = Br.T@xrAll
            t4 = time.time()
            logging.info("Low-fidelity data generated. Time used: {}s ".format(t4-t3))

            high_f.append(y)
            low_f.append(y_mor)
            in_all.append(uAll)
        
        low_f = np.array(low_f)
        high_f = np.array(high_f)
        inall = np.array(in_all)
        time1 = np.array(time1)
        np.save(save_path+'/mf_time.npy', time1)
        np.save(save_path+'/mf_low_f.npy', low_f)
        np.save(save_path+'/mf_high_f.npy', high_f)
        np.save(save_path+'/mf_inall.npy', inall)
        logging.info("Data saved")
    
    else:
        IS, VS = generate_udiff(port_num, args.circuit, seed=1)
        x0 = np.zeros((C1.shape[0], 1))
        xAll, time1, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, C1, G1, B1, VS, IS, x0, srcType)
        y = B1.T@xAll

        xr0 = np.zeros((Cr.shape[0], 1))
        xrAll, time2, dtAll, urAll = tdIntLinBE_new(t0, tf, dt, Cr, Gr, Br, VS, IS, xr0, srcType)
        y_mor = Br.T@xrAll

        yy = y[0,:]
        yy_mor = y_mor[0,:]
        for i in range(1, port_num):
            yy = yy + y[i,:]
            yy_mor = yy_mor + y_mor[i,:]

        plt.plot(time1, yy, color='#ED3F27', linestyle='-.', marker='*', label='GT', markevery = 35, markersize=6, linewidth=1.5)
        plt.plot(time2, yy_mor, color='#6B3F69', linestyle='--', marker='*', label='MUL-SIP-BDSM', markevery = 25, markersize=6, linewidth=1.5)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Response result (V)", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid()
        plt.title("SIP-BDSM test", fontsize=14)
        plt.tight_layout()
        plt.savefig('SIP-BDSM_{}t_{}_1w.png'.format(args.circuit,f), dpi=300)
        pass



