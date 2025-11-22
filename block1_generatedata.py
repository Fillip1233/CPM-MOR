'''
2025-11-16
'''
import argparse
import os
import sys
import logging
import numpy as np
import scipy.io as spio
from scipy.sparse import coo_matrix, block_diag
import matplotlib.pyplot as plt
from generate_mf_mor_data import generate_u, generate_udiff, generate_uover
from utils.tdIntLinBE_new import *
from verify_msipbdsm import coo_from_triplets, load_coo_txt
from memory_profiler import memory_usage
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate mf_mor data for model")
    parser.add_argument("--port_num", type=int, default= 40000)
    parser.add_argument("--circuit", type=int, default = 6)
    parser.add_argument("--generate", type=int, default= 0)
    parser.add_argument("--block", type=int, default= 1)
    parser.add_argument("--data_num", type=int, default= 400)
    args = parser.parse_args()
    save_path = os.path.join(sys.path[0], 'MSIP_BDSM/train_data/{}t_2per_1block'.format(args.circuit))
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

    # Cr = load_coo_txt(f'/home/fillip/home/My_MUMPS_5.7.1/examples/4w1/SIP_C_sip',to = 'csr')
    # Gr = load_coo_txt(f'/home/fillip/home/My_MUMPS_5.7.1/examples/4w1/SIP_G_sip',to = 'csr')
    Cr = load_coo_txt(f'/home/fillip/home/My_MUMPS_5.7.1/examples/4w3/SIP_C',to = 'csr')
    Gr = load_coo_txt(f'/home/fillip/home/My_MUMPS_5.7.1/examples/4w3/SIP_G',to = 'csr')

    # Cr = load_coo_txt(f'/home/fillip/home/My_MUMPS_5.7.1/examples/result/{args.circuit}t/SIP_C',to = 'csr')
    # Gr = load_coo_txt(f'/home/fillip/home/My_MUMPS_5.7.1/examples/result/{args.circuit}t/SIP_G',to = 'csr')

    Cr = Cr.tocsc()
    Gr = Gr.tocsc()
    Br = B1[-Gr.shape[0]:,:port_num]

    Cr = Cr[:5000,:5000]
    Gr = Gr[:5000,:5000]
    Br = Br[:5000,:5000]

    f = 9
    t0 = 0
    tf = 2e-09
    dt = 1e-11
    srcType = 'pulse'

    low_f = []

    if args.generate == 1:
        for i in range(data_num):
            logging.info("Generating {}th input data".format(i))
            seed = i
            t1 = time.time()
            IS, VS = generate_udiff(port_num, args.circuit, seed)

            t3 = time.time()
            xr0 = np.zeros((Cr.shape[0], 1))
            xrAll, time2, dtAll, urAll = tdIntLinBE_new(t0, tf, dt, Cr, Gr, Br, VS, IS, xr0, srcType)
            y_mor = Br.T@xrAll
            t4 = time.time()
            logging.info("Low-fidelity data generated. Time used: {}s ".format(t4-t3))

            low_f.append(y_mor)
        
        low_f = np.array(low_f)
        np.save(save_path+'/mf_low_f_b4.npy', low_f)
        logging.info("Data saved")
    
    else:
        IS, VS = generate_udiff(5000, args.circuit, seed=1)

        xr0 = np.zeros((Cr.shape[0], 1))
        # xrAll, time2, dtAll, urAll = tdIntLinBE_new(t0, tf, dt, Cr, Gr, Br, VS, IS, xr0, srcType)
        mem_usage = memory_usage((tdIntLinBE_new, (t0, tf, dt, Cr, Gr, Br, VS, IS, xr0, srcType)), max_usage=True)
        print(f"Maximum memory usage: {mem_usage/1024} GB")
        # y_mor = Br.T@xrAll

        # yy_mor = y_mor[0,:]
        # for i in range(1, port_num):
        #     yy_mor = yy_mor + y_mor[i,:]

        # plt.plot(time2, yy_mor, color='#6B3F69', linestyle='--', marker='*', label='MUL-SIP-BDSM', markevery = 25, markersize=6, linewidth=1.5)
        # plt.xlabel("Time (s)", fontsize=12)
        # plt.ylabel("Response result (V)", fontsize=12)
        # plt.legend(fontsize=12)
        # plt.grid()
        # plt.title("SIP-BDSM test", fontsize=14)
        # plt.tight_layout()
        # plt.savefig('SIP-4block.png'.format(args.circuit,f), dpi=300)
        pass

