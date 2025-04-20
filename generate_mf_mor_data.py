'''
2025-3-8
generate mf_mor data for model
2025-4-6
add generate sin and over size data
'''
import scipy.io as spio
import random
from utils.tdIntLinBE_new import *
import numpy as np
from SVDMOR import svdmor
from svd_try import svd_try
import utils.PRIMA as PRIMA
import matplotlib.pyplot as plt
import time
import logging
import argparse
import os
import sys

def generate_u(port_num, circuit_size, seed):
    '''
    生成各个端口的输入都一致的波形
    '''
    VS = []
    VS = np.array(VS)
    # 根据spice的参数设置
    if circuit_size == 1:
        vl_range = (1.748e-5, 4.139e-5)
        vh_range = (0.036, 0.121)
    elif circuit_size == 2:
        vl_range = (5.308e-6, 9.672e-6)
        vh_range = (0.013, 0.024)
    elif circuit_size == 3:
        vl_range = (6.367e-7, 1.830e-6)
        vh_range = (0.002, 0.005)
    # elif circuit_size == 4:
    #     vl = 4.178e-8
    #     vh = 0.0001
    elif circuit_size == 5:
        vl_range = (3.764e-8, 6.408e-8)
        vh_range = (9.410e-5, 3.108e-4)
    elif circuit_size == 6:
        vl_range = (4.228e-8, 2.075e-7)
        vh_range = (1.488e-4, 5.244e-4)
    # td_range = (0, 1.2e-9)
    td_range = (0, 5e-11)
    tr_range = (5e-11, 8e-11)
    tf_range = (5e-11, 8e-11) 
    pw_range = (5e-11, 8e-11)
    per = 3e-10
    # tr = 1e-10
    # tf = 1e-10
    # pw = 1e-11
    # per_range = (2e-9, 3e-9)
    
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
    # in_data = []
    # in_data.append([vl, vh, td, tr, tf, pw, per])
    for _ in range(port_num):
        IS.append([vl, vh, td, tr, tf, pw, per])
    IS = np.vstack(IS)
    return IS, VS

def generate_udiff(port_num, circuit_size, seed):
    '''
    生成各个端口输入不一致的数据
    '''
    VS = []
    VS = np.array(VS)
    if circuit_size == 1:
        vl_range = (1.748e-5, 4.139e-5)
        vh_range = (0.036, 0.121)
    elif circuit_size == 2:
        vl_range = (5.308e-6, 9.672e-6)
        vh_range = (0.013, 0.024)
    elif circuit_size == 3 or circuit_size == 4:
        vl_range = (6.367e-7, 1.830e-6)
        vh_range = (0.002, 0.005)
    # elif circuit_size == 4:
    #     vl = 4.178e-8
    #     vh = 0.0001
    elif circuit_size == 5:
        vl_range = (3.764e-8, 6.408e-8)
        vh_range = (9.410e-5, 3.108e-4)
    elif circuit_size == 6:
        vl_range = (4.228e-8, 2.075e-7)
        vh_range = (1.488e-4, 5.244e-4)
    # td_range = (0, 1.2e-9)
    td_range = (0, 5e-11)
    tr_range = (5e-11, 8e-11)
    tf_range = (5e-11, 8e-11) 
    pw_range = (5e-11, 8e-11)
    # per = 3e-10
    per_range = (3e-10, 7e-10)
    # tr = 1e-10
    # tf = 1e-10
    # pw = 1e-11
    # per_range = (2e-9, 3e-9)
    
    IS = []
    for k in range(port_num):
        random.seed(seed+200+k)
        vl = random.uniform(*vl_range)
        vh = random.uniform(*vh_range)
        td = random.uniform(*td_range)
        tr = random.uniform(*tr_range)
        # tr = 1e-10
        tf = random.uniform(*tf_range)
        # tf = 1e-10
        pw = random.uniform(*pw_range)
        # pw = 1e-11
        per = random.uniform(*per_range)
        IS.append([vl, vh, td, tr, tf, pw, per])
    IS = np.vstack(IS)
    return IS, VS

def generate_uover(port_num, circuit_size, seed):
    '''
    生成超过训练数据范围的测试数据
    '''
    VS = []
    VS = np.array(VS)
    if circuit_size == 1:
        # vl_range = (1.748e-5, 4.139e-5)
        # vh_range = (0.036, 0.121)
        vl_range = (1.748e-3, 4.139e-3)
        vh_range = (3.6, 12.1)
    elif circuit_size == 2:
        vl_range = (5.308e-6, 9.672e-6)
        vh_range = (0.013, 0.024)
    elif circuit_size == 3:
        vl_range = (6.367e-7, 1.830e-6)
        vh_range = (0.002, 0.005)
    # elif circuit_size == 4:
    #     vl = 4.178e-8
    #     vh = 0.0001
    elif circuit_size == 5:
        vl_range = (3.764e-8, 6.408e-8)
        vh_range = (9.410e-5, 3.108e-4)
    elif circuit_size == 6:
        vl_range = (4.228e-8, 2.075e-7)
        vh_range = (1.488e-4, 5.244e-4)

    # td_range = (0, 3e-11)
    # tr_range = (5e-11, 8e-11)
    # tf_range = (5e-11, 8e-11) 
    # pw_range = (5e-11, 8e-11)
    # per_range = (3e-10, 7e-10)

    td_range = (0, 2e-11)
    tr_range = (2e-11, 4e-11)
    tf_range = (2e-11, 4e-11) 
    pw_range = (3e-11, 9e-11)
    per_range = (3e-10, 8e-10)
    
    IS = []
    for k in range(port_num):
        random.seed(seed+200+k)
        vl = random.uniform(*vl_range)
        vh = random.uniform(*vh_range)
        td = random.uniform(*td_range)
        tr = random.uniform(*tr_range)
        # tr = 1e-10
        tf = random.uniform(*tf_range)
        # tf = 1e-10
        pw = random.uniform(*pw_range)
        # pw = 1e-11
        per = random.uniform(*per_range)
        IS.append([vl, vh, td, tr, tf, pw, per])
    IS = np.vstack(IS)
    return IS, VS

def generate_sin(port_num, circuit_size, seed):
    VS = []
    VS = np.array(VS)
    ##sourcesAll = [node1, node2, freq, phase, offset, amp]
    # usin = offset+amp⋅sin(2π⋅freq⋅t+phase)
    if circuit_size == 1:
        offset_range = (1.748e-5, 4.139e-5)
        amp_range = (0.036, 0.121)
        freq_range = (4e9, 5e9)
    node1 = 1
    node2 = 0
    phase = 0
    # freq = 1e10
    IS = []
    for k in range(port_num):
        random.seed(seed+200+k)
        offset = random.uniform(*offset_range)
        amp = random.uniform(*amp_range)
        freq = random.uniform(*freq_range)
        IS.append([node1, node2, freq, phase, offset, amp])
    IS = np.vstack(IS)
    return IS, VS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generate mf_mor data for model")
    parser.add_argument("--port_num", type=int, default= 2000)
    parser.add_argument("--circuit_size", type=int, default= 1)
    parser.add_argument("--threshold", type=float, default= 1.0)
    parser.add_argument("--svd_type", type=int, default= 1)
    parser.add_argument("--load", type=int, default= 0)
    parser.add_argument("--generate", type=int, default= 0)
    parser.add_argument("--data_num", type=int, default= 200)
    ## srcType: 'pulse' or 'sin'
    parser.add_argument("--srcType", type=str, default= 'sin')
    args = parser.parse_args()
    save_path = os.path.join(sys.path[0], 'train_data/1t/sim_100_port1000_multiper_diff')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(f"{save_path}/data_generate.log")])
    logging.info(args)
    data = spio.loadmat("./IBM_transient/ibmpg1t.mat")
    circuit_size = args.circuit_size
    port_num = args.port_num
    C, G, B = data['E'] * 1e-0, data['A'], data['B']
    B = B.tocsc()
    C = C.tocsc()
    G = G.tocsc()
    B = B[:, 0:0+port_num]
    # output matrix
    O = B
    
    t0 = 0
    t_all = 1e-09
    dt = 1e-11
    data_num = args.data_num
    logging.info("t0: {}, t_all: {}, dt: {}, data_num: {}".format(t0, t_all, dt, data_num))

    N = C.shape[0]

    tic = datetime.now()

    if args.load == 0:
        if args.svd_type == 0:
            # svd mor
            left, right, B_r, O_r = svdmor(G, B, B, threshold = args.threshold, load_path ="/home/fillip/桌面/CPM-MOR/SVD_data/5t")
        else:
            # my svd mor
            B_r ,right = svd_try(B, threshold = args.threshold)
            O_r = B_r

        toc = datetime.now()
        ts = toc - tic
        logging.info("SVD completed. Time used:{} s ".format(ts))

        f = np.array([1e9])
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
        logging.info("SVD-PRIMA-MOR:")
        logging.info('MOR completed. Time used:{} s '.format(tprima))
        logging.info('The original order is{}, the reduced order is {}'.format(N, nr_2))

        Br_2 = Br_2@right
        if args.svd_type == 0:
            Or_2 = Or_2@left
        else:
            ## for svd_try
            Or_2 = Or_2@right
        # save the reduced order model
        np.savez(save_path + '/mf_mor_data.npz', Cr_2 = Cr_2, Gr_2 = Gr_2, Br_2 = Br_2, Or_2 = Or_2, XX_2 = XX_2)

        f = np.array([1e9])
        m = 2
        s = 1j * 2 * np.pi * f

        q = m * B.shape[1]
        # perform svd mor
        tic = datetime.now()
        XX_1 = PRIMA.PRIMA_mp(C, G, B, s, q)
        Cr_1 = (XX_1.T@ C)@XX_1
        Gr_1 = (XX_1.T@ G)@XX_1
        Br_1 = (XX_1.T@ B)
        Or_1 = XX_1.T@ O
        nr_1 = Cr_1.shape[0]
        
        toc = datetime.now()
        tprima = toc - tic
        logging.info("PRIMA-MOR:")
        logging.info('MOR completed. Time used:{} s '.format(tprima))
        logging.info('The original order is{}, the reduced order is {}'.format(N, nr_1))
        np.savez(save_path + '/prima_mor_data.npz', Cr_1 = Cr_1, Gr_1 = Gr_1, Br_1 = Br_1, Or_1 = Or_1, XX_1 = XX_1)


    else:
        # load
        # save_path2 = os.path.join(sys.path[0], 'train_data/1t/sim_100_port2000_multiper_diff')
        data = np.load(save_path + '/mf_mor_data.npz')
        Cr_2 = data['Cr_2']
        Gr_2 = data['Gr_2']
        Br_2 = data['Br_2']
        Or_2 = data['Or_2']
        XX_2 = data['XX_2']

        data1 = np.load(save_path + '/prima_mor_data.npz')
        Cr_1 = data1['Cr_1']
        Gr_1 = data1['Gr_1']
        Br_1 = data1['Br_1']
        Or_1 = data1['Or_1']
        XX_1 = data1['XX_1']
    
    low_f = []
    high_f = []
    prima = []
    input_data = []
    in_all = []

    x0 = np.zeros((N, 1))
    xr1 = XX_1.T@ x0
    xr2 = XX_2.T@ x0

    if args.generate == 1:
        for i in range(data_num):
            logging.info("Generating {}th input data".format(i))
            seed = i
            # IS, VS = generate_u(port_num, circuit_size, seed)
            # IS, VS = generate_udiff(port_num, circuit_size, seed)
            # IS, VS = generate_uover(port_num, circuit_size, seed)
            IS, VS = generate_sin(port_num, circuit_size, seed)
            t1 = time.time()
            xAll, time1, dtAll, uAll = tdIntLinBE_new(t0, t_all, dt, C, -G, B, VS, IS, x0, srcType = args.srcType)
            y = O.T@xAll
            t2 = time.time()
            logging.info("High-fidelity data generated. Time used: {}s ".format(t2-t1))

            t3 = time.time()
            xAll_svd, time_svd, dtAll_svd, urAll_svd = tdIntLinBE_new(t0, t_all, dt, Cr_2, -Gr_2, Br_2, VS, IS, xr2, srcType = args.srcType)
            y_svd = Or_2.T@xAll_svd
            t4 = time.time()
            logging.info("Low-fidelity data generated. Time used: {}s ".format(t4-t3))

            # if i > 99 and i < 110:
            if i < 10:
                t5 = time.time()
                xAll_mor, time_mor, dtAll_mor, urAll_mor = tdIntLinBE_new(t0, t_all, dt, Cr_1, -Gr_1, Br_1, VS, IS, xr1, srcType = args.srcType)
                y_mor = Or_1.T@xAll_mor
                t6 = time.time()
                logging.info("PRIMA data generated. Time used: {}s ".format(t6-t5))
                prima.append(y_mor)

            low_f.append(y_svd)
            high_f.append(y)
            # input_data.append(in_data)
            in_all.append(uAll)
        low_f = np.array(low_f)
        high_f = np.array(high_f)
        inall = np.array(in_all)
        prima = np.array(prima)
        input_data = np.array(input_data)
        input_data = np.squeeze(input_data)
        time1 = np.array(time1)
        np.save(save_path+'/mf_time.npy', time1)
        # np.save('train_data/mf_in.npy', input_data)
        np.save(save_path+'/mf_low_f.npy', low_f)
        np.save(save_path+'/mf_high_f.npy', high_f)
        np.save(save_path+'/mf_inall.npy', inall)
        np.save(save_path+'/prima.npy',prima)
        logging.info("Data saved")


    else:
        # use to check the data
        # IS, VS = generate_u(port_num, circuit_size, seed = 1)
        # IS, VS = generate_udiff(port_num,circuit_size, seed=1)
        # IS, VS = generate_uover(port_num,circuit_size, seed=1)
        IS, VS = generate_sin(port_num, circuit_size, seed = 1)
        
        xAll, time1, dtAll, uAll = tdIntLinBE_new(t0, t_all, dt, C, -G, B, VS, IS, x0, srcType = args.srcType)
        y = O.T@xAll

        xAll_svd, time_svd, dtAll_svd, urAll_svd = tdIntLinBE_new(t0, t_all, dt, Cr_2, -Gr_2, Br_2, VS, IS, xr2, srcType = args.srcType)
        y_svd = Or_2.T@xAll_svd

        xAll_mor, time_mor, dtAll_mor, urAll_mor = tdIntLinBE_new(t0, t_all, dt, Cr_1, -Gr_1, Br_1, VS, IS, xr1, srcType = args.srcType)
        y_mor = Or_1.T@xAll_mor

        yy = np.zeros((y.shape[1]))
        for i in range(y.shape[0]):
            yy += np.real(y[i, :])
        yy_svd = np.zeros((y_svd.shape[1]))
        for i in range(y_svd.shape[0]):
            yy_svd += np.real(y_svd[i, :])   
        yy_mor = np.zeros((y_mor.shape[1]))
        for i in range(y_mor.shape[0]):
            yy_mor += np.real(y_mor[i,:]) 
        plt.figure(figsize=(8, 5))
        plt.plot(time1, yy, color='black', linestyle='-.', marker='*', label='High-fidelity-data', markevery = 35, markersize=6, linewidth=1.5)
        plt.plot(time_svd, yy_svd, color='red', linestyle='--', marker='s', label='Low-fidelity-data', markersize=6, markevery = 45, linewidth=1.5)
        plt.plot(time_mor, yy_mor, color='green', linestyle='-', marker='o', label='PRIMA', markersize=6, markevery = 40, linewidth=1.5)
        plt.legend(fontsize=14)
        plt.title("MF-data", fontsize=14)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("result", fontsize=12)
        plt.grid(alpha=0.5)
        plt.tight_layout()
        # plt.show()
        plt.savefig("1t_sin.png")
        plt.close()    
    pass