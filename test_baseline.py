'''
2025-7-16
test baseline
'''
import argparse
import os
import logging
import scipy.io as spio
import datetime
import numpy as np
import pandas as pd
from generate_mf_mor_data import generate_u, generate_udiff 
from SVDMOR import svdmor_2
from McPack import mcpack_2
from DeMOR_generate_data import generate_DeMOR_data
import utils.PRIMA as PRIMA
from utils.calculate_metrix import calculate_metrix
from utils.tdIntLinBE_new import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVDMOR')
    parser.add_argument('--circuit', type=int, default= 1, help='Circuit number')
    parser.add_argument("--port_num", type=int, default= 2000)
    parser.add_argument("--threshold", type=int, default= 1)
    parser.add_argument("--method", type=str, default='demor', help='Method to use for MOR')
    parser.add_argument("--test_num", type=int, default= 10, help='Number of test samples')
    parser.add_argument("--process", type=int, default=10, help='DeMOR process number')
    args = parser.parse_args()
    save_path = os.path.join('/home/fillip/home/CPM-MOR/Exp_res/test_baseline/{}t/'.format(args.circuit))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(f"{save_path}/SVDMOR.log")])
    logging.info(args)
    data = spio.loadmat("./IBM_transient/ibmpg{}t.mat".format(args.circuit))
    C, G, B = data['E'] * 1e-0, data['A'], data['B']
    port_num = args.port_num
    B = B.tocsc()
    C = C.tocsc()
    G = G.tocsc()
    B = B[:, 0:port_num]
    # output matrix
    O = B

    N = C.shape[0]
    f = np.array([1e9])
    m = 2
    s = 1j * 2 * np.pi * f
    t0 = 0
    tf = 1e-09
    dt = 1e-11
    srcType = 'pulse'
    x0 = np.zeros((N, 1))
    if args.method == 'svdmor':
        tic = datetime.now()
        left, right, B_r, O_r = svdmor_2(G, B, O, threshold=args.threshold,load_path='/home/fillip/home/CPM-MOR/SVDMOR2_res/{}t/'.format(args.circuit))
        toc = datetime.now()
        ts = toc - tic
        logging.info("SVD completed. Time used: {}s".format(ts))

        q_svd = m * B_r.shape[1]
        tic = datetime.now()
        XX = PRIMA.PRIMA_mp(C, G, B_r, s, q_svd)
        Cr_2 = (XX.T@ C)@XX
        Gr_2 = (XX.T@ G)@XX
        Br_2 = (XX.T@ B_r)
        Or_2 = XX.T@ O_r
        nr_2 = Cr_2.shape[0]
        toc = datetime.now()
        tprima = toc - tic
        logging.info("SVD-PRIMA-MOR:")
        logging.info('MOR completed. Time used: {}'.format(tprima))
        logging.info('The original order is {}'.format(N))
        logging.info('The reduced order is {}'.format(nr_2))
        xr1 = XX.T@ x0
        Br_2 = Br_2@right
        logging.info("B reduced order: {}".format(Br_2.shape[0]))

        y_all = []
        y_svd_all = []
        for i in range(args.test_num):
            IS, VS = generate_udiff(args.port_num, args.circuit, seed = i)
            xAll, time, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, C, -G, B, VS, IS, x0, srcType)
            y = O.T@xAll
        
            xAll_svd, time_svd, dtAll_svd, urAll_svd = tdIntLinBE_new(t0, tf, dt, Cr_2, -Gr_2, Br_2, VS, IS, xr1, srcType)
            y_svd = (Or_2@left).T@ xAll_svd

            y_all.append(y)
            y_svd_all.append(y_svd)

    if args.method == 'mcpack':
        s = 1j * 2 * np.pi * 1e10
        t1 = datetime.now()
        Cr, Gr, Br, Lr, XX = mcpack_2(C, G, B, O, s, m, m, r=50, threshold=args.threshold)
        t2 = datetime.now()
        logging.info("McPack time: {}".format(t2 - t1))
        nr_size = Cr.shape[0]
        logging.info('The original order is {}'.format(N))
        logging.info("Reduced order size: {}".format(nr_size))

        y_all = []
        y_svd_all = []
        xr1 = XX.T@ x0
        for i in range(args.test_num):
            IS, VS = generate_udiff(args.port_num, args.circuit, seed = i)
            xAll, time, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, C, -G, B, VS, IS, x0, srcType)
            y = O.T@xAll
        
            xAll_mc, time_svd, dtAll_svd, urAll_svd = tdIntLinBE_new(t0, tf, dt, Cr, -Gr, Br, VS, IS, xr1, srcType)
            y_svd = Lr.T@ xAll_mc

            y_all.append(y)
            y_svd_all.append(y_svd)
    
    if args.method == 'demor':
        y_all = []
        y_svd_all = []
        for i in range(args.test_num):
            IS, VS = generate_udiff(args.port_num, args.circuit, seed = i)
            xAll, time, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, C, -G, B, VS, IS, x0, srcType)
            y = O.T@xAll
            t3 = datetime.now()
            y_svd = generate_DeMOR_data(G, C, B, y, args, f"/home/fillip/home/CPM-MOR/Exp_res/DeMOR_data/{args.circuit}t/", seed=i)
            t4 = datetime.now()
            logging.info("DeMOR time: {}".format(t4 - t3))
            y_all.append(y)
            y_svd_all.append(y_svd)
        
    y_all = np.array(y_all)
    y_svd_all = np.array(y_svd_all)

    recording = {'rmse':[], 'nrmse':[], 'r2':[],'mae':[]}
    metrics = calculate_metrix(y_test = y_all.real, y_mean_pre = y_svd_all.real)
    recording['rmse'].append(metrics['rmse'])
    recording['nrmse'].append(metrics['nrmse'])
    recording['r2'].append(metrics['r2'])
    recording['mae'].append(metrics['mae'])
    record = pd.DataFrame(recording)
    record.to_csv(save_path + '/{}_res.csv'.format(args.method), index = False)
        
    pass