'''
2025-3-8
generate mf_mor data for model
'''
import scipy.io as spio
import random
from utils.tdIntLinBE_new import *
import numpy as np


if __name__ == '__main__':
    data = spio.loadmat("./IBM_transient/ibmpg1t.mat")
    C, G, B = data['E'] * 1e-0, data['A'], data['B']
    port_num = 40
    B = B.tocsc()
    C = C.tocsc()
    G = G.tocsc()
    B = B[:, 0:port_num]
    # output matrix
    O = B
    
    t0 = 0
    t_all = 2e-09
    dt = 1e-11
    port_num = 40
    data_length = 200
    seed = 0
    #generate ui

    VS = []
    VS = np.array(VS)
    N = C.shape[0]
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
    IS = []
    for _ in range(port_num):
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
        IS.append([vl, vh, td, tr, tf, pw, per])
    x0 = np.zeros((N, 1))
    xAll, time, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, C, -G, B, VS, IS, x0, srcType = 'pulse')
    pass