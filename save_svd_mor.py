from scipy.sparse.linalg import splu
import scipy.sparse as sp
import scipy.io as spio
import numpy as np
import logging
import argparse
import os
import sys
import time
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve_triangular
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity
from scipy.sparse.linalg import lgmres
from scipy.sparse.linalg import inv
from tqdm import tqdm

def svdmor(G, in_mat, out_mat, threshold):
    
    t1 = time.time()
    lu = splu(G)

    ##方法1
    ## 计算量大，内存占用大
    # R = spsolve(lu.L, in_mat)
    # L = spsolve(lu.U.T, out_mat)
    
    ##方法2
    ## 大规模矩阵计算不适用
    # lu2 = splu(lu.L)
    # lu3 = splu(lu.U.T)
    # R1 = lu2.solve(in_mat.toarray(), trans='N')
    # L1 = lu3.solve(out_mat.toarray(),trans='N')

    #方法3
    # #重复计算且数值不对
    # L1 = lu.solve(out_mat.toarray(), trans='T')
    # R1 = lu.solve(in_mat.toarray(), trans='N')
    # L1= lu.L.T@L1
    # R1 = lu.U@R1

    #方法4
    in_mat = in_mat.toarray()
    out_mat = out_mat.toarray()
    LuL = lu.L
    LuUT = lu.U.T
    L_inv_in = sp.lil_matrix((in_mat.shape[0], in_mat.shape[1]))
    for j in tqdm(range(in_mat.shape[1])):
        bj = in_mat[:, j]
        L_inv_in[:, j] = sp.linalg.spsolve_triangular(LuL, bj, lower=True)

    UT_inv_out  =  sp.lil_matrix((out_mat.shape[0], out_mat.shape[1])) 
    for k in tqdm(range(out_mat.shape[1])):
        bk = out_mat[:, k]
        UT_inv_out[:, k] = sp.linalg.spsolve_triangular(LuUT, bk, lower=True)

    # R = R.toarray()
    # L = L.toarray()
    # # R1 = R1.toarray()
    # L_inv_in = L_inv_in.toarray()
    # UT_inv_out = UT_inv_out.toarray()
    # print(np.allclose(R, L_inv_in, atol=1e-9))
    # print(np.allclose(L, UT_inv_out, atol=1e-9))

    B = sp.hstack([UT_inv_out, L_inv_in])
    t2 = time.time()
    logging.info("Finish B ,time: {}".format(t2-t1))

    t3 = time.time()
    B = B.toarray()
    U, S, V = np.linalg.svd(B, full_matrices=False)
    t4 = time.time()
    logging.info("Finish SVD ,time: {}".format(t4-t3))

    return U,S,V

def svd_try(B):

    B_1 = csc_matrix(B, dtype=np.float64)
    t1 = time.time()
    B_dense = B_1.toarray()
    U, S, V = np.linalg.svd(B_dense,full_matrices=False)
    t2 = time.time()
    logging.info("Finish SVD, time: {}".format(t2-t1))

    return U,S,V

if __name__ == '__main__':
    save_path = os.path.join(sys.path[0], 'SVD_data/4t')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(f"{save_path}/data_generate.log")])
    
    data = spio.loadmat("./IBM_transient/ibmpg4t.mat")
    C, G, B = data['E'] * 1e-0, data['A'], data['B']
    port_num = 100
    B = B.tocsc()
    C = C.tocsc()
    G = G.tocsc()
    B = B[:, 0:port_num]
    # output matrix
    O = B

    U,S,V = svdmor(G, B, B, threshold = 2)
    # U,S,V = svd_try(B)
    np.save("F:\zhenjie\code\CPM-MOR\code_try/U1.npy",U)
    np.save("F:\zhenjie\code\CPM-MOR\code_try/S1.npy",S)
    np.save("F:\zhenjie\code\CPM-MOR\code_try/V1.npy",V)
    logging.info("Finish!")
    pass