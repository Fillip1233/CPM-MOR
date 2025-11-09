# coding: utf-8
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
import psutil
from tqdm import tqdm
from scipy.sparse.linalg import svds
import gc
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # 返回 MB
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
    # R = R.toarray()
    # L = L.toarray()
    
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

    ## 不逐列求解的话，返回的是np.array 不是稀疏矩阵，还是不能用在大规模的矩阵上
    L_inv_in = sp.linalg.spsolve_triangular(LuL, in_mat, lower=True)
    L_inv_in = sp.csc_matrix(L_inv_in)
    UT_inv_out = sp.linalg.spsolve_triangular(LuUT, out_mat, lower=True)
    UT_inv_out = sp.csc_matrix(UT_inv_out)

    B = sp.hstack([UT_inv_out, L_inv_in])

    # R = R.toarray()
    # L = L.toarray()
    # # # R1 = R1.toarray()
    L_inv_in = L_inv_in.toarray()
    UT_inv_out = UT_inv_out.toarray()
    logging.info(np.allclose(R, L_inv_in, atol=1e-9))
    logging.info(np.allclose(L, UT_inv_out, atol=1e-9))

    t2 = time.time()
    logging.info("Finish B ,time: {}".format(t2-t1))

    t3 = time.time()
    B = B.toarray()
    U, S, V = np.linalg.svd(B, full_matrices=False)
    t4 = time.time()
    logging.info("Finish SVD ,time: {}".format(t4-t3))

    return U,S,V

def svdmor_2(G, in_mat, out_mat):
    t1 = time.time()
    L = out_mat
    R = spsolve(G, in_mat)
    # chunk_size = 200  # 根据内存调整
    # R = sp.lil_matrix(in_mat.shape, dtype=in_mat.dtype)  # 使用 lil_matrix 便于按列填充
    
    # for i in tqdm(range(0, in_mat.shape[1], chunk_size)):
    #     # 提取当前块（保持稀疏性）
    #     chunk = in_mat[:, i:i+chunk_size].tocsc()  # CSC 格式列切片效率更高
        
    #     solution_chunk = spsolve(G, chunk)
    #     R[:, i:i+chunk_size] = solution_chunk.tolil()
    #     del chunk, solution_chunk
    #     gc.collect()
    
    # R = R.toarray()
    # R1 = R1.toarray()
    # logging.info(np.allclose(R, R1, atol=1e-9))
    logging.info("Finish R")
    B = sp.hstack([L, R])
    # sp.save_npz("B_sparse.npz", B, compressed=True)
    logging.info("Finish B sparse save")
    # B = sp.load_npz("B_sparse.npz")
    # logging.info("Finish B sparse load")
    B = B.toarray()
    U, S, V = np.linalg.svd(B, full_matrices=False)
    t2 = time.time()
    logging.info("Finish SVD ,time: {}".format(t2-t1))
    return U, S, V
    pass

def svdmor_3(G, in_mat, out_mat=None):
    '''
    for thumpg1t but fail(oom)
    try to solve by solve 4 parts
    '''
    # n = in_mat.shape[1]
    # chunk_size = 100
    # half_n = n // 4  # 前一半的列数

    # # 初始化 R_part1 (前一半列)
    # R_part1 = sp.lil_matrix((in_mat.shape[0], half_n), dtype=in_mat.dtype)

    # for i in tqdm(range(3*half_n, 4*half_n, chunk_size)):
    #     chunk = in_mat[:, i:i+chunk_size].tocsc()
    #     solution_chunk = spsolve(G, chunk)
    #     R_part1[:, i-3*half_n:i-3*half_n+chunk_size] = solution_chunk.tolil()
    #     del chunk, solution_chunk
    #     gc.collect()

    # # 保存前一半结果
    # logging.info("Finish R_part4")
    # sp.save_npz("R_part4_thumpg.npz", R_part1.tocsc(), compressed=True)
    # t1 = time.time()
    # R1 = sp.load_npz("R_part1_thumpg.npz")
    # R2 = sp.load_npz("R_part2_thumpg.npz")
    # R3 = sp.load_npz("R_part3_thumpg.npz")
    # R4 = sp.load_npz("R_part4_thumpg.npz")
    # logging.info("Finish B sparse load")
    # R = sp.hstack([R1, R2, R3, R4])
    # logging.info("Finish R sparse hstack")
    # del R1, R2, R3, R4
    # gc.collect()
    # L = out_mat
    # B = sp.hstack([L, R])
    # sp.save_npz("B.npz", B.tocsc(), compressed=True)
    # logging.info("Finish B")
    # del L, R
    # gc.collect()
    B = sp.load_npz("B.npz")
    U, S, Vt = svds(B, k=200)
    # B = B.toarray()
    # U, S, V = np.linalg.svd(B, full_matrices=False)
    # t2 = time.time()
    # logging.info("Finish SVD ,time: {}".format(t2-t1))
    return U, S, Vt

def svd_try(B):

    B_1 = csc_matrix(B, dtype=np.float64)
    t1 = time.time()
    B_dense = B_1.toarray()
    U, S, V = np.linalg.svd(B_dense,full_matrices=False)
    t2 = time.time()
    logging.info("Finish SVD, time: {}".format(t2-t1))

    return U,S,V

if __name__ == '__main__':
    start_mem = get_memory_usage()
    parser = argparse.ArgumentParser(description='SVD MOR')
    parser.add_argument('--threshold', type=float, default=2, help='Threshold for SVD')
    parser.add_argument('--port_num', type=int, default=10000, help='Number of ports')
    parser.add_argument('--circuit', type=int, default=1, help='Circuit number')
    args = parser.parse_args()
    # save_path = os.path.join('/home/fillip/home/CPM-MOR/SVDMOR2_res/{}t'.format(args.circuit))
    save_path = os.path.join('/home/fillip/home/CPM-MOR/Baseline/SVDMOR/{}t/'.format(args.circuit))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(f"{save_path}/data_generate.log")])
    logging.info(args)
    # data = spio.loadmat("./IBM_transient/ibmpg{}t.mat".format(args.circuit))
    data = spio.loadmat("./IBM_transient/{}t_SIP_gt.mat".format(args.circuit))
    C, G, B = data['C_final'] * 1e-0, data['G_final'], data['B_final']
    B = B.tocsc()
    # C = C.tocsc()
    G = G.tocsc()
    B = B[:, 0:args.port_num]
    # output matrix
    O = B
    logging.info("Finish B, C, G")
    # U,S,V = svdmor_2(G, B, O)
    # svdmor_2(G, B, O)
    U, S, V = svdmor_2(G, B, O)
    # U,S,V = svd_try(B)
    np.save(save_path + "/U.npy",U)
    np.save(save_path + "/S.npy",S)
    np.save(save_path + "/V.npy",V)
    end_mem = get_memory_usage()
    logging.info("Memory usage: {:.2f} MB".format(end_mem - start_mem))
    logging.info("Finish!")
    pass