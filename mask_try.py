import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import argparse
import os
import logging
import scipy.io as spio
from utils.load_data import *
import heapq
import time

def build_X_from_B(B):
    """
    构建节点->端口映射矩阵 X (n_nodes x n_ports)
    B 每列的非零行表示该端口对应节点（one-hot映射）
    返回稀疏 csr 矩阵
    """
    n_nodes, n_ports = B.shape
    rows = []
    cols = []
    data = []
    for j in range(n_ports):
        if sp.issparse(B):
            col = B[:, j].tocoo()
            rows.extend(col.row.tolist())
            cols.extend([j] * len(col.row))
            # 若 B 有权重，也可以把权重乘进去：
            data.extend(np.abs(col.data).tolist())
        else:
            nz = np.nonzero(B[:, j])[0]
            rows.extend(nz.tolist())
            cols.extend([j] * len(nz))
            data.extend([1.0] * len(nz))
    X = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_ports)).tocsr()
    return X

def corr_via_XAX_streaming(G, C, B, alpha=1.0, topk=10, normalize=True, diffusion_steps=0):
    
    n_nodes = G.shape[0]
    n_ports = B.shape[1]

    # 1. A
    if sp.issparse(G) or sp.issparse(C):
        A = (abs(G) if not sp.issparse(G) else abs(G)) + alpha * (abs(C) if not sp.issparse(C) else abs(C))
        A = sp.csr_matrix(A)
    else:
        A = np.abs(G) + alpha * np.abs(C)

    # diffusion_steps 如果需要，可先把 A 替换为 A + A^2 + ...
    # 为简单这里不展开 diffusion（可按需加）

    # 2. X (csr)
    X = build_X_from_B(B).tocsr()

    # 3. 为每行维护一个小根堆 (保留 topk 最大)
    # we keep heap of size topk per row i, storing (score, j)
    heaps = [ [] for _ in range(n_ports) ]  # 若 n_ports 超大，占内存，改成文件/分块策略

    # 4. 逐列计算
    print("step 4")
    for j in range(n_ports):
        print(f"Processing port {j+1}/{n_ports}", end='\r')
        xj = X[:, j]            # sparse column (n_nodes x 1) as csr with many zeros
        v = A.dot(xj)           # sparse vector (n_nodes x 1)
        colj = X.T.dot(v)       # (n_ports x 1)  - sparse vector but result type may be dense for some libs
        # colj 可能为稀疏矩阵或 ndarray
        if sp.issparse(colj):
            # iterate nonzero entries
            coo = colj.tocoo()
            for i_idx, val in zip(coo.row, coo.data):
                score = float(abs(val))
                heap = heaps[i_idx]
                if len(heap) < topk:
                    heapq.heappush(heap, (score, j))
                else:
                    if score > heap[0][0]:
                        heapq.heapreplace(heap, (score, j))
        else:
            # dense vector (小端口数或密集情况下)
            colj = np.abs(np.asarray(colj)).ravel()
            for i_idx in range(n_ports):
                score = colj[i_idx]
                heap = heaps[i_idx]
                if len(heap) < topk:
                    heapq.heappush(heap, (score, j))
                else:
                    if score > heap[0][0]:
                        heapq.heapreplace(heap, (score, j))

    # 5. 从 heap 构造 topk_idx
    topk_idx = np.zeros((n_ports, topk), dtype=int)
    print("\nstep 5")
    for i in range(n_ports):
        print(f"Constructing topk for port {i+1}/{n_ports}", end='\r')
        heap = heaps[i]
        # heap contains (score, j) with smallest score at heap[0]
        topk_sorted = sorted(heap, key=lambda x: -x[0])  # 从大到小
        for k_idx, (_, jidx) in enumerate(topk_sorted):
            topk_idx[i, k_idx] = jidx
        # 若某些行非满 topk，可以把剩余填 -1 或自身
        if len(topk_sorted) < topk:
            rem = topk - len(topk_sorted)
            topk_idx[i, len(topk_sorted):] = -1

    # 6. normalize 可选（若要用相对值）
    # 该 streaming 版本直接返回索引；若需 corr 值可稍作扩展存储
    return topk_idx

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Baseline mask')
    parser.add_argument('--circuit', type=int, default=1, help='Circuit number')
    parser.add_argument("--port_num", type=int, default= 10000)
    parser.add_argument("--threshold", type=int, default= 0)
    parser.add_argument("--generate", type=int, default= 1)
    parser.add_argument("--data_num", type=int, default= 1)
    parser.add_argument("--process", type=int, default= 10)
    args = parser.parse_args()
    save_path = os.path.join('./Baseline/mask/{}t/'.format(args.circuit))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_path = "/home/fillip/home/CPM-MOR/Baseline/mask/{}t".format(args.circuit)
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(f"{save_path}/mask.log")])
    logging.info(args)
    data = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/{}t_SIP_gt.mat".format(args.circuit))
    
    port_num = args.port_num
    logging.info("Circuit : {}".format(args.circuit))
    C, G, B = data['C_final'] * 1e-0, data['G_final'], data['B_final']
    Node_size = C.shape[0]
    B = B.tocsc()
    C = C.tocsc()
    G = G.tocsc()
    B = B[:, 0:0+port_num]
    # output matrix
    O = B
    data_path = os.path.join(f'./MSIP_BDSM/train_data/{args.circuit}t_2per/')
    # x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time1, pr = prepare_data(data_path, train_data_num=10, prima=False)
  # 假设 G, B 为你的电路矩阵
    t1 = time.time()
    topk_idx  = corr_via_XAX_streaming(G, C, B, topk=10)
    t2 = time.time()
    logging.info("Time taken for correlation computation: {:.2f} seconds".format(t2 - t1))
    np.save(os.path.join(save_path, 'coridx.npy'), topk_idx)

    print("Top-10 related ports for each port:")
    for i, idx in enumerate(topk_idx):
        print(f"Port {i}: {idx}")
        if i >= 100:
            break  # 仅打印前10个端口的相关端口