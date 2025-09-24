import scipy.io as spio
from scipy.sparse import triu
from scipy.sparse import identity
import numpy as np
if __name__ == "__main__":
    
    data = spio.loadmat("/home/fillip/home/CPM-MOR/IBM_transient/1t_final.mat")

    C, G, B = data['C_final'] * 1e-0, data['G_final'], data['B_final']
    Node_size = C.shape[0]
    
    B = B.tocsc()
    
    C = C.tocsc()
    C = C + 1e-20 * identity(C.shape[0], format="csc")
    G = G.tocsc()
    G = G + 1e-10 * identity(G.shape[0], format="csc")

    GC = G + C
    GC1 = triu(GC, format='csc')
    Ap0 = GC1.indptr.astype(np.int64)   # 列指针 (n+1)
    Ai0 = GC1.indices.astype(np.int64)  # 行索引 (nnz)
    Ax0 = GC1.data.astype(np.float64)
    np.savetxt("GC_Ap.txt", Ap0, fmt="%d")
    np.savetxt("GC_Ai.txt", Ai0, fmt="%d")
    np.savetxt("GC_Ax.txt", Ax0, fmt="%.18e")

    G1 = triu(G, format='csc')
    C1 = triu(C, format='csc')

    Ap = G1.indptr.astype(np.int64)   # 列指针 (n+1)
    Ai = G1.indices.astype(np.int64)  # 行索引 (nnz)
    Ax = G1.data.astype(np.float64)
    np.savetxt("G_Ap.txt", Ap, fmt="%d")
    np.savetxt("G_Ai.txt", Ai, fmt="%d")
    np.savetxt("G_Ax.txt", Ax, fmt="%.18e")

    Ap1 = C1.indptr.astype(np.int64)   # 列指针 (n+1)
    Ai1 = C1.indices.astype(np.int64)  # 行索引 (nnz)
    Ax1 = C1.data.astype(np.float64)
    np.savetxt("C_Ap.txt", Ap1, fmt="%d")
    np.savetxt("C_Ai.txt", Ai1, fmt="%d")
    np.savetxt("C_Ax.txt", Ax1, fmt="%.18e")

    Ap2 = B.indptr.astype(np.int64)   # 列指针 (n+1)
    Ai2 = B.indices.astype(np.int64)  # 行索引 (nnz)
    Ax2 = B.data.astype(np.float64)
    np.savetxt("B_Ap.txt", Ap2, fmt="%d")
    np.savetxt("B_Ai.txt", Ai2, fmt="%d")
    np.savetxt("B_Ax.txt", Ax2, fmt="%.18e")
    pass
