'''
2025-6-26
simplify PRIMA for demor multiprocessing
'''
from datetime import datetime
import numpy as np
import scipy as sp
import math
def PRIMA_sp(C = None, B = None, q = None, lu = None):
    n, N = np.shape(B)
    if type(B) is not np.ndarray:
        B = B.toarray()
    R = lu.solve(B)
    nr = math.ceil(q / N)

    qrtol = 1e-12
    X = []
    x0, _ = qr_deflated(R, qrtol)
    X.append(x0)

    if(C.nnz != 0): # if a circuit has only R , E is empty
        for k in range(nr): # nr-1
            V = C @ X[k]
            X.append(lu.solve(V))
            print('iteration', k + 1)
            for j in range(k + 1):
                X[k + 1] = X[k + 1] - X[k - j] @ (X[k - j].conj().T @ X[k + 1])
            X[k + 1], _ = qr_deflated(X[k + 1], qrtol)
    XX = X[0]
    if(C.nnz != 0):
        for i in range(1, nr):
            XX = np.hstack([XX, X[i]])
    XX,_ = qr_deflated(XX, tol=1e-3)

    return XX

# deflated QR factorization.
# remove columns in Q if corresponding diagonals in R are too small  
def qr_deflated(V, tol):
    (Q, R) = sp.linalg.qr(V, mode='economic')
    d = np.abs(np.diag(R))
    p = d > max(d) * tol  # mask for the columns to be kept
    return Q[:, p], R[p, :]
if __name__ == "__main__":

    pass