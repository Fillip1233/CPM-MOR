'''
2025-3-1
祖传代码,PRIMA方法
'''
# from PRIMA_sp import *
# from spaceit import *
import numpy as np
import math
import scipy.sparse
import scipy.sparse.linalg
import scipy as sp
from datetime import datetime


# multi-point PRIMA
def PRIMA_mp(E=None, A=None, B=None, s=None, q=None):
    t_start = datetime.now()
    ls = len(s)
    n, N = np.shape(B)
    V_temp = None  # 初始化V_temp

    for i in range(0, ls):
        t1 = datetime.now()
        X = PRIMA_sp(E, A, B, s[i], q)  # use deflated version of single-point PRIMA
        t2 = datetime.now()
        print('single-point PRIMA time:', t2 - t1)
        # X = PRIMA_sp(E, A, B, s[i], q)
        if V_temp is None:
            V = X
        else:
            t3 = datetime.now()
            V, _ = spaceit(V_temp, X)  # concatenate single-point subspace and orthonomalize the resultant subspace
            t4 = datetime.now()
            print('spaceit time:', t4 - t3)
        V_temp = V

    toc = datetime.now()
    t_mor = toc - t_start
    return V


# def PRIMA_sp(E=None, A=None, B=None, s=None, q=None):
#     n, N = np.shape(B)
#     M = s * E + A
#     lu = sp.sparse.linalg.splu(M)
#     B = B.toarray()
#     R = lu.solve(B)
#     nr = math.ceil(q / N)
#     X = np.zeros([n, N, 1 + nr], dtype=complex)
#     X[:, :, 0], _ = np.linalg.qr(R, mode='full')
#     for k in range(nr):
#         V = E @ X[:, :, k]
#         X[:, :, k + 1] = lu.solve(V)
#         print('iteration', k + 1)
#         for j in range(k + 1):
#             X[:, :, k + 1] = X[:, :, k + 1] - X[:, :, k - j] @ (X[:, :, k - j].conj().T @ X[:, :, k + 1])
#         X[:, :, k + 1], _ = np.linalg.qr(X[:, :, k + 1], mode='full')
#     XX = X[:, :, 0]
#     for i in range(1, nr):
#         XX = np.hstack([XX, X[:, :, i]])
#     XX = np.real(XX[:, 0:q])
#     print('The original order is', n, 'the reduced order is', q)

#     return XX

# single point PRIMA with deflated QR
def PRIMA_sp(E=None, A=None, B=None, s=None, q=None):
    n, N = np.shape(B)
    M = s * E + A  #s?
    lu = sp.sparse.linalg.splu(M)
    if type(B) is not np.ndarray:
        B = B.toarray()
    R = lu.solve(B)
    nr = math.ceil(q / N)

    qrtol = 1e-12
    X = []
    x0, _ = qr_deflated(R, qrtol)
    X.append(x0)

    for k in range(nr): # nr-1
        V = E @ X[k]
        X.append(lu.solve(V))
        print('iteration', k + 1)
        for j in range(k + 1):
            X[k + 1] = X[k + 1] - X[k - j] @ (X[k - j].conj().T @ X[k + 1])
        # X[k + 1], _ = np.linalg.qr(X[k + 1], mode='full')
        X[k + 1], _ = qr_deflated(X[k + 1], qrtol)
    XX = X[0]
    for i in range(1, nr):
        XX = np.hstack([XX, X[i]])
    # XX = np.real(XX)
    # print('The original order is', n, 'the reduced order is', q)

    return XX


# append X to V and re-orthnomalize the whole matrix
# we assume V is already orthonormal, so only the newly added X is processed
def spaceit(V, X):
    tol = 1e-12
    m = X.shape[1]
    for k in range(m):
        X[:, k] = X[:, k] / np.linalg.norm(X[:, k])
        N = V.shape[1]
        for j in range(N):
            # hkj = (X[:, k].conj().transpose() @ V[:, j])
            hkj = V[:, j].conj().transpose() @ X[:, k]  # can't swap the two vector if they are complex
            X[:, k] = X[:, k] - hkj * V[:, j]  # orthonormalize Xk w.r.t all columns in V
        hkk = np.sqrt(X[:, k].conj().transpose() @ X[:, k])  # norm of Xk
        if hkk > tol:
            X[:, k] = X[:, k] / hkk
            V = np.hstack((V, X[:, k].reshape(-1, 1)))  # append the new orthonormal basis to V
        else:
            pass  # no need to add this basis if it is linearly dependent on previous ones
            # X[:, k] = np.zeros(len(X[:, k]))

    return V, X


# deflated QR factorization.
# remove columns in Q if corresponding diagonals in R are too small  
def qr_deflated(V, tol):
    (Q, R) = sp.linalg.qr(V, mode='economic')
    d = np.abs(np.diag(R))
    p = d > max(d) * tol  # mask for the columns to be kept
    return Q[:, p], R[p, :]


if __name__ == '__main__':
    pass
