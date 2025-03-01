# Aenerated with SMOP  0
from filecmp import cmp
import math
import numpy as np
import time
# import pandas as pd
import matplotlib.pyplot as plt
import operator
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import scipy.io as spio
from numpy.linalg import norm
from scipy.sparse import csc_matrix
from datetime import datetime
import PRIMA as PRIMA
from tdIntLinBE_new import *
from tdIntLinBE_forsvd import *

data = spio.loadmat("./IBM_transient/ibmpg1t.mat")

E, A, B = data['E'] * 1e-0, data['A'], data['B']

N = E.shape[0]
NB = B.shape[1]

Nb = 40 
B = B[:, 0:Nb]
C = B
t0 = 0
tf = 1e-08
dt = 1e-11

srcType = 'pulse'

#     PULSE(V1 V2 TD TR TF PW PER)
VS = []
VS = np.array(VS)
# main.m:32
# 电流源的输入都是一样可能看不出啥
IS = np.hstack(
    [np.zeros([Nb, 1]), 
        np.dot(np.ones([Nb, 1]), 0.1),
        np.dot(np.ones([Nb, 1]), tf) / 5,
        np.dot(np.ones([Nb, 1]), 1e-09),
        np.dot(np.ones([Nb, 1]), 1e-09),
        np.dot(np.ones([Nb, 1]), 5e-09),
        np.dot(np.ones([Nb, 1]), 1e-08)])
x0 = np.zeros((N, 1))
# B[:,1] = 0


# xAll, time, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, E, -A, B, VS, IS, x0, srcType)
# y = C.T@xAll


# simplify B using svd
## method 1 -> turn to dense matrix
B_1 = csc_matrix(B, dtype=np.float64)
t1 = time.time()
B_dense = B_1.toarray()
U, S, V = np.linalg.svd(B_dense,full_matrices=False)
t2 = time.time()
print('time for svd:', t2-t1)
threshold = 1.0
indices = S >= threshold
U_new = U[:, indices]               
S_new = S[indices]                  
V_new = V[indices, :] 
# ass_mat = np.diag(S_new) @ V_new #assignment matrix


# xAll_svd, time_svd, dtAll_svd, urAll_svd = tdIntLinBE_svd(t0, tf, dt, E, -A, U_new, VS, IS, x0, srcType, ass_mat)
# y_svd = C.T@ xAll_svd


# mor with PRIMA
f = np.array([1e2, 1e9])  # array of targeting frequencies
m = 2  # number of moments to match per expansion point (same for all points here, though can be made different)
s = 1j * 2 * np.pi * f  # array of expansion points
q = m * Nb
tic = datetime.now()

XX = PRIMA.PRIMA_mp(E, A, B, s, q)
YY = XX
Er = (YY.T @ E) @ XX
Ar = (YY.T @ A) @ XX
Br = YY.T @ B
Cr = XX.T @ C
nr = Er.shape[0]
toc = datetime.now()
tprima = toc - tic

print("Origin PRIMA MOR:")
print('PRIMA completed. Time used: ', str(tprima), 's')
print('The original order is', N, 'the reduced order is', nr)
# xr0 = XX.T@ x0


# xrAll, time, dtAll, urAll = tdIntLinBE_new(t0, tf, dt, Er, -Ar, Br, VS, IS, xr0, srcType)
# xAll_mor = XX@xrAll
# y_mor = Cr.T@xrAll

# plt.plot(time, y_mor[0, :], 'r-o')



# mor-prima -> U_new
q_new = m * U_new.shape[1]

tic = datetime.now()
XX = PRIMA.PRIMA_mp(E, A, U_new, s, q_new)
YY = XX
Er = (YY.T @ E) @ XX
Ar = (YY.T @ A) @ XX
Br_1 = YY.T @ U_new
Cr = XX.T @ C
nr = Er.shape[0]
toc = datetime.now()
tprima = toc - tic

print("SVD-PRIMA MOR:")
print('PRIMA completed. Time used: ', str(tprima), 's')
print('The original order is', N, 'the reduced order is', nr)
xr0 = XX.T@ x0


# xrAll, time2, dtAll2, urAll2 = tdIntLinBE_svd(t0, tf, dt, Er, -Ar, Br, VS, IS, xr0, srcType,ass_mat)
xrAll, time2, dtAll2, urAll2 = tdIntLinBE_new(t0, tf, dt, Er, -Ar, Br_1, VS, IS[:U_new.shape[1]], xr0, srcType)
xAll_mor2 = XX@xrAll
y_mor2 = Cr.T@xrAll


plt.figure(figsize=(8, 5))
plt.plot(time, y_mor[0,:], color='blue', linestyle='-', marker='o', label='Original', markersize=6, linewidth=1.5)
plt.plot(time_svd, y_svd[0,:], color='red', linestyle='--', marker='s', label='SVD Reduced', markersize=6, linewidth=1.5)

plt.legend(fontsize=12)
plt.title("Comparison of Original and SVD Reduced Data", fontsize=14)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("result", fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()