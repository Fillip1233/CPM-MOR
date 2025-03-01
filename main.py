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

# This will be the main function through which we define both tic() and toc()

data = spio.loadmat("./IBM_transient/ibmpg1t.mat")

# the 1e-6 factor is for MNA_5 only. Remove it for other cases
E, A, B = data['E'] * 1e-0, data['A'], data['B']  # rename the matrices
# B = B.toarray()

# s, _ = sp.linalg.eig(A.todense(), E.todense())

N = E.shape[0]
NB = B.shape[1]

Nb = 40   # set the number of port
B = B[:, 0:Nb]
C = B
t0 = 0
# main.m:24
tf = 1e-08
# main.m:24
dt = 1e-11
# main.m:24

srcType = 'pulse'

if operator.eq(srcType, 'sin'):

    # main.m:28
    VS = []
    VS = np.array(VS)
    IS = np.hstack([np.ones([Nb, 1]), np.zeros([Nb, 1]), np.dot(10000000000.0, np.ones([Nb, 1])), np.zeros([Nb, 1]),
                    np.dot(np.ones([Nb, 1]), tf) / 10, np.dot(np.ones([Nb, 1]), 1)])
else:
    if operator.eq(srcType, 'pulse'):
        #     PULSE(V1 V2 TD TR TF PW PER)
        VS = []
        VS = np.array(VS)
        # main.m:32
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

from tdIntLinBE_new import *

xAll, time, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, E, -A, B, VS, IS, x0, srcType)
y = C.T@xAll

plt.plot(time, y[0, :], 'b-o')
plt.show()

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

print('PRIMA completed. Time used: ', str(tprima), 's')
print('The original order is', N, 'the reduced order is', nr)
f_plot = np.logspace(0, 10, 100)  # freq points for plotting transfer function
f_plot = f
s_plot = 1j * 2 * np.pi * f_plot
relErr = []
magorg, magmor, angorg, angmor = [], [], [], []  # mag and angle for original and reduced models
idx = (1, 1)
for i in range(len(s_plot)):
    H = C.T @ (scipy.sparse.linalg.spsolve((s_plot[i] * E - A), B))
    H = H.toarray()
    # main.m:61
    Hr = Cr.T @ (np.linalg.solve((s_plot[i] * Er - Ar), Br))

    magorg.append(np.abs(H[idx]))
    magmor.append(np.abs(Hr[idx]))
    angorg.append(np.angle(H[idx], deg=True))
    angmor.append(np.angle(Hr[idx], deg=True))

    err = H - Hr
    relErr.append(np.linalg.norm(err) / np.linalg.norm(H))
# main.m:63

print('Relative Error is')
print(np.linalg.norm(relErr))
'''
plt.figure()
plt.subplot(1, 2, 1)
plt.semilogx(f_plot, magorg, f_plot, magmor, '-*', label='FOM')
plt.subplot(1, 2, 2)
plt.semilogx(f_plot, angorg, f_plot, angmor, '-*')
plt.show()
plt.figure()
plt.plot(f_plot, relErr)
plt.show()
'''


xr0 = XX.T@ x0
# main.m:66
xrAll, time, dtAll, urAll = tdIntLinBE_new(t0, tf, dt, Er, -Ar, Br, VS, IS, xr0, srcType)
# main.m:67
xAll_mor = XX@xrAll

y_mor = Cr.T@xrAll
# main.m:69
nd = 500
# main.m:71
# plt.plot(time, y_mor[0, :], 'r-o')

plt.plot(time, y[0, :], 'b-o', time, y_mor[0, :], 'r-o')
plt.show()
