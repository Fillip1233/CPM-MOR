'''
2025-3-1
祖传代码,后向欧拉法
'''
import numpy as np
from datetime import datetime
import scipy.sparse
import scipy.sparse.linalg
from utils.computeInputs_new import *


def tdIntLinBE_new(t0=None, tf=None, dt=None, Cl=None, Gl=None, Bl=None, VS=None, IS=None, x0=None, srcType=None):
    print('Start backward Euler solve...with tf = ', str(tf), ', dt = ', str(dt), '.')
    print('Current time step is:     ')
    tic = datetime.now()
    xtold = x0
    xAll = xtold
    ui, uv = computeInputs_new(VS, IS, t0, srcType)
    
    uAll = np.vstack((ui, uv))
    told = t0
    dt0 = dt
    it = 2
    dtAll = [0]
    time = [0]
    M = Cl + dt * Gl 
    lu = scipy.sparse.linalg.splu(M)
    while told < tf - np.spacing(1):
        ui, uv = computeInputs_new(VS, IS, told, srcType)
        b = Bl @ np.vstack((ui, uv)) #akhab setting is current sources first and voltage sources later
        f = Cl @ xtold + dt * b
        # interesting thing
        if type(M) is np.ndarray:
            xnew = np.linalg.solve(M, f)
        else:
            # xnew = scipy.sparse.linalg.spsolve(M, f)
            xnew = lu.solve(f)
            xnew = np.reshape(xnew, (M.shape[1], f.shape[1]))
        xAll = np.hstack((xAll, xnew))
        uAll = np.hstack((uAll, np.vstack((ui, uv))))

        dtAll.append(dt)
        uold = np.vstack((ui, uv))
        told = told + dt
        xtold = xnew
        time.append(told)
        it = it + 1
    #     toc;
    dtAll = np.array(dtAll)
    time = np.array(time)
    NT = it
    toc = datetime.now()
    t = toc - tic
    print('End backward Euler solve...done in ', str(t), ' seconds, ', str(NT), ' time steps.')

    return xAll, time, dtAll, uAll

def tdIntLinBE_re(t0=None, tf=None, dt=None, Cl=None, Gl=None, Bl=None, VS=None, IS=None, x0=None, srcType=None, idx_h=None, idx_l=None):
    '''
    采用点乘的方法计算,不知道能不能保持moment
    '''
    print('Start backward Euler solve...with tf = ', str(tf), ', dt = ', str(dt), '.')
    print('Current time step is:     ')
    tic = datetime.now()
    xtold = x0
    xAll = xtold
    ui, uv = computeInputs_new(VS, IS, t0, srcType)
    uAll = np.vstack((ui, uv))
    told = t0
    dt0 = dt
    it = 2
    dtAll = [0]
    time = [0]
    M = Cl + dt * Gl
    lu = scipy.sparse.linalg.splu(M)
    B_1 = np.zeros((Bl.shape[0], Bl.shape[1]))
    while told < tf - np.spacing(1):
        ui, uv = computeInputs_new(VS, IS, told, srcType)
        if idx_h is not None:
            B_1[idx_h] = Bl[idx_h] * np.vstack((ui, uv))[idx_l]
        else:
            B_1 = Bl * np.vstack((ui, uv))
        b = B_1
        f = Cl @ xtold + dt * b
        # interesting thing
        if type(M) is np.ndarray:
            xnew = np.linalg.solve(M, f)
        else:
            # xnew = scipy.sparse.linalg.spsolve(M, f)
            xnew = lu.solve(f)
            xnew = np.reshape(xnew, (M.shape[1], f.shape[1]))
        xAll = np.hstack((xAll, xnew))
        uAll = np.hstack((uAll, np.vstack((ui, uv))))

        dtAll.append(dt)
        uold = np.vstack((ui, uv))
        told = told + dt
        xtold = xnew
        time.append(told)
        it = it + 1
    #     toc;
    dtAll = np.array(dtAll)
    time = np.array(time)
    NT = it
    toc = datetime.now()
    t = toc - tic
    print('End backward Euler solve...done in ', str(t), ' seconds, ', str(NT), ' time steps.')

    return xAll, time, dtAll, uAll


if __name__ == '__main__':
    pass
