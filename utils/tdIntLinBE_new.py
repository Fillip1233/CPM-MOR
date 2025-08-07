'''
2025-3-1
祖传代码,后向欧拉法
'''
import numpy as np
from datetime import datetime
import scipy.sparse
import scipy.sparse.linalg
from utils.computeInputs_new import *
from numpy.linalg import norm


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

def tdIntLinBE_adaptive(t0=None, tf=None, dt=None, Cl=None, Gl=None, Bl=None, VS=None, IS=None, x0=None, srcType=None, tol=1e-5, dt_min=1e-12, dt_max=1e-2):
    print('Start adaptive backward Euler solve...with tf = ', str(tf), ', dt0 = ', str(dt), '.')
    tic = datetime.now()

    told = t0
    xtold = x0
    xAll = xtold
    ui, uv = computeInputs_new(VS, IS, t0, srcType)
    uAll = np.vstack((ui, uv))

    dtAll = [0]
    time = [t0]
    it = 2

    while told < tf - np.spacing(1):
        accept_step = False
        dt_try = dt
        while not accept_step:
            # Step 1: 一步 dt
            M1 = Cl + dt_try * Gl
            lu1 = scipy.sparse.linalg.splu(M1)

            ui1, uv1 = computeInputs_new(VS, IS, told, srcType)
            u1 = np.vstack((ui1, uv1))
            b1 = Bl @ u1
            f1 = Cl @ xtold + dt_try * b1
            x1 = lu1.solve(f1)
            x1 = x1.reshape(-1, 1)

            # Step 2: 两步 dt/2
            dt_half = dt_try / 2
            M2 = Cl + dt_half * Gl
            lu2 = scipy.sparse.linalg.splu(M2)

            # 第一次半步
            ui_half1, uv_half1 = computeInputs_new(VS, IS, told, srcType)
            u_half1 = np.vstack((ui_half1, uv_half1))
            b_half1 = Bl @ u_half1
            f_half1 = Cl @ xtold + dt_half * b_half1
            x_half1 = lu2.solve(f_half1)
            x_half1 = x_half1.reshape(-1, 1)

            # 第二次半步
            ui_half2, uv_half2 = computeInputs_new(VS, IS, told + dt_half, srcType)
            u_half2 = np.vstack((ui_half2, uv_half2))
            b_half2 = Bl @ u_half2
            f_half2 = Cl @ x_half1 + dt_half * b_half2
            x_half2 = lu2.solve(f_half2)
            x_half2 = x_half2.reshape(-1, 1)

            # 误差估计
            err = norm(x1 - x_half2) / max(norm(x_half2), 1e-12)

            # 更新步长
            if err < tol:
                accept_step = True
                dt = update_dt(dt_try, err, tol, dt_min, dt_max)
                told += dt_try
                xtold = x_half2  # 使用更精确的 x_half2
                xAll = np.hstack((xAll, x_half2))
                uAll = np.hstack((uAll, u_half2))
                dtAll.append(dt_try)
                time.append(told)
                print(f"t = {told:.4e}, dt = {dt_try:.1e}, err = {err:.1e}")
            else:
                dt_try = update_dt(dt_try, err, tol, dt_min, dt_max)
                if dt_try < dt_min:
                    print("Warning: minimum dt reached, accepting step anyway.")
                    accept_step = True
                    told += dt_try
                    xtold = x_half2
                    xAll = np.hstack((xAll, x_half2))
                    uAll = np.hstack((uAll, u_half2))
                    dtAll.append(dt_try)
                    time.append(told)

        it += 1

    toc = datetime.now()
    t = toc - tic
    print('End adaptive backward Euler...done in ', str(t), ' seconds, ', str(it), ' time steps.')
    return xAll, np.array(time), np.array(dtAll), uAll


def update_dt(dt, err, tol, dt_min, dt_max, fac=0.9):
    if err == 0:
        return min(dt * 2, dt_max)
    else:
        dt_new = dt * min(2.0, max(0.5, fac * (tol / err)**0.5))
        return max(dt_min, min(dt_max, dt_new))

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
