import scipy.io as spio
import numpy as np
from tdIntLinBE_new import *
import matplotlib.pyplot as plt
import PRIMA as PRIMA

if __name__ == '__main__':
    data = spio.loadmat("./IBM_transient/ibmpg1t.mat")
    E, A, B = data['E'] * 1e-0, data['A'], data['B']
    port_num = 10
    Nb = 10
    B = B[:, 2000:2000+port_num]
    B = B.toarray()
    idx_h = np.nonzero(B)[0].reshape(B.shape[1], -1)[:, 0]
    idx_l = np.nonzero(B)[1].reshape(B.shape[1], -1)[:, 0]
    C = B
    b = np.sum(B, axis=1, keepdims=True)
    c = b

    t0 = 0
    tf = 1e-08
    dt = 1e-11
    VS = []
    VS = np.array(VS)
    is_num = int(Nb/2)
    # main.m:32
    IS = np.hstack(
        [np.zeros([is_num, 1]), 
            np.dot(np.ones([is_num, 1]), 0.1),
            np.dot(np.ones([is_num, 1]), tf) / 5,
            np.dot(np.ones([is_num, 1]), 1e-09),
            np.dot(np.ones([is_num, 1]), 1e-09),
            np.dot(np.ones([is_num, 1]), 5e-09),
            np.dot(np.ones([is_num, 1]), 1e-08)])
    IS2 = np.hstack([
        np.zeros([is_num, 1]),
        np.dot(np.ones([is_num, 1]), 0.2),
        np.dot(np.ones([is_num, 1]), tf) / 6,
        np.dot(np.ones([is_num, 1]), 2e-09),
        np.dot(np.ones([is_num, 1]), 2e-09),
        np.dot(np.ones([is_num, 1]), 4e-09),
        np.dot(np.ones([is_num, 1]), 1e-08)])
    IS = np.vstack([IS, IS2])
    N = E.shape[0]
    x0 = np.zeros((N, 1))
    srcType = 'pulse'
    xAll, time, dtAll, uAll = tdIntLinBE_new(t0, tf, dt, E, -A, B, VS, IS, x0, srcType)
    y = C.T@xAll

    k = np.zeros((y.shape[1]))
    for i in range(port_num):
        k += y[i, :]
    plt.plot(time, k, 'b-',label='Original Output')
    # # # # # plt.show()


    # xAll_1, time_1, dtAll, uAll = tdIntLinBE_re(t0, tf, dt, E, -A, b, VS, IS, x0, srcType, idx_h, idx_l)
    # y_re = c.T@xAll_1

    # plt.plot(time_1, y_re[0, :], 'r--',label='Compression port Output')
    # plt.legend()
    # plt.show()

    f = np.array([1e2,1e9])  # array of targeting frequencies
    m = 2 
    s = 1j * 2 * np.pi * f  # array of expansion points

    q = m * Nb
    tic = datetime.now()
    XX = PRIMA.PRIMA_mp(E, A, B, s, q)
    
    Er = (XX.T @ E) @ XX
    Ar = (XX.T @ A) @ XX
    Br = XX.T @ B
    Cr = XX.T @ C
    nr = Er.shape[0]
    toc = datetime.now()
    tprima = toc - tic
    print('PRIMA completed. Time used: ', str(tprima), 's')
    print('The original order is', N, 'the reduced order is', nr)
    xr0 = XX.T@ x0
    xrAll, time, dtAll, urAll = tdIntLinBE_new(t0, tf, dt, Er, -Ar, Br, VS, IS, xr0, srcType)
    xAll_mor = XX@xrAll
    y_mor = Cr.T@xrAll

    k_re = np.zeros((y_mor.shape[1]))
    for i in range(port_num):
        k_re += np.real(y_mor[i, :])
    plt.plot(time, k_re, 'g--',label='Mor Output')
    # plt.plot(time, y_mor[0, :], 'g--', label='Mor Output')

    # f_plot = np.logspace(0, 10, 100)  # freq points for plotting transfer function
    # f_plot = f
    # s_plot = 1j * 2 * np.pi * f_plot
    # relErr = []
    # magorg, magmor, angorg, angmor = [], [], [], []  # mag and angle for original and reduced models
    # idx = (1, 1)
    # for i in range(len(s_plot)):
    #     H = C.T @ (scipy.sparse.linalg.spsolve((s_plot[i] * E - A), B))
    #     # H = H.toarray()
    #     Hr = Cr.T @ (np.linalg.solve((s_plot[i] * Er - Ar), Br))

    #     err = H - Hr
    #     relErr.append(np.linalg.norm(err) / np.linalg.norm(H))

    # print('Relative Error is')
    # print(np.linalg.norm(relErr))

    Nb_new = b.shape[1]
    q_new = m * Nb_new
    tic = datetime.now()
    XX_new = PRIMA.PRIMA_mp(E, A, b, s, q_new)
    Er_new = (XX_new.T @ E) @ XX_new
    Ar_new = (XX_new.T @ A) @ XX_new
    Br_new = XX_new.T @ b
    Cr_new = XX_new.T @ c
    nr_new = Er_new.shape[0]
    toc = datetime.now()
    tprima_new = toc - tic
    print('PRIMA completed. Time used: ', str(tprima_new), 's')
    print('The original order is', N, 'the reduced order is', nr_new)
    is2_num = int(nr_new/2)
    IS_new = np.hstack([
        np.zeros([is2_num, 1]),
        np.dot(np.ones([is2_num, 1]), 0.1),
        np.dot(np.ones([is2_num, 1]), tf) / 5,
        np.dot(np.ones([is2_num, 1]), 1e-09),
        np.dot(np.ones([is2_num, 1]), 1e-09),
        np.dot(np.ones([is2_num, 1]), 5e-09),
        np.dot(np.ones([is2_num, 1]), 1e-08)])
    IS2_new = np.hstack(
        [np.zeros([is2_num, 1]), 
            np.dot(np.ones([is2_num, 1]), 0.2),
            np.dot(np.ones([is2_num, 1]), tf) / 6,
            np.dot(np.ones([is2_num, 1]), 2e-09),
            np.dot(np.ones([is2_num, 1]), 2e-09),
            np.dot(np.ones([is2_num, 1]), 4e-09),
            np.dot(np.ones([is2_num, 1]), 1e-08)])
    IS_new = np.vstack([IS_new, IS2_new])
    xr0_new = XX_new.T@ x0
    xrAll_new, time, dtAll, urAll = tdIntLinBE_re(t0, tf, dt, Er_new, -Ar_new, Br_new, VS, IS_new, xr0_new, srcType)
    xAll_mor_new = XX_new@xrAll_new
    y_mor_new = Cr_new.T@xrAll_new


    # f_plot = np.logspace(0, 10, 100)  # freq points for plotting transfer function
    # f_plot = f
    # s_plot = 1j * 2 * np.pi * f_plot
    # relErr = []
    # for i in range(len(s_plot)):
    #     H = c.T @ (scipy.sparse.linalg.spsolve((s_plot[i] * E - A), b))
    #     # H = H.toarray()
    #     Hr_new = Cr_new.T @ (np.linalg.solve((s_plot[i] * Er_new - Ar_new), Br_new))

    #     err = H - Hr_new
    #     relErr.append(np.linalg.norm(err) / np.linalg.norm(H))

    # print('Relative Error is')
    # print(np.linalg.norm(relErr))

    plt.plot(time, y_mor_new[0, :], 'k-.', label='Compressed mor Output')
    plt.legend()
    plt.show()
