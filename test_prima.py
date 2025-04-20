'''
2025/4/10
test prima
'''
import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time
from utils.load_data import *
from utils.calculate_metrix import calculate_metrix


if __name__ == "__main__":

    torch.manual_seed(1)
    data_path = os.path.join(sys.path[0], 'train_data/3t/sim_100_port500_multiper_diff')
    # data1 = np.load(data_path + '/prima_mor_data.npz')
    # Cr_1 = data1['Cr_1']
    # print('Cr_1:', Cr_1.shape)
    x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time1, pr = prepare_data(data_path,prima=True)
    # y_test = np.load(data_path + '/y_test.npy')

    # plt.figure(figsize=(8, 5))
    # for i in range(10):
    #     y_te = y_test[i]
    #     yy_te = torch.zeros((y_te.shape[1]))
    #     for k in range(y_te.shape[0]):
    #         yy_te += y_te[k, :]
    #     if pr is not None:
    #         pr1 = pr[i]
    #         ppr = torch.zeros((pr1.shape[1]))
    #         for p in range(pr1.shape[0]):
    #             ppr += pr1[p, :]
    #     plt.plot(time1, yy_te.cpu(), color='blue', linestyle='-.', marker='*', label='GroundTruth', markevery = 25, markersize=6, linewidth=1.5)
    #     if pr is not None:
    #         plt.plot(time1, ppr.cpu(), color='green', linestyle='--', marker='o', label='PRIMA', markevery = 30, markersize=6, linewidth=1.5)
    #     plt.legend(fontsize=12)
    #     plt.xlabel("Time (s)", fontsize=12)
    #     plt.ylabel("Response result (V)", fontsize=12)
    #     plt.grid()
    #     plt.tight_layout()
    #     plt.show()
    recording = {'rmse':[], 'nrmse':[], 'r2':[],'mae':[]}
    metrics = calculate_metrix(y_test = y_test[:10,:], y_mean_pre = pr[:10,:])
    recording['rmse'].append(metrics['rmse'])
    recording['nrmse'].append(metrics['nrmse'])
    recording['r2'].append(metrics['r2'])
    recording['mae'].append(metrics['mae'])
    record = pd.DataFrame(recording)
    record.to_csv(data_path + '/prima_res.csv', index = False)
