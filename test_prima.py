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
    data_path = os.path.join(sys.path[0], 'train_data/1t/sim_100_port2000_multiper_sin')
    # data1 = np.load(data_path + '/prima_mor_data.npz')
    # Cr_1 = data1['Cr_1']
    # print('Cr_1:', Cr_1.shape)
    x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time1, pr = prepare_data(data_path,prima=True)
    # y_test = np.load(data_path + '/y_test.npy')

    recording = {'rmse':[], 'nrmse':[], 'r2':[],'mae':[]}
    metrics = calculate_metrix(y_test = y_h[:10,:], y_mean_pre = pr[:,:])
    recording['rmse'].append(metrics['rmse'])
    recording['nrmse'].append(metrics['nrmse'])
    recording['r2'].append(metrics['r2'])
    recording['mae'].append(metrics['mae'])
    record = pd.DataFrame(recording)
    record.to_csv(data_path + '/prima_res.csv', index = False)
