'''
2025-3-19
test mf_mor
'''
import torch
import numpy as np
from utils.Ann_mor import *
import matplotlib.pyplot as plt
import tensorly
from utils.calculate_metrix import calculate_metrix
import pandas as pd
import os
import sys
import time
tensorly.set_backend('pytorch')

def prepare_data(data_path):

    x1 = np.load(data_path+'/mf_inall.npy')
    x1 = torch.tensor(x1, dtype=torch.float32)
    yl1= np.load(data_path+'/mf_low_f.npy')
    yl = torch.tensor(yl1, dtype=torch.float32)

    yh1 = np.load(data_path+'/mf_high_f.npy')
    yh = torch.tensor(yh1, dtype=torch.float32)

    time = np.load(data_path+'/mf_time.npy')

    x_trainl = x1[:100, :]
    x_trainh = x1[:100, :]
    y_l = yl[:100, :]
    y_h = yh[:100, :]

    x_test = x1[100:, :]
    y_test = yh[100:, :]
    yl_test = yl[100:, :]

    x_te = np.load("F:/zhenjie/code/CPM-MOR/train_data/1t/sim_100_port1500_multiper/mf_inall.npy")
    x_te = torch.tensor(x_te, dtype=torch.float32)
    y_te_h = np.load("F:/zhenjie/code/CPM-MOR/train_data/1t/sim_100_port1500_multiper/mf_high_f.npy")
    y_te_h = torch.tensor(y_te_h, dtype=torch.float32)
    y_te_l = np.load("F:/zhenjie/code/CPM-MOR/train_data/1t/sim_100_port1500_multiper/mf_low_f.npy")
    y_te_l = torch.tensor(y_te_l, dtype=torch.float32)
    x_test = x_te[100:, :]
    y_test = y_te_h[100:, :]
    yl_test = y_te_l[100:, :]

    return x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time
    
if __name__ == "__main__":
    test_type = 1
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(sys.path[0], 'train_data/1t/sim_100_port1500')
    x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time1 = prepare_data(data_path)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    yl_test = yl_test.to(device)

    data_shape = [y_l[0].shape, y_h[0].shape]
    fidelity_num = 2

    mynn = ann_mor(fidelity_num , data_shape, hidden_size=128, d_num=101).to(device)
    mynn.load_state_dict(torch.load(data_path + '/ann_mor.pth'))
    
    with torch.no_grad():
        pre_t1 = time.time()
        ypred = mynn(x_test)
        pre_t2 = time.time()

    yte = y_test

    recording = {'rmse':[], 'nrmse':[], 'r2':[],'mae':[], 'time':[]}
    metrics = calculate_metrix(y_test = yte, y_mean_pre = ypred)
    recording['rmse'].append(metrics['rmse'])
    recording['nrmse'].append(metrics['nrmse'])
    recording['r2'].append(metrics['r2'])
    recording['mae'].append(metrics['mae'])
    recording['time'].append(pre_t2 - pre_t1)
    record = pd.DataFrame(recording)
    record.to_csv(data_path + '/predit_result_ann.csv', index = False)

    ## plot the results with all ports -> test type 0
    if test_type == 0:
        for i in range(100):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # cbar_ax1 = fig.add_axes([0.02, 0.3, 0.03, 0.4])
            cbar_ax2 = fig.add_axes([0.94, 0.3, 0.03, 0.4])
            vmin = torch.min(yte[i])
            vmax = torch.max(yte[i])

            im1 = axs[0].imshow(yte[i].cpu(), cmap='viridis', interpolation='nearest', vmin = vmin, vmax = vmax)
            axs[0].set_title('Groundtruth')

            axs[1].imshow(ypred[i].cpu(), cmap='viridis', interpolation ='nearest', vmin = vmin, vmax = vmax)
            axs[1].set_title('Predict')

            im2 = axs[2].imshow((yte[i]-ypred[1]).cpu().abs(), cmap = 'viridis', interpolation='nearest', vmin = vmin, vmax = vmax)
            axs[2].set_title('Difference')

            # cbar1 = fig.colorbar(im1, cax=cbar_ax1)
            cbar2 = fig.colorbar(im2, cax=cbar_ax2)
            plt.show()
            plt.clf()

    ## plot the results with the sum of all ports -> test type 1
    if test_type == 1:
        plt.figure(figsize=(8, 5))
        for i in range(100):
            y_1 = ypred[i]
            yy_1 = torch.zeros((y_1.shape[1])).to(device)
            for j in range(y_1.shape[0]):
                yy_1 += y_1[j, :]
            y_te = yte[i]
            yy_te = torch.zeros((y_te.shape[1])).to(device)
            for k in range(y_te.shape[0]):
                yy_te += y_te[k, :]
            y_low = yl_test[i]
            yy_low = torch.zeros((y_low.shape[1])).to(device)
            for e in range(y_low.shape[0]):
                yy_low += y_low[e, :]
            plt.plot(time1, yy_low.cpu(), color='black', linestyle='-.', marker='*', label='Low-fidelity-GT', markevery = 35, markersize=6, linewidth=1.5)
            plt.plot(time1, yy_te.cpu(), color='blue', linestyle='-.', marker='*', label='GroundTruth', markevery = 25, markersize=6, linewidth=1.5)
            plt.plot(time1, yy_1.cpu(), color='red', linestyle='-.', marker='*', label='Predict', markevery = 28, markersize=6, linewidth=1.5)
            plt.legend(fontsize=12)
            plt.title("MF-result", fontsize=14)
            plt.xlabel("Time (s)", fontsize=12)
            plt.ylabel("result", fontsize=12)
            plt.grid()
            plt.tight_layout()
            plt.show()
            plt.clf()
        