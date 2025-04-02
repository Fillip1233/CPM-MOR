'''
2025-3-27
only use the res ann for mor
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorly
from utils.res_ann import *
import os
import sys
import argparse
tensorly.set_backend('pytorch')
from utils.calculate_metrix import calculate_metrix
import pandas as pd
import time

# def prepare_data(data_path):

#     x1 = np.load(data_path+'/mf_inall.npy')
#     x2 = np.load(data_path+'_multiper'+'/mf_inall.npy')
#     x1 = torch.tensor(x1, dtype=torch.float32)
#     x2 = torch.tensor(x2, dtype=torch.float32)
#     x_trainl = torch.cat((x1[:100,:], x2[:100,:]), dim = 0)
#     x_trainh = x_trainl
#     yl1= np.load(data_path+'/mf_low_f.npy')
#     yl2= np.load(data_path+'_multiper'+'/mf_low_f.npy')
#     yl1 = torch.tensor(yl1, dtype=torch.float32)
#     yl2 = torch.tensor(yl2, dtype=torch.float32)
#     y_l = torch.cat((yl1[:100,:], yl2[:100,:]), dim = 0)

#     yh1 = np.load(data_path+'/mf_high_f.npy')
#     yh1 = torch.tensor(yh1, dtype=torch.float32)
#     yh2 = np.load(data_path+'_multiper'+'/mf_high_f.npy')
#     yh2 = torch.tensor(yh2, dtype=torch.float32)
#     y_h = torch.cat((yh1[:100, :], yh2[:100,:]), dim = 0)

#     time = np.load(data_path+'/mf_time.npy')

#     x_test = torch.cat((x1[100:, :], x2[100:, :]), dim = 0)
#     y_test = torch.cat((yh1[100:, :], yh2[100:, :]), dim = 0)
#     yl_test = torch.cat((yl1[100:, :], yl2[100:, :]), dim = 0)

#     return x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time

def prepare_data(data_path):

    x1 = np.load(data_path+'/mf_inall.npy')
    x = torch.tensor(x1, dtype=torch.float32)
   
    yl1= np.load(data_path+'/mf_low_f.npy')
    yl = torch.tensor(yl1, dtype=torch.float32)

    yh1 = np.load(data_path+'/mf_high_f.npy')
    yh = torch.tensor(yh1, dtype=torch.float32)

    time = np.load(data_path+'/mf_time.npy')

    x_trainl = x[:100, :]
    x_trainh = x[:100, :]
    y_l = yl[:100, :]
    y_h = yh[:100, :]

    x_test = x[100:, :]
    y_test = yh[100:, :]
    yl_test = yl[100:, :]

    return x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time

class waveform_data(Dataset):
    def __init__(self, u, y):
        self.u = u
        self.y = y

    def __len__(self):
        return len(self.u)
    
    def __getitem__(self, idx):
        return self.u[idx], self.y[idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An example program with command-line arguments")
    parser.add_argument("--lr", type= float, default= 1e-2)
    parser.add_argument("--epoch", type= int, default= 300)
    parser.add_argument("--bs", type= int, default= 100)
    parser.add_argument("--hidden_size", type= int, default= 128)
    parser.add_argument("--draw_type", type= int, default= 0)
    args = parser.parse_args()
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(sys.path[0], 'train_data/3t/sim_100_port2000_multiper')
    x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time1 = prepare_data(data_path)

    x_test = x_test.to(device)
    y_test = y_test.to(device)
    yl_test = yl_test.to(device)

    data_shape = [y_l[0].shape, y_h[0].shape]
    y_res = y_h - y_l
    waveform_dataset = waveform_data(x_trainh.to(device), y_res.to(device))

    mynn = res_ann(data_shape, hidden_size=args.hidden_size, d_num=101).to(device)

    tr_time1 = time.time()
    train_res_ann(mynn, x_trainh.to(device), y_l.to(device), y_h.to(device), lr = args.lr, epoch = args.epoch)
    tr_time2 = time.time()

    total_params = sum(p.numel() for p in mynn.parameters())
    print(f"Total number of parameters: {total_params}")
    torch.save(mynn.state_dict(), data_path + '/res_ann.pth')

    with torch.no_grad():
        pre_t1 = time.time()
        ypred = mynn.forward_h(x_test, yl_test)
        # ypred = ypred + yl_test
        pre_t2 = time.time()

    ##plot the results
    yte = y_test

    recording = {'rmse':[], 'nrmse':[], 'r2':[],'mae':[], 'pred_time':[],'train_time':[]}
    metrics = calculate_metrix(y_test = yte, y_mean_pre = ypred)
    recording['rmse'].append(metrics['rmse'])
    recording['nrmse'].append(metrics['nrmse'])
    recording['r2'].append(metrics['r2'])
    recording['mae'].append(metrics['mae'])
    recording['pred_time'].append(pre_t2 - pre_t1)
    recording['train_time'].append(tr_time2 - tr_time1)
    record = pd.DataFrame(recording)
    record.to_csv(data_path + '/predit_result_res.csv', index = False)
    
    if args.draw_type == 0:
        #全端口波形图
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

            im2 = axs[2].imshow((yte[i].cpu()-ypred[1].cpu()).abs(), cmap = 'viridis', interpolation='nearest', vmin = vmin, vmax = vmax)
            axs[2].set_title('Difference')

            # cbar1 = fig.colorbar(im1, cax=cbar_ax1)
            cbar2 = fig.colorbar(im2, cax=cbar_ax2)
            plt.show()
            plt.clf()
    
    if args.draw_type == 1:
        #端口响应总和
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
    if args.draw_type == 2:
        #单个端口响应波形图
        plt.figure(figsize=(8, 5))
        example = 1
        y_1 = ypred[example]
        y_te = yte[example]
        y_low = yl_test[example]
        for i in range(100):
            yy_1 = torch.zeros((y_1.shape[1])).to(device)
            yy_1 = y_1[i, :]
            yy_te = torch.zeros((y_te.shape[1])).to(device)
            yy_te = y_te[i, :]
            yy_low = torch.zeros((y_low.shape[1])).to(device)
            yy_low = y_low[i, :]
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