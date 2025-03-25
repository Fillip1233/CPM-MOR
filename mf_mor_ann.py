'''
2025-3-24
ann network for mf_mor
'''
import torch
import numpy as np
from utils.MF_data import MultiFidelityDataManager
import matplotlib.pyplot as plt
import tensorly
from utils.Ann_mor import *
import os
import sys
tensorly.set_backend('pytorch')

def prepare_data(data_path):

    # x1 = np.load(data_path+'/mf_inall.npy')
    # x1 = torch.tensor(x1, dtype=torch.float32)
    # yl1= np.load(data_path+'/mf_low_f.npy')
    # yl = torch.tensor(yl1, dtype=torch.float32)

    # yh1 = np.load(data_path+'/mf_high_f.npy')
    # yh = torch.tensor(yh1, dtype=torch.float32)

    # time = np.load(data_path+'/mf_time.npy')

    # x_trainl = x1[:100, :]
    # x_trainh = x1[:100, :]
    # y_l = yl[:100, :]
    # y_h = yh[:100, :]

    # x_test = x1[100:, :]
    # y_test = yh[100:, :]
    # yl_test = yl[100:, :]

    x1 = np.load(data_path+'/mf_inall.npy')
    x2 = np.load(data_path+'_multiper'+'/mf_inall.npy')
    x1 = torch.tensor(x1, dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)
    x_trainl = torch.cat((x1[:100,:], x2[:100,:]), dim = 0)
    x_trainh = x_trainl
    yl1= np.load(data_path+'/mf_low_f.npy')
    yl2= np.load(data_path+'_multiper'+'/mf_low_f.npy')
    yl1 = torch.tensor(yl1, dtype=torch.float32)
    yl2 = torch.tensor(yl2, dtype=torch.float32)
    y_l = torch.cat((yl1[:100,:], yl2[:100,:]), dim = 0)

    yh1 = np.load(data_path+'/mf_high_f.npy')
    yh1 = torch.tensor(yh1, dtype=torch.float32)
    yh2 = np.load(data_path+'_multiper'+'/mf_high_f.npy')
    yh2 = torch.tensor(yh2, dtype=torch.float32)
    y_h = torch.cat((yh1[:100, :], yh2[:100,:]), dim = 0)

    time = np.load(data_path+'/mf_time.npy')

    x_test = torch.cat((x1[100:, :], x2[100:, :]), dim = 0)
    y_test = torch.cat((yh1[100:, :], yh2[100:, :]), dim = 0)
    yl_test = torch.cat((yl1[100:, :], yl2[100:, :]), dim = 0)

    return x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time
    
if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(sys.path[0], 'train_data/1t/sim_100_port1500')
    x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time = prepare_data(data_path)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    yl_test = yl_test.to(device)

    data_shape = [y_l[0].shape, y_h[0].shape]

    initial_data = [
        {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_trainl.to(device), 'Y': y_l.to(device)},
        {'fidelity_indicator': 1,'raw_fidelity_name': '1', 'X': x_trainh.to(device), 'Y': y_h.to(device)},
        
    ]
    fidelity_num = len(initial_data)
    fidelity_manager = MultiFidelityDataManager(initial_data)

    mynn = ann_mor(fidelity_num, data_shape, hidden_size=128, d_num=101).to(device)

    train_ann_mor(mynn, fidelity_manager, epoch = 200, lr_init = 1e-2, normal = False)

    total_params = sum(p.numel() for p in mynn.parameters())
    print(f"Total number of parameters: {total_params}")
    torch.save(mynn.state_dict(), data_path + '/ann_mor.pth')

    with torch.no_grad():
        ypred = mynn(x_test)

    ##plot the results
    yte = y_test
    
    # for i in range(100):
    #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    #     # cbar_ax1 = fig.add_axes([0.02, 0.3, 0.03, 0.4])
    #     cbar_ax2 = fig.add_axes([0.94, 0.3, 0.03, 0.4])
    #     vmin = torch.min(yte[i])
    #     vmax = torch.max(yte[i])

    #     im1 = axs[0].imshow(yte[i].cpu(), cmap='viridis', interpolation='nearest', vmin = vmin, vmax = vmax)
    #     axs[0].set_title('Groundtruth')

    #     axs[1].imshow(ypred[i].cpu(), cmap='viridis', interpolation ='nearest', vmin = vmin, vmax = vmax)
    #     axs[1].set_title('Predict')

    #     im2 = axs[2].imshow((yte[i].cpu()-ypred[1].cpu()).abs(), cmap = 'viridis', interpolation='nearest', vmin = vmin, vmax = vmax)
    #     axs[2].set_title('Difference')

    #     # cbar1 = fig.colorbar(im1, cax=cbar_ax1)
    #     cbar2 = fig.colorbar(im2, cax=cbar_ax2)
    #     plt.show()
    #     plt.clf()

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
        plt.plot(time, yy_low.cpu(), color='black', linestyle='-.', marker='*', label='Low-fidelity-GT', markevery = 35, markersize=6, linewidth=1.5)
        plt.plot(time, yy_te.cpu(), color='blue', linestyle='-.', marker='*', label='GroundTruth', markevery = 25, markersize=6, linewidth=1.5)
        plt.plot(time, yy_1.cpu(), color='red', linestyle='-.', marker='*', label='Predict', markevery = 28, markersize=6, linewidth=1.5)
        plt.legend(fontsize=12)
        plt.title("MF-result", fontsize=14)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("result", fontsize=12)
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.clf()