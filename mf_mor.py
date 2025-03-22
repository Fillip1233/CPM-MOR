'''
2025-3-11
use GAR to perform mf_mor
'''
import torch
import numpy as np
from utils.MF_data import MultiFidelityDataManager
import matplotlib.pyplot as plt
import tensorly
from utils.GAR import *
import os
import sys
tensorly.set_backend('pytorch')

def prepare_data(data_path):

    x1 = np.load(data_path+'/mf_inall.npy')
    # x1 = np.repeat(x1[:, np.newaxis, :], 100, axis=1)
    x1 = torch.tensor(x1, dtype=torch.float32)
    # x1 = torch.fft.fft(x1,dim = -1)
    # x1 = torch.abs(x1)
    x = x1.reshape(x1.shape[0], -1)
    yl1= np.load(data_path+'/mf_low_f.npy')
    yl = torch.tensor(yl1, dtype=torch.float32)
    # yl = torch.fft.fft(yl,dim = -1)
    # yl = torch.abs(yl)

    yh1 = np.load(data_path+'/mf_high_f.npy')
    yh = torch.tensor(yh1, dtype=torch.float32)
    # yh = torch.fft.fft(yh,dim = -1)
    # yh = torch.abs(yh)

    time = np.load(data_path+'/mf_time.npy')

    x_trainl = x[:100, :]
    x_trainh = x[:100, :]
    y_l = yl[:100, :]
    y_h = yh[:100, :]

    x_test = x[100:, :]
    y_test = yh[100:, :]
    yl_test = yl[100:, :]

    return x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time
    
if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(sys.path[0], 'train_data/3t/sim_100_port2000_multiper')
    # device = torch.device("cpu")
    x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time = prepare_data(data_path)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    yl_test = yl_test.to(device)

    data_shape = [y_l[0].shape, y_h[0].shape]

    initial_data = [
        {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_trainl.to(device), 'Y': y_l.to(device)},
        {'fidelity_indicator': 1,'raw_fidelity_name': '1', 'X': x_trainh.to(device), 'Y': y_h.to(device)},
        # {'fidelity_indicator': 2,'raw_fidelity_name': '2', 'X': x_train.to(device), 'Y': y_h2.to(device)}
    ]
    fidelity_num = len(initial_data)
    fidelity_manager = MultiFidelityDataManager(initial_data)

    myGAR = GAR(fidelity_num, data_shape, if_nonsubset = False).to(device)

    train_GAR(myGAR, fidelity_manager, max_iter = 400, lr_init = 1e-2, normal = False, debugger = None)

    total_params = sum(p.numel() for p in myGAR.parameters())
    print(f"Total number of parameters: {total_params}")
    torch.save({ "model_state":myGAR.state_dict(),
                 "hogp1_g": myGAR.hogp_list[0].g,
                 "hogp1_A": myGAR.hogp_list[0].A,
                 "hogp1_K": myGAR.hogp_list[0].K,
                 "hogp1_K_eigen": myGAR.hogp_list[0].K_eigen,
                 "hogp2_K": myGAR.hogp_list[1].K,
                 "hogp2_K_eigen": myGAR.hogp_list[1].K_eigen,
                 "hogp2_A": myGAR.hogp_list[1].A,
                 "hogp2_g": myGAR.hogp_list[1].g,
                }, data_path + "\model.pth")

    with torch.no_grad():
        # x_test = fidelity_manager.normalizelayer[myGAR.fidelity_num-1].normalize_x(x_test)
        ypred, ypred_var = myGAR(fidelity_manager, x_test)
        # ypred, ypred_var = fidelity_manager.normalizelayer[myGAR.fidelity_num-1].denormalize(ypred, ypred_var)

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
        
    # plt.show()