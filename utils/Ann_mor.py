import torch.nn as nn
import torch
from utils.GAR import *
import os
import sys
import numpy as np

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

    return x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time

class ann_mor(nn.Module):
    def __init__(self, fidelity_num, data_shape_list ,hidden_size, d_num = 100):
        super(ann_mor, self).__init__()
        self.fidelity_num = fidelity_num
        self.f_list = []
        for _ in range(fidelity_num):
            self.f_list.append( torch.nn.Sequential(
            nn.Linear(d_num, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, d_num),
        ))
        self.f_list = torch.nn.ModuleList(self.f_list)
        # self.Tensor_linear_list = []
        # for i in range(self.fidelity_num - 1):
        #     self.Tensor_linear_list.append(Tensor_linear(data_shape_list[i], data_shape_list[i + 1]))
        # self.Tensor_linear_list = torch.nn.ModuleList(self.Tensor_linear_list)

    def forward(self, u , to_fidelity = None):
        if to_fidelity is not None:
            fidelity_level = to_fidelity
        else:
            fidelity_level = self.fidelity_num - 1
        batch_size, port_num, d_num = u.shape
        u = u.view(-1, d_num)
        for i_fidelity in range(fidelity_level + 1):
            if i_fidelity == 0:
                yf_l = self.f_list[i_fidelity](u)
                if fidelity_level == 0:
                    yf_h = yf_l
            else:
                # yf_l = yf_l.view(batch_size, port_num, d_num)
                # yf_out = self.Tensor_linear_list[i_fidelity - 1](yf_l)
                # yf_out = yf_out.view(-1, d_num)
                yf_res = self.f_list[i_fidelity](u)
                # yf_h = yf_out + yf_res  
                yf_h = yf_l + yf_res       
        u_f = yf_h.view(batch_size, port_num, d_num)
        return u_f
    def forward_nn(self, u, i_fidelity = None):
        batch_size, port_num, d_num = u.shape
        u = u.view(-1, d_num)
        y_pred = self.f_list[i_fidelity](u)
        u_f = y_pred.view(batch_size, port_num, d_num)
        return u_f

        
    
def train_ann_mor(ann_mor, data_manager, lr = 0.01, epoch = 100, normal = False):
    
    criterion = nn.MSELoss()
    for i_fidelity in range(ann_mor.fidelity_num):
        optimizer = torch.optim.Adam(ann_mor.parameters(), lr = lr)
        if i_fidelity == 0:
            x_low, y_low = data_manager.get_data(i_fidelity, normal = normal)
            for i in range(epoch):
                optimizer.zero_grad()
                pred = ann_mor(x_low, to_fidelity = i_fidelity)
                loss = criterion(pred, y_low)
                loss.backward()
                optimizer.step()
                print('fidelity {}, epoch {}, loss {}'.format(i_fidelity, i, loss), end='\r')
            print('')
        else:
            x_low, y_low = data_manager.get_data(i_fidelity - 1, normal = normal)
            x_high, y_high = data_manager.get_data(i_fidelity, normal = normal)
            for i in range(epoch):
                optimizer.zero_grad()
                # y_res = y_high - ann_mor.Tensor_linear_list[i_fidelity - 1](y_low)
                y_res = y_high - y_low
                pred = ann_mor.forward_nn(x_high,i_fidelity = i_fidelity)
                loss = criterion(pred, y_res)
                loss.backward()
                optimizer.step()
                print('fidelity {}, epoch {}, loss {}'.format(i_fidelity, i, loss), end='\r')
            print('\n')

if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(sys.path[0], 'train_data/1t/sim_100_port1500_multiper')
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