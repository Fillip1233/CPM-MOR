'''
2025/4/12
a script to draw the tensor mapping results 
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
from utils.tensor_ann import *
from utils.alpha_ann import *
from utils.res_ann import *
from utils.tensor_rnn import *
from utils.tensor_lstm import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An example program with command-line arguments")
    parser.add_argument("--lr", type= float, default= 1e-2)
    parser.add_argument("--epoch", type= int, default= 200)
    parser.add_argument("--bs", type= int, default= 100)
    parser.add_argument("--hidden_size", type= int, default= 128)
    parser.add_argument("--draw_type", type= int, default= 0)
    parser.add_argument("--module_name", type= str, default= "tensor_ann")
    parser.add_argument("--test_over", type= int, default= 0)

    args = parser.parse_args()
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(sys.path[0], 'train_data/1t/sim_100_port2000_multiper_diff')
    x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time1, pr = prepare_data(data_path,prima=True)
    if args.test_over:
        data_path1 = os.path.join(sys.path[0], 'train_data/1t/sim_100_port2000_multiper_sin')
        x_test, y_test, yl_test, time1, pr = load_over_data(data_path1)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    yl_test = yl_test.to(device)
    pr = pr.to(device)

    data_shape = [y_l[0].shape, y_h[0].shape]
    
    if args.module_name == 'res_ann':
        mynn = res_ann(hidden_size=args.hidden_size, d_num=101).to(device)
        
    elif args.module_name == 'alpha_ann':
        mynn = alpha_ann(hidden_size=args.hidden_size, d_num=101).to(device)
        
    elif args.module_name == 'tensor_ann':
        mynn = tensor_ann(data_shape, hidden_size=args.hidden_size, d_num=101).to(device)
        mynn.load_state_dict(torch.load(data_path + '/tensor_ann.pth'))
        
    elif args.module_name == 'tensor_rnn':
        mynn = tensor_rnn(data_shape, hidden_size=args.hidden_size, d_num=2000, num_layers=1).to(device)
        
    elif args.module_name == 'tensor_lstm':
        mynn = tensor_lstm(data_shape, hidden_size=args.hidden_size, d_num=2000, num_layers=1).to(device)


    total_params = sum(p.numel() for p in mynn.parameters())
    print(f"Total number of parameters: {total_params}")
    with torch.no_grad():
        pre_t1 = time.time()
        yafter1, yafter2 = mynn.draw(yl_test)
        pre_t2 = time.time()

    ##plot the results
    yte = yafter2

    if args.draw_type == 0:
        #全端口波形图
        for i in range(100):
            fig, axs = plt.subplots(1, 4, figsize=(20, 6))

            # cbar_ax1 = fig.add_axes([0.02, 0.3, 0.03, 0.4])
            cbar_ax2 = fig.add_axes([0.94, 0.3, 0.03, 0.4])
            vmin = torch.min(y_test[i])
            vmax = torch.max(y_test[i])

            im1 = axs[0].imshow(yafter1[i].cpu(), cmap='viridis', interpolation='nearest', vmin = vmin, vmax = vmax, aspect='auto')
            axs[0].set_title('yafter1')

            axs[1].imshow(yafter2[i].cpu(), cmap='viridis', interpolation ='nearest', vmin = vmin, vmax = vmax, aspect='auto')
            axs[1].set_title('yafter2')

            axs[2].imshow(yl_test[i].cpu(), cmap='viridis', interpolation ='nearest', vmin = vmin, vmax = vmax, aspect='auto')
            axs[2].set_title('y_l')

            axs[3].imshow((y_test[i]-yafter2[i]).cpu(), cmap='viridis', interpolation ='nearest', vmin = vmin, vmax = vmax, aspect='auto')
            axs[3].set_title('y_h')

            # cbar1 = fig.colorbar(im1, cax=cbar_ax1)
            cbar2 = fig.colorbar(im1, cax=cbar_ax2,location='right', fraction=0.02, pad=0.02)
            # plt.tight_layout()
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
            pr1 = pr[i]
            ppr = torch.zeros((pr1.shape[1])).to(device)
            for p in range(pr1.shape[0]):
                ppr += pr1[p, :]
            plt.plot(time1, yy_low.cpu(), color='black', linestyle='-.', marker='*', label='Low-fidelity-GT', markevery = 35, markersize=6, linewidth=1.5)
            plt.plot(time1, yy_te.cpu(), color='blue', linestyle='-.', marker='*', label='GroundTruth', markevery = 25, markersize=6, linewidth=1.5)
            
            plt.plot(time1, ppr.cpu(), color='green', linestyle='--', marker='o', label='PRIMA', markevery = 30, markersize=6, linewidth=1.5)
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
        pr1 = pr[example+1]
        for i in range(100):
            yy_1 = torch.zeros((y_1.shape[1])).to(device)
            yy_1 = y_1[i, :]
            yy_te = torch.zeros((y_te.shape[1])).to(device)
            yy_te = y_te[i, :]
            yy_low = torch.zeros((y_low.shape[1])).to(device)
            yy_low = y_low[i, :]
            ppr = torch.zeros((pr1.shape[1])).to(device)
            ppr += pr1[i, :]
            plt.plot(time1, yy_low.cpu(), color='black', linestyle='-.', marker='*', label='Low-fidelity-GT', markevery = 35, markersize=6, linewidth=1.5)
            plt.plot(time1, yy_te.cpu(), color='blue', linestyle='-.', marker='*', label='GroundTruth', markevery = 25, markersize=6, linewidth=1.5)
            plt.plot(time1, ppr.cpu(), color='green', linestyle='--', marker='o', label='PRIMA', markevery = 30, markersize=6, linewidth=1.5)
            plt.plot(time1, yy_1.cpu(), color='red', linestyle='-.', marker='*', label='Predict', markevery = 28, markersize=6, linewidth=1.5)
            plt.legend(fontsize=12)
            plt.title("MF-result", fontsize=14)
            plt.xlabel("Time (s)", fontsize=12)
            plt.ylabel("result", fontsize=12)
            plt.grid()
            plt.tight_layout()
            plt.show()
            plt.clf()