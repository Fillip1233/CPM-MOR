'''
2025/10/15
top module
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
from utils.single_gar import *
import matplotlib.ticker as ticker
from scipy.io import savemat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An example program with command-line arguments")
    parser.add_argument("--lr", type= float, default= 1e-2)
    parser.add_argument("--epoch", type= int, default= 600)
    parser.add_argument("--bs", type= int, default= 100)
    parser.add_argument("--hidden_size", type= int, default= 128)
    parser.add_argument("--draw_type", type= int, default= 0)
    parser.add_argument("--module_name", type= str, default= "tensor_ann")
    parser.add_argument("--test_over", type= int, default= 0)
    parser.add_argument("--cir", type= int, default= 1)

    args = parser.parse_args()
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data_path = os.path.join(f'./Exp_res/DeMOR_data/{args.cir}t/')
    data_path = os.path.join(f'./MSIP_BDSM/train_data/{args.cir}t/')
    # x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time1,pr = prepare_data_mix(data_path,prima = True)
    x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time1, pr = prepare_data(data_path, prima=False)
    if args.test_over:
        data_path1 = os.path.join(sys.path[0], 'train_data/1t/sim_100_port2000_multiper_sin')
        x_test, y_test, yl_test, time1, pr = load_over_data(data_path1)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    yl_test = yl_test.to(device)
    if pr is not None:
        pr = pr.to(device)

    data_shape = [y_l[0].shape, y_h[0].shape]
    initial_data = [
        {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_trainl.to(device), 'Y': y_l.to(device)},
        {'fidelity_indicator': 1,'raw_fidelity_name': '1', 'X': x_trainh.to(device), 'Y': y_h.to(device)},
        
    ]
    fidelity_num = len(initial_data)
    fidelity_manager = MultiFidelityDataManager(initial_data)
    if args.module_name == 'res_ann':
        mynn = res_ann(hidden_size=args.hidden_size, d_num=101).to(device)
        tr_time1 = time.time()
        train_res_ann(mynn, fidelity_manager, lr = args.lr, epoch = args.epoch)
        tr_time2 = time.time()
        # torch.save(mynn.state_dict(), data_path + '/res_ann.pth')
        
    elif args.module_name == 'alpha_ann':
        mynn = alpha_ann(hidden_size=args.hidden_size, d_num=101).to(device)
        tr_time1 = time.time()
        train_alpha_ann(mynn, fidelity_manager, lr = args.lr, epoch = args.epoch)
        # torch.save(mynn.state_dict(), data_path + '/alpha_ann.pth')
        print('alpha:', mynn.alpha)
        tr_time2 = time.time()
    elif args.module_name == 'tensor_ann':
        mynn = tensor_ann(data_shape, hidden_size=args.hidden_size, d_num=101).to(device)
        tr_time1 = time.time()
        train_tensor_ann(mynn, fidelity_manager, lr = args.lr, epoch = args.epoch)
        tr_time2 = time.time()
        torch.save(mynn.state_dict(), data_path + '/tensor_ann.pth')
        
    elif args.module_name == 'tensor_rnn':
        mynn = tensor_rnn(data_shape, hidden_size=args.hidden_size, d_num=2000, num_layers=1).to(device)
        tr_time1 = time.time()
        train_tensor_rnn(mynn, fidelity_manager, lr = args.lr, epoch = args.epoch)
        tr_time2 = time.time()
        # torch.save(mynn.state_dict(), data_path + '/tensor_rnn.pth')
        
    elif args.module_name == 'tensor_lstm':
        mynn = tensor_lstm(data_shape, hidden_size=args.hidden_size, d_num=2000, num_layers=1).to(device)
        tr_time1 = time.time()
        train_tensor_lstm(mynn, fidelity_manager, lr = args.lr, epoch = args.epoch)
        tr_time2 = time.time()
        torch.save(mynn.state_dict(), data_path + '/tensor_lstm.pth')
    elif args.module_name == 'gar':
        mynn = gar(data_shape).to(device)
        tr_time1 = time.time()
        train_gar(mynn, fidelity_manager, max_iter = args.epoch, lr_init = args.lr)
        tr_time2 = time.time()
        

    total_params = sum(p.numel() for p in mynn.parameters())
    print(f"Total number of parameters: {total_params}")
    with torch.no_grad():
        pre_t1 = time.time()
        if args.module_name == 'gar':
            ypred = mynn.forward(fidelity_manager, x_test, yl_test)
        else:
            ypred = mynn.forward_h(x_test, yl_test)
        pre_t2 = time.time()

    ##plot the results
    yte = y_test

    # savemat('ypred.mat', {'ypred': ypred.cpu().numpy()})
    # savemat('yte.mat', {'yte': yte.cpu().numpy()})

    recording = {'rmse':[], 'nrmse':[], 'r2':[],'mae':[], 'pred_time':[],'train_time':[]}
    # metrics = calculate_metrix(y_test = yte[:10,:], y_mean_pre = ypred[:10,:])
    metrics = calculate_metrix(y_test = yte, y_mean_pre = ypred)
    recording['rmse'].append(metrics['rmse'])
    recording['nrmse'].append(metrics['nrmse'])
    recording['r2'].append(metrics['r2'])
    recording['mae'].append(metrics['mae'])
    recording['pred_time'].append(pre_t2 - pre_t1)
    recording['train_time'].append(tr_time2 - tr_time1)
    record = pd.DataFrame(recording)
    if args.test_over:
        record.to_csv(data_path1 + '/{}_over_res.csv'.format(args.module_name), index = False)
    else:
        record.to_csv(data_path + '/{}_res.csv'.format(args.module_name), index = False)

    if args.draw_type == 0:
        #全端口波形图
        for i in range(100):
            i = 99
            fig, axs = plt.subplots(1, 3, figsize=(10, 6))

            # cbar_ax1 = fig.add_axes([0.02, 0.3, 0.03, 0.4])
            cbar_ax2 = fig.add_axes([0.94, 0.3, 0.03, 0.4])
            vmin = torch.min(yte[i])
            vmax = torch.max(yte[i])

            im1 = axs[0].imshow(yte[i].cpu(), cmap='viridis', interpolation='nearest', vmin = vmin, vmax = vmax, aspect='auto')
            axs[0].set_title('Groundtruth')

            axs[1].imshow(ypred[i].cpu(), cmap='viridis', interpolation ='nearest', vmin = vmin, vmax = vmax, aspect='auto')
            axs[1].set_title('Predict')

            im2 = axs[2].imshow((yte[i].cpu()-ypred[i].cpu()).abs(), cmap = 'viridis', interpolation='nearest', vmin = vmin, vmax = vmax, aspect='auto')
            axs[2].set_title('Difference')

            # cbar1 = fig.colorbar(im1, cax=cbar_ax1)
            cbar2 = fig.colorbar(im2, cax=cbar_ax2,location='right', fraction=0.02, pad=0.02)
            cbar2.formatter = ticker.ScalarFormatter(useMathText=True)  # 启用科学计数法
            cbar2.formatter.set_powerlimits((0, 0))  # 设置科学计数法的阈值（这里对所有数值生效）
            cbar2.update_ticks()
            fig.suptitle(f"ibmpg{args.cir}t time and voltage model predictions and errors for each port", fontsize=14)
            # plt.tight_layout()
            fig.subplots_adjust(left=0.05, right=0.92, top=0.85, bottom=0.1)
            # plt.show()
            plt.savefig(f'./MSIP_BDSM/Exp_res/{args.cir}t/ibmpg{args.cir}t_example_{i}_response.png')
            plt.clf()
            break
    
    if args.draw_type == 1:
        #端口响应总和
        plt.figure(figsize=(8, 5))
        # for i in range(100):
        i = 99
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
        if pr is not None:
            pr1 = pr[i]
            ppr = torch.zeros((pr1.shape[1])).to(device)
            for p in range(pr1.shape[0]):
                ppr += pr1[p, :]
        plt.plot(time1, yy_low.cpu(), color='black', linestyle='-.', marker='*', label='Low-fidelity', markevery = 35, markersize=6, linewidth=1.5)
        plt.plot(time1, yy_te.cpu(), color='blue', linestyle='-.', marker='*', label='GroundTruth', markevery = 25, markersize=6, linewidth=1.5)
        if pr is not None:
            plt.plot(time1, ppr.cpu(), color='green', linestyle='--', marker='o', label='PRIMA', markevery = 30, markersize=6, linewidth=1.5)
        
        plt.plot(time1, yy_1.cpu(), color='red', linestyle='-.', marker='*', label='MF-MOR', markevery = 28, markersize=6, linewidth=1.5)
        plt.legend(fontsize=12)
        plt.title(f"ibmpg{args.cir}t total port response", fontsize=14)
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Response result (V)", fontsize=12)
        plt.grid()
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./MSIP_BDSM/Exp_res/{args.cir}t/ibmpg{args.cir}t_total_port_response.png')
        plt.clf()
        # break
    if args.draw_type == 2:
        #单个端口响应波形图
        plt.figure(figsize=(8, 5))
        example = 99
        y_1 = ypred[example]
        y_te = yte[example]
        y_low = yl_test[example]
        if pr is not None:
            pr1 = pr[example]
        for i in range(100):
            i = 100
            yy_1 = torch.zeros((y_1.shape[1])).to(device)
            yy_1 = y_1[i, :]
            yy_te = torch.zeros((y_te.shape[1])).to(device)
            yy_te = y_te[i, :]
            yy_low = torch.zeros((y_low.shape[1])).to(device)
            yy_low = y_low[i, :]
            if pr is not None:
                ppr = torch.zeros((pr1.shape[1])).to(device)
                ppr = pr1[i, :]
            
            plt.plot(time1, yy_low.cpu(), color='black', linestyle='-.', marker='*', label='Low-fidelity', markevery = 35, markersize=6, linewidth=1.5)
            plt.plot(time1, yy_te.cpu(), color='blue', linestyle='-.', marker='*', label='GroundTruth', markevery = 25, markersize=6, linewidth=1.5)
            if pr is not None:
                plt.plot(time1, ppr.cpu(), color='green', linestyle='--', marker='o', label='PRIMA', markevery = 30, markersize=6, linewidth=1.5)
            plt.plot(time1, yy_1.cpu(), color='red', linestyle='-.', marker='*', label='MF-MOR', markevery = 28, markersize=6, linewidth=1.5)
            # plt.plot(time1, yy_low.cpu(), color='black', linestyle='-.', marker='*', label='LF', markevery = 35, markersize=6, linewidth=1.5)
            # plt.plot(time1, yy_te.cpu(), color='blue', linestyle='-.', marker='*', label='GT', markevery = 25, markersize=6, linewidth=1.5)
            # if pr is not None:
            #     plt.plot(time1, ppr.cpu(), color='green', linestyle='--', marker='o', label='PRIMA', markevery = 30, markersize=6, linewidth=1.5)
            # plt.plot(time1, yy_1.cpu(), color='red', linestyle='-.', marker='*', label='LF+', markevery = 28, markersize=6, linewidth=1.5)
            plt.legend(fontsize=12)
            plt.title(f"ibmpg{args.cir}t single-Port output response", fontsize=14)
            plt.xlabel("Time (s)", fontsize=12)
            plt.ylabel("Response result (V)", fontsize=12)
            plt.grid()
            plt.tight_layout()
            plt.savefig(f'./MSIP_BDSM/Exp_res/{args.cir}t/ibmpg{args.cir}t_example_{example}_response_port.png', dpi=300)
            plt.clf()
            break
    if args.draw_type == 3:
        pass