'''
2025-3-1
nn拟合波形以及alpha想法相关思路尝试
'''
import scipy.io as spio
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from utils.normalization import MaxMinNormalization
import argparse
import os
import sys
import logging
    
class port_module(nn.Module):
    def __init__(self, u_num, hidden_size, d_num = 200):
        super(port_module, self).__init__()
        self.alpha_list = nn.ParameterList([nn.Parameter(torch.Tensor([1.0])) for _ in range(u_num)])  # alpha_list is a list of learnable parameters
        self.f = torch.nn.Sequential(
            nn.Linear(d_num, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, d_num),
        )
    def forward(self, u):
        batch_size, port_num, d_num = u.shape
        u = u.view(-1, d_num)
        u_f = self.f(u)
        u_f = u_f.view(batch_size, port_num, d_num)
        u_final = torch.zeros_like(u_f)
        for i in range(len(self.alpha_list)):
            u_final[:,i,:] = u_f[:,i,:] * self.alpha_list[i]
        return u_final
    
class waveform_data(Dataset):
    def __init__(self, u):
        self.u = u

    def __len__(self):
        return len(self.u)
    
    def __getitem__(self, idx):
        return self.u[idx], idx

def train_port_module(port_module, dataset, u_target, batch_size, lr_init = 0.01, epoch = 100):
    optimizer = torch.optim.Adam(port_module.parameters(), lr = lr_init)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    criterion = nn.MSELoss() 
    loss_list = []
    for i in range(epoch):
        total_loss = 0
        for _, u in tqdm(enumerate(dataloader)):
            u_in = u[0]
            idx = u[1]
            u_pred = port_module(u_in)
            u_tar = u_target[idx]
            loss = criterion(u_pred,u_tar)
            # loss = torch.norm(u_pred - u_tar)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {i}, loss {total_loss}")   
        logging.info(f"Epoch {i}, loss {total_loss}")
        loss_list.append(total_loss)
    return loss_list
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="An example program with command-line arguments")
    parser.add_argument("--port_num", type=int, default= 10)
    parser.add_argument("--lr", type= float, default= 0.001)
    parser.add_argument("--epoch", type= int, default= 100)
    parser.add_argument("--bs", type= int, default= 1)
    parser.add_argument("--exp_marker", type= str, default= "Exp_port10_1per")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Exp_marker = args.exp_marker
    file_path = os.path.join(sys.path[0], 'Experiment', Exp_marker)
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    logging.basicConfig(filename = file_path + '/train.log', level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(args)

    
    data = spio.loadmat("./IBM_transient/ibmpg1t.mat")
    E, A, B = data['E'] * 1e-0, data['A'], data['B']
    
    port_num = args.port_num
    hidden_size = 256
    B = B[:, 0:port_num]
    port_model = port_module(port_num, hidden_size).to(device)
    
    u = np.load(f'./train_data/Uin_{port_num}port_1per.npy'.format(port_num))
    u = torch.tensor(u, dtype = torch.float32).to(device)
    # normalizer = MaxMinNormalization()
    # u = normalizer(u)
    # u = torch.log10(u)
    waveform_dataset = waveform_data(u[:100])
    u_target = u
    
    logging.info("Start training")
    loss = train_port_module(port_model, waveform_dataset, u_target,batch_size=args.bs,
                              lr_init=args.lr, epoch=args.epoch)
    step = range(args.epoch)
    plt.plot(step, torch.tensor(loss).detach(), label='Loss', color = '#FF4500')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(file_path+'/'+f'loss_{port_num}port.png'.format(port_num))
    torch.save(port_model.state_dict(), file_path + '/'+ f'port_model_{port_num}.pth'.format(port_num))
    logging.info(torch.tensor(port_model.alpha_list))
    logging.info("Training finished")

    