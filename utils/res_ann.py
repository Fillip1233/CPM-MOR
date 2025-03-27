import torch.nn as nn
import torch
# from utils.GAR import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
class res_ann(nn.Module):
    def __init__(self, data_shape_list, hidden_size, d_num = 100):
        super(res_ann, self).__init__()
        self.f = torch.nn.Sequential(
            nn.Linear(d_num, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, d_num),
        )
        
        # self.Tensor_linear_list = []
        # for i in range(len(data_shape_list) - 1):
        #     self.Tensor_linear_list.append(Tensor_linear(data_shape_list[i], data_shape_list[i + 1]))
        # self.Tensor_linear_list = torch.nn.ModuleList(self.Tensor_linear_list)
    def forward(self, u):
        batch_size, port_num, d_num = u.shape
        u = u.view(-1, d_num)
        u_f = self.f(u)
        u_f = u_f.view(batch_size, port_num, d_num)
        return u_f
    def forward_h(self, u, y_low):
        res = self(u)
        return res + y_low

def train_res_ann(res_ann , dataset, batch_size, lr, epoch):
    optimizer = torch.optim.Adam(res_ann.parameters(), lr = lr)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    criterion = nn.MSELoss()
    loss_list = [] 
    for i in range(epoch):
        total_loss = 0
        for _, u in enumerate(dataloader):
            u_in = u[0]
            y = u[1]
            u_pred = res_ann(u_in)
            loss = criterion(u_pred, y)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {i}, loss {total_loss}")   
        loss_list.append(total_loss)
    return loss_list