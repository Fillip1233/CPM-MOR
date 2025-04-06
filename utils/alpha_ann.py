'''
2025/4/2
alpha ann
'''
import torch.nn as nn
import torch
from utils.GAR import *
class alpha_ann(nn.Module):
    def __init__(self, hidden_size, d_num = 100):
        super(alpha_ann, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([1.0]))
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
        return u_f
    def forward_h(self, u, y_low):
        y_l_after = self.alpha * y_low
        res = self(u)
        return res + y_l_after

def train_alpha_ann(alpha_ann ,data_manager, lr, epoch, normal = False):
    
    optimizer = torch.optim.Adam(alpha_ann.parameters(), lr = lr)
    criterion = nn.MSELoss()
    _, y_l = data_manager.get_data(0, normal = normal)
    x_h, y_h = data_manager.get_data(1, normal = normal)
    for i in range(epoch):
    
        optimizer.zero_grad()
        y_res = y_h - alpha_ann.alpha * y_l
        u_pred = alpha_ann(x_h)
        loss = criterion(u_pred, y_res)
        
        loss.backward()
        optimizer.step()
        print(f"Epoch {i}, loss {loss}", end='\r')   
    print(' ')