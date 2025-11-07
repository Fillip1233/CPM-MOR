'''
single ann with tensor linear layer
modify: 2025/4/1: find tensor linear layer is useful
'''
import torch.nn as nn
import torch
from utils.GAR import *
from torch.utils.tensorboard import SummaryWriter

class tensor_ann(nn.Module):
    def __init__(self, data_shape_list, hidden_size, d_num = 100):
        super(tensor_ann, self).__init__()
        self.f = torch.nn.Sequential(
            nn.Linear(d_num, hidden_size),
            nn.SiLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.SiLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.SiLU(),
            nn.Linear(hidden_size, d_num),
        )
        
        self.Tensor_linear_list = []
        for i in range(len(data_shape_list) - 1):
            self.Tensor_linear_list.append(Tensor_linear(data_shape_list[i], data_shape_list[i + 1]))
        self.Tensor_linear_list = torch.nn.ModuleList(self.Tensor_linear_list)
    
    def forward(self, u):
        # u = u.permute(0, 2, 1)
        # batch_size, d_num, port_num = u.shape
        # u = u.reshape(-1, port_num)
        # u_f = self.f(u)
        # u_f = u_f.reshape(batch_size, d_num,  port_num)
        # u_f = u_f.permute(0, 2, 1)
        
        batch_size, port_num, d_num = u.shape
        u = u.view(-1, d_num)
        # u = u.view(batch_size, port_num*d_num)
        u_f = self.f(u)
        u_f = u_f.view(batch_size, port_num, d_num)

        return u_f
    def forward_h(self, u, y_low):
        y_l_after = self.Tensor_linear_list[0](y_low)
        res = self(u)
        return res + y_l_after
        # return y_l_after 
    
    def draw(self, y_low):
        y_l_after1  = tensorly.tenalg.mode_dot(y_low, self.Tensor_linear_list[0].vectors[0], 1)
        y_l_after2  = tensorly.tenalg.mode_dot(y_l_after1, self.Tensor_linear_list[0].vectors[1], 2)
        return y_l_after1, y_l_after2

def train_tensor_ann(tensor_ann ,data_manager, lr, epoch, normal = False):
    
    optimizer = torch.optim.Adam(tensor_ann.parameters(), lr = lr)
    criterion = nn.MSELoss()
    _, y_l = data_manager.get_data(0, normal = normal)
    x_h, y_h = data_manager.get_data(1, normal = normal)
    for i in range(epoch):
        optimizer.zero_grad()
        y_res = y_h - tensor_ann.Tensor_linear_list[0](y_l)
        u_pred = tensor_ann(x_h)
        loss = criterion(u_pred, y_res)

        # y_l_after = tensor_ann.Tensor_linear_list[0](y_l)
        # loss = criterion(y_h, y_l_after)
        
        loss.backward()
        optimizer.step()
        print(f"Epoch {i}, loss {loss}", end='\r')   
    print(' ')

def train_tensor_ann_fft(tensor_ann ,data_manager, lr, epoch, alpha=1.0, beta=1.0, normal = False):
    
    writer = SummaryWriter(log_dir='./runs/tensor_ann_fft')
    optimizer = torch.optim.Adam(tensor_ann.parameters(), lr = lr)
    criterion = nn.MSELoss()
    _, y_l = data_manager.get_data(0, normal = normal)
    x_h, y_h = data_manager.get_data(1, normal = normal)
    for i in range(epoch):
        optimizer.zero_grad()
        y_res = y_h - tensor_ann.Tensor_linear_list[0](y_l)
        u_pred = tensor_ann(x_h)
        loss_time = criterion(u_pred, y_res)

        U_pred = torch.fft.rfft(u_pred, dim=-1)
        Y_res = torch.fft.rfft(y_res, dim=-1)

        loss_freq = criterion(torch.abs(U_pred), torch.abs(Y_res))
        loss = alpha * loss_time + beta * loss_freq
        
        loss.backward()
        optimizer.step()
        print(f"Epoch {i}, loss={loss.item():.6f}, time={loss_time.item():.6f}, freq={loss_freq.item():.6f}", end='\r')
        writer.add_scalar('Loss/total', loss.item(), i)
        writer.add_scalar('Loss/time', loss_time.item(), i)
        writer.add_scalar('Loss/freq', loss_freq.item(), i)
          
    print(' ')