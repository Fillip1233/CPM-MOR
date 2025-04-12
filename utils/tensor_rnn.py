import torch
import torch.nn as nn
from utils.GAR import *

class tensor_rnn(nn.Module):
    def __init__(self, data_shape_list, hidden_size, d_num=100, num_layers=2):
        super(tensor_rnn, self).__init__()
        
        self.rnn = nn.GRU(
            input_size=d_num,         
            hidden_size=hidden_size,  
            num_layers=num_layers,    
            batch_first=True 
        )
        
        self.output_layer = nn.Linear(hidden_size, d_num)

        self.Tensor_linear_list = []
        for i in range(len(data_shape_list) - 1):
            self.Tensor_linear_list.append(Tensor_linear(data_shape_list[i], data_shape_list[i + 1]))
        self.Tensor_linear_list = torch.nn.ModuleList(self.Tensor_linear_list)
    
    def forward(self, u):
        # u: (batch_size, port_num, d_num)
        u = u.permute(0, 2, 1)
        rnn_out, _ = self.rnn(u)  # rnn_out: (batch_size, port_num, hidden_size)
        out = self.output_layer(rnn_out)
        out = out.permute(0, 2, 1) 
        return out

    def forward_h(self, u, y_low):
        y_l_after = self.Tensor_linear_list[0](y_low)
        res = self(u)
        return res + y_l_after

def train_tensor_rnn(tensor_rnn ,data_manager, lr, epoch, normal = False):
    
    optimizer = torch.optim.Adam(tensor_rnn.parameters(), lr = lr)
    criterion = nn.MSELoss()
    _, y_l = data_manager.get_data(0, normal = normal)
    x_h, y_h = data_manager.get_data(1, normal = normal)
    for i in range(epoch):
    
        optimizer.zero_grad()
        y_res = y_h - tensor_rnn.Tensor_linear_list[0](y_l)
        u_pred = tensor_rnn(x_h)
        loss = criterion(u_pred, y_res)
        
        loss.backward()
        optimizer.step()
        print(f"Epoch {i}, loss {loss}", end='\r')   
    print(' ')