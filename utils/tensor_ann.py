'''
single ann with tensor linear layer
modify: 2025/4/1: find tensor linear layer is useful
2025/12/25: add fno version
'''
import torch.nn as nn
import torch
from utils.GAR import *
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from neuralop.models import FNO
import torch.fft


class tensor_fno_mask(nn.Module):
    def __init__(self, data_shape_list, mask=None, d_num=100,
                 fno_width=32, fno_modes=16):
        super().__init__()

        self.f = FNO(n_modes=(64,),
               hidden_channels=64,
               in_channels=10000,
               out_channels=10000)

        self.Tensor_linear_list = []
        for i in range(len(data_shape_list) - 1):
            self.Tensor_linear_list.append(
                Tensor_linear_mask(
                    data_shape_list[i],
                    data_shape_list[i + 1],
                    mask=mask
                )
            )
        self.Tensor_linear_list = nn.ModuleList(self.Tensor_linear_list)

    def forward(self, u):
        # u: (batch, port, d_num)
        # batch_size, port_num, d_num = u.shape

        # u = u.view(batch_size*port_num , 1 , d_num)      # (batch*port, d_num)
        u_f = self.f(u)
        # u_f = u_f.view(batch_size, port_num, d_num)

        return u_f

    def forward_h(self, u, y_low):
        y_l_after = self.Tensor_linear_list[0](y_low)
        res = self(u)
        return res + y_l_after
    
    def forward_t(self, u, y_low):
        y_l_after = self.Tensor_linear_list[0](y_low)
        return y_l_after

class tensor_ann_mask(nn.Module):
    def __init__(self, data_shape_list, hidden_size, mask = None, d_num = 100):
        super(tensor_ann_mask, self).__init__()
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
            self.Tensor_linear_list.append(Tensor_linear_mask(data_shape_list[i], data_shape_list[i + 1], mask=mask))
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
    
    def forward_t(self, u, y_low):
        y_l_after = self.Tensor_linear_list[0](y_low)
        return y_l_after
    
    def draw(self, y_low):
        y_l_after1  = tensorly.tenalg.mode_dot(y_low, self.Tensor_linear_list[0].vectors[0], 1)
        y_l_after2  = tensorly.tenalg.mode_dot(y_l_after1, self.Tensor_linear_list[0].vectors[1], 2)
        return y_l_after1, y_l_after2
    
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

def train_tensor_ann_fft(tensor_ann, data_manager, lr, epoch, batch_size=64, alpha=1.0, beta=0.0, normal=False):
    
    writer = SummaryWriter(log_dir='./runs/tensor_ann_fft')

    optimizer = torch.optim.Adam(tensor_ann.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_h, y_h = data_manager.get_data(1, normal=normal)
    _, y_l = data_manager.get_data(0, normal=normal)

    dataset = TensorDataset(x_h, y_l, y_h)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for ep in range(epoch):
        epoch_loss, epoch_time_loss, epoch_freq_loss = 0, 0, 0

        for batch_x, batch_y_l, batch_y_h in dataloader:
            optimizer.zero_grad()
            pred = tensor_ann(batch_x) + tensor_ann.Tensor_linear_list[0](batch_y_l)

            loss_time = criterion(pred, batch_y_h)

            U_pred = torch.fft.rfft(pred, dim=-1)
            Y_res = torch.fft.rfft(batch_y_h, dim=-1)
            loss_freq = criterion(torch.abs(U_pred), torch.abs(Y_res))

            loss = alpha * loss_time + beta * loss_freq
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_time_loss += loss_time.item()
            epoch_freq_loss += loss_freq.item()

        n_batches = len(dataloader)
        print(f"Epoch {ep:03d} | Loss={epoch_loss/n_batches:.6f} | "
              f"Time={epoch_time_loss/n_batches:.6f} | Freq={epoch_freq_loss/n_batches:.6f}")

        writer.add_scalar('Loss/total', epoch_loss/n_batches, ep)
        writer.add_scalar('Loss/time', epoch_time_loss/n_batches, ep)
        writer.add_scalar('Loss/freq', epoch_freq_loss/n_batches, ep)

    writer.close()
    print("finish training")


def train_fft_com(tensor_ann, data_manager, lr, epoch, batch_size=64, alpha=1.0, beta=0.0, normal=False):
    # writer = SummaryWriter(log_dir='./runs/fft_com')

    optimizer = torch.optim.Adam(tensor_ann.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_h, y_h = data_manager.get_data(1, normal=normal)
    _, y_l = data_manager.get_data(0, normal=normal)

    dataset = TensorDataset(x_h, y_l, y_h)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for ep in range(epoch):
        epoch_loss, epoch_time_loss, epoch_freq_loss = 0, 0, 0

        for batch_x, batch_y_l, batch_y_h in dataloader:
            optimizer.zero_grad()
            pred = tensor_ann(batch_x) + tensor_ann.Tensor_linear_list[0](batch_y_l)
            # pred = tensor_ann(tensor_ann.Tensor_linear_list[0](batch_y_l))

            loss_time = criterion(pred, batch_y_h)

            # U_pred = torch.fft.rfft(pred, dim=-1)
            # Y_res = torch.fft.rfft(batch_y_h, dim=-1)
            # loss_freq = criterion(torch.abs(U_pred), torch.abs(Y_res))

            # loss = alpha * loss_time + beta * loss_freq
            loss = loss_time
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_time_loss += loss_time.item()
            # epoch_freq_loss += loss_freq.item()

        n_batches = len(dataloader)
        # print(f"Epoch {ep:03d} | Loss={epoch_loss/n_batches:.6f} | "
        #       f"Time={epoch_time_loss/n_batches:.6f} | Freq={epoch_freq_loss/n_batches:.6f}")
        print(f"Epoch {ep:03d} | Loss={epoch_loss/n_batches:.6f} | "
              f"Time={epoch_time_loss/n_batches:.6f}")

    #     writer.add_scalar('Loss/total', epoch_loss/n_batches, ep)
    #     writer.add_scalar('Loss/time', epoch_time_loss/n_batches, ep)
    #     writer.add_scalar('Loss/freq', epoch_freq_loss/n_batches, ep)

    # writer.close()
    print("finish training")
