import torch.nn as nn
import torch
from utils.GAR import *   
class gar(nn.Module):

    def __init__(self,data_shape_list):
        super().__init__()
        self.hogp_list = []
        self.hogp_list.append(HOGP_simple(kernel=[kernel.SquaredExponentialKernel() for _ in range(len(data_shape_list[0])+1)], noise_variance=1.0, output_shape=data_shape_list[0], learnable_grid=False, learnable_map=False))
        self.hogp_list = torch.nn.ModuleList(self.hogp_list)

        self.Tensor_linear_list = []
        self.Tensor_linear_list.append(Tensor_linear(data_shape_list[0], data_shape_list[0 + 1]))
        self.Tensor_linear_list = torch.nn.ModuleList(self.Tensor_linear_list)

    def forward(self, data_manager, x_test, y_low):

        x_train, _ = data_manager.get_data(1, normal=False)
        mean_res, _ = self.hogp_list[0].forward(x_train, x_test)
        mean_high = self.Tensor_linear_list[0](y_low) + mean_res

        return mean_high
        
def train_gar(GARmodel, data_manager, max_iter=1000, lr_init=1e-1, normal = False):

    optimizer = torch.optim.Adam(GARmodel.parameters(), lr=lr_init)
    x_low, y_low = data_manager.get_data(0, normal=False)
    x_high, y_high = data_manager.get_data(1, normal=False)
    for i in range(max_iter):
        optimizer.zero_grad()
        y_residual = y_high - GARmodel.Tensor_linear_list[0](y_low)
        loss = GARmodel.hogp_list[0].log_likelihood(x_high, y_residual)
        loss.backward()
        optimizer.step()
        print('epoch {}/{}, nll: {}'.format(i+1, max_iter, loss.item()), end='\r')
    print('')


    