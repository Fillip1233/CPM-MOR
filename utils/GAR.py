import torch
import numpy as np
from utils.hogp_simple import HOGP_simple
import utils.kernel as kernel
from utils.MF_data import MultiFidelityDataManager
import matplotlib.pyplot as plt
import tensorly
tensorly.set_backend('pytorch')

class Tensor_linear(torch.nn.Module):
    def __init__(self,l_shape,h_shape):
        super().__init__()
        self.l_shape=l_shape
        self.h_shape=h_shape
        self.vectors = []
        for i in range(len(self.l_shape)):
            if self.l_shape[i] < self.h_shape[i]:
                init_tensor = torch.eye(self.l_shape[i])
                init_tensor = torch.nn.functional.interpolate(init_tensor.reshape(1, 1, *init_tensor.shape), 
                                                            (self.l_shape[i],self.h_shape[i]), mode='bilinear')
                init_tensor = init_tensor.squeeze().T
            elif self.l_shape[i] == self.h_shape[i]:
                init_tensor = torch.eye(self.l_shape[i])
            self.vectors.append(torch.nn.Parameter(init_tensor))
        self.vectors = torch.nn.ParameterList(self.vectors)

    def forward(self,x):
        for i in range(len(self.l_shape)):
            x = tensorly.tenalg.mode_dot(x, self.vectors[i], i+1)
        return x
        
class GAR(torch.nn.Module):
    """
    GeneralizedAutoAR (GAR) model.

    Args:
        fidelity_num (int): The number of fidelity levels.
        kernel_list (list): List of kernel values for each fidelity level.
        data_shape_list (list): List of data shapes for each fidelity level.
        if_nonsubset (bool, optional): Flag indicating if non-subset data is used. Defaults to False.
    """

    def __init__(self, fidelity_num, data_shape_list, if_nonsubset=False):
        super().__init__()
        self.fidelity_num = fidelity_num
        self.hogp_list = []
        for i in range(self.fidelity_num):
            self.hogp_list.append(HOGP_simple(kernel=[kernel.SquaredExponentialKernel() for _ in range(len(data_shape_list[i])+1)], noise_variance=1.0, output_shape=data_shape_list[i], learnable_grid=False, learnable_map=False))
        self.hogp_list = torch.nn.ModuleList(self.hogp_list)

        self.Tensor_linear_list = []
        for i in range(self.fidelity_num - 1):
            self.Tensor_linear_list.append(Tensor_linear(data_shape_list[i], data_shape_list[i + 1]))
        self.Tensor_linear_list = torch.nn.ModuleList(self.Tensor_linear_list)

        self.if_nonsubset = if_nonsubset

    def forward(self, data_manager, x_test, to_fidelity=None, normal=False):
        """
        Forward pass of the GAR model.

        Args:
            data_manager: The data manager object.
            x_test: The test input data.
            to_fidelity (int, optional): The fidelity level to compute. Defaults to None.

        Returns:
            tuple: A tuple containing the mean and variance of the output.
        """
        if to_fidelity is not None:
            fidelity_level = to_fidelity
        else:
            fidelity_level = self.fidelity_num - 1

        for i_fidelity in range(fidelity_level + 1):
            if i_fidelity == 0:
                x_train, _ = data_manager.get_data(i_fidelity, normal=normal)
                mean_low, var_low = self.hogp_list[i_fidelity].forward(x_train, x_test)
                if fidelity_level == 0:
                    mean_high = mean_low
                    var_high = var_low
            else:
                x_train, _ = data_manager.get_data_by_name('res-{}'.format(i_fidelity))
                if x_train is None:
                    print("warning: can not find the res data")
                    x_train, _ = data_manager.get_data(i_fidelity, normal=normal)
                mean_res, var_res = self.hogp_list[i_fidelity].forward(x_train, x_test)

                mean_high = self.Tensor_linear_list[i_fidelity - 1](mean_low) + mean_res
                var_high = self.Tensor_linear_list[i_fidelity - 1](var_low) + var_res

                mean_low = mean_high
                var_low = var_high

        return mean_high, var_high
        
def train_GAR(GARmodel, data_manager, max_iter=1000, lr_init=1e-1, normal = False, debugger=None):
    """
    Trains the GARmodel using the specified data_manager.

    Args:
        GARmodel: The GAR model to be trained.
        data_manager: The data manager object that provides the training data.
        max_iter (optional): The maximum number of iterations for training. Default is 1000.
        lr_init (optional): The initial learning rate for the optimizer. Default is 0.1.
        debugger (optional): The debugger object for debugging purposes. Default is None.
    """

    for i_fidelity in range(GARmodel.fidelity_num):
        optimizer = torch.optim.Adam(GARmodel.parameters(), lr=lr_init)
        if i_fidelity == 0:
            x_low, y_low = data_manager.get_data(i_fidelity, normal=normal)
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = GARmodel.hogp_list[i_fidelity].log_likelihood(x_low, y_low)
                if debugger is not None:
                    debugger.get_status(GARmodel, optimizer, i, loss)
                loss.backward()
                optimizer.step()
                # print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
                print('fidelity {}, epoch {}/{}, nll: {}'.format(i_fidelity, i+1, max_iter, loss.item()), end='\r')
            print('')
        else:
            if GARmodel.if_nonsubset:
                with torch.no_grad():
                    subset_x, y_low, y_high = data_manager.get_nonsubset_fill_data(GARmodel, i_fidelity - 1, i_fidelity, normal=normal)
            else:
                _, y_low, subset_x, y_high = data_manager.get_overlap_input_data(i_fidelity - 1, i_fidelity, normal=normal)
            for i in range(max_iter):
                optimizer.zero_grad()
                if GARmodel.if_nonsubset:
                    y_residual_mean = y_high[0] - GARmodel.Tensor_linear_list[i_fidelity - 1](y_low[0])  # tensor linear layer
                    y_residual_var = abs(y_high[1] - y_low[1])
                else:
                    y_residual_mean = y_high - GARmodel.Tensor_linear_list[i_fidelity - 1](y_low)
                    y_residual_var = None

                if i == max_iter - 1:
                    if y_residual_var is not None:
                        y_residual_var = y_residual_var.detach()
                    data_manager.add_data(raw_fidelity_name='res-{}'.format(i_fidelity), fidelity_index=None, x=subset_x.detach(), y=[y_residual_mean.detach(), y_residual_var])
                loss = GARmodel.hogp_list[i_fidelity].log_likelihood(subset_x, [y_residual_mean, y_residual_var])
                if debugger is not None:
                    debugger.get_status(GARmodel, optimizer, i, loss)
                loss.backward()
                optimizer.step()
                # print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
                print('fidelity {}, epoch {}/{}, nll: {}'.format(i_fidelity, i+1, max_iter, loss.item()), end='\r')
            print('')


if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x1 = np.load('train_data\mf_inall.npy')
    x1 = torch.tensor(x1, dtype=torch.float32)
    x1 = torch.fft.fft(x1,dim = -1)
    x1 = torch.abs(x1)
    x = x1.reshape(x1.shape[0], -1)
    yl1= np.load('train_data\mf_low_f.npy')
    yl1 = torch.tensor(yl1, dtype=torch.float32)
    yl = torch.fft.fft(yl1,dim = -1)
    yl = torch.abs(yl)

    yh1 = np.load('train_data\mf_high_f.npy')
    yh1 = torch.tensor(yh1, dtype=torch.float32)
    yh = torch.fft.fft(yh1,dim = -1)
    yh = torch.abs(yh)

    time = np.load('train_data\mf_time.npy')


    x_train = x[:100, :]
    y_l = yl[:100, :]
    y_h = yh[:100, :]

    x_test = x[100:, :].to(device)
    y_test = yh[100:, :]
    yl_test = yl[100:, :]

    # data_shape = [y_l[0].shape, y_h[0].shape, y_h2[0].shape]
    data_shape = [y_l[0].shape, y_h[0].shape]

    initial_data = [
        {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_train.to(device), 'Y': y_l.to(device)},
        {'fidelity_indicator': 1,'raw_fidelity_name': '1', 'X': x_train.to(device), 'Y': y_h.to(device)},
        # {'fidelity_indicator': 2,'raw_fidelity_name': '2', 'X': x_train.to(device), 'Y': y_h2.to(device)}
    ]
    fidelity_num = len(initial_data)
    fidelity_manager = MultiFidelityDataManager(initial_data)

    myGAR = GAR(fidelity_num, data_shape, if_nonsubset = False).to(device)

    train_GAR(myGAR, fidelity_manager, max_iter = 200, lr_init = 1e-2, debugger = None)

    with torch.no_grad():
        x_test = fidelity_manager.normalizelayer[myGAR.fidelity_num-1].normalize_x(x_test)
        ypred, ypred_var = myGAR(fidelity_manager, x_test)
        ypred, ypred_var = fidelity_manager.normalizelayer[myGAR.fidelity_num-1].denormalize(ypred, ypred_var)

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
        yy_1 = torch.zeros((y_1.shape[1]))
        for j in range(y_1.shape[0]):
            yy_1 += y_1[j, :]
        y_te = yte[i]
        yy_te = torch.zeros((y_te.shape[1]))
        for k in range(y_te.shape[0]):
            yy_te += y_te[k, :]
        y_low = yl_test[i]
        yy_low = torch.zeros((y_low.shape[1]))
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
        
    plt.show()

    