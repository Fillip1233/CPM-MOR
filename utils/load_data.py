import torch
import numpy as np

def prepare_data_mix(data_path, train_data_num1=30, train_data_num2=70, prima = False):

    x1 = np.load(data_path+'/mf_inall.npy')
    x2 = np.load(data_path+'_diff'+'/mf_inall.npy')
    x1 = torch.tensor(x1, dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)
    x_trainl = torch.cat((x1[:train_data_num1,:], x2[:train_data_num2,:]), dim = 0)
    x_trainh = x_trainl
    yl1= np.load(data_path+'/mf_low_f.npy')
    yl2= np.load(data_path+'_diff'+'/mf_low_f.npy')
    yl1 = torch.tensor(yl1, dtype=torch.float32)
    yl2 = torch.tensor(yl2, dtype=torch.float32)
    y_l = torch.cat((yl1[:train_data_num1,:], yl2[:train_data_num2,:]), dim = 0)

    yh1 = np.load(data_path+'/mf_high_f.npy')
    yh1 = torch.tensor(yh1, dtype=torch.float32)
    yh2 = np.load(data_path+'_diff'+'/mf_high_f.npy')
    yh2 = torch.tensor(yh2, dtype=torch.float32)
    y_h = torch.cat((yh1[:train_data_num1, :], yh2[:train_data_num2,:]), dim = 0)

    if prima:
        pr = np.load(data_path+'_diff'+'/prima.npy')
        pr = torch.tensor(pr, dtype=torch.float32)
    else:
        pr = None

    time = np.load(data_path+'/mf_time.npy')

    x_test = torch.cat((x1[train_data_num1:, :], x2[train_data_num2:, :]), dim = 0)
    y_test = torch.cat((yh1[train_data_num1:, :], yh2[train_data_num2:, :]), dim = 0)
    yl_test = torch.cat((yl1[train_data_num1:, :], yl2[train_data_num2:, :]), dim = 0)

    return x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time, pr

def prepare_data(data_path, train_data_num=100, prima = False):

    x1 = np.load(data_path+'/mf_inall.npy')
    x = torch.tensor(x1, dtype=torch.float32)
   
    yl1= np.load(data_path+'/mf_low_f.npy')
    yl = torch.tensor(yl1, dtype=torch.float32)

    yh1 = np.load(data_path+'/mf_high_f.npy')
    yh = torch.tensor(yh1, dtype=torch.float32)

    time = np.load(data_path+'/mf_time.npy')

    if prima:
        pr = np.load(data_path+'/prima.npy')
        pr = torch.tensor(pr, dtype=torch.float32)

    x_trainl = x[:train_data_num, :]
    x_trainh = x[:train_data_num, :]
    y_l = yl[:train_data_num, :]
    y_h = yh[:train_data_num, :]

    x_test = x[train_data_num:, :]
    y_test = yh[train_data_num:, :]
    yl_test = yl[train_data_num:, :]

    if not prima:
        pr = None
    
    return x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time, pr

def load_over_data(data_path):
    x1 = np.load(data_path+'/mf_inall.npy')
    x = torch.tensor(x1, dtype=torch.float32)
   
    yl1= np.load(data_path+'/mf_low_f.npy')
    yl = torch.tensor(yl1, dtype=torch.float32)

    yh1 = np.load(data_path+'/mf_high_f.npy')
    yh = torch.tensor(yh1, dtype=torch.float32)

    time = np.load(data_path+'/mf_time.npy')
    pr = np.load(data_path+'/prima.npy')
    pr = torch.tensor(pr, dtype=torch.float32)

    x_test = x
    y_test = yh
    yl_test = yl

    return x_test, y_test, yl_test, time, pr