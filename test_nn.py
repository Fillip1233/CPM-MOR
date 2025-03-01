import torch
import numpy as np
import scipy.io as spio
from torch.utils.data import DataLoader
from tqdm import tqdm
from nn_forB import port_module, waveform_data
import matplotlib.pyplot as plt
from utils.normalization import MaxMinNormalization

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()

def test_port_module(port_model, dataset, u_target, normalizer):
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)
    total_loss = 0
    with torch.no_grad():
        for uid, u in tqdm(enumerate(dataloader), desc="Testing"):
            u1 = u[0]
            u_pred = port_model(u1)
            u_pred = normalizer.denormalize(u_pred)
            u_tar = u_target[uid]
            loss = torch.norm(u_pred - u_tar)
            total_loss += loss
    print(f"Total loss: {total_loss}")

def draw_result(port_model, dataset, u_target, normalizer):
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)
    with torch.no_grad():
        for i, u in tqdm(enumerate(dataloader), desc="Testing"):
            u1 = u[0]
            idx = u[1]
            u_pred = port_model(u1)
            # u_pred = normalizer.denormalize(u_pred)
            # u_pred = 10**u_pred
            u_tar = u_target[idx]
            time = np.linspace(0, 2e-9, 200)
            fig, ax = plt.subplots(2,5,figsize=(30,10))
            for i in range(10):
                ax[i//5,i%5].plot(time, u_pred[0][i], label='Predicted',color='#BA55D3')
                ax[i//5,i%5].plot(time, u_tar[0][i], label='Target',color = '#FF4500')
                ax[i//5,i%5].legend()
            fig.suptitle('Port model prediction')
            plt.show()

if __name__ == '__main__':
    data = spio.loadmat("./IBM_transient/ibmpg1t.mat")
    E, A, B = data['E'] * 1e-0, data['A'], data['B']

    port_num = 10
    hidden_size = 256
    B = B[:, 0:port_num]
    
    port_model = port_module(port_num, hidden_size)
    load_model(port_model, './Experiment/Exp_port10_1per/port_model_10.pth')
    
    u = np.load('./train_data/Uin_10port_1per.npy')
    u = torch.tensor(u, dtype=torch.float32)
    u_in = u[200:300]
    u_target = u_in

    # normalizer = MaxMinNormalization()
    # u = normalizer(u)
    # u = torch.log10(u)
    waveform_dataset = waveform_data(u_in)
    
    # test_port_module(port_model, waveform_dataset, u_target, normalizer)
    draw_result(port_model, waveform_dataset, u_target, normalizer = None)