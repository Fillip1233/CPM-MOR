'''
2025/11
draw paper fig for y_l y_h
'''
import argparse
import logging
import os
import sys
import torch
import matplotlib.pyplot as plt
import pandas as pd
import time
from utils.load_data import *
from utils.calculate_metrix import calculate_metrix
from utils.tensor_ann import *
from utils.alpha_ann import *
from utils.res_ann import *
from utils.tensor_rnn import *
from utils.tensor_lstm import *
from utils.single_gar import *
from utils.MF_data import *
import matplotlib.ticker as ticker
from scipy.io import savemat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An example program with command-line arguments")
    parser.add_argument("--lr", type= float, default= 1e-2)
    parser.add_argument("--epoch", type= int, default= 100)
    parser.add_argument("--bs", type= int, default= 32)
    parser.add_argument("--hidden_size", type= int, default= 128)
    parser.add_argument("--draw_type", type= int, default= 2)
    parser.add_argument("--module_name", type= str, default= "tensor_ann_mask")
    parser.add_argument("--test_over", type= int, default= 0)
    parser.add_argument("--cir", type= int, default= 6)
    parser.add_argument("--alpha", type= float, default= 1.0)
    parser.add_argument("--beta", type= float, default= 0.0)
    parser.add_argument("--topk", type= int, default= 3)
    parser.add_argument("--exp_marker", type= str, default= "top")

    args = parser.parse_args()
    save_path = f'./DAC/{args.cir}t_2per/'+args.exp_marker
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),logging.FileHandler(f"{save_path}/data_generate.log")])
    logging.info(args)
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(f'./MSIP_BDSM/train_data/{args.cir}t_2per/')
    x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time1, pr = prepare_data(data_path, train_data_num=300, prima=False)

    y_l = y_h[99,:,:]
   # 对于这种细长形状的数据，也可以考虑热图
    plt.figure(figsize=(12, 8))
    plt.imshow(y_l.cpu(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('y_l - Heatmap')
    plt.xlabel('Dimension 2 (200)')
    plt.ylabel('Dimension 1 (10000)')
    plt.tight_layout()
    plt.savefig('y_l_heatmap1.pdf', bbox_inches='tight', dpi=300)
    # plt.show()

