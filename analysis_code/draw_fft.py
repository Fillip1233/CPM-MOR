import torch
import argparse
import os
import matplotlib.pyplot as plt
from utils.load_data import *

def plot_fft_comparison(y_l, y_h, sample_idx=0, channel_idx=0, fs=1.0, save_path='fft_comparison.png'):
    """
    绘制时域和频域对比图
    
    Parameters:
    - y_l, y_h: 输入数据 [batch_size, channels, time_steps]
    - sample_idx: 要可视化的样本索引
    - channel_idx: 要可视化的通道索引  
    - fs: 采样频率（用于正确的频率轴）
    """
    
    y_l_single = y_l[sample_idx, channel_idx, :]
    y_h_single = y_h[sample_idx, channel_idx, :]
    
    y_l_fft = torch.fft.rfft(y_l_single)
    y_h_fft = torch.fft.rfft(y_h_single)
    
    n_l = len(y_l_single)
    n_h = len(y_h_single)
    freqs_l = torch.fft.rfftfreq(n_l, d=1/fs)
    freqs_h = torch.fft.rfftfreq(n_h, d=1/fs)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(y_l_single.detach().numpy(), 'b-', alpha=0.7, label='Low-fidelity')
    axes[0, 0].set_title('Time Domain - Low-fidelity')
    axes[0, 0].set_xlabel('Time Sample')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    axes[0, 1].plot(y_h_single.detach().numpy(), 'r-', alpha=0.7, label='High-fidelity')
    axes[0, 1].set_title('Time Domain - High-fidelity')
    axes[0, 1].set_xlabel('Time Sample')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # 频域图（幅度谱）
    axes[1, 0].plot(freqs_l, torch.abs(y_l_fft).detach().numpy(), 'b-', alpha=0.7, label='Low-fidelity')
    axes[1, 0].set_title('Frequency Domain - Magnitude')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    axes[1, 1].plot(freqs_h, torch.abs(y_h_fft).detach().numpy(), 'r-', alpha=0.7, label='High-fidelity')
    axes[1, 1].set_title('Frequency Domain - Magnitude')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fft drawing")
    parser.add_argument("--cir", type=int, default=1)
    args = parser.parse_args()
    
    data_path = os.path.join(f'./MSIP_BDSM/train_data/{args.cir}t_2per/')
    x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time1, pr = prepare_data(
        data_path, train_data_num=300, prima=False
    )
    
    plot_fft_comparison(
        y_l, y_h, 
        sample_idx=0, 
        channel_idx=0,
        fs=1e10,  # 根据你的实际采样频率调整
        save_path=f'fft_detailed_comparison_{args.cir}t.png'
    )
    