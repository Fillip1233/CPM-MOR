import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils.load_data import *
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test MSIP_BDSM_MF-MOR")
    parser.add_argument("--cir", type= int, default= 1)
    args = parser.parse_args()
    data_path = os.path.join(f'./MSIP_BDSM/train_data/{args.cir}t_2per/')
    x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time1, pr = prepare_data(data_path, train_data_num=30, prima=False)
    dt = 1e-11
    fs = 1 / dt
    
    yl = y_l[0, 0, :].numpy()
    yh = y_h[0, 0, :].numpy()

    # ===== FFT =====
    N = len(yl)
    Y1 = np.fft.fft(yl)
    Y2 = np.fft.fft(yh)
    freq = np.fft.fftfreq(N, d=dt)

    # 单边频谱
    half = N // 2
    freq_half = freq[:half]
    amp_half1 = np.abs(Y1[:half]) / N * 2
    amp_half2 = np.abs(Y2[:half]) / N * 2
    # ===== 选频段（非常关键）=====
    fmax = 4e9   # 根据你的电路自己改
    mask = (freq_half >= 0) & (freq_half <= fmax)

    freq_plot = freq_half[mask]
    amp_plot1 = amp_half1[mask]
    amp_plot2 = amp_half2[mask]

    # ===== 柱状图 =====
    df = freq_plot[1] - freq_plot[0]

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    
    axes[0].bar(freq_plot, amp_plot1, width=df)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("low_f Frequency Spectrum")
    axes[1].bar(freq_plot, amp_plot2, width=df)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("high_f Frequency Spectrum")
    plt.savefig(f'freq_spectrum_cir{args.cir}_0.png', dpi=300)
    plt.close()
