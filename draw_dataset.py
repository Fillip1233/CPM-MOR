import os
import sys
import matplotlib.pyplot as plt
from utils.load_data import *
if __name__ == "__main__":
    data_path = os.path.join(sys.path[0], 'train_data/1t/sim_100_port2000_multiper_diff')
    x_trainl, x_trainh, y_l, y_h, x_test, y_test, yl_test, time1, pr = prepare_data(data_path,prima=False)
    yte = y_l
    for i in range(100):
        fig, axs = plt.subplots(1, 2, figsize=(10, 20))

        # cbar_ax1 = fig.add_axes([0.02, 0.3, 0.03, 0.4])
        cbar_ax2 = fig.add_axes([0.94, 0.3, 0.03, 0.4])
        vmin = torch.min(yte[i])
        vmax = torch.max(yte[i])

        im1 = axs[0].imshow(yte[i].cpu(), cmap='viridis', interpolation='nearest', vmin = vmin, vmax = vmax, aspect='auto')
        axs[0].set_title('y_l')

        axs[1].imshow(y_h[i].cpu(), cmap='viridis', interpolation ='nearest', vmin = vmin, vmax = vmax, aspect='auto')
        axs[1].set_title('y_h')

        # cbar1 = fig.colorbar(im1, cax=cbar_ax1)
        cbar2 = fig.colorbar(im1, cax=cbar_ax2,location='right', fraction=0.02, pad=0.02)
        # plt.tight_layout()
        plt.show()
        plt.clf()
    # for i in range(100):
    #         y_te = yte[i]
    #         yy_te = torch.zeros((y_te.shape[1]))
    #         for k in range(y_te.shape[0]):
    #             yy_te += y_te[k, :]
    #         y_low = yl_test[i]
    #         yy_low = torch.zeros((y_low.shape[1]))
    #         for e in range(y_low.shape[0]):
    #             yy_low += y_low[e, :]
    #         plt.plot(time1, yy_low.cpu(), color='#5F8B4C', linestyle='-.', marker='*', label='SVD-PRIMA low-fidelity result', markevery = 20, markersize=7, linewidth=2.0)
    #         plt.plot(time1, yy_te.cpu(), color='#945034', linestyle='-.', marker='*', label='Original simulation results', markevery = 15, markersize=7, linewidth=2.0)
    #         plt.legend(fontsize=12)
    #         plt.title("MF-result", fontsize=14)
    #         plt.xlabel("Time (s)", fontsize=12)
    #         plt.ylabel("result", fontsize=12)
    #         plt.grid()
    #         plt.tight_layout()
    #         # plt.show()
    #         plt.savefig('MF_data_example.png', dpi=300, bbox_inches='tight')
    #         plt.clf()