import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw baseline")
    parser.add_argument("--draw_type", type= int, default= 0, help="0: normal comparison; 1: over comparison")
    args = parser.parse_args()
    draw_type = args.draw_type

    if draw_type == 0:
        time1 = np.load('./MSIP_BDSM/Exp_res/compare1/time.npy')
        mf_mor = np.load('./MSIP_BDSM/Exp_res/compare1/MF-MOR.npy')
        smf_mor = np.load('./MSIP_BDSM/Exp_res/compare1/SMF-MOR.npy')
        high = np.load('./MSIP_BDSM/Exp_res/compare1/high.npy')
        sip = np.load('./MSIP_BDSM/Exp_res/compare/sip.npy')
        svdmor = np.load('./MSIP_BDSM/Exp_res/compare1/svdmor.npy')
        demor = np.load('./MSIP_BDSM/Exp_res/compare1/DeMOR2.npy')

        plt.figure(figsize=(8,5))
        
        plt.plot(time1, high, color="#23AF1EFF", linestyle='--', marker='s', label='GT', markevery = 28, markersize=6, linewidth=1.5)
        plt.plot(time1, sip, color="#E20CAC", linestyle='--', marker='o', label='SIP', markevery = 24, markersize=6, linewidth=1.5)
        plt.plot(time1, svdmor, color="#FFA500", linestyle='--', marker='^', label='SVD-MOR', markevery = 30, markersize=6, linewidth=1.5)
        plt.plot(time1, demor[0], color="#129CAC", linestyle='--', marker='v', label='DeMOR', markevery = 34, markersize=6, linewidth=1.5)
        plt.plot(time1, mf_mor, color="#3F2ED1", linestyle='-.', marker='*', label='MF-MOR', markevery = 32, markersize=6, linewidth=1.5)
        plt.plot(time1, smf_mor, color="#DD123E", linestyle='-.', marker='*', label='SMF-MOR', markevery = 26, markersize=6, linewidth=2)
        
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Voltage (V)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('./MSIP_BDSM/Exp_res/compare1/baseline_comparison.pdf', dpi=300)
        plt.close()
        
    elif draw_type == 1:

        time1 = np.load('./MSIP_BDSM/Exp_res/over/time.npy')
        mf_mor = np.load('./MSIP_BDSM/Exp_res/over/MF-MOR.npy')
        smf_mor = np.load('./MSIP_BDSM/Exp_res/over/SMF-MOR.npy')
        high = np.load('./MSIP_BDSM/Exp_res/over/high.npy')
        sip = np.load('./MSIP_BDSM/Exp_res/over/sip_over.npy')
        svdmor = np.load('./MSIP_BDSM/Exp_res/over/svdmor_over.npy')
        demor = np.load('./MSIP_BDSM/Exp_res/over/DeMOR2_over.npy')
        plt.figure(figsize=(8,5))
        
        plt.plot(time1, high, color="#23AF1EFF", linestyle='--', marker='s', label='GT', markevery = 28, markersize=6, linewidth=1.5)
        plt.plot(time1, sip, color="#E20CAC", linestyle='--', marker='o', label='SIP', markevery = 24, markersize=6, linewidth=1.5)
        # plt.plot(time1, svdmor, color="#FFA500", linestyle='--', marker='^', label='SVD-MOR', markevery = 30, markersize=6, linewidth=1.5)
        # plt.plot(time1, demor[0], color="#129CAC", linestyle='--', marker='v', label='DeMOR', markevery = 34, markersize=6, linewidth=1.5)
        plt.plot(time1, mf_mor, color="#3F2ED1", linestyle='-.', marker='*', label='MF-MOR', markevery = 32, markersize=6, linewidth=1.5)
        plt.plot(time1, smf_mor, color="#DD123E", linestyle='-.', marker='*', label='SMF-MOR', markevery = 26, markersize=6, linewidth=2)
        
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Voltage (V)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('./MSIP_BDSM/Exp_res/over/baseline_over_comparison.pdf', dpi=300)
        plt.close()
    pass