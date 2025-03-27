'''
2025-3-7
McPack baseline
'''
import scipy.io as spio
def McPack():
    pass

if __name__ == '__main__':
    # C = E, G = A
    data = spio.loadmat("./IBM_transient/ibmpg1t.mat")
    C, G, B = data['E'] * 1e-0, data['A'], data['B']
    port_num = 40
    B = B.tocsc()
    C = C.tocsc()
    G = G.tocsc()
    B = B[:, 0:port_num]
    # output matrix
    O = B