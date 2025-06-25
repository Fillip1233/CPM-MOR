import matplotlib.pyplot as plt
import scipy.io as spio
if __name__ == "__main__":
    # a simple sample to draw the G matrix

    # Load the G matrix from the .mat file
    plt.figure(figsize=(20, 20))
    data = spio.loadmat("G.mat")
    G = data["G"]
    plt.spy(G, markersize=1)
    plt.title("G matrix")
    plt.savefig("G.png")