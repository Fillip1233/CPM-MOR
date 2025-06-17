import numpy as np
import matplotlib.pyplot as plt

singular_values = {
    "ibmpg1t": np.load("/home/fillip/home/CPM-MOR/SVD_B/1t/S.npy"),
    "ibmpg2t": np.load("/home/fillip/home/CPM-MOR/SVD_B/2t/S.npy"),
    "ibmpg3t": np.load("/home/fillip/home/CPM-MOR/SVD_B/3t/S.npy"),
    "ibmpg4t": np.load("/home/fillip/home/CPM-MOR/SVD_B/4t/S.npy"),
    "ibmpg5t": np.load("/home/fillip/home/CPM-MOR/SVD_B/5t/S.npy"),
    "ibmpg6t": np.load("/home/fillip/home/CPM-MOR/SVD_B/6t/S.npy"),
    "thupg1t": np.load("/home/fillip/home/CPM-MOR/SVD_B/thupg1t/S.npy"),
}

thresholds = [0, 0.5, 1, 1.1, 1.5, 2 ,2.5 ]

sv_counts = {name: [np.sum(sv >= t) for t in thresholds] for name, sv in singular_values.items()}

plt.figure(figsize=(8, 5))
for name, counts in sv_counts.items():
    plt.plot(thresholds, counts, marker="o", linestyle="-", label=name)

plt.xlabel("Singular Value Threshold")
plt.ylabel("Number of Singular Values")
plt.title("Singular Values Count Above Thresholds (2000 ports)")
plt.xticks(thresholds)
plt.yticks(range(0, max(max(sv_counts.values())) + 1, max(1, max(max(sv_counts.values())) // 10))) 
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("B1.png")
