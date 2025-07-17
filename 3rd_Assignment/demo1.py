from scipy.io import loadmat
from spectral_clustering import spectral_clustering

data = loadmat("dip_hw_3.mat")
d1a = data["d1a"]

k_values = [2, 3, 4]
for k in k_values:
    cluster_idx = spectral_clustering(d1a, k)
    print(f"Ετικέτες κόμβων για k={k}: {cluster_idx}")