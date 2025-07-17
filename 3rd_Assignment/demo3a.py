from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from image_to_graph import image_to_graph
from n_cuts import n_cuts

data = loadmat("dip_hw_3.mat")
d2a = data["d2a"]
d2b = data["d2b"]

plt.imshow(d2a)
plt.title("Original Image: d2a")
plt.axis('off')
plt.savefig("d2a_original.png", bbox_inches='tight')

plt.imshow(d2b)
plt.title("Original Image: d2b")
plt.axis('off')
plt.savefig("d2b_original.png", bbox_inches='tight')

k_values = [2, 3, 4]

#d2a
affinity_mat = image_to_graph(d2a)
for k in k_values:
    cluster_idx = n_cuts(affinity_mat, k)

    M, N, _ = d2a.shape
    segmented = cluster_idx.reshape((M, N))

    plt.imshow(segmented, cmap='nipy_spectral')
    plt.axis('off')
    plt.title(f"Normalized-cuts (k={k}) on d2a")
    plt.savefig(f"d2a_k={k}_n_cuts.png", bbox_inches='tight')


#d2b
affinity_mat = image_to_graph(d2b)
for k in k_values:
    cluster_idx = n_cuts(affinity_mat, k)

    M, N, _ = d2b.shape
    segmented = cluster_idx.reshape((M, N))

    plt.imshow(segmented, cmap='nipy_spectral')
    plt.axis('off')
    plt.title(f"Normalized-cuts (k={k}) on d2b")
    plt.savefig(f"d2b_k={k}_n_cuts.png", bbox_inches='tight')

