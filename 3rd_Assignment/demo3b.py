from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from image_to_graph import image_to_graph
from n_cuts import n_cuts
from n_cuts import calculate_n_cut_value


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


#d2a
affinity_mat = image_to_graph(d2a)
cluster_idx = n_cuts(affinity_mat, 2)
print(f"Τιμή μετρικής n_cuts για d2a: {calculate_n_cut_value(affinity_mat, cluster_idx)}")

M, N, _ = d2a.shape
segmented = cluster_idx.reshape((M, N))
plt.imshow(segmented, cmap='nipy_spectral')
plt.axis('off')
plt.title(f"One iteration of recursive n_cuts on d2a")
plt.savefig("d2a_rec_n_cuts_1_iter.png", bbox_inches='tight')


#d2b
affinity_mat = image_to_graph(d2b)
cluster_idx = n_cuts(affinity_mat, 2)
print(f"Τιμή μετρικής n_cuts για d2b: {calculate_n_cut_value(affinity_mat, cluster_idx)}")

M, N, _ = d2b.shape
segmented = cluster_idx.reshape((M, N))
plt.imshow(segmented, cmap='nipy_spectral')
plt.axis('off')
plt.title(f"One iteration of recursive n_cuts on d2b")
plt.savefig("d2b_rec_n_cuts_1_iter.png", bbox_inches='tight')

