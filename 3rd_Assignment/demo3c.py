import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from image_to_graph import image_to_graph
from n_cuts import n_cuts_recursive

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
cluster_idx = n_cuts_recursive(affinity_mat, 5, 0.95, 0)

M, N, _ = d2a.shape
segmented = cluster_idx.reshape((M, N))
plt.imshow(segmented, cmap='nipy_spectral')
plt.axis('off')
plt.title(f"Recursive n_cuts on d2a")
plt.savefig("d2a_k2_rec_n_cuts.png", bbox_inches='tight')


#d2b
affinity_mat = image_to_graph(d2b)
cluster_idx = n_cuts_recursive(affinity_mat, 5, 0.95, 0)

M, N, _ = d2b.shape
segmented = cluster_idx.reshape((M, N))
plt.imshow(segmented, cmap='nipy_spectral')
plt.axis('off')
plt.title(f"Recursive n_cuts on d2b")
plt.savefig("d2b_k2_rec_n_cuts.png", bbox_inches='tight')

#Εμφάνιση κατατμημένης εικόνας d2b με το χρώμα κάθε cluster να είναι ο μέσος όρος των χρωμάτων των pixels
# που ανήκουν στο cluster
M, N, C = d2b.shape
flat_img = d2b.reshape(-1, C)
flat_labels = cluster_idx.flatten()
unique_labels = np.unique(flat_labels)
segmented_rgb = np.zeros_like(flat_img)

for label in unique_labels:
    indices = np.where(flat_labels == label)[0]
    mean_color = np.mean(flat_img[indices], axis=0)
    segmented_rgb[indices] = mean_color

segmented_rgb_image = segmented_rgb.reshape((M, N, C))

plt.imshow(segmented_rgb_image)
plt.axis('off')
plt.title("Recursive n_cuts on d2b (true colors)")
plt.savefig("d2b_segmented_true_colors.png", bbox_inches='tight')
