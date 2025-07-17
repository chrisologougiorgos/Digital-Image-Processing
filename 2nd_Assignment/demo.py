import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sobel_edge import sobel_edge
from log_edge import log_edge
from circ_hough import circ_hough

filename = "basketball_large.png"
img = Image.open(fp = filename)
bw_img = img.convert("L")
img_array = np.array(bw_img).astype(float) / 255.0
img_uint8_original = (img_array*255).astype(np.uint8)
Image.fromarray(img_uint8_original).save("input_image.png")

#

#Sobel
'''
threshold_values = np.linspace(0.1, 0.9, 9)
edge_pixel_counts = []
for thres in threshold_values:
    binary_edges_img = sobel_edge(img_array, thres)
    binary_edges_uint8 = (binary_edges_img * 255).astype(np.uint8)

    filename_out = f"sobel_edges_thres_{thres:.1f}.png"
    Image.fromarray(binary_edges_uint8).save(filename_out)

    edge_count = np.count_nonzero(binary_edges_img)
    edge_pixel_counts.append(edge_count)

plt.figure(figsize=(8, 6))
plt.bar([f"{thres:.1f}" for thres in threshold_values], edge_pixel_counts, color='skyblue')
plt.xlabel("Threshold value")
plt.ylabel("Number of edge pixels")
plt.title("Effect of threshold value on number of detected edge pixels")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("sobel_edge_threshold_barplot.png")
'''


#LoG
'''
threshold_values = np.linspace(0.1, 1, 10)
for mode in range(1, 4):
    edge_pixel_counts = []
    for thres in threshold_values:
        binary_edges_img = log_edge(img_array, thres, mode)
        binary_edges_uint8 = (binary_edges_img * 255).astype(np.uint8)

        filename_out = f"LoG_edges_mode_{mode}_thres_{thres:.1f}.png"
        Image.fromarray(binary_edges_uint8).save(filename_out)

        edge_count = np.count_nonzero(binary_edges_img)
        edge_pixel_counts.append(edge_count)

    plt.figure(figsize=(8, 6))
    plt.bar([f"{thres:.1f}" for thres in threshold_values], edge_pixel_counts, color='skyblue')
    plt.xlabel("Threshold value")
    plt.ylabel("Number of edge pixels")
    plt.title("Effect of threshold value on number of detected edge pixels")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"LoG_edge_threshold_barplot_mode_{mode}.png")
'''


#Sobel-Hough
'''
binary_edges_img = sobel_edge(img_array, 0.4)
R_max = 400
dim = [200, 200, 120]
V_min = 1050
centers, radii = circ_hough(binary_edges_img, R_max, dim, V_min)

plt.figure(figsize=(10, 10))
fig, ax = plt.subplots()
ax.imshow(img_array, cmap='gray')

for (x, y), r in zip(centers, radii):
    circle = patches.Circle((x, y), r, fill=False, edgecolor='red', linewidth=0.5)
    ax.add_patch(circle)

plt.title("Ανιχνευμένοι κύκλοι")
plt.axis('off')
plt.savefig("detected_circles_Sobel.png")
'''


#Log-Hough
'''
binary_edges_img = log_edge(img_array, 0.3, 3)
R_max = 400
dim = [200, 200, 120]
V_min = 390
centers, radii = circ_hough(binary_edges_img, R_max, dim, V_min)

plt.figure(figsize=(10, 10))
fig, ax = plt.subplots()
ax.imshow(img_array, cmap='gray')

for (x, y), r in zip(centers, radii):
    circle = patches.Circle((x, y), r, fill=False, edgecolor='red', linewidth=0.5)
    ax.add_patch(circle)

plt.title("Ανιχνευμένοι κύκλοι")
plt.axis('off')
plt.savefig("detected_circles_LoG.png")
'''