import numpy as np
from fir_conv import fir_conv

def sobel_edge(in_img_array: np.ndarray, thres: float):
    # Δημιουργία 1ης μάσκας
    G_x1 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # Δημιουργία 2ης μάσκας
    G_x2 = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    in_origin = np.array([0, 0]) # θέση της αρχή των αξόνων της εικόνας εισόδου
    mask_origin = np.array([1, 1])  # θέση της αρχής των αξόνων της μάσκας

    g1 = fir_conv(in_img_array, G_x1, in_origin, mask_origin)[0] # Αποτέλεσμα συνέλιξης της εικόνας εισόδου με την 1η μάσκα
    g2 = fir_conv(in_img_array, G_x2, in_origin, mask_origin)[0] # Αποτέλεσμα συνέλιξης της εικόνας εισόδου με την 2η μάσκα

    g = np.sqrt(g1**2 + g2**2)

    out_img_array = np.zeros((g.shape[0], g.shape[1]))
    for n1 in range(g.shape[0]):
        for n2 in range(g.shape[1]):
            if g[n1][n2] >= thres: # Αν η τιμή του pixel είναι μεγαλύτερη ή ίση με το threshold, θεωρείται pixel ακμής
                out_img_array[n1][n2] = 1

    return out_img_array
