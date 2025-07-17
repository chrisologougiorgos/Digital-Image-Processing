import numpy as np

def image_to_graph(img_array: np.ndarray):
    N = img_array.shape[0]
    M = img_array.shape[1]
    C = img_array.shape[2]
    affinity_mat = np.zeros((N*M, N*M))

    for I in range(0, N):
        for J in range(0, M):
            pixel = img_array[I][J][:] # το pixel για το οποίο θα υπολογιστούν οι αποστάσεις με όλα τα pixels
            for i in range(0, N):
                for j in range(0, M):
                    d = np.linalg.norm(pixel - img_array[i][j][:]) # υπολογισμός απόστασης
                    affinity_mat[I*M+J][i*M+j] = 1 / np.exp(d)
    return affinity_mat