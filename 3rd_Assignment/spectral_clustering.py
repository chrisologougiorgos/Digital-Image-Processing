import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans

def spectral_clustering(affinity_mat: np.ndarray, k: int):
    D = np.zeros(affinity_mat.shape)

    for i in range(affinity_mat.shape[0]):
        D[i][i] = np.sum(affinity_mat[i][:])

    L = D - affinity_mat

    eigenvalues, U = eigs(L, k = k, which='SM')
    U = np.real(U)

    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(U)
    cluster_idx = kmeans.labels_

    return cluster_idx



