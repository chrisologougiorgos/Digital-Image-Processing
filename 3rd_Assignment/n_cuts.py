import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans

def n_cuts(affinity_mat: np.ndarray, k: int):
    D = np.zeros(affinity_mat.shape)

    for i in range(affinity_mat.shape[0]):
        D[i][i] = np.sum(affinity_mat[i][:])

    L = D - affinity_mat

    eigenvalues, U = eigs(L, k=k, M=D, which='SM')
    U = np.real(U)

    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(U)
    cluster_idx = kmeans.labels_

    return cluster_idx


def calculate_n_cut_value(affinity_mat: np.ndarray, cluster_idx: np.ndarray):
    labelA, labelB = np.unique(cluster_idx)
    clusterA = np.where(cluster_idx == labelA)[0]
    clusterB = np.where(cluster_idx == labelB)[0]

    assoc_AA = 0
    for idx1 in clusterA:
        for idx2 in clusterA:
            assoc_AA += affinity_mat[idx1][idx2]

    assoc_BB = 0
    for idx1 in clusterB:
        for idx2 in clusterB:
            assoc_BB += affinity_mat[idx1][idx2]

    assoc_AV = 0
    for idx1 in clusterA:
        for idx2 in range(0, len(cluster_idx)):
           assoc_AV += affinity_mat[idx1][idx2]

    assoc_BV = 0
    for idx1 in clusterB:
        for idx2 in range(0, len(cluster_idx)):
            assoc_BV += affinity_mat[idx1][idx2]

    Nassoc_AB = assoc_AA / assoc_AV + assoc_BB / assoc_BV
    n_cut_value = 2 - Nassoc_AB
    return n_cut_value


def n_cuts_recursive(affinity_mat: np.ndarray, T1: int, T2: float, parent_label: int):
    cluster_idx = n_cuts(affinity_mat, 2)
    labelA, labelB = np.unique(cluster_idx)

    # Δίνονται καινούρια τυχαία labels ώστε να μην υπάρχει πιθανότητα ύπαρξης διαφορετικών clusters με το ίδιο label
    new_labelA = np.random.randint(1, 99999)
    new_labelB = np.random.randint(1, 99999)
    cluster_idx[cluster_idx == labelA] = new_labelA
    cluster_idx[cluster_idx == labelB] = new_labelB

    clusterA = np.where(cluster_idx == new_labelA)[0]
    clusterB = np.where(cluster_idx == new_labelB)[0]

    if len(clusterA) < T1 or len(clusterB) < T1 or calculate_n_cut_value(affinity_mat, cluster_idx) > T2:
        # Τα δύο υπο-clusters συγχωνεύονται πίσω στο αρχικό cluster
        cluster_idx[cluster_idx == new_labelA] = parent_label
        cluster_idx[cluster_idx == new_labelB] = parent_label
        return cluster_idx

    # Ορισμός των affinity πινάκων που αντιστοιχούν στους κόμβους των δύο clusters
    sub_affinity_A = affinity_mat[np.ix_(clusterA, clusterA)]
    sub_affinity_B = affinity_mat[np.ix_(clusterB, clusterB)]

    clusterA_new = n_cuts_recursive(sub_affinity_A, T1, T2, new_labelA)
    clusterB_new = n_cuts_recursive(sub_affinity_B, T1, T2, new_labelB)

    # Βρίσκω όλα τα pixels που ανήκουν σε καθένα από τα clusters A,B και τους δίνω το label του νέου υπο-cluster στο οποίο ανήκουν
    for index in range(0, len(clusterA_new)):
        value = clusterA_new[index]
        cluster_idx[clusterA[index]] = value

    for index in range(0, len(clusterB_new)):
        value = clusterB_new[index]
        cluster_idx[clusterB[index]] = value

    return cluster_idx


