import numpy as np

def fir_conv(in_img_array: np.ndarray, h: np.ndarray, in_origin: np.ndarray, mask_origin: np.ndarray):
    # zero padding της εικόνας εισόδου
    in_img_array_padded = np.pad(in_img_array,
                                 pad_width = ((mask_origin[0], h.shape[1] - mask_origin[0] - 1), (mask_origin[1], h.shape[0] - mask_origin[1] - 1)),
                                 mode = 'constant',
                                 constant_values = 0)
    out_img_array = np.zeros((in_img_array.shape[0], in_img_array.shape[1]))

    for n1 in range(in_img_array.shape[0]):
        for n2 in range(in_img_array.shape[1]):
            # Μετατόπιση της θέσης του κάθε pixel της εικόνας εισόδου στην αντίστοιχη θέση του στην padded εικόνα
            m1 = n1 + mask_origin[1]
            m2 = n2 + mask_origin[0]
            # Απομόνωση του τμήματος της padded εικόνας που ''βρίσκεται κάτω'' από την μάσκα
            img_under_mask = in_img_array_padded[(m1 - mask_origin[1]) : (m1 + h.shape[0] - mask_origin[1]),
                                                 (m2 - mask_origin[0]) : (m2 + h.shape[1] - mask_origin[0])]
            # Συνέλιξη
            out_img_array[n1][n2] = np.sum(img_under_mask * h)

    # Ορισμός της θέσης της αρχής των αξόνων στην εικόνα εξόδου με βάση τις αντίστοιχες στην εικόνα εισόδου και την μάσκα
    out_origin = in_origin + mask_origin
    return out_img_array, out_origin




