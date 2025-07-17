import numpy as np
from fir_conv import fir_conv

def log_edge(in_img_array: np.ndarray, thres: float, mode: int):
    # Δημιουργία μάσκας
    log_mask = np.array([
        [0,  0, -1,  0,  0],
        [0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1,  0],
        [0,  0, -1,  0,  0]
    ])

    in_origin = np.array([0, 0])  # θέση της αρχή των αξόνων της εικόνας εισόδου
    mask_origin = np.array([2, 2])  # θέση της αρχής των αξόνων της μάσκας
    conv_img = fir_conv(in_img_array, log_mask, in_origin, mask_origin)[0] # Αποτέλεσμα συνέλιξης της εικόνας εισόδου με την μάσκα

    out_img_array = np.zeros((conv_img.shape[0], conv_img.shape[1]))
    # Έλεγχος για zero-crossing
    if mode==1: # 1ος τρόπος ελέγχου (3x3 grid)
        for n1 in range(1, conv_img.shape[0]-1):
            for n2 in range(1, conv_img.shape[1]-1):
                if conv_img[n1-1][n2]*conv_img[n1+1][n2]<0 and np.abs(conv_img[n1-1][n2] - conv_img[n1+1][n2]) > thres:
                    out_img_array[n1][n2] = 1
                elif conv_img[n1][n2-1]*conv_img[n1][n2+1]<0 and np.abs(conv_img[n1][n2-1] - conv_img[n1][n2+1]) > thres:
                    out_img_array[n1][n2] = 1
                elif conv_img[n1-1][n2-1]*conv_img[n1+1][n2+1]<0 and np.abs(conv_img[n1-1][n2-1] - conv_img[n1+1][n2+1]) > thres:
                    out_img_array[n1][n2] = 1
                elif conv_img[n1-1][n2+1]*conv_img[n1+1][n2-1]<0 and np.abs(conv_img[n1-1][n2+1] - conv_img[n1+1][n2-1]) > thres:
                    out_img_array[n1][n2] = 1
    elif mode==2: # 2ος τρόπος ελέγχου (3x3 grid)
        for n1 in range(1, conv_img.shape[0] - 1):
            for n2 in range(1, conv_img.shape[1] - 1):
                if conv_img[n1-1][n2-1] * conv_img[n1-1][n2+1]<0: #pattern1-> αριστερή στήλη και δεξιά στήλη αντίθετων προσήμων
                    if conv_img[n1-1][n2-1] * conv_img[n1][n2-1]>0:
                        if conv_img[n1-1][n2-1] * conv_img[n1+1][n2-1]>0:
                            if conv_img[n1-1][n2+1] * conv_img[n1][n2+1]>0:
                                if conv_img[n1-1][n2+1] * conv_img[n1+1][n2+1]>0:
                                    mean_left = np.mean([conv_img[n1-1][n2-1], conv_img[n1][n2-1], conv_img[n1+1][n2-1]])
                                    mean_right = np.mean([conv_img[n1-1][n2+1], conv_img[n1][n2+1], conv_img[n1+1][n2+1]])
                                    if np.abs(mean_left - mean_right) > thres:
                                        out_img_array[n1][n2] = 1

                if conv_img[n1-1][n2-1] * conv_img[n1+1][n2-1]<0: #pattern2-> πάνω γραμμή και κάτω γραμμή αντίθετων προσήμων
                    if conv_img[n1-1][n2-1] * conv_img[n1-1][n2]>0:
                        if conv_img[n1-1][n2-1] * conv_img[n1-1][n2+1]>0:
                            if conv_img[n1+1][n2-1] * conv_img[n1+1][n2]>0:
                                if conv_img[n1+1][n2-1] * conv_img[n1+1][n2+1]>0:
                                    mean_top = np.mean([conv_img[n1-1][n2-1], conv_img[n1-1][n2], conv_img[n1-1][n2+1]])
                                    mean_bottom = np.mean([conv_img[n1+1][n2-1], conv_img[n1+1][n2], conv_img[n1+1][n2+1]])
                                    if np.abs(mean_top - mean_bottom) > thres:
                                        out_img_array[n1][n2] = 1

                if conv_img[n1-1][n2-1] * conv_img[n1+1][n2+1]<0: #pattern3-> πάνω αριστερά pixels και κάτω δεξιά pixels αντίθετων προσήμων
                    if conv_img[n1-1][n2-1] * conv_img[n1][n2-1]>0:
                        if conv_img[n1-1][n2-1] * conv_img[n1-1][n2]>0:
                            if conv_img[n1+1][n2+1] * conv_img[n1][n2+1]>0:
                                if conv_img[n1+1][n2+1] * conv_img[n1+1][n2]>0:
                                    mean_topleft = np.mean([conv_img[n1-1][n2-1], conv_img[n1][n2-1], conv_img[n1-1][n2]])
                                    mean_bottomright = np.mean([conv_img[n1+1][n2+1], conv_img[n1][n2+1], conv_img[n1+1][n2]])
                                    if np.abs(mean_topleft - mean_bottomright) > thres:
                                        out_img_array[n1][n2] = 1

                if conv_img[n1+1][n2-1] * conv_img[n1-1][n2+1]<0: #pattern4-> κάτω αριστερά pixels και πάνω δεξιά pixels αντίθετων προσήμων
                    if conv_img[n1+1][n2-1] * conv_img[n1][n2-1]>0:
                        if conv_img[n1+1][n2-1] * conv_img[n1+1][n2]>0:
                            if conv_img[n1-1][n2+1] * conv_img[n1-1][n2]>0:
                                if conv_img[n1-1][n2+1] * conv_img[n1][n2+1]>0:
                                    mean_bottomleft = np.mean([conv_img[n1+1][n2-1], conv_img[n1][n2-1], conv_img[n1+1][n2]])
                                    mean_topright = np.mean([conv_img[n1-1][n2+1], conv_img[n1-1][n2], conv_img[n1][n2+1]])
                                    if np.abs(mean_bottomleft - mean_topright) > thres:
                                        out_img_array[n1][n2] = 1
    elif mode==3: # 3ος τρόπος ελέγχου (5x5 grid)
        for n1 in range(2, conv_img.shape[0] - 2):
            for n2 in range(2, conv_img.shape[1] - 2):

                if conv_img[n1-1][n2-1] * conv_img[n1-1][n2+1]<0: # pattern1-> 2 αριστερές στήλη και 2 δεξιά στήλες αντίθετων προσήμων
                    if conv_img[n1-1][n2-1] * conv_img[n1][n2-1]>0 and conv_img[n1-1][n2-1] * conv_img[n1+1][n2-1]>0 and conv_img[n1-1][n2-1] * conv_img[n1-2][n2-1]>0 and conv_img[n1-1][n2-1] * conv_img[n1+2][n2-1]>0:
                        if conv_img[n1-1][n2-1] * conv_img[n1-1][n2-2]>0 and conv_img[n1-1][n2-1] * conv_img[n1][n2-2]>0 and conv_img[n1-1][n2-1] * conv_img[n1+1][n2-2]>0 and conv_img[n1-1][n2-1] * conv_img[n1-2][n2-2]>0 and conv_img[n1-1][n2-1] * conv_img[n1+2][n2-2]>0:
                            if conv_img[n1-1][n2+1] * conv_img[n1][n2+1]>0 and conv_img[n1-1][n2+1] * conv_img[n1+1][n2+1]>0 and conv_img[n1-1][n2+1] * conv_img[n1-2][n2+1]>0 and conv_img[n1-1][n2+1] * conv_img[n1+2][n2+1]>0:
                                if conv_img[n1-1][n2+1] * conv_img[n1-1][n2+2]>0 and conv_img[n1-1][n2+1] * conv_img[n1][n2+2]>0 and conv_img[n1-1][n2+1] * conv_img[n1+1][n2+2]>0 and conv_img[n1-1][n2+1] * conv_img[n1-2][n2+2]>0 and conv_img[n1-1][n2+1] * conv_img[n1+2][n2+2]>0:
                                    mean_left = np.mean([conv_img[n1-1][n2-1], conv_img[n1][n2-1], conv_img[n1+1][n2-1], conv_img[n1-2][n2-1], conv_img[n1+2][n2-1],
                                                         conv_img[n1-1][n2-2], conv_img[n1][n2-2], conv_img[n1+1][n2-2], conv_img[n1-2][n2-2], conv_img[n1+2][n2-2]])
                                    mean_right = np.mean([conv_img[n1-1][n2+1], conv_img[n1][n2+1], conv_img[n1+1][n2+1], conv_img[n1-2][n2+1], conv_img[n1+2][n2+1],
                                                          conv_img[n1-1][n2+2], conv_img[n1][n2+2], conv_img[n1+1][n2+2], conv_img[n1-2][n2+2], conv_img[n1+2][n2+2]])
                                    if np.abs(mean_left - mean_right) > thres:
                                        out_img_array[n1][n2] = 1

                if conv_img[n1-1][n2-1] * conv_img[n1+1][n2-1]<0: # pattern2-> 2 πάνω γραμμές και 2 κάτω γραμμές αντίθετων προσήμων
                    if conv_img[n1-1][n2-1] * conv_img[n1-1][n2]>0 and conv_img[n1-1][n2-1] * conv_img[n1-1][n2+1]>0 and conv_img[n1-1][n2-1] * conv_img[n1-1][n2-2]>0 and conv_img[n1-1][n2-1] * conv_img[n1-1][n2+2]>0:
                        if conv_img[n1-1][n2-1] * conv_img[n1-2][n2-1]>0 and conv_img[n1-1][n2-1] * conv_img[n1-2][n2]>0 and conv_img[n1-1][n2-1] * conv_img[n1-2][n2+1]>0 and conv_img[n1-1][n2-1] * conv_img[n1-2][n2-2]>0 and conv_img[n1-1][n2-1] * conv_img[n1-2][n2+2]>0:
                            if conv_img[n1+1][n2-1] * conv_img[n1+1][n2]>0 and conv_img[n1+1][n2-1] * conv_img[n1+1][n2+1]>0 and conv_img[n1+1][n2-1] * conv_img[n1+1][n2-2]>0 and conv_img[n1+1][n2-1] * conv_img[n1+1][n2+2]>0:
                                if conv_img[n1+1][n2-1] * conv_img[n1+2][n2-1]>0 and conv_img[n1+1][n2-1] * conv_img[n1+2][n2]>0 and conv_img[n1+1][n2-1] * conv_img[n1+2][n2+1]>0 and conv_img[n1+1][n2-1] * conv_img[n1+2][n2-2]>0 and conv_img[n1+1][n2-1] * conv_img[n1+2][n2+2]>0:
                                    mean_top = np.mean([conv_img[n1-1][n2-1], conv_img[n1-1][n2], conv_img[n1-1][n2+1], conv_img[n1-1][n2-2], conv_img[n1-1][n2+2],
                                                        conv_img[n1-2][n2-1], conv_img[n1-2][n2], conv_img[n1-2][n2+1], conv_img[n1-2][n2-2], conv_img[n1-2][n2+2]])
                                    mean_bottom = np.mean([conv_img[n1+1][n2-1], conv_img[n1+1][n2], conv_img[n1+1][n2+1], conv_img[n1+1][n2-2], conv_img[n1+1][n2+2],
                                                           conv_img[n1+2][n2-1], conv_img[n1+2][n2], conv_img[n1+2][n2+1], conv_img[n1+2][n2-2], conv_img[n1+2][n2+2]])
                                    if np.abs(mean_top - mean_bottom) > thres:
                                        out_img_array[n1][n2] = 1

                if conv_img[n1-1][n2-1] * conv_img[n1+1][n2+1]<0: #pattern3-> πάνω αριστερά pixels και κάτω δεξιά pixels αντίθετων προσήμων
                    if conv_img[n1-1][n2-1] * conv_img[n1][n2-1]>0 and conv_img[n1-1][n2-1] * conv_img[n1-1][n2]>0:
                        if conv_img[n1-1][n2-1] * conv_img[n1-2][n2-2]>0 and conv_img[n1-1][n2-1] * conv_img[n1-2][n2-1]>0 and conv_img[n1-1][n2-1] * conv_img[n1-2][n2]>0 and conv_img[n1-1][n2-1] * conv_img[n1-1][n2-2]>0 and conv_img[n1-1][n2-1] * conv_img[n1][n2-2]>0:
                            if conv_img[n1+1][n2+1] * conv_img[n1][n2+1]>0 and conv_img[n1+1][n2+1] * conv_img[n1+1][n2]>0:
                                if conv_img[n1+1][n2+1] * conv_img[n1+2][n2+2]>0 and conv_img[n1+1][n2+1] * conv_img[n1+2][n2+1]>0 and conv_img[n1+1][n2+1] * conv_img[n1+2][n2]>0 and conv_img[n1+1][n2+1] * conv_img[n1+1][n2+2]>0 and conv_img[n1+1][n2+1] * conv_img[n1][n2+2]>0:
                                    mean_topleft = np.mean([conv_img[n1-1][n2-1], conv_img[n1][n2-1], conv_img[n1-1][n2], conv_img[n1-2][n2-2],
                                                            conv_img[n1-2][n2-1], conv_img[n1-2][n2], conv_img[n1-1][n2-2], conv_img[n1][n2-2]])
                                    mean_bottomright = np.mean([conv_img[n1+1][n2+1], conv_img[n1][n2+1], conv_img[n1+1][n2], conv_img[n1+2][n2+2],
                                                                conv_img[n1+2][n2+1], conv_img[n1+2][n2], conv_img[n1+1][n2+2], conv_img[n1][n2+2]])
                                    if np.abs(mean_topleft - mean_bottomright) > thres:
                                        out_img_array[n1][n2] = 1

                if conv_img[n1+1][n2-1] * conv_img[n1-1][n2+1]<0: #pattern4-> κάτω αριστερά pixels και πάνω δεξιά pixels αντίθετων προσήμων
                    if conv_img[n1+1][n2-1] * conv_img[n1][n2-1]>0 and conv_img[n1+1][n2-1] * conv_img[n1+1][n2]>0:
                        if conv_img[n1+1][n2-1] * conv_img[n1+2][n2-2]>0 and conv_img[n1+1][n2-1] * conv_img[n1+2][n2-1]>0 and conv_img[n1+1][n2-1] * conv_img[n1+2][n2]>0 and conv_img[n1+1][n2-1] * conv_img[n1+1][n2-2]>0 and conv_img[n1+1][n2-1] * conv_img[n1][n2-2]>0:
                            if conv_img[n1-1][n2+1] * conv_img[n1-1][n2]>0 and conv_img[n1-1][n2+1] * conv_img[n1][n2+1]>0:
                                if conv_img[n1-1][n2+1] * conv_img[n1-2][n2+2]>0 and conv_img[n1-1][n2+1] * conv_img[n1-2][n2+1]>0 and conv_img[n1-1][n2+1] * conv_img[n1-2][n2]>0 and conv_img[n1-1][n2+1] * conv_img[n1-1][n2+2]>0 and conv_img[n1-1][n2+1] * conv_img[n1][n2+2]>0:
                                    mean_bottomleft = np.mean([conv_img[n1+1][n2-1], conv_img[n1][n2-1], conv_img[n1+1][n2], conv_img[n1+2][n2-2],
                                                               conv_img[n1+2][n2-1], conv_img[n1+2][n2], conv_img[n1+1][n2-2], conv_img[n1][n2-2]])
                                    mean_topright = np.mean([conv_img[n1-1][n2+1], conv_img[n1-1][n2], conv_img[n1][n2+1], conv_img[n1-2][n2+2],
                                                             conv_img[n1-2][n2+1], conv_img[n1-2][n2], conv_img[n1-1][n2+2], conv_img[n1][n2+2]])
                                    if np.abs(mean_bottomleft - mean_topright) > thres:
                                        out_img_array[n1][n2] = 1

    return out_img_array