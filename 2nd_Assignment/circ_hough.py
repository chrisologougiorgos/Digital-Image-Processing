import numpy as np

def circ_hough(in_img_array: np.ndarray, R_max: float, dim: np.ndarray, V_min: int):

    Height, Width = in_img_array.shape  # Αποθήκευση ύψους και πλάτους εικόνας
    Votes_array = np.zeros((dim[0], dim[1], dim[2]), dtype=np.uint32)

    horizontal_values = np.linspace(1, Width - 1, dim[0])  # Διακριτές θέσεις pixels στην οριζόντια διάσταση (υποψήφιες θέσεις κέντρων)
    vertical_values = np.linspace(1, Height - 1, dim[1])  # Διακριτές θέσεις pixels στην κατακόρυφη διάσταση (υποψήφιες θέσεις κέντρων)
    r_values = np.linspace(50, R_max, dim[2]) # Διακριτές τιμές ακτινών (υποψήφιες τιμές ακτινών)
    angles = np.linspace(0, 2 * np.pi, num=72, endpoint=False)

    # Υπολογισμός ημιτόνων και συνημιτόνων για την εξασφάλιση ταχύτητας
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    for y in range(Height):
        for x in range(Width):
            if in_img_array[y, x] == 1: #Έλεγχος αν πρόκειται για pixel ακμής
                for r_index, r in enumerate(r_values):
                    # Υπολογισμός θέσεων των υποψήφιων κεντρών των κύκλων που διέρχονται από το pixel ακμή της επανάληψης
                    cx = x - r * cos_angles
                    cy = y - r * sin_angles

                    # Εύρεση των υποψήφιων κεντρών που βρίσκονται εντός των ορίων της εικόνας
                    valid_centers = (cx >= horizontal_values[0]) & (cx <= horizontal_values[-1]) & \
                            (cy >= vertical_values[0]) & (cy <= vertical_values[-1])

                    # Αντιστοίχηση των υποψήφιων κεντρών στις πλησιέστερες διακριτές θέσεις (που έχουν καθοριστεί παραπάνω)
                    x_idx = np.argmin(np.abs(horizontal_values[:, np.newaxis] - cx), axis=0)
                    y_idx = np.argmin(np.abs(vertical_values[:, np.newaxis] - cy), axis=0)

                    # Αύξηση κατά 1 των ψήφων κάθε υποψήφιου κύκλου με έγκυρα κέντρα
                    for i in np.where(valid_centers)[0]:
                        Votes_array[x_idx[i], y_idx[i], r_index] += 1

    centers = []
    radii = []

    # Με αυτήν την τριπλή for διατρέχονται όλοι οι υποψήφιοι κύκλοι
    for x_index in range(dim[0]):
        for y_index in range(dim[1]):
            for r_index, r in enumerate(r_values):
                votes = Votes_array[x_index, y_index, r_index]
                if votes >= V_min: # Έλεγχος αν ο υποψήφιος κύκλος έχει αρκετούς ψήφους
                    x_c = horizontal_values[x_index]
                    y_c = vertical_values[y_index]
                    centers.append([x_c, y_c])
                    radii.append(r)
                    print("----")
                    print(f"x: {x_c}, y: {y_c}, r: {r}, votes: {votes}")
                    print("-----\n")

    '''
    # Εύρεση του μέγιστου κύκλου
    max_index = np.unravel_index(np.argmax(Votes_array), Votes_array.shape)
    x_index, y_index, r_index = max_index

    x_center = horizontal_values[x_index]
    y_center = vertical_values[y_index]
    radius = r_values[r_index]
    max_votes = Votes_array[x_index, y_index, r_index]

    print("----")
    print("Κύκλος μέγιστων ψήφων:")
    print(f"x: {x_center}, y: {y_center}, r: {radius}, votes: {max_votes}")
    print("-----\n")
    '''

    return centers, radii
