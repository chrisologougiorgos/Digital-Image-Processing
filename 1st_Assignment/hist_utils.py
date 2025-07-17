import numpy as np

def calculate_hist_of_img(img_array: np.ndarray, return_normalized: bool):
    hist = {}
    N = img_array.shape[0] * img_array.shape[1] #Υπολογισμός των συνολικών εικονοστοιχείων της εικόνας εισόδου
    if return_normalized:
        for row in range(img_array.shape[0]):
            for column in range(img_array.shape[1]):
                if img_array[row][column] in hist:
                    # Αύξηση της σχετικής συχνότητας εμφάνισης μίας στάθμης φωτεινότητας όταν συναντάται εικονοστοιχείου με την τιμή της
                    hist[img_array[row][column]] = hist[img_array[row][column]] + 1 / N
                else:
                    # Αρχικοποίηση ενός κλειδιού του dictionary hist όταν η τιμή μίας στάθμης φωτεινότητας συναντάται για 1η φορά
                    hist[img_array[row][column]] = 1 / N
    else:
        for row in range(img_array.shape[0]):
            for column in range(img_array.shape[1]):
                if img_array[row][column] in hist:
                    # Αύξηση της απόλυτης συχνότητας εμφάνισης μίας στάθμης φωτεινότητας όταν συναντάται εικονοστοιχείου με την τιμή της
                    hist[img_array[row][column]] = hist[img_array[row][column]] + 1
                else:
                    # Αρχικοποίηση ενός κλειδιού του dictionary hist όταν η τιμή μίας στάθμης φωτεινότητας συναντάται για 1η φορά
                    hist[img_array[row][column]] = 1
    hist = dict(sorted(hist.items())) #Ταξινόμηση με βάση την αυξανόμενη τιμή του κλειδιού (όχι την τιμή στην οποία αντιστοιχίζεται το κλειδί)
    return hist


def apply_hist_modification_transform(img_array: np.ndarray, modification_transform: dict):
    modified_img = np.zeros((img_array.shape[0], img_array.shape[1]))
    for row in range(img_array.shape[0]):
        for column in range(img_array.shape[1]):
            modified_img[row][column] = modification_transform[img_array[row][column]] #Αντιστοίχιση κάθε εικονοστοιχείου μίας στάθμης εισόδου με την θεμιτή στάθμη εξόδου
    return modified_img
