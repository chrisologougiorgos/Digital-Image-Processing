import numpy as np
from hist_utils import calculate_hist_of_img
from hist_utils import apply_hist_modification_transform

def perform_hist_modification(img_array: np.ndarray, hist_ref: dict, mode: str):
    # Αρχικοποιήσεις
    hist_ref = dict(sorted(hist_ref.items()))
    modified_hist = {}
    for ref_key in hist_ref:
        modified_hist[ref_key] = 0
    modification_transform = {}
    used_keys = set()

    # Μέθοδος greedy
    if mode == 'greedy':
        hist = calculate_hist_of_img(img_array, True)
        for ref_key in hist_ref:
            for key in hist:
                # Έλεγχος αν η συχνότητα εμφάνισης της στάθμης φωτεινότητας στην τελική εικόνα έχει
                # υπερβεί την αντίστοιχη στην εικόνα αναφοράς
                if modified_hist[ref_key] >= hist_ref[ref_key]:
                    break
                elif key not in used_keys:
                    modified_hist[ref_key] += hist[key]

                    modification_transform[key] = ref_key #Αντιστοίχιση στάθμης εισόδου με στάθμη εξόδου
                    used_keys.add(key)
        modified_img = apply_hist_modification_transform(img_array, modification_transform)
        return modified_img


    # Μέθοδος non-greedy
    if mode == 'non-greedy':
        hist = calculate_hist_of_img(img_array, True)
        for ref_key in hist_ref:
            for key in hist:
                deficiency = hist_ref[ref_key] - modified_hist[ref_key] #Ορισμός deficiency
                # Έλεγχος αν η συχνότητα εμφάνισης της στάθμης φωτεινότητας στην τελική εικόνα έχει
                # υπερβεί την αντίστοιχη στην εικόνα αναφοράς
                if modified_hist[ref_key] >= hist_ref[ref_key]:
                    break
                elif key not in used_keys:
                    # Έλεγχος αν το deficiency είναι αρκετά μικρό ώστε να διακοπεί η διαδικασία για την
                    # συγκεκριμένη στάθμη φωτεινότητας της τελικής εικόνας
                    # Σε περίπτωση που το deficiency είναι αρκετά μικρό αλλά στην στάθμη της τελικής εικόνας
                    # δεν έχει αντιστοιχιθεί καμία από τις διαθέσιμες στάθμες εισόδου, τότε πραγματοποιείται
                    # αντιστοίχιση
                    if deficiency < hist[key] / 2 and modified_hist[ref_key]!=0:
                        break
                    modified_hist[ref_key] += hist[key]
                    modification_transform[key] = ref_key #Αντιστοίχιση στάθμης εισόδου με στάθμη εξόδου
                    used_keys.add(key)
        modified_img = apply_hist_modification_transform(img_array, modification_transform)
        return modified_img


    # Μέθοδος post-disturbance
    if mode == 'post-disturbance':
        levels = np.unique(img_array)
        d = levels[1] - levels[0]
        u = np.random.uniform(low=-d / 2, high=d / 2, size=(img_array.shape[0], img_array.shape[1]))
        disturbed_img_array = img_array + u # Προσθήκη θορύβου στον πίνακα της εικόνας εισόδου
        hist = calculate_hist_of_img(disturbed_img_array, True)
        #Μέθοδος greedy
        for ref_key in hist_ref:
            for key in hist:
                # Έλεγχος αν η συχνότητα εμφάνισης της στάθμης φωτεινότητας στην τελική εικόνα έχει
                # υπερβεί την αντίστοιχη στην εικόνα αναφοράς
                if modified_hist[ref_key] >= hist_ref[ref_key]:
                    break
                elif key not in used_keys:
                    modified_hist[ref_key] += hist[key]
                    modification_transform[key] = ref_key #Αντιστοίχιση στάθμης εισόδου με στάθμη εξόδου
                    used_keys.add(key)
        modified_img = apply_hist_modification_transform(disturbed_img_array, modification_transform)
        return modified_img


def perform_hist_eq(img_array: np.ndarray, mode: str):
    hist_ref = {}
    img_array_levels = np.unique(img_array)
    # Δημιουργία dictionary που αναπαριστά ένα απόλυτα εξισορροπημένο ιστόγραμμα
    for level in img_array_levels:
        hist_ref[level] = 1 / len(img_array_levels)
    equalized_img = perform_hist_modification(img_array, hist_ref, mode)
    return equalized_img


def perform_hist_matching(img_array: np.ndarray, img_array_ref: np.ndarray, mode: str):
    hist_ref = calculate_hist_of_img(img_array_ref, True)
    processed_img = perform_hist_modification(img_array, hist_ref, mode)
    return processed_img


