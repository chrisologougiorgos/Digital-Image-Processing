import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hist_utils import calculate_hist_of_img
from hist_modif import perform_hist_eq
from hist_modif import perform_hist_matching

# Στην συνάρτηση αυτή δημιουργούνται όλες οι ζητούμενες εικόνες μαζί με τα ιστογράμματα τους. Η σύγκριση
# πραγματοποιείται στην αναφορά.

#Εικόνα εισόδου
filename = "input_img.jpg"
img = Image.open(fp = filename)
bw_img = img.convert("L")
img_array = np.array(bw_img).astype(float) / 255.0
img_uint8_original = (img_array*255).astype(np.uint8)
Image.fromarray(img_uint8_original).save("input_image.png")


#Εικόνα αναφοράς
filename = "ref_img.jpg"
img = Image.open(fp = filename)
bw_img = img.convert("L")
img_array_ref = np.array(bw_img).astype(float) / 255.0
img_uint8_original = (img_array_ref*255).astype(np.uint8)
Image.fromarray(img_uint8_original).save("ref_image.png")


#Ιστόγραμμα εικόνας εισόδου
hist_input = calculate_hist_of_img(img_array, True)
keys = list(hist_input.keys())
values = list(hist_input.values())
plt.bar(keys, values, width = 0.003)
plt.savefig("input_histogram.png")
plt.clf()

#Ιστόγραμμα εικόνας αναφοράς
hist_ref = calculate_hist_of_img(img_array_ref, True)
keys = list(hist_ref.keys())
values = list(hist_ref.values())
plt.bar(keys, values, width = 0.003)
plt.savefig("ref_histogram.png")
plt.clf()

#Εξισορρόπηση ιστογράμματος
for mode in ['greedy', 'non-greedy', 'post-disturbance']:
    eq_img = perform_hist_eq(img_array, mode)
    eq_hist = calculate_hist_of_img(eq_img, True)

    # Προσθήκη της στάθμης 1.0 με τιμή σχετικής συχνότητας εμφάνισης 0, αν δεν υπάρχει ήδη, ώστε
    # να έχουν όλες οι γραφικές παραστάσεις εύρος τιμών στον άξονα x από 0 μέχρι 1 (και όχι από 0 εώς την
    # μέγιστη στάθμη που εμφανίζεται στο ιστόγραμμα)
    hist_copy = eq_hist.copy()
    if 1.0 not in hist_copy:
        hist_copy[1.0] = 0.0

    keys = list(hist_copy.keys())
    values = list(hist_copy.values())
    plt.bar(keys, values, width = 0.003)
    plt.savefig(f"{mode}_equalized_histogram.png")
    plt.clf()

    img_uint8 = (eq_img*255).astype(np.uint8)
    output_filename = f"{mode}_equalized_image.png"
    Image.fromarray(img_uint8).save(output_filename)

#Αντιστοίχιση ιστογράμματος
for mode in ['greedy', 'non-greedy', 'post-disturbance']:
    processed_img = perform_hist_matching(img_array, img_array_ref, mode)
    processed_hist = calculate_hist_of_img(processed_img, True)

    # Προσθήκη της στάθμης 1.0 με τιμή σχετικής συχνότητας εμφάνισης 0, αν δεν υπάρχει ήδη, ώστε
    # να έχουν όλες οι γραφικές παραστάσεις εύρος τιμών στον άξονα x από 0 μέχρι 1 (και όχι από 0 εώς την
    # μέγιστηστάθμη που εμφανίζεται στο ιστόγραμμα)
    hist_copy = processed_hist.copy()
    if 1.0 not in hist_copy:
        hist_copy[1.0] = 0.0

    keys = list(hist_copy.keys())
    values = list(hist_copy.values())
    plt.bar(keys, values, width = 0.003)
    plt.savefig(f"{mode}_processed_histogram.png")
    plt.clf()

    img_uint8 = (processed_img*255).astype(np.uint8)
    output_filename = f"{mode}_processed_image.png"
    Image.fromarray(img_uint8).save(output_filename)

