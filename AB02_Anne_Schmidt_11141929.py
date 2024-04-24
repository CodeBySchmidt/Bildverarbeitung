import cv2
import numpy as np
import matplotlib.pyplot as plt

input_img = cv2.imread('Utils/test.jpg')


def histogram(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = np.zeros(256)
    height, width = img.shape[:2]

    # Pixelwerte zählen
    for i in range(height):
        for j in range(width):
            pixel_value = int(img[i, j])  # Umwandlung des Pixelwerts in Ganzzahl
            hist[pixel_value] = hist[pixel_value] + 1

    return hist


def kumulatives_histogramm(histogramm):
    kumulatives_histogramm = np.zeros(len(histogramm))
    kumulatives_histogramm[0] = histogramm[0]  # Initialisierung des ersten Elements

    for i in range(1, len(histogramm)):
        kumulatives_histogramm[i] = kumulatives_histogramm[i - 1] + histogramm[i]

    return kumulatives_histogramm


def lineare_kontrastspreizung(img, t0, t1):
    kontrastbild = np.copy(img)
    kontrastbild[kontrastbild < t0] = t0
    kontrastbild[kontrastbild > t1] = t1
    kontrastbild = (kontrastbild - t0) * (255 / (t1 - t0))

    cont_hist = histogram(kontrastbild)

    return kontrastbild.astype(np.uint8), cont_hist


def linear_histogramm_ausgleich(histogramm):
    total_pixels = np.sum(histogramm)
    p_a = kumulatives_histogramm(histogramm) / total_pixels
    k = 256
    linear_ausgleich = (p_a * (k - 1)).astype(np.uint8)

    return linear_ausgleich


def binarisierung(img, threshold):

    # bin_img = np.copy(img)
    # bin_img[bin_img < threshold] = 0
    # bin_img[bin_img > threshold] = 255

    height, width = img.shape[:2]
    bin_img = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            if img[y, x] < threshold:
                bin_img[y, x] = 0
            else:
                bin_img[y, x] = 255
    return bin_img


def hist_to_plotter(hist, str):
    plt.bar(range(256), hist, width=1, color='black')
    plt.xlabel('Pixel Wert')
    plt.ylabel('Häufigkeit')
    plt.title(str)
    plt.show()


hist = histogram(input_img)
# Histogramm plotten
hist_to_plotter(hist, "Histogram: Original Image")

kumuliertes_hist = kumulatives_histogramm(hist)
hist_to_plotter(kumuliertes_hist, "Kumuliertes Histogramm: LennaCol")
#
kont = lineare_kontrastspreizung(input_img, 50, 225)
hist_to_plotter(kont[1], "Lineare Konstarstspreizung Histogramm: LennaCol")

# Anwenden des linearen Histogrammausgleichs
linear_ausgleich = linear_histogramm_ausgleich(kumuliertes_hist)

# Histogramm des ausbalancierten Bildes plotten
hist_to_plotter(linear_ausgleich, "Histogramm nach linearem Ausgleich: LennaCol")


img2 = binarisierung(input_img, 143)
cv2.imshow("Original Bild", input_img)
cv2.imshow("Kontrastspreizung Bild", kont[0])
cv2.imshow("Binär Bild", img2)

cv2.imwrite("Original Bild.jpg", input_img)
cv2.imwrite("Kontrastspreizung Bild.jpg", kont[0])
cv2.imwrite("Binary Bild.jpg", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
