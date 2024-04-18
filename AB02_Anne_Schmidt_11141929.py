import cv2
import numpy as np
import matplotlib.pyplot as plt

input_img = cv2.imread('Utils/LennaCol.png', cv2.IMREAD_GRAYSCALE)


def histogram(img):
    hist = np.zeros(256)
    height, width = img.shape[:2]

    # Pixelwerte zählen
    for i in range(height):
        for j in range(width):
            pixel_value = int(img[i, j])  # Umwandlung des Pixelwerts in Ganzzahl
            hist[pixel_value] = hist[pixel_value] + 1

    return hist


def kumuliertes_histogramm(histogramm):
    kumulatives_histogramm = np.zeros(len(histogramm))
    for i in range(len(histogramm)):
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
    cumulative_histogram = np.cumsum(histogramm) / total_pixels * 255
    linear_ausgleich = np.zeros_like(histogramm, dtype=np.uint8)

    for i in range(len(histogramm)):
        linear_ausgleich[i] = np.round(cumulative_histogram[i])

    # Histogramme plotten
    plt.bar(range(256), linear_ausgleich, width=2, color='black')
    plt.title('Histogramm nach linearem Ausgleich')
    plt.show()

    return linear_ausgleich


def binarisierung(img, threshold):
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
hist_to_plotter(hist, "Histogramm: LennaCol")

kumuliertes_hist = kumuliertes_histogramm(hist)
hist_to_plotter(kumuliertes_hist, "Kumuliertes Histogramm: LennaCol")

kont = lineare_kontrastspreizung(input_img, 50, 225)
hist_to_plotter(kont[1], "Lineare Konstarstspreizung Histogramm: LennaCol")


img2 = binarisierung(input_img, 143)
cv2.imshow("Original Bild", input_img)
cv2.imshow("Kontrastspreizung Bild", kont[0])
cv2.imshow("Binär Bild", img2)

cv2.imwrite("Original Bild.jpg", input_img)
cv2.imwrite("Kontrastspreizung Bild.jpg", kont[0])
cv2.imwrite("Binary Bild.jpg", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
