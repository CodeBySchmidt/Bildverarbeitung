import cv2
import numpy as np
import matplotlib.pyplot as plt

input_img = cv2.imread('Utils/LennaCol2.png')
# cv2.imshow("Bild", input_img)
hist = np.zeros(256)
height, width = input_img.shape[:2]


def histogram(img):
    # Pixelwerte zählen
    for i in range(height):
        for j in range(width):
            pixel_value = img[i, j]
            hist[pixel_value] = hist[pixel_value] + 1

    # Histogramm plotten
    plt.bar(range(0, 256), hist, width=2, color='black')
    plt.xlabel('Pixel Wert')
    plt.ylabel('Häufigkeit')
    plt.title('Histogram')
    plt.show()

    return hist


def kumuliertes_histogramm(histogramm):
    kumulatives_histogramm = [0] * len(histogramm)
    kumulatives_histogramm[0] = histogramm[0]

    for i in range(1, len(histogramm)):
        kumulatives_histogramm[i] = kumulatives_histogramm[i - 1] + histogramm[i]

    # Histogramm plotten
    plt.bar(range(0, 256), kumulatives_histogramm, width=2, color='black')
    plt.xlabel('Pixel Wert')
    plt.ylabel('Häufigkeit')
    plt.title('Kumulatives Histogram')
    plt.show()

    return kumulatives_histogramm


def lineare_kontrastspreizung(histogramm, t0, t1):

    temp_hist = np.zeros(abs(t1 - t0))

    for i in range(t0, t1):  # Wir müssen auch den größten Wert einschließen
        temp_hist[i - t0] = histogramm[i]



    # Histogramm plotten
    plt.bar(range(256), temp_hist, width=2, color='black')  # Den gesamten Bereich von t0 bis t1 abdecken
    plt.xlabel('Pixel Wert')
    plt.ylabel('Häufigkeit')
    plt.title('Lineare Spreizung - Histogram')
    plt.show()

    return

# def linearKonstastpreizungAusgleich(img):

# def binarisierunga(img):


hist = histogram(input_img)
kumuli_hist = kumuliertes_histogramm(hist)
lineare_kontrastspreizung(hist, 30, 225)
cv2.waitKey(0)
cv2.destroyAllWindows()
