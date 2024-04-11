import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Utils/LennaCol2.png')

hist = np.zeros(256)
height, width = img.shape[:2]
cv2.imshow("Bild", img)


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

# def kumuliertesHistogram(img):

# def linearKonstastpreizung(img):

# def linearKonstastpreizungAusgleich(img):

# def binarisierunga(img):


# histogram(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
