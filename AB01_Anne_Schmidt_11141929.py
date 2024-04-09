# Anne Kathrin Schmidt - 11141929 - BV1 - Übungsblatt 01

import cv2
import numpy as np

# Aufgabe 1
print("Hello World!")

# Aufgabe 2
for i  in range(100):
    print(i+1)

# Aufgabe 3
def ist_primzahl(zahl):
    if zahl < 2:
        return False
    for i in range(2, int(zahl ** 0.5) + 1):
        if zahl % i == 0:
            return False
    return True


def erste_n_primzahlen(n):
    primzahlen = []
    aktuelle_zahl = 2
    while len(primzahlen) < n:
        if ist_primzahl(aktuelle_zahl):
            primzahlen.append(aktuelle_zahl)
        aktuelle_zahl += 1
    return primzahlen


n = int(input("Geben Sie die Anzahl der ersten Primzahlen ein: "))
primzahlen = erste_n_primzahlen(n)
print("Die ersten", n, "Primzahlen sind:", primzahlen)

# Aufgabe 4
img = cv2.imread("Utils\LennaCol.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Bild", img)
cv2.imwrite("Utils\GrayLennaCol.png", img)
cv2.waitKey(0)


# Aufgabe 5
def invertimage(img):
    height, width = img.shape
    canvas = np.zeros((height, width), dtype=np.uint8)
    for y in range(0, height):
        for x in range(0, width):
            pixelvalue = img[y, x]
            canvas[y, x] = 255 - pixelvalue
    return canvas


invertMyImg = invertimage(img)

cv2.imshow("Invertiert", invertMyImg)
cv2.imwrite("Utils\InventierteLennaCol.png", invertMyImg)
cv2.waitKey(0)
