import cv2 as cv

img = cv.imread("Utils/LennaCol.png", cv.IMREAD_GRAYSCALE)
cv.imshow("Testbild", img)
cv.waitKey(0)
cv.imwrite("Utils/LennaCol2.png", img)
