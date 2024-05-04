import cv2
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
import time
import random

# Read input image
img = cv2.imread("IMG_1972.jpg")
input_img = cv2.resize(img, (640, 480))

plt.title("Original Image")
plt.imshow(input_img)
plt.show()

# Function to compute histogram of grayscale image
def histogram(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = np.zeros(256)
    height, width = img.shape[:2]

    # Count pixel values
    for i in range(height):
        for j in range(width):
            pixel_value = int(img[i, j])  # Convert pixel value to integer
            hist[pixel_value] = hist[pixel_value] + 1

    return hist




# Function to perform Otsu's thresholding
def otsu_thresholding(histogram):
    K = len(histogram)
    I_size = np.sum(histogram)
    optimal_threshold = -1
    var_max = -1

    for i in range(K):
        w0 = np.sum(histogram[:i + 1]) / I_size
        w1 = np.sum(histogram[i + 1:]) / I_size

        if w0 == 0 or w1 == 0:
            continue

        mean0 = compute_mean(histogram, 0, i)
        mean1 = compute_mean(histogram, i, K)

        var_between = w0 * w1 * ((mean0 - mean1) ** 2)

        if var_max < var_between:
            var_max = var_between
            optimal_threshold = i

    return optimal_threshold


# Function to compute mean within a given range of histogram
def compute_mean(histogram, start, end):
    total = np.sum(histogram[start:end])
    weights = np.arange(start, end) * histogram[start:end]
    return np.sum(weights) / total if total != 0 else 0


# Function to perform binarization based on a threshold
def binarization(img, threshold):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = img.shape[:2]
    bin_img = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if img[i, j] < threshold:
                bin_img[i, j] = 0
            else:
                bin_img[i, j] = 1
    return bin_img


# Aufgabe 3
def flood_fill_dfs(image, x, y, label):
    height, width = image.shape[:2]
    stack = []
    stack.append((x, y))
    while stack:
        x, y = stack.pop()
        if 0 <= x < height and 0 <= y < width and image[x][y] == 1:
            image[x][y] = label
            stack.append((x + 1, y))
            stack.append((x, y + 1))
            stack.append((x, y - 1))
            stack.append((x - 1, y))


def flood_fill_bfs(image, x, y, label):
    height, width = image.shape[:2]
    queue = Queue()
    queue.put((x, y))
    while not queue.empty():
        x, y = queue.get()
        if 0 <= x < height and 0 <= y < width and image[x][y] == 1:
            image[x][y] = label
            queue.put((x + 1, y))
            queue.put((x, y + 1))
            queue.put((x, y - 1))
            queue.put((x - 1, y))


def region_labeling(image):
    label = 2
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y] == 1:
                # flood_fill_dfs(image, x, y, label)
                flood_fill_bfs(image, x, y, label)
                label = label + 1


def flood_fill_recursive(image, x, y, label):
    height, width = image.shape[:2]

    if 0 <= x < width and 0 <= y < height and image[x][y] == 1:

        image[x][y] = label

        flood_fill_recursive(image, x + 1, y, label)
        flood_fill_recursive(image, x - 1, y, label)
        flood_fill_recursive(image, x, y + 1, label)
        flood_fill_recursive(image, x, y - 1, label)

    else:
        return


def region_labeling_recursive(image):
    label = 2
    height, width = image.shape[:2]

    for y in range(height):
        for x in range(width):

            flood_fill_recursive(image, x, y, label)
            label += 1


def region_coloring(image):
    # Maximalen Labelwert finden
    max_label = np.max(image)

    # Zufällige Farben für jeden möglichen Labelwert generieren und in einem Dictionary speichern
    colors = {}
    for label in range(1, max_label + 1):
        colors[label] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    # Bild für die gefärbte Darstellung initialisieren
    image_color = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Farben entsprechend den Regionen zuweisen
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            label = image[x][y]
            if label != (0 and 1):
                image_color[x][y] = colors[label]

    return image_color


def scale_picture(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


# Compute histogram of input image
hist = histogram(input_img)

# Apply Otsu's thresholding to find optimal threshold
optimal_threshold = otsu_thresholding(hist)

# Binarize image using the optimal threshold
otsu_img = binarization(input_img, optimal_threshold)


# Creating kernel
kernel = np.ones((3, 3), np.uint8)

otsu_img_eroded = cv2.erode(otsu_img, kernel, iterations=1)
otsu_img_dilated = cv2.dilate(otsu_img_eroded, kernel, iterations=1)


region_img = otsu_img_dilated.copy()


# start and stop time for the region_labeling method
start_time = time.time()
region_labeling(region_img)
end_time = time.time()

# calculation for the taken time
execution_time = end_time - start_time
print("Execution time for Region labeling: ", execution_time, "seconds")


start_time = time.time()
img_color = region_coloring(region_img)
end_time = time.time()

execution_time = end_time - start_time
print("Execution time for Coloring: ", execution_time, "seconds")


# img_scaled = scale_picture(img_color, 30)
# img_scaled2 = scale_picture(region_img, 30)
#
# cv2.imshow("Image Region Label", img_scaled2 * 255)
# cv2.imshow("Image Region Label Color", img_scaled * 255)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


plt.title("Binary Image with Otsu Threshold")
plt.imshow(otsu_img, cmap='gray')
plt.show()
plt.title("eroded Image with 1 Iterations")
plt.imshow(otsu_img_eroded, cmap='gray')
plt.show()
plt.title("dilated Image with 1 Iterations")
plt.imshow(otsu_img_dilated, cmap='gray')
plt.show()
plt.title("Region labeled Image")
plt.imshow(region_img, cmap='gray')
plt.show()
plt.title("colored Region Image")
plt.imshow(img_color, cmap="gray")
plt.show()
