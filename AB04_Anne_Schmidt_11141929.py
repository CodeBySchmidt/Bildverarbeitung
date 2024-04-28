import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read input image
input_img = cv2.imread("Utils/SetGameNew.jpg")


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

    for y in range(height):
        for x in range(width):
            if img[y, x] < threshold:
                bin_img[y, x] = 0
            else:
                bin_img[y, x] = 255
    return bin_img


# Compute histogram of input image
hist = histogram(input_img)

# Apply Otsu's thresholding to find optimal threshold
optimal_threshold = otsu_thresholding(hist)

# Binarize image using the optimal threshold
otsu_img = binarization(input_img, optimal_threshold)


# Creating kernel
kernel = np.ones((5, 5), np.uint8)

otsu_img_eroded = cv2.erode(otsu_img, kernel, iterations=4)

otsu_img_dilated = cv2.dilate(otsu_img_eroded, kernel, iterations=4)


# Display binary image using otsu threshold
# plt.title("Binary Image with Otsu Threshold: " + str(optimal_threshold))
# plt.imshow(otsu_img, cmap='gray')
# plt.show()

# Display binary image using otsu threshold
plt.title("Eroded Image with Otsu Threshold: " + str(optimal_threshold))
plt.imshow(otsu_img_eroded, cmap='gray')
plt.show()

# Display binary image using otsu threshold
plt.title("Dilated Image with Otsu Threshold: " + str(optimal_threshold))
plt.imshow(otsu_img_dilated, cmap='gray')
plt.show()
