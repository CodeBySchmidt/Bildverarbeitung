import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read input image
# input_img = cv2.imread("Utils/test.jpg")


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


def calculate_median(img):
    # Compute the median value of pixel intensities in the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median_value = np.median(img)
    return median_value


def apply_median_filter(img, m, n):
    kernel_size = m * n
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    # Get image dimensions
    height, width = img.shape[:2]

    # Define padding size to handle border pixels
    padding_size = kernel_size // 2

    # Create padded image with reflected border pixels
    padded_image = cv2.copyMakeBorder(img, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_REFLECT)

    # Initialize filtered image
    filtered_image = np.zeros_like(img)

    # Apply median filter to each pixel
    for y in range(padding_size, height + padding_size):
        for x in range(padding_size, width + padding_size):
            # Extract kernel region
            kernel_region = padded_image[y - padding_size:y + padding_size + 1, x - padding_size:x + padding_size + 1]

            # Compute median value of kernel region and assign to filtered image
            filtered_image[y - padding_size, x - padding_size] = np.median(kernel_region)

    return filtered_image, padded_image


# # Compute histogram of input image
# hist = histogram(input_img)
#
# # Apply Otsu's thresholding to find optimal threshold
# optimal_threshold = otsu_thresholding(hist)
#
# # Binarize image using the optimal threshold
# otsu_img = binarization(input_img, optimal_threshold)
#
# # Display binary image using otsu threshold
# plt.title("Binary Image with Otsu Threshold: " + str(optimal_threshold))
# plt.imshow(otsu_img, cmap='gray')
# plt.show()
#
# median_value = calculate_median(input_img)
# median_img = binarization(input_img, median_value)
#
# # Display binary image using median threshold
# plt.title("Binary Image with Median Threshold: " + str(median_value))
# plt.imshow(median_img, cmap='gray')
# plt.show()

input_img = cv2.imread("Utils/LennaCol.png", cv2.IMREAD_GRAYSCALE)

m = 3
n = 3
filtered_img = apply_median_filter(input_img, m, n)

# Display binary image using median threshold
plt.title("Filtered Image with Kernel Size (M: " + str(m) + " & N: " + str(n) + ")")
plt.imshow(filtered_img[0], cmap="gray")
plt.show()

plt.title("Padded Image")
plt.imshow(filtered_img[1], cmap="gray")
plt.show()

plt.title("Original Image")
plt.imshow(input_img, cmap="gray")
plt.show()