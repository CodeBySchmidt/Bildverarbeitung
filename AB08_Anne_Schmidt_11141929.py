import cv2
import numpy as np
import os


def load_images_from_folder(folder, size):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, size)
            images.append(img)
    return images


def calculate_average_color(image):
    return image.mean(axis=(0, 1))


def find_best_tile(target_avg, tile_avgs):
    min_dist = float('inf')
    best_idx = -1
    for idx, avg in enumerate(tile_avgs):
        dist = np.linalg.norm(target_avg - avg)
        if dist < min_dist:
            min_dist = dist
            best_idx = idx
    return best_idx


def create_mosaic(target_image_path, tile_images, tile_size):
    # Zielbild laden
    target_image = cv2.imread(target_image_path)
    target_h, target_w, _ = target_image.shape

    # Gittergröße berechnen, durch die Verhältnisse von Bildhöhe / Kachelhöhe = Anzahl der Kachel die auf die Bildhöhe passen
    # -> das gleiche für die Breite
    grid_size = (target_h // tile_size[0], target_w // tile_size[1])

    # Zielbildgröße anpassen, damit es genau in das Gitter passt, da man sehr wahrscheinlich Kommazahlen raus bekommt und man möchte ja "ganze" Kacheln und keine reste
    resized_target_image = cv2.resize(target_image, (grid_size[1] * tile_size[1], grid_size[0] * tile_size[0]))

    # Mosaik-Array initialisieren
    mosaic = np.zeros(resized_target_image.shape, dtype=np.uint8)

    # Kachelbilder anpassen und durchschnittliche Farbwerte berechnen
    tile_images_resized = [cv2.resize(img, tile_size) for img in tile_images]
    tile_avgs = [calculate_average_color(img) for img in tile_images_resized]

    # Mosaik zusammensetzen
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            target_patch = resized_target_image[i * tile_size[0]:(i + 1) * tile_size[0],
                           j * tile_size[1]:(j + 1) * tile_size[1], :]
            target_avg = calculate_average_color(target_patch)

            best_idx = find_best_tile(target_avg, tile_avgs)
            best_tile = tile_images_resized[best_idx]

            mosaic[i * tile_size[0]:(i + 1) * tile_size[0],
            j * tile_size[1]:(j + 1) * tile_size[1], :] = best_tile

    return mosaic


# Beispielaufruf
target_image_path = "me3.JPG"
tile_images_folder = "cat"
tile_size = (32, 32)

tile_images = load_images_from_folder(tile_images_folder, tile_size)
mosaic_image = create_mosaic(target_image_path, tile_images, tile_size)

# Speichern des Mosaikbildes
cv2.imwrite('mosaic_output.jpg', mosaic_image)
