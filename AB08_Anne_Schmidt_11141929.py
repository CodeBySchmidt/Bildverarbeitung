import cv2
import numpy as np
import os

from matplotlib import pyplot as plt


def load_images_from_folder(folder, size):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, size)
            images.append(img)
    return images


def calculate_average_color(image):  # berechnet den durchschnittlichen Farbwert eines Bildes
    img_avg = image.mean(axis=(0, 1))  # image.mean(axis=(0, 1)) berechnet den Mittelwert entlang der Achsen 0 und 1, also entlang der Höhe und Breite des Bildes.
    return img_avg                     # außerdem wird der Mittelwert für jeden Farbkanal separat berechnet wird, indem über alle Pixel des Bildes gemittelt wird.




def find_best_tile(target_avg, tile_avgs):  # Mit den durchschnittlichen Farbwerten von den tiles (Kacheln) und unserem Zielbild (Target) können die "passenden" tiles gesucht werden

    min_dist = float("inf")  # min_dist wird auf unendlich gesetzt, um sicherzustellen, dass jede berechnete Distanz kleiner sein wird
    best_idx = -1  # best_idx wird auf -1 gesetzt, um anzuzeigen, dass noch kein Index gefunden wurde.

    for idx, avg in enumerate(tile_avgs):  # geht durch jede Kachel in der Liste tile_avgs. idx ist der Index der aktuellen Kachel, und avg ist der durchschnittliche Wert der Kachel
        dist = np.linalg.norm(target_avg - avg)  # Hier wird die euklidische Distanz zwischen dem Zielwert (target_avg) und dem aktuellen Kachelwert (avg) berechnet.
        if dist < min_dist:  # Wenn die berechnete Distanz kleiner ist als die bisher kleinste gefundene Distanz (min_dist), wird min_dist aktualisiert und best_idx auf den aktuellen Index gesetzt
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
    total_summ_tiles = grid_size[0] * grid_size[1]
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

    return mosaic, total_summ_tiles


# Beispielaufruf
target_image_path = "me3.JPG"
tile_images_folder = "cat"
tile_size = (32, 32)

tile_images = load_images_from_folder(tile_images_folder, tile_size)
mosaic_image = create_mosaic(target_image_path, tile_images, tile_size)

# Speichern des Mosaikbildes
cv2.imwrite("mosaic_output.jpg", mosaic_image[0])


# Anzeigen der Bilder
plt.figure(figsize=(15, 8))

# Eingabebild anzeigen
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(target_image_path), cv2.COLOR_BGR2RGB))
plt.title("Eingabebild")
plt.axis("off")

# Mosaikbild anzeigen
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(mosaic_image[0], cv2.COLOR_BGR2RGB))
plt.title(f"Mosaikbild (Anzahl Kacheln: {mosaic_image[1]})")
plt.axis("off")

plt.tight_layout()
plt.show()
