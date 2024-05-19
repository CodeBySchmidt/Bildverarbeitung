#  Übungsblatt 05
import cv2
import numpy as np
import random


def find_cards(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, imgBin = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cards = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        card = imgBin[y:y + h, x:x + w]
        cards.append((x, y, w, h, card))

    return cards


def count_symbols(card):
    contours, _ = cv2.findContours(card, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    symbol_count = len(contours)
    return symbol_count


def determine_pattern_and_shape(card):
    contours, _ = cv2.findContours(card, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    patterns = []
    shapes = []
    roundness = 0

    for contour in contours:
        mask = np.zeros_like(card)
        cv2.drawContours(mask, [contour], -1, color=255, thickness=1)
        filled_area = np.sum(mask == 255)
        bounding_rect_area = cv2.contourArea(contour)
        fill_rate = filled_area / bounding_rect_area

        # Calculate roundness
        perimeter = cv2.arcLength(contour, closed=True)
        area = cv2.contourArea(contour)
        roundness = (4 * np.pi * area) / (perimeter ** 2)


        if fill_rate > 0.6:  # Threshold for filled
            patterns.append("filled")
        else:
            patterns.append("striped")

        huMoments = cv2.HuMoments(cv2.moments(contour)).flatten()
        shapes.append(huMoments)

    return patterns, shapes, roundness


def classify_shapes(huMoments, roundness):

    roundness_threshold_low = 0.693
    roundness_threshold_high = 0.77
    # Fallunterscheidung basierend auf den Roundness-Werten und HuMoments

    if roundness < roundness_threshold_low:
        return "Raute"

    elif roundness_threshold_low <= roundness < roundness_threshold_high:

        if -2.00600827e-09 > huMoments[5] > -3.71612183e-09:
            return "Raute"
        if huMoments[3] > 5.16398607e-07:
            return "Raute"

        elif 1.70038716e-07 < huMoments[3] < 4.25120757e-07:
            if huMoments[4] > 1.10003454e-14:
                return "Rechteck"
            else:
                return "Wellen"

        if 4.06948654e-08 < huMoments[3] < 4.25120757e-08:
            return "Rechteck"
        else:
            return "Wellen"


def main(image_path):
    cards = find_cards(image_path)
    img = cv2.imread(image_path)

    fixed_colors_brg = [
        (0, 0, 255),  # Rot (BGR)
        (0, 255, 0),  # Grün (BGR)
        (255, 0, 0),  # Blau (BGR)
        (0, 255, 255),  # Gelb (BGR)
        (255, 255, 0),  # Cyan (BGR)
        (255, 0, 255),  # Magenta (BGR)
        (192, 192, 192),  # Silber (BGR)
        (128, 128, 128),  # Grau (BGR)
        (0, 0, 128),  # Dunkelrot (BGR)
        (0, 128, 128),  # Olivgrün (BGR)
        (0, 128, 0),  # Dunkelgrün (BGR)
        (128, 0, 0)  # Petrol (BGR)
    ]

    for i, (x, y, w, h, card) in enumerate(cards):
        symbol_count = count_symbols(card)
        patterns, shapes, roundness = determine_pattern_and_shape(card)

        print(f"Card {i + 1}:")
        print(f"    - Symbols: {symbol_count}")
        for j, pattern in enumerate(patterns):
            shape = classify_shapes(shapes[j], roundness)
            # print(f"   Symbol {j + 1}:")
            # print(f"     - Pattern: {pattern}")
            print(f"    - Shape: {shape}")
        print()

        # Draw rectangle around card
        cv2.rectangle(img, (x, y), (x + w, y + h), (fixed_colors_brg[i]), 2)

        text = f"Card {i + 1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = x + (w - text_size[0]) + 2
        text_y = y + (h + text_size[1]) + 2
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    cv2.imshow('Detected Cards', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main('Utils/Set01.jpg')

