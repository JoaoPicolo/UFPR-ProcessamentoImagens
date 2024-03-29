import os
import sys
import cv2
import numpy as np
import scipy.signal as sg
import scipy.ndimage as nd


def processImageName(image_name):
    information = {}

    split_name = image_name.split('_')
    writer = split_name[0][1:]
    number = split_name[1]
    lines = split_name[2].split('.')[0]

    information["writer"] = writer
    information["number"] = number
    information["lines"] = lines

    return information


def getImagesInfo(dir_path):
    letters = []

    for filename in os.listdir(dir_path):
        if ".jpg" in filename:
            information = processImageName(filename)
            information["image"] = filename
            letters.append(information)

    return letters


def determineScore(arr, angle):
    data = nd.interpolation.rotate(arr, angle, reshape=False, order=0)
    histogram = np.sum(data, axis=1, dtype=float)
    score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
    return score


def correctRotation(image, delta=2, limit=7):
    _, thresh = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        score = determineScore(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)

    return corrected


def saveImage(image, image_name, peaks_pos):
    height, width = image.shape
    for j in range(height):
        if j in peaks_pos or j+1 in peaks_pos or j-1 in peaks_pos:
            for i in range(width):
                image[j][i] = 0

    cv2.imwrite("./"+image_name, image)


def preprocessImage(image, k_size):
    rotated = correctRotation(image)

    _, th2 = cv2.threshold(
        rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((k_size, k_size), np.uint8)
    opened = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)

    return rotated, opened


def cropImage(image):
    image_cp = image
    h, w = image.shape
    top = -1
    bottom = -1

    n_objects = 0
    for i in range(h):
        for j in range(w):
            if image[i, j] == 0:
                _, image, _, rect = cv2.floodFill(
                    image, None, (j, i), n_objects)

                # Min height, so we won't consider commas, etc
                _, y_top, wdt, hgt = rect[0], rect[1], rect[2], rect[3]
                if hgt > 15 and wdt > 18:
                    top = y_top
                    break

        if top != -1:
            break

    for i in range(h-1, 0, -1):
        for j in range(w-1, 0, -1):
            if image[i, j] == 0:
                _, image, _, rect = cv2.floodFill(
                    image, None, (j, i), n_objects)

                # Min height, so we won't consider commas, etc
                _, y_top, wdt, hgt = rect[0], rect[1], rect[2], rect[3]
                if hgt > 15 and wdt > 18:
                    bottom = y_top + hgt
                    break

        if bottom != -1:
            break

    image_cp = image_cp[top:bottom, 0:w]
    return image_cp


def processLines(image):
    # Gets hist on y-axis
    image = cv2.bitwise_not(image)
    data = np.sum(image, axis=1).astype(int).tolist()
    data = data / np.linalg.norm(data)

    # Gets histogram peaks
    # Height is the number of words
    # Distance is the height of the line
    h, w = image.shape
    peaks_pos, _ = sg.find_peaks(data, height=0.0025, distance=(h*0.0283))

    return peaks_pos


def getImageLines(image):
    _, dilated = preprocessImage(image, 16)
    dilated = cropImage(dilated)
    peaks_pos = processLines(dilated)

    return len(peaks_pos)


def countLines():
    letters = getImagesInfo(".")
    correct = 0

    for letter in letters:
        image = cv2.imread(letter["image"], 0)
        total = getImageLines(image)

        writer = letter["writer"]
        lines = int(letter["lines"])
        print(f"c{ writer } { total } { lines }")

        correct = correct + 1 if lines == total else correct

    print(f"Cartas corretas: { correct }/{ len(letters) }")


def highlightWords():
    letters = getImagesInfo(".")

    for letter in letters:
        image = cv2.imread(letter["image"], 0)
        rotated, dilated = preprocessImage(image, 18)
        dilated = cv2.bitwise_not(dilated)
        rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB)

        borders, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        n_objects = 0
        for border in borders:
            x1, y1, wdt, hgt = cv2.boundingRect(border)
            if hgt > 15 and wdt > 18:
                n_objects += 1
                cv2.rectangle(rotated, (x1, y1),
                              (x1 + wdt, y1 + hgt), (255, 0, 0), 2)

        writer = letter["writer"]
        print(f"c{ writer } { n_objects }")
        cv2.imwrite("./c"+letter["writer"]+".jpg", rotated)


def main(args):
    op_type = args[0]

    if (op_type == "-l"):
        countLines()
    elif (op_type == "-w"):
        highlightWords()
    else:
        print("Please enter a valid option: -l or -w")


if __name__ == "__main__":
    main(sys.argv[1:])
