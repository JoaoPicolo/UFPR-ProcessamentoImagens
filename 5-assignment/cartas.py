import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
            information["image"] = dir_path + filename
            letters.append(information)

    return letters


def determineScore(arr, angle):
    data = nd.interpolation.rotate(arr, angle, reshape=False, order=0)
    histogram = np.sum(data, axis=1, dtype=float)
    score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
    return score


def correctRotation(image, delta=0.8, limit=5):
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


def processLines(image):
    # Gets hist on y-axis
    data = np.sum(image, axis=1).astype(int).tolist()
    data = data / np.linalg.norm(data)

    # Gets histogram peaks
    # Height is the number of words
    # Distance is the height of the line
    h, _ = image.shape
    peaks_pos, _ = sg.find_peaks(data, distance=(h*0.0283))

    return peaks_pos


def processImage(image, idx):
    rotated = correctRotation(image)

    _, th2 = cv2.threshold(
        rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((16, 16), np.uint8)
    dilated = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)

    peaks_pos = processLines(dilated)

    height, width = dilated.shape
    for j in range(height):
        if j in peaks_pos or j+1 in peaks_pos or j-1 in peaks_pos:
            for i in range(width):
                dilated[j][i] = 0

    cv2.imwrite("./results/image"+str(idx)+".jpg", dilated)

    # plt.plot(data)
    #plt.plot(peaks_pos, data[peaks_pos], "x")
    # plt.show()

    return len(peaks_pos)


def countLines():
    letters = getImagesInfo("./training/")
    correct = 0

    for idx, letter in enumerate(letters):
        image = cv2.imread(letter["image"], 0)
        total = processImage(image, idx)

        writer = letter["writer"]
        lines = int(letter["lines"])
        #print("Writer, calculated, real")
        #print(f"c{ writer } { total } { lines }")

        correct = correct + 1 if lines == total else correct

    print(f"Cartas corretas: { correct }/{ len(letters) }")


def main(args):
    op_type = args[0]

    if (op_type == "-l"):
        countLines()
    elif (op_type == "-w"):
        print("words")
    else:
        print("Please enter a valid option: -l or -w")


if __name__ == "__main__":
    main(sys.argv[1:])
