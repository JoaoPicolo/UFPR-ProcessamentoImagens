import os
import sys
import cv2
from cv2 import threshold
import numpy as np
import matplotlib.pyplot as plt


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


def countLines():
    letters = getImagesInfo("./training/")

    for letter in letters:
        image = cv2.imread(letter["image"], 0)
        image = cv2.equalizeHist(image)
        _, th2 = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th2 = cv2.bitwise_not(th2)

        # Gets hist on y-axis
        line_sum = np.sum(th2, axis=1)

        # Gets reference values and threshold
        median_hist = np.median(line_sum)
        avg_hist = np.average(line_sum)
        threshold = abs(median_hist - avg_hist)
        # Zero all values below threshld
        line_sum[line_sum < threshold] = 0

        line_sum = line_sum.tolist()
        plt.plot(line_sum)
        plt.show()


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
