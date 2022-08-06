import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg


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
    correct = 0

    for letter in letters:
        image = cv2.imread(letter["image"], 0)
        _, th2 = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th2 = cv2.bitwise_not(th2)

        # Gets hist on y-axis
        line_sum = np.sum(th2, axis=1).astype(int).tolist()
        line_sum = line_sum / np.linalg.norm(line_sum)

        # Gets reference values and threshold
        median_hist = np.median(line_sum)
        avg_hist = np.average(line_sum)
        dvt_hist = np.std(line_sum)
        threshold = median_hist - dvt_hist

        # Gets histogram peaks
        peaksPos, peaksVal = sg.find_peaks(
            line_sum, height=0.0001, distance=100)
        peaksVal = peaksVal["peak_heights"]

        finalPos = []
        totalPeaks = len(peaksPos)
        for i in range(0, totalPeaks):
            if (peaksVal[i] > threshold):
                finalPos.append(peaksPos[i])

        # print(letter)
        #print(len(finalPos), threshold)

        # plt.plot(line_sum)
        #plt.plot(finalPos, line_sum[finalPos], "x")
        # plt.show()

        writer = letter["writer"]
        lines = int(letter["lines"])
        calculated = len(finalPos)
        print(f"c{ writer } { calculated } { lines }")
        correct = correct + 1 if lines == calculated else correct

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
