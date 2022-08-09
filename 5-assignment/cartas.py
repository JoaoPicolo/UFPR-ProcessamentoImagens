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


def getImageLines(image, idx):
    _, th2 = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((16, 16), np.uint8)
    dilated = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
    dilated = cv2.bitwise_not(dilated)

    # Gets hist on y-axis
    line_sum = np.sum(th2, axis=1).astype(int).tolist()
    line_sum = line_sum / np.linalg.norm(line_sum)

    # Gets reference values and threshold
    median_hist = np.median(line_sum)
    avg_hist = np.average(line_sum)
    dvt_hist = np.std(line_sum)
    var_hist = np.var(line_sum)

    # Gets histogram peaks
    # Height is the number of words
    # Distance is the height of the line
    peaksPos, _ = sg.find_peaks(
        line_sum, height=var_hist, distance=100, prominence=var_hist)
    peaksPos = peaksPos.tolist()
    peaksPos.pop()

    print("\n\nBefore, After, average, median, std, var")
    print(len(peaksPos), len(peaksPos), avg_hist,
          median_hist, dvt_hist, var_hist)

    height, width = image.shape
    for j in range(height):
        if j in peaksPos or j+1 in peaksPos or j-1 in peaksPos:
            for i in range(width):
                image[j][i] = 0

    cv2.imwrite("./results/image"+str(idx)+".jpg", image)

    # plt.plot(line_sum)
    # plt.plot(peaksPos, line_sum[peaksPos], "x")
    # plt.show()

    return len(peaksPos)


def countLines():
    letters = getImagesInfo("./training/")
    correct = 0

    for idx, letter in enumerate(letters):
        image = cv2.imread(letter["image"], 0)
        total = getImageLines(image, idx)

        writer = letter["writer"]
        lines = int(letter["lines"])
        print("Writer, calculated, real")
        print(f"c{ writer } { total } { lines }")

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
