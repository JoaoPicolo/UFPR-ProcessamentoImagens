import os
import sys
import cv2
import numpy as np


def getDirImages(dir_path):
    images = []
    for filename in os.listdir(dir_path):
        if ".png" in filename:
            images.append(filename)

    return images


def showImage(fst_image, scd_image):
    images = np.concatenate((fst_image, scd_image), axis=1)
    cv2.imshow("Image", images)
    cv2.waitKey(0)


def processImage(image_name):
    image = cv2.imread(image_name)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    h, s, v = sorted(h.flatten()), sorted(s.flatten()), sorted(v.flatten())
    h_avg, s_avg, v_avg = np.average(h), np.average(s), np.average(v)

    lower_bound = (h_avg - 20, s_avg - 20, v_avg - 20)
    upper_bound = (h_avg + 10, s_avg + 10, v_avg + 10)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    result = cv2.bitwise_and(image, image, mask=mask)
    result = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)

    showImage(image, result)


def main(args):
    dir_path = args[0]
    names = getDirImages(dir_path)

    for name in names:
        processImage(dir_path + name)


if __name__ == "__main__":
    main(sys.argv[1:])