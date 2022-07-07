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


def showImage(image):
    images = np.concatenate((image, image), axis=1)
    cv2.imshow("Image", images)
    cv2.waitKey(0)


def main(args):
    dir_path = args[0]
    names = getDirImages(dir_path)

    for name in names:
        image = cv2.imread(dir_path + name, cv2.IMREAD_COLOR)
        showImage(image)


if __name__ == "__main__":
    main(sys.argv[1:])