import os
import sys
import cv2
import numpy as np

def getDirImages(dir_path):
    images = []
    for filename in os.listdir(dir_path):
        if ".jpg" in filename:
            images.append(filename)

    return images


def showImage(fst_image, scd_image):
    images = np.concatenate((fst_image, scd_image), axis=1)
    cv2.imshow("Image", images)
    cv2.waitKey(0)

def main(args):
    in_path = args[0]
    # out_path = args[1]
    images = getDirImages(in_path)

    for image_name in images:
        image = cv2.imread(in_path + image_name, 0)
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(32, 32))
        equalized = cv2.medianBlur(image, 3)
        equalized = cv2.GaussianBlur(equalized, (5, 5), 0)
        equalized = clahe.apply(equalized)
        showImage(image, equalized)


if __name__ == "__main__":
    main(sys.argv[1:])
