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


def medianBlur(image):
    processed = cv2.medianBlur(image, 3)
    showImage(image, processed)


def frequencyLowPass(image):
    kernel = np.ones((35, 35), np.float32) / 25
    processed = cv2.filter2D(image, -1, kernel)
    processed = image - processed
    showImage(image, processed)


def gaussian(image):
    processed = cv2.GaussianBlur(image, (7, 7), 0)
    showImage(image, processed)


def laplacian(image):
    processed = cv2.Laplacian(processed, cv2.CV_64F, ksize=3)
    showImage(image, processed)


def sobel(image):
    processedX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    processedY = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    processed = processedX
    showImage(image, processed)


def canny(image):
    processed = cv2.medianBlur(image, 3)
    processed = cv2.Canny(processed, 10, 5)
    showImage(image, processed)


def fourier(image):
    original = np.copy(image)
    print(original is image)
    f = np.fft.fft2(image)
    fshift1 = np.fft.fftshift(f)

    magnitude_specttrum = 20 * np.log(np.abs(fshift1))

    f_ishift = np.fft.ifftshift(fshift1)
    img_back = np.fft.fft2(f_ishift)
    img_back = np.abs(img_back)

    showImage(original, img_back)


def main(args):
    in_path = args[0]
    # out_path = args[1]
    images = getDirImages(in_path)

    for image_name in images:
        image = cv2.imread(in_path + image_name, 0)
        fourier(image)


if __name__ == "__main__":
    main(sys.argv[1:])
