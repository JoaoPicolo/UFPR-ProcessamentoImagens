import os
import sys
import cv2
import statistics
import numpy as np
import matplotlib.pyplot as plt


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

def equalizeImage(image):
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
    image_eq = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    return image_eq


def processImageHSVStats(image_name):
    image = cv2.imread(image_name)
    image_eq = equalizeImage(image)
    hsv_image = cv2.cvtColor(image_eq, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    h, s, v = h.flatten(), s.flatten(), v.flatten()
    h_avg, s_avg, v_avg = int(statistics.mean(h)), int(statistics.mean(s)), int(statistics.mean(v))

    lower_bound = (h_avg - 40, s_avg - 40, v_avg - 40)
    upper_bound = (h_avg + 40, s_avg + 40, v_avg + 40)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    result = cv2.bitwise_and(image, image, mask=mask)
    showImage(image_eq, result)


def processImageHSVRange(image_name):
    image = cv2.imread(image_name)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    h, s, v = h.flatten(), s.flatten(), v.flatten()

    h_list, s_list, v_list = [], [], []
    for idx, hue in enumerate(h):
        if hue < 30 or hue > 85:
            h_list.append(h[idx])
            s_list.append(s[idx])
            v_list.append(v[idx])

    h_avg, s_avg, v_avg = int(statistics.mean(h_list)), int(statistics.mean(s_list)), int(statistics.mean(v_list))

    lower_bound = (h_avg - 30, s_avg -  30, v_avg - 30)
    upper_bound = (h_avg + 30, s_avg + 30, v_avg + 30)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    result = cv2.bitwise_and(image, image, mask=mask)
    showImage(image, result)


def processImageBGRStats(image_name):
    bgr_image = cv2.imread(image_name)
    b, g, r = cv2.split(bgr_image)
    b, g, r = b.flatten(), g.flatten(), r.flatten()
    b_avg, g_avg, r_avg = np.average(b), np.average(g), np.average(r)

    lower_bound = (b_avg - 35, g_avg - 35, r_avg - 35)
    upper_bound = (b_avg + 35, g_avg + 35, r_avg + 35)
    mask = cv2.inRange(bgr_image, lower_bound, upper_bound)

    result = cv2.bitwise_and(bgr_image, bgr_image, mask=mask)
    showImage(bgr_image, result)


def main(args):
    dir_path = args[0]
    names = getDirImages(dir_path)

    for name in names:
        processImageHSVStats(dir_path + name)

if __name__ == "__main__":
    main(sys.argv[1:])