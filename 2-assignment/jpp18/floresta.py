import sys
import cv2
import numpy as np


def equalizeImage(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, image)
    return image


def getHSVChannels(image):
    h, s, v = cv2.split(image)
    h, s, v = h.flatten(), s.flatten(), v.flatten()

    return h, s, v


def getMeans(h, s, v):
    h_mean = int(np.mean(h))
    s_mean = int(np.mean(s))
    v_mean = int(np.mean(v))

    return h_mean, s_mean, v_mean


def getMedians(h, s, v):
    h_median = int(np.median(h))
    s_median = int(np.median(s))
    v_median = int(np.median(v))

    return h_median, s_median, v_median


def getDeviations(h, s, v):
    h_std = int(np.std(h))
    s_std = int(np.std(s))
    v_std = int(np.std(v))

    return h_std, s_std, v_std


def processImageHSV(image_name):
    image = cv2.imread(image_name)
    processed_img = cv2.imread(image_name)
    processed_img = equalizeImage(processed_img)

    hsv_image = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
    h, s, v = getHSVChannels(hsv_image)
    h_mean, s_mean, v_mean = getMeans(h, s, v)
    h_median, s_median, v_median = getMedians(h, s, v)
    h_std, s_std, v_std = getDeviations(h, s, v)
    h_dist, s_dist, v_dist = abs(
        h_mean - h_median), abs(s_mean - s_median), abs(v_mean - v_median)

    lower_bound = (
        h_median - h_std*1.2 + h_dist*1.1,
        s_median - s_std*0.8 + s_dist*1.6,
        v_median - v_std*1.9 + v_dist*1.5)
    upper_bound = (
        h_median + h_std*1.6 - h_dist*0.2,
        s_median + s_std*1.9 - s_dist*0.2,
        v_median + v_std*1.9 - v_dist*1.5)

    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)

    return result


def main(args):
    in_path = args[0]
    out_path = args[1]

    result = processImageHSV(in_path)
    cv2.imwrite(out_path, result)


if __name__ == "__main__":
    main(sys.argv[1:])
