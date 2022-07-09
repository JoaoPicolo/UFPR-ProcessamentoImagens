import cv2
import statistics
import numpy as np
import matplotlib.pyplot as plt


def plotRGBHistogram(image):
    b_hist = cv2.calcHist([image], [2], None, [256], [0, 255])
    g_hist = cv2.calcHist([image], [1], None, [256], [0, 255])
    r_hist = cv2.calcHist([image], [0], None, [256], [0, 255])

    plt.subplot(4, 1, 1)
    plt.imshow(image)
    plt.title('image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(4, 1, 2)
    plt.plot(r_hist, color='r')
    plt.xlim([0, 255])
    plt.title('red histogram')

    plt.subplot(4, 1, 3)
    plt.plot(g_hist, color='g')
    plt.xlim([0, 255])
    plt.title('green histogram')

    plt.subplot(4, 1, 4)
    plt.plot(b_hist, color='b')
    plt.xlim([0, 255])
    plt.title('blue histogram')

    plt.tight_layout()
    plt.show()


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

    h_avg, s_avg, v_avg = int(statistics.mean(h_list)), int(
        statistics.mean(s_list)), int(statistics.mean(v_list))

    lower_bound = (h_avg - 30, s_avg - 30, v_avg - 30)
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
