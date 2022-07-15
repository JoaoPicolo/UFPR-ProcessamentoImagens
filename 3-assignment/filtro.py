import sys
import cv2
import random
import numpy as np
import time


def processArgs(args):
    in_path = args[0]
    noise_lvl = args[1]
    filter_name = args[2]
    out_path = args[3]

    return in_path, noise_lvl, filter_name, out_path


def sp_noise(image, prob):
    st = time.time()
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]

    et = time.time()
    return output, et-st


def filter2DReport(image, noise_lvl):
    # (1, 1) a (5, 5)
    #kernel = np.ones((5, 5), np.float32) / 25
    # return cv2.filter2D(noise_image, -1, kernel)
    kernel_size = 0

    psnr_max = -1
    psnr_current = 0

    report = open(f"filter2D-{noise_lvl}.txt", "w")

    while psnr_current > psnr_max:
        psnr_max = psnr_current
        kernel_size += 1
        noise_image, noise_time = sp_noise(image, noise_lvl)

        st = time.time()
        kernel = np.ones((kernel_size, kernel_size), np.float32) / 25
        filtered_img = cv2.filter2D(noise_image, -1, kernel)
        et = time.time()
        exe_time = et - st
        psnr_current = cv2.PSNR(image, filtered_img)
        report.write(
            f"Kernel { kernel_size } - PSNR { psnr_current } - TIME { exe_time } - PSNR/TIME { round(psnr_current/exe_time, 3) }\n")
        print(
            f"Kernel { kernel_size } - PSNR { psnr_current } - TIME { exe_time } - PSNR/TIME { round(psnr_current/exe_time, 3) }\n")

    report.close()


def applyMeanFilter(image, noise_lvl):
    noise_image, time = sp_noise(image, noise_lvl)

    # (1,1) a (15, 15)
    # return cv2.blur(noise_image, (4, 4))

    # (5,5) a (35, 35) -> Soh impar
    # return cv2.GaussianBlur(noise_image, (35, 35), 0)


def applyMedianFilter(image, noise_lvl):
    noise_image, noise_time = sp_noise(image, noise_lvl)

    # 1 a 31 -> so impar
    return cv2.medianBlur(noise_image, 31)


def applyStackingFilter(image, noise_lvl, layers):
    if layers == 0:
        return []

    stacked_img = np.zeros(image.shape, np.float32)

    for _ in range(0, layers):
        noise_image, noise_time = sp_noise(image, noise_lvl)
        noise_image = np.divide(noise_image.astype(np.float32), float(layers))
        stacked_img = np.add(stacked_img, noise_image)

    return stacked_img.astype(np.uint8)


def filterImage(image, noise_lvl, filter_name):
    if filter_name == '[0]':
        return applyMeanFilter(image, noise_lvl)
    elif filter_name == '[1]':
        return applyMedianFilter(image, noise_lvl)
    elif filter_name == '[2]':
        return applyStackingFilter(image, noise_lvl, 20)
    else:
        print("Please enter a valid filter option: [0], [1], [2]")
        return []


def main(args):
    in_path, noise_lvl, filter_name, out_path = processArgs(args)

    in_image = cv2.imread(in_path)

    filter2DReport(in_image, float(noise_lvl))
    exit(0)

    out_image = filterImage(in_image, float(noise_lvl), filter_name)

    if len(out_image):
        psnr = cv2.PSNR(in_image, out_image)
        print(
            f"For image { in_path } and level { noise_lvl } PSNR is { round(psnr, 3) }")
        cv2.imwrite(out_path, out_image)


if __name__ == "__main__":
    main(sys.argv[1:])
