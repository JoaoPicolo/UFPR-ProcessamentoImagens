import sys
import cv2
import random
import numpy as np
import time


def processArgs(args):
    in_path = args[0]

    return in_path


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


def stackingFilter(image, noise_lvl, layers):
    if layers == 0:
        return [], 0

    stacked_img = np.zeros(image.shape, np.float32)
    gen_time = 0

    for _ in range(0, layers):
        noise_image, noise_time = sp_noise(image, noise_lvl)
        gen_time += noise_time
        noise_image = np.divide(noise_image.astype(np.float32), float(layers))
        stacked_img = np.add(stacked_img, noise_image)

    return stacked_img.astype(np.uint8), gen_time


def filter2DReport(image, noise_lvl):
    kernel_size = 0

    psnr_max = -1
    psnr_current = 0

    report = open(f"filter2D-{noise_lvl}.txt", "w")
    report.write(f"KERNEL;PSNR;EXE_TIME;PSNR/EXE_TIME;NOISE_TIME\n")

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
            f"{ kernel_size };{ psnr_current };{ exe_time };{ round(psnr_current/exe_time, 3) };{ noise_time }\n")
        print(f"{ kernel_size };{ psnr_current };{ exe_time };{ round(psnr_current/exe_time, 3) };{ noise_time }")

    report.close()


def blurReport(image, noise_lvl):
    kernel_size = 0

    psnr_max = -1
    psnr_current = 0

    report = open(f"blur-{noise_lvl}.txt", "w")
    report.write(f"KERNEL;PSNR;EXE_TIME;PSNR/EXE_TIME;NOISE_TIME\n")

    while psnr_current > psnr_max:
        psnr_max = psnr_current
        kernel_size += 1
        noise_image, noise_time = sp_noise(image, noise_lvl)

        st = time.time()
        filtered_img = cv2.blur(noise_image, (kernel_size, kernel_size))
        et = time.time()
        exe_time = et - st
        psnr_current = cv2.PSNR(image, filtered_img)
        report.write(
            f"{ kernel_size };{ psnr_current };{ exe_time };{ round(psnr_current/exe_time, 3) };{ noise_time }\n")
        print(f"{ kernel_size };{ psnr_current };{ exe_time };{ round(psnr_current/exe_time, 3) };{ noise_time }")

    report.close()


def gaussianReport(image, noise_lvl):
    kernel_size = -1

    psnr_max = -1
    psnr_current = 0

    report = open(f"gaussian-{noise_lvl}.txt", "w")
    report.write(f"KERNEL;PSNR;EXE_TIME;PSNR/EXE_TIME;NOISE_TIME\n")

    while psnr_current > psnr_max:
        psnr_max = psnr_current
        kernel_size += 2
        noise_image, noise_time = sp_noise(image, noise_lvl)

        st = time.time()
        filtered_img = cv2.GaussianBlur(
            noise_image, (kernel_size, kernel_size), 0)
        et = time.time()
        exe_time = et - st
        psnr_current = cv2.PSNR(image, filtered_img)
        report.write(
            f"{ kernel_size };{ psnr_current };{ exe_time };{ round(psnr_current/exe_time, 3) };{ noise_time }\n")
        print(f"{ kernel_size };{ psnr_current };{ exe_time };{ round(psnr_current/exe_time, 3) };{ noise_time }")

    report.close()


def medianReport(image, noise_lvl):
    kernel_size = -1

    psnr_max = -1
    psnr_current = 0

    report = open(f"median-{noise_lvl}.txt", "w")
    report.write(f"KERNEL;PSNR;EXE_TIME;PSNR/EXE_TIME;NOISE_TIME\n")

    while psnr_current > psnr_max:
        psnr_max = psnr_current
        kernel_size += 2
        noise_image, noise_time = sp_noise(image, noise_lvl)

        st = time.time()
        filtered_img = cv2.medianBlur(noise_image, kernel_size)
        et = time.time()
        exe_time = et - st
        psnr_current = cv2.PSNR(image, filtered_img)
        report.write(
            f"{ kernel_size };{ psnr_current };{ exe_time };{ round(psnr_current/exe_time, 3) };{ noise_time }\n")
        print(f"{ kernel_size };{ psnr_current };{ exe_time };{ round(psnr_current/exe_time, 3) };{ noise_time }")

    report.close()


def stackingReport(image, noise_lvl):
    kernel_size = 0
    exe_time = 0

    report = open(f"stacking-{noise_lvl}.txt", "w")
    report.write(f"KERNEL;PSNR;EXE_TIME;PSNR/EXE_TIME;NOISE_TIME\n")

    while exe_time < 0.8:
        kernel_size += 1

        st = time.time()
        filtered_img, gen_time = stackingFilter(
            image, noise_lvl, kernel_size)
        et = time.time()
        exe_time = et - st - gen_time
        psnr_current = cv2.PSNR(image, filtered_img)
        report.write(
            f"{ kernel_size };{ psnr_current };{ exe_time };{ round(psnr_current/exe_time, 3) };{ gen_time }\n")
        print(f"{ kernel_size };{ psnr_current };{ exe_time };{ round(psnr_current/exe_time, 3) };{ gen_time }")

    report.close()


def main(args):
    in_path = processArgs(args)
    in_image = cv2.imread(in_path)

    levels = [0.01, 0.02, 0.05, 0.07, 0.1]

    for level in levels:
        #print(f"\n\nFilter 2D report for noise {level}\n")
        #filter2DReport(in_image, level)

        #print(f"\n\nBlur report for noise {level}\n")
        #blurReport(in_image, level)

        #print(f"\n\nGaussian report for noise {level}\n")
        #gaussianReport(in_image, level)

        #print(f"\n\nMedian report for noise {level}\n")
        #medianReport(in_image, level)

        print(f"\n\nStacking report for noise {level}\n")
        stackingReport(in_image, level)


if __name__ == "__main__":
    main(sys.argv[1:])
