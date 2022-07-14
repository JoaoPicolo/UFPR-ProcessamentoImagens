import sys
import cv2
import random
import numpy as np


def processArgs(args):
    in_path = args[0]
    noise_lvl = args[1]
    filter_name = args[2]
    out_path = args[3]

    return in_path, noise_lvl, filter_name, out_path


def sp_noise(image, prob):
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
    return output


def applyMeanFilter(image, noise_lvl):
    noise_image = sp_noise(image, noise_lvl)
    return cv2.GaussianBlur(noise_image, (5, 5), 0)


def applyMedianFilter(image, noise_lvl):
    noise_image = sp_noise(image, noise_lvl)
    return cv2.medianBlur(noise_image, 7)


def applyStackingFilter(image, noise_lvl, layers):
    if layers == 0:
        return image

    stacked_img = np.zeros(image.shape, np.uint8)

    # layers - 1 since the first layer is the copy of the image
    for _ in range(0, layers):
        noise_image = sp_noise(image, noise_lvl)
        noise_image = cv2.divide(noise_image, layers)
        stacked_img = cv2.add(noise_image, stacked_img)

    return stacked_img


def filterImage(image, noise_lvl, filter_name):
    if filter_name == '[0]':
        return applyMeanFilter(image, noise_lvl)
    elif filter_name == '[1]':
        return applyMedianFilter(image, noise_lvl)
    elif filter_name == '[2]':
        return applyStackingFilter(image, noise_lvl, 2)
    else:
        print("Please enter a valid filter option: [0], [1], [2]")
        return []


def main(args):
    in_path, noise_lvl, filter_name, out_path = processArgs(args)

    in_image = cv2.imread(in_path, 0)
    out_image = filterImage(in_image, float(noise_lvl), filter_name)

    if len(out_image):
        # psnr = cv2.PSNR(in_image, out_image)
        #print(
        #    f"For image { in_path } and level { noise_lvl } PSNR is { round(psnr, 3) }")
        # cv2.imwrite(out_path, out_image)

        cv2.imshow("image", in_image)
        cv2.waitKey(0)
        cv2.imshow("Filtered", out_image)
        cv2.waitKey(0)

        


if __name__ == "__main__":
    main(sys.argv[1:])
