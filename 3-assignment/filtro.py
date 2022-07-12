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


def stackingFilter(image, layers):
    stacked_img = np.copy(image) / layers
    height, width, channels = image.shape

    # layers - 1 since the first layer is the copy of the image
    for iteration in range(layers - 1):
        for wdt_px in range(width):
            for hgt_px in range(height):
                rgb_sum = np.add(stacked_img[hgt_px][wdt_px], image[hgt_px][wdt_px])
                stacked_img[hgt_px][wdt_px] = rgb_sum

    return stacked_img


def filterImage(image, filter_name):
    if filter_name == '[0]':
        return cv2.GaussianBlur(image, (5,5), 0)
    elif filter_name == '[1]':
        return cv2.medianBlur(image, 7)
    elif filter_name == '[2]':
        return stackingFilter(image, 3)
    else:
        print("Please enter a valid filter option: [0], [1], [2]")
        return []


def generateReport(in_path, noise_lvl):
    print()


def main(args):
    in_path, noise_lvl, filter_name, out_path = processArgs(args)
    
    in_image = cv2.imread(in_path) 
    noise_image = sp_noise(in_image, float(noise_lvl))
    out_image = filterImage(noise_image, filter_name)

    if len(out_image):
        # psnr = cv2.PSNR(in_image, out_image)
        # print(f"For image { in_path } and level { noise_lvl } PSNR is { round(psnr, 3) }")
        # cv2.imwrite(out_path, out_image)

        cv2.imshow("Noised", noise_image)
        cv2.waitKey(0)
        cv2.imshow("Filtered", out_image)
        cv2.waitKey(0)
    


if __name__ == "__main__":
    main(sys.argv[1:])