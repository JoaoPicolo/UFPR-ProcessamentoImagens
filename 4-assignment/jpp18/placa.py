import sys
import cv2
import numpy as np


def main(args):
    in_path = args[0]
    out_path = args[1]

    image = cv2.imread(in_path, 0)

    equalized = cv2.medianBlur(image, 3)
    equalized = cv2.GaussianBlur(equalized, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(32, 32))
    equalized = clahe.apply(equalized)

    fshift = np.fft.fftshift(np.fft.fft2(equalized))
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    magnitude_specttrum = 0.09 * np.log(np.abs(f_ishift))
    out_img = np.real(img_back * magnitude_specttrum)

    cv2.imwrite(out_path, out_img)


if __name__ == "__main__":
    main(sys.argv[1:])
