import sys
import cv2


def main(args):
    in_path = args[0]
    out_path = args[1]

    image = cv2.imread(in_path, 0)
    equalized = cv2.medianBlur(image, 3)
    equalized = cv2.GaussianBlur(equalized, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(32, 32))
    out_img = clahe.apply(equalized)
    cv2.imwrite(out_path, out_img)


if __name__ == "__main__":
    main(sys.argv[1:])
