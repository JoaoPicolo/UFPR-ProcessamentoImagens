import sys
import cv2
import statistics as stats


<<<<<<< HEAD
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
=======
def equalizeImage(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img
>>>>>>> b63336150e89852f125e6839a68a30d1ba496efc


def getHSVChannels(image):
    h, s, v = cv2.split(image)
    h, s, v = h.flatten(), s.flatten(), v.flatten()

    return h, s, v

<<<<<<< HEAD
    result = cv2.bitwise_and(image, image, mask=mask)
    showImage(image_eq, result)
=======
>>>>>>> b63336150e89852f125e6839a68a30d1ba496efc

def getMeans(h, s, v):
    h_mean = int(stats.mean(h))
    s_mean = int(stats.mean(s))
    v_mean = int(stats.mean(v))

    return h_mean, s_mean, v_mean


def getMedians(h, s, v):
    h_median = int(stats.median(h))
    s_median = int(stats.median(s))
    v_median = int(stats.median(v))

    return h_median, s_median, v_median


def processImageHSV(image_name):
    image = cv2.imread(image_name)
    processed_img = cv2.imread(image_name)
    processed_img = equalizeImage(processed_img)

    hsv_image = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
    h, s, v = getHSVChannels(hsv_image)
    h_mean, s_mean, v_mean = getMeans(h, s, v)
    h_median, s_median, v_median = getMedians(h, s, v)
    h_dist, s_dist, v_dist = abs(
        h_mean - h_median), abs(s_mean - s_median), abs(v_mean - v_median)

    lower_bound = (h_median - h_dist*10, s_median -
                   s_dist*4, v_median - v_dist*4)
    upper_bound = (h_median + h_dist*4, s_median +
                   s_dist*8, v_median + v_dist*2)

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
