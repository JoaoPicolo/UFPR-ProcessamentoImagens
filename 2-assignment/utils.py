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