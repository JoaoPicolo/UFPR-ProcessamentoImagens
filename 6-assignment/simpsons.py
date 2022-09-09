import os

import cv2
import numpy as np

from skimage import feature

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# TODO: Falar no relatorio que nao usou concavidade pq nao faz sentido
# TODO: Falar no relatorio que nao usou textura pq nao faz sentido
# TODO: Tentar aplicar zoneamento


CLASS_MAP = {
    'bart': 0,
    'family': 1,
    'homer': 2,
    'lisa': 3,
    'maggie': 4,
    'marge': 5
}


def processImageName(dir_path, image_name):
    information = {}

    split_name = image_name.split('.')
    value = split_name[0]
    img_class = CLASS_MAP[value[:-3]]

    information["class"] = img_class
    information["path"] = dir_path + '/' + image_name

    return information


def readImages(dir_path):
    images = []

    for filename in os.listdir(dir_path):
        if ".bmp" in filename:
            information = processImageName(dir_path, filename)
            images.append(information)

    return images


# TODO: Testar binarizado
def applyHuMoments(images):
    x_values, y_values = [], []

    for image_data in images:
        image = cv2.imread(image_data["path"], 0)
        moments = cv2.HuMoments(cv2.moments(image)).flatten()
        x_values.append(moments)
        y_values.append(image_data["class"])

    return x_values, y_values


# TODO: Fazer rodar na KNN
# TODO: Testar binarizado
def applyContours(images):
    x_values, y_values = [], []

    for image_data in images:
        image = cv2.imread(image_data["path"], 0)
        contours, _ = cv2.findContours(
            image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        x_values.append(contours)
        y_values.append(image_data["class"])

    return x_values, y_values


# TODO: Testar binarizado
# TODO: Testar com cor
# TODO: Testar com canal
def applyHistogramGray(images):
    x_values, y_values = [], []

    for image_data in images:
        image = cv2.imread(image_data["path"], 0)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        x_values.append(hist)
        y_values.append(image_data["class"])

    return x_values, y_values


# TODO: Mesmo do de cima
def applyHistogramColor(images):
    x_values, y_values = [], []

    for image_data in images:
        print(image_data["path"])
        image = cv2.imread(image_data["path"])
        hist = cv2.calcHist([image], [0, 1, 2], None, [64, 64, 64], [
                            0, 255, 0, 255, 0, 255])
        hist = cv2.normalize(hist, hist).flatten()
        x_values.append(hist)
        y_values.append(image_data["class"])

    return x_values, y_values


# TODO: Mesmo de cima
# TODO: Testar com outros tamanhos de image
# TODo: Testar em outros axis
# Default is x-axis
def applyProjectedHistogramGray(images, dim=(100, 100), axis=0):
    x_values, y_values = [], []

    for image_data in images:
        image = cv2.imread(image_data["path"], 0)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        hist = np.sum(image, axis=axis).astype(int).flatten()
        x_values.append(hist)
        y_values.append(image_data["class"])

    return x_values, y_values


# TODO: Mesmo de cima
# Default is x-axis
def applyProjectedHistogramColor(images, dim=(100, 100), axis=0):
    x_values, y_values = [], []

    for image_data in images:
        image = cv2.imread(image_data["path"])
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        hist = np.sum(image, axis=axis).astype(int).flatten()
        x_values.append(hist)
        y_values.append(image_data["class"])

    return x_values, y_values


# TODO: Testar com outros tamanhos de image
# TODO: Testar diferentes numeros de coisa e radio
def applyLBP(images, dim=(100, 100)):
    x_values, y_values = [], []

    for image_data in images:
        image = cv2.imread(image_data["path"], 0)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        lbp = feature.local_binary_pattern(
            image, 8, 8*3, method="uniform").flatten()
        x_values.append(lbp)
        y_values.append(image_data["class"])

    return x_values, y_values


def getSets(dir_path, method):
    images = readImages(dir_path)
    x_values, y_values = [], []

    if method == 'HuMoments':
        x_values, y_values = applyHuMoments(images)
    elif method == 'Contours':
        x_values, y_values = applyContours(images)
    elif method == 'HistogramGray':
        x_values, y_values = applyHistogramGray(images)
    elif method == 'HistogramColor':
        x_values, y_values = applyHistogramColor(images)
    elif method == 'ProjectedHistogramGray':
        x_values, y_values = applyProjectedHistogramGray(images)
    elif method == 'ProjectedHistogramColor':
        x_values, y_values = applyProjectedHistogramColor(images)
    elif method == 'LBP':
        x_values, y_values = applyLBP(images)
    else:
        print("Please provide a valid method")
        exit(0)

    return x_values, y_values


def knnClassify(X_train, y_train, X_test, y_test, neighbors, metric):
    neigh = KNeighborsClassifier(n_neighbors=neighbors, metric=metric)

    print('Fitting knn')
    neigh.fit(X_train, y_train)

    print('Predicting...')
    y_pred = neigh.predict(X_test)

    print('Accuracy: ',  neigh.score(X_test, y_test))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test, y_pred))


def main():
    X_train, y_train = getSets('./train', 'ProjectedHistogramColor')
    X_test, y_test = getSets('./validation', 'ProjectedHistogramColor')

    knnClassify(X_train, y_train, X_test, y_test, 1, 'euclidean')


if __name__ == "__main__":
    main()
