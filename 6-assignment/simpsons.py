import os

import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


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


def applyHuMoments(images):
    x_values, y_values = [], []

    for image_data in images:
        image = cv2.imread(image_data["path"], 0)
        moments = cv2.HuMoments(cv2.moments(image)).flatten()
        x_values.append(moments)
        y_values.append(image_data["class"])

    return x_values, y_values


def getSets(dir_path, method):
    images = readImages(dir_path)
    x_values, y_values = [], []

    if method == 'huMoments':
        x_values, y_values = applyHuMoments(images)
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
    X_train, y_train = getSets('./train', 'huMoments')
    X_test, y_test = getSets('./validation', 'huMoments')

    knnClassify(X_train, y_train, X_test, y_test, 1, 'euclidean')


if __name__ == "__main__":
    main()
