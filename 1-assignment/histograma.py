import os
import cv2

BGR = [0, 1, 2]    # CV2 read BGR, not RGB

CORRELATION = 0    # The higher, more accurate
CHI_SQUARE = 1     # The lower, more accurate
INTERSECTION = 2   # The higher, more accurate
BHATTACHARYYA = 3  # The lower, more accurate


def getDirImages(dirPath):
    images = []
    for filename in os.listdir(dirPath):
        if ".bmp" in filename:
            images.append(filename)

    return images


def getNormalizedHist(image):
    hist = cv2.calcHist([image], BGR, None, [8, 8, 8], [
                        0, 255, 0, 255, 0, 255])
    hist = cv2.normalize(hist, hist)

    return hist


def getClass(name):
    base_name = name.split('.')[0]
    class_name = ''.join([char for char in base_name if not char.isdigit()])

    return class_name


def getHistogramsAndClasses(dir_images):
    histograms = {}
    classes = []
    bigger_name = 0

    for image_name in dir_images:
        img = cv2.imread(image_name)
        hist = getNormalizedHist(img)
        image_class = getClass(image_name)

        bigger_name = max(bigger_name, len(image_class))

        histograms[image_name] = {
            "hist": hist,
            "class": image_class
        }

        if image_class not in classes:
            classes.append(image_class)

    return histograms, classes, bigger_name


def initConfusionMatrix(classes):
    matrices = {}

    for class_name in classes:
        matrices[class_name] = {}

        for option in classes:
            matrices[class_name][option] = 0

    return matrices


def updatePrecision(candidate, new_class, new_value, higher_metric):
    if candidate["class"]:
        if higher_metric and new_value > candidate["value"]:
            candidate["class"] = new_class
            candidate["value"] = new_value
        elif not higher_metric and new_value < candidate["value"]:
            candidate["class"] = new_class
            candidate["value"] = new_value
    else:
        candidate["class"] = new_class
        candidate["value"] = new_value


def getConfusionMatrices(histograms, classes):
    correlation_matrix = initConfusionMatrix(classes)
    chi_square_matrix = initConfusionMatrix(classes)
    intersection_matrix = initConfusionMatrix(classes)
    bhattacharyya_matrix = initConfusionMatrix(classes)

    for test in histograms:
        test_hist = histograms[test]["hist"]
        test_class = histograms[test]["class"]

        correlation_candidate = {"class": '', "value": 0}
        chi_square_candidate = {"class": '', "value": 0}
        intersection_candidate = {"class": '', "value": 0}
        bhattacharyya_candidate = {"class": '', "value": 0}

        for current in histograms:
            if test != current:
                current_hist = histograms[current]["hist"]
                current_class = histograms[current]["class"]

                correlation = cv2.compareHist(
                    test_hist, current_hist, CORRELATION)
                updatePrecision(correlation_candidate,
                                current_class, correlation, True)

                chi_square = cv2.compareHist(
                    test_hist, current_hist, CHI_SQUARE)
                updatePrecision(chi_square_candidate,
                                current_class, chi_square, False)

                intersection = cv2.compareHist(
                    test_hist, current_hist, INTERSECTION)
                updatePrecision(intersection_candidate,
                                current_class, intersection, True)

                bhattacharyya = cv2.compareHist(
                    test_hist, current_hist, BHATTACHARYYA)
                updatePrecision(bhattacharyya_candidate,
                                current_class, bhattacharyya, False)

        correlation_matrix[test_class][correlation_candidate["class"]] += 1
        chi_square_matrix[test_class][chi_square_candidate["class"]] += 1
        intersection_matrix[test_class][intersection_candidate["class"]] += 1
        bhattacharyya_matrix[test_class][bhattacharyya_candidate["class"]] += 1

    return correlation_matrix, chi_square_matrix, intersection_matrix, bhattacharyya_matrix


def printConfusionMatrix(method, matrix, bigger, samples):
    total = 0
    for item in matrix:
        total += matrix[item][item]

    space = " " * bigger

    print(f"{ method } accuracy: { total / samples } ({ total }/{ samples })")

    print(space, end="")
    for item in matrix:
        print(f"{ item }", end=" ")
    print()

    for line in matrix:
        print(f"{ line }" + " " * (bigger - len(line)), end=" ")
        for col in matrix:
            print(f"{ matrix[line][col] }", end=" ")
        print()

    print("\n\n")


def main():
    dir_images = getDirImages('.')
    histograms, classes, bigger_name = getHistogramsAndClasses(dir_images)
    correlation, chi_square, intersection, bhattacharrya = getConfusionMatrices(
        histograms, classes)

    printConfusionMatrix("Correlation", correlation, bigger_name, len(histograms))
    printConfusionMatrix("Chi-square", chi_square, bigger_name, len(histograms))
    printConfusionMatrix("Intersection", intersection, bigger_name, len(histograms))
    printConfusionMatrix("Bhattacharrya distance",
                         bhattacharrya, bigger_name, len(histograms))


if __name__ == "__main__":
    main()
