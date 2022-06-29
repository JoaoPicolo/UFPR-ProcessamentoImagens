# Ler todas as imagens do diretorio corrente
# Para cada imagem_original:
#   calcular histograma e normalizar
#   Para cada imagem_atual:
#       calcular histograma e normalizar
#       calcular resultado em cada um dos 4 metodos
#       atualizar o resultado no objeto que guarda as infos para cada metodo


import os
import cv2


IMG_EXT = ".bpm"

BGR = [0, 1, 2]   # CV2 read BGR, not RGB

CORRELATION = 0   # The higher, more accurate
CHI_SQUARE = 1    # The lower, more accurate
INTERSECTION = 2  # The higher, more accurate
BHATTACHARYYA = 3 # The lower, more accurate

def getDirImages(dirPath):
    images = []
    for filename in os.listdir(dirPath):
        if ".bmp" in filename:
            images.append(filename)

    return images


def getNormalizedHist(image):
    hist = cv2.calcHist([image], BGR, None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist)

    return hist


def getClass(name):
    base_name = name.split('.')[0]
    class_name = ''.join([char for char in base_name if not char.isdigit()])
    
    
    return class_name


def calculateHistograms(dir_images):
    histograms = {}

    for image_name in dir_images:
        img = cv2.imread(image_name)
        hist = getNormalizedHist(img)
        histograms[image_name] = {
            "hist": hist,
            "class": getClass(image_name)
        }

    return histograms


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


def calculatePrecisions(histograms):
    correlation_hits = 0
    chi_square_hits = 0
    intersection_hits = 0
    bhattacharyya_hits = 0

    for test in histograms:
        test_hist = histograms[test]["hist"]
        test_class = histograms[test]["class"]

        correlation_candidate = { "class": '', "value": 0 }
        chi_square_candidate = { "class": '', "value": 0 }
        intersection_candidate = { "class": '', "value": 0 }
        bhattacharyya_candidate = { "class": '', "value": 0 }

        for current in histograms:
            current_hist = histograms[current]["hist"]
            current_class = histograms[current]["class"]

            if test != current:
                correlation = cv2.compareHist(test_hist, current_hist, CORRELATION)
                updatePrecision(correlation_candidate, current_class, correlation, True)

                chi_square = cv2.compareHist(test_hist, current_hist, CHI_SQUARE)
                updatePrecision(chi_square_candidate, current_class, chi_square, False)
                
                intersection = cv2.compareHist(test_hist, current_hist, INTERSECTION)
                updatePrecision(intersection_candidate, current_class, intersection, True)

                bhattacharyya = cv2.compareHist(test_hist, current_hist, BHATTACHARYYA)
                updatePrecision(bhattacharyya_candidate, current_class, bhattacharyya, False)

        correlation_hits += (1 if (correlation_candidate["class"] == test_class) else 0)
        chi_square_hits += (1 if (chi_square_candidate["class"] == test_class) else 0)
        intersection_hits += (1 if (intersection_candidate["class"] == test_class) else 0)
        bhattacharyya_hits += (1 if (bhattacharyya_candidate["class"] == test_class) else 0)
        # print(test_class)
        # print(correlation_candidate)
        # print(correlation_hits)
        # print(chi_square_candidate)
        # print(chi_square_hits)
        # print(intersection_candidate)
        # print(intersection_hits)
        # print(bhattacharyya_candidate)
        # print(bhattacharyya_hits)
        # exit(0)

    print(correlation_hits)
    print(chi_square_hits)
    print(intersection_hits)
    print(bhattacharyya_hits)


def main():
    dir_images = getDirImages('.')
    histograms = calculateHistograms(dir_images)
    calculatePrecisions(histograms)

if __name__ == "__main__":
    main()
