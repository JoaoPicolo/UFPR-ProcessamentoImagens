# Ler todas as imagens do diretorio corrente
# Para cada imagem_original:
#   calcular histograma e normalizar
#   Para cada imagem_atual:
#       calcular histograma e normalizar
#       calcular resultado em cada um dos 4 metodos
#       atualizar o resultado no objeto que guarda as infos para cada metodo


import os
import cv2


BGR = [0, 1, 2] # CV2 read BGR, not RGB

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


def calculateHistograms(dir_images):
    histograms = {}

    for image_name in dir_images:
        img = cv2.imread(image_name)
        hist = getNormalizedHist(img)
        histograms[image_name] = hist

    return histograms


def calculatePrecisions(histograms):
    for test_hist in histograms:
        for current_hist in histograms:
            if test_hist != current_hist:
                correlation = compareHist(test_hist, current_hist, "CV_COMP_CORREL")
                chi_square = compareHist(test_hist, current_hist, "CV_COMP_CHISQR")
                intersection = compareHist(test_hist, current_hist, "CV_COMP_INTERSECT")
                bhattacharyya = compareHist(test_hist, current_hist, "CV_COMP_BHATTACHARYYA")


def main():
    dir_images = getDirImages('.')
    histograms = calculateHistograms(dir_images)
    calculatePrecisions(histograms)

if __name__ == "__main__":
    main()
