# Processamento de Imagens - 2021/02 - Erick Sales, Felipe Augusto, Helen Machado, Juan Luiz
import os
import cv2
import glob
import imutils
from nn import nnClassify
from svm import svmClassify
from imutils.contours import sort_contours


def removeAntigas():
    # Cria o diretório caso não exista
    try:
        os.makedirs("./imagens/")
    except FileExistsError:
        pass
    # remove todos os arquivos do diretório
    files = glob.glob('./imagens/*')
    for f in files:
        os.remove(f)


def getImageAndNames():
    # Para normalizar o filepath entre sistemas operacionais
    # filepath = unicodedata.normalize('NFKD', uni_img).encode('ascii', 'ignore')
    # filepath = filepath.decode('utf-8')

    files = glob.glob('./imagens/*')
    imgNames = []
    loadedImgs = []
    for f in files:
        imgNames.append(f)
        loadedImgs.append(cv2.imread(f))

    return loadedImgs, imgNames


def separa(image):
    # Carrega a imagem em tons de cinza e usa o blur para limpar a imagem
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Usa o algoritimo de canny para encontrar bordas e depois contorna-las
    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    removeAntigas()
    # Percorrer a lista de objetos contornados
    for c in cnts:
        # pegando coordenadas, largura e altura.
        (x, y, w, h) = cv2.boundingRect(c)

        # filtrando por tamanho cada objeto, para que não seja nem muito grande nem muito pequeno
        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            roi = gray[y:y + h, x:x + w]
            # extrai o caractere e aplica o threshold usando o algoritimo de otsu
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # pega o tamanho da imagem após o otsu e aplica um rezise proporcional
            (tH, tW) = thresh.shape
            if tW > tH:
                thresh = imutils.resize(thresh, width=28)
            else:
                thresh = imutils.resize(thresh, height=28)

            # escreve as imagens no diretório. o nome do arquivo será composto de informações uteis para quando recuperarmos no futuro
            (tH, tW) = thresh.shape
            dX = int(max(0, 28 - tW) / 2.0)
            dY = int(max(0, 28 - tH) / 2.0)

            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                        left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            padded = cv2.resize(padded, (28, 28))
            norm = cv2.normalize(padded, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite("./imagens/y=" + str(y) + "-h=" + str(h) + "-x=" + str(x) + "-w=" + str(w) + ".jpg", norm)


def marcacaoDaImgSVM(image):
    croppedimg, croppedImgPaths = getImageAndNames()
    borderSize = 40
    padded2 = cv2.copyMakeBorder(image, top=borderSize, bottom=borderSize,
                                 left=borderSize, right=borderSize, borderType=cv2.BORDER_CONSTANT,
                                 value=(255, 255, 255, 255))

    for name in croppedImgPaths:
        value = svmClassify(name)
        # Em geral o caminho de leitura no linux é diferente
        arr = ((name.replace("./imagens/y=", ""))
               .replace("./imagens\\y=", "")
               .replace("h=", "")
               .replace("x=", "")
               .replace("w=", "")
               .replace(".jpg", "")
               .split("-"))
        (y, h, x, w) = [int(numeric_string) for numeric_string in arr]
        cv2.rectangle(padded2, (x + borderSize, y + borderSize), (x + w + borderSize, y + h + borderSize), (0, 0, 240),
                      1)
        cv2.putText(padded2, str(value), (borderSize + x - 10, borderSize + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (240, 0, 0), 2)

    cv2.imshow("Predicao SVM", padded2)


def marcacaoDaImgNN(image):
    croppedimg, croppedImgPaths = getImageAndNames()
    borderSize = 40
    padded2 = cv2.copyMakeBorder(image, top=borderSize, bottom=borderSize,
                                 left=borderSize, right=borderSize, borderType=cv2.BORDER_CONSTANT,
                                 value=(255, 255, 255, 255))

    for name in croppedImgPaths:
        value = nnClassify(name)
        arr = ((name.replace("./imagens/y=", ""))
               .replace("./imagens\\y=", "")
               .replace("h=", "")
               .replace("x=", "")
               .replace("w=", "")
               .replace(".jpg", "")
               .split("-"))
        (y, h, x, w) = [int(numeric_string) for numeric_string in arr]
        cv2.rectangle(padded2, (x + borderSize, y + borderSize), (x + w + borderSize, y + h + borderSize), (0, 0, 240),
                      1)
        cv2.putText(padded2, str(value), (borderSize + x - 10, borderSize + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (240, 0, 0), 2)

    cv2.imshow("Predicao Rede Neural", padded2)
