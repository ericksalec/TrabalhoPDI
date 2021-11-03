#Processamento de Imagens - 2021/02 - Erick Sales, Felipe Augusto, Helen Machado, Juan Luiz
import cv2
import numpy as np
from matplotlib import pyplot as plt
import charSeparation
import glob

#Cria as projecoes verticais e horizontais da imagem de entrada
def projecao(pathImage=None, name = 0):

	#Leia a imagem e substitua-a por uma matriz operacional
	img=cv2.imread(pathImage, cv2.IMREAD_GRAYSCALE)
	#Converter imagem BGR para imagem em tons de cinza
	#Alterar os pontos entre a binarização da imagem (130255) a 255 (plano de fundo)
	ret,proj_vertical=cv2.threshold(img,130,255,cv2.THRESH_BINARY)
	proj_vertical_arr = proj_vertical

	#Retornar altura e largura
	(h,w)=proj_vertical.shape
	a = [0 for z in range(0, w)]
	#Calcula projecao vertical
	proj2= np.sum(proj_vertical_arr, 0)
	#Desenha a projecao veritxal
	#Registre os pontos max de cada coluna
	for j in range(0,w): #Percorrendo uma coluna
		for i in range(0,h): #Percorrendo a linha
			if proj_vertical[i,j]==0: #Se você mudar o ponto para preto
				a[j]+=1
				proj_vertical[i,j]=255 #Volte para fundo após a gravação

	for j in range(0,w): #Percorrendo uma coluna
		for i in range((h-a[j]),h): #Comece do ponto superior da coluna que deve virar preto até a parte inferior
			proj_vertical[i,j]=0

	#Inverte o fundo
	img = 255 - img
	#Calcula a projecao horizontal
	proj = np.sum(img, 1)

	#Crie a imagem de saída com a mesma altura do texto
	m = np.max(proj)
	w = 28
	zero = np.zeros((proj.shape[0], 1000))
	result = np.zeros((proj.shape[0], 28))

	#Desenha a projecao horizontal
	for row in range(img.shape[0]):
		cv2.line(result, (0, row), (int(proj[row] * w / m), row), (255, 255, 255), 1)
	a = np.zeros((28,28), dtype=int)
	projecao_horizontal = result

	proj_vertical = np.invert(proj_vertical)
	img_interpolada = np.interp(a, proj, proj2)

	#Concatena as projecoes
	proj_concat = np.hstack((proj_vertical, projecao_horizontal))
	#Concatena as imagens
	img_concat = np.hstack((img, proj_concat))
	#Salva a projecao
	cv2.imwrite("./projecoes/" + str(name) + ".jpg", img_concat)
	return img_concat

#Exibe as projecoes concatenadas me tela
def printProjecao():
	image, pathImage = charSeparation.getImageAndNames()
	projecoes = []
	images: list[np.ndarray]
	name = 0
	tam = 0
	for img in pathImage:
		projecoes = projecao(img, name)
		name = name + 1
	proj, pathProj = getProjAndNames()

	output =  concat_n_images(pathProj)
	#cv2.imshow("Projecoes", output)
	#plt.imshow(output,cmap=plt.gray())
	#plt.show()

#Concatena duas imagens
def concat_images(imga, imgb, i):
	img_concat = np.hstack((imga, imgb))
	return img_concat

#Concatena todas as projecoes listadas
def concat_n_images(image_path_list):
    output = None
    for i, img_path in enumerate(image_path_list):
        img = plt.imread(img_path)[:,:]
        if i==0:
            output = img
        else:
            output = concat_images(output, img, i)
    return output

#Retorna todas as imagens das projecoes salvas
def getProjAndNames():
	files = glob.glob('./projecoes/*')
	imgNames = []
	loadedImgs = []
	for f in files:
		imgNames.append(f)
		loadedImgs.append(cv2.imread(f))

	return loadedImgs, imgNames










