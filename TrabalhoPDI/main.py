from ResizeRegion import resizeImage
from Zoom import MainWindow
from Quantization import quantize
from Save import _save_file_dialogs
import easygui
import unicodedata
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import cv2 as cv2
import matplotlib.pyplot as plt
import os

def pathImage():
    global filepath, enteredImage, baseImage, hasBaseImage, app
    uni_img = easygui.fileopenbox()
    try:
        filepath = unicodedata.normalize('NFKD', uni_img).encode('ascii','ignore')
    except:
        return
    filepath = filepath.decode('utf-8')
    enteredImage = True
    hasBaseImage = True
    baseImage = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    app = MainWindow(group1, filepath)

def pathSave():
     global app
     verificaBaseImage(app)
     if not hasBaseImage:
        messagebox.showerror("Erro", "Abra uma imagem primeiro")
     else:
        path = _save_file_dialogs()
        if path != None:
            result=cv2.imwrite(path, baseImage)
            if result==True:
                messagebox.showinfo("Salvo", "Arquivo Salvo")
            else:
                messagebox.showerror("Erro", "Erro ao salvar")

def pathTrain():
    messagebox.showinfo("Treinando", "Treinando Classificadores");

def pathResize():
    global app
    verificaBaseImage(app)
    if not hasBaseImage:
        messagebox.showerror("Erro", "Abra uma imagem primeiro")
    else:
        scale = easygui.enterbox("Qual a escala será aplicada na imagem (%)?")
        if scale != None:
            global baseImage
            imageResized = resizeImage(baseImage, int(scale))
            baseImage = imageResized
            result=cv2.imwrite('temp.png', baseImage)
            app = MainWindow(group1, 'temp.png')

def pathQuantize():
    global app
    verificaBaseImage(app)
    if not hasBaseImage:
        messagebox.showerror("Erro", "Abra uma imagem primeiro")
    else:
        levels = easygui.enterbox("Qual a quantidade de tons de cinza aplicada na imagem (tipo int)?")
        if levels != None:
            global baseImage
            imageQuantized = quantize(baseImage, int(levels))
            baseImage = imageQuantized
            result=cv2.imwrite('temp.png', baseImage)
            app = MainWindow(group1, 'temp.png')
            plt.show()

def pathInvert():
    global app
    verificaBaseImage(app)
    if not hasBaseImage:
        messagebox.showerror("Erro", "Abra uma imagem primeiro")
    else:
        global baseImage
        imageInvert = cv2.bitwise_not(baseImage)
        baseImage = imageInvert
        result = cv2.imwrite('temp.png', baseImage)
        app = MainWindow(group1, 'temp.png')
        plt.show()

def pathSair():
    master_window.destroy()
    temp = cv2.imread('temp.png')
    try:
        if temp.size != 0:
            os.remove('temp.png')
    except:   
        print('A imagem temporaria não foi apagada')

def pathClassify():
    messagebox.showinfo("Classificando", "Classificando OCR da imagem");

def verificaBaseImage(app):
    if app.canvas.verifica:
        global hasBaseImage, baseImage
        hasBaseImage = True
        baseImage = cv2.imread('temp.png')
        app.canvas.verifica = False


filepath = 'KMabWEXvea4P6QSXqDM6.png'
baseImage = cv2.imread('KMabWEXvea4P6QSXqDM6.png', cv2.IMREAD_UNCHANGED)
enteredImage = False
hasBaseImage = False
changeZoom = True
app = None
modeloTreinado = None

master_window = tk.Tk()
master_window.title("Menu")

menubar = Menu(master_window)

opcoesFile = Menu(menubar, tearoff=0)
opcoesFile.add_command(label='Abrir', command=pathImage)
opcoesFile.add_command(label="Salvar", command=pathSave)
opcoesFile.add_command(label="Sair", command=pathSair)
menubar.add_cascade(label="File", menu=opcoesFile)

opcaoEntropia = BooleanVar()
opcaoEntropia.set(False)
opcaoHomogeneidade = BooleanVar()
opcaoHomogeneidade.set(False)
opcaoEnergia = BooleanVar()
opcaoEnergia.set(False)
opcaoContraste = BooleanVar()
opcaoContraste.set(True)
opcaoHu =  BooleanVar()
opcaoHu.set(False)

opcoesOCR = Menu(menubar, tearoff=0)
opcoesOCR.add_command(label="Classificar", command=pathClassify)
opcoesOCR.add_cascade(label="Treinar", command=pathTrain)
menubar.add_cascade(label="OCR", menu=opcoesOCR)

optionVar = tk.StringVar(menubar)

opcoes = Menu(menubar, tearoff=0)
opcoes.add_command(label="Redimensionar", command=pathResize)
opcoes.add_command(label="Quantizar", command=pathQuantize)
opcoes.add_command(label="Inverter", command=pathInvert)

menubar.add_cascade(label="Opções", menu=opcoes)

master_window.config(menu=menubar)

# Frame Group1 ----------------------------------------------------
group1 = tk.LabelFrame(master_window, text="Imagem", padx=5, pady=5)
group1.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky='ewns')

master_window.columnconfigure(0, weight=1)
master_window.rowconfigure(0, weight=1)

group1.rowconfigure(0, weight=1)
group1.columnconfigure(0, weight=1)

# Cria o Canvas da imagem
app = MainWindow(group1, filepath)

master_window.mainloop()
