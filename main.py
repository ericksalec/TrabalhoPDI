import cv2 as cv2
import tkinter as tk
from zoom import MainWindow
import easygui
import unicodedata
import nn
from tkinter import messagebox
from charSeparation import separa
from charSeparation import marcacaoDaImgNN
from charSeparation import marcacaoDaImgSVM
import svm

def pathImage():
    global filepath, enteredImage, baseImage, hasBaseImage, app
    uni_img = easygui.fileopenbox()
    try:
        filepath = unicodedata.normalize('NFKD', uni_img).encode('ascii', 'ignore')
    except:
        return
    filepath = filepath.decode('utf-8')
    enteredImage = True
    hasBaseImage = True
    baseImage = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    separa(baseImage)
    app = MainWindow(group1, filepath)


def pathSair():
    master_window.destroy()

def pathClassifyNN():
    if enteredImage:
        if nnTrained:
            marcacaoDaImgNN(baseImage)
        else:
            messagebox.showerror("Erro", "Execute o treinamento primeiro")
    else:
        messagebox.showerror("Erro", "Abra uma imagem primeiro")
    return

def pathClassifySVM():
    if svmTrained:
        if enteredImage:
            marcacaoDaImgSVM(baseImage)
        else:
            messagebox.showerror("Erro", "Abra uma imagem primeiro")
    else:
        messagebox.showerror("Erro","Execute o treinamento primeiro")
    return

def pathTrainNN():
    global nnTrained
    nnTrained = nn.nnTrain()
    return

def pathTrainSVM():
    global svmTrained
    svmTrained = svm.svmtrain()
    return

filepath = 'dot.png'
baseImage = cv2.imread('dot.png', cv2.IMREAD_UNCHANGED)
enteredImage = False
hasBaseImage = False
changeZoom = True
app = None
nnTrained = False
svmTrained = False

master_window = tk.Tk()
master_window.title("Menu")

menubar = tk.Menu(master_window)

opcoesFile = tk.Menu(menubar, tearoff=0)
opcoesFile.add_command(label='Abrir', command=pathImage)
opcoesFile.add_command(label="Sair", command=pathSair)
menubar.add_cascade(label="File", menu=opcoesFile)

opcoesClassificacao = tk.Menu(menubar, tearoff=0)
opcoesClassificacao.add_command(label='SVM', command=pathClassifySVM)
opcoesClassificacao.add_command(label="Rede Neural", command=pathClassifyNN)

opcoesTreinamento = tk.Menu(menubar, tearoff=0)
opcoesTreinamento.add_command(label='SVM', command=pathTrainSVM)
opcoesTreinamento.add_command(label="Rede Neural", command=pathTrainNN)

opcoesOCR = tk.Menu(menubar, tearoff=0)
opcoesOCR.add_cascade(label="Classificar", menu=opcoesClassificacao)
opcoesOCR.add_cascade(label="Treinar", menu=opcoesTreinamento)
menubar.add_cascade(label="OCR", menu=opcoesOCR)

optionVar = tk.StringVar(menubar)

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
