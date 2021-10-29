import cv2 as cv2
import tkinter as tk
from zoom import MainWindow
import easygui
import unicodedata
import dnn

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
    app = MainWindow(group1, filepath)


def pathSair():
    master_window.destroy()

def pathClassify():
    return

def pathTrain():
    dnn.nntrain()
    return

filepath = 'dot.png'
baseImage = cv2.imread('dot.png', cv2.IMREAD_UNCHANGED)
enteredImage = False
hasBaseImage = False
changeZoom = True
app = None
modeloTreinado = None

master_window = tk.Tk()
master_window.title("Menu")

menubar = tk.Menu(master_window)

opcoesFile = tk.Menu(menubar, tearoff=0)
opcoesFile.add_command(label='Abrir', command=pathImage)
opcoesFile.add_command(label="Sair", command=pathSair)
menubar.add_cascade(label="File", menu=opcoesFile)

opcoesOCR = tk.Menu(menubar, tearoff=0)
opcoesOCR.add_command(label="Classificar", command=pathClassify)
opcoesOCR.add_cascade(label="Treinar", command=pathTrain)
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