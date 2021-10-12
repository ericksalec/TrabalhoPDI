import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def nntrain():
    #Carrefa a base de dados MNIST internamente pela biblioteca tensorflow
    mnist = tf.keras.datasets.mnist
    #Separa os dados da base para treino e para teste
    #X é o dado em si, conjunto de pixels
    #Y é a label do digito, o que ele representa
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #Normalizando os valores dos pixels
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    print("Teste")
