#Processamento de Imagens - 2021/02 - Erick Sales, Felipe Augusto, Helen Machado, Juan Luiz
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn import metrics
from tkinter import messagebox
import confusion_matrix as conf




def nnTrain():
    #Marca o inicio do tempo de execução do treinamento
    start = time.time()
    # Carrega a base de dados MNIST internamente pela biblioteca tensorflow
    mnist = tf.keras.datasets.mnist
    # Separa os dados da base para treino e para teste
    # X é o dado em si, conjunto de pixels
    # Y é a label do digito, o que ele representa
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalizando os valores dos pixels
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # Cria o modelo sequencial da rede neural
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # Compila e otimiza o modelo com base na acurácia
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Treina o modelo
    model.fit(x_train, y_train, epochs=3)

    #Avalia o modelo criado
    loss, acc = model.evaluate(x_test, y_test)

    #Faz a predição do conjunto de teste para plotar a matriz de concorrencia
    y_predicted = model.predict(x_test)

    #Pega a ordem de todas os valores de ativação da ultima camada da rede para os valores previstos
    y_predicted_arr = []
    for value in y_predicted:
        y_predicted_arr.append(np.argmax(value))
    y_predicted_arr = np.array(y_predicted_arr)

    # Salva o modelo criado para ser usado posteriormente
    model.save('models/nn.model')

    #Marca o fim do tempo de  treinamento
    end = time.time()
    #Calcula quanto tempo demorou o treinamento e exibe uma msgbox informando esse tempo para o usuário
    executionTime = end - start
    tempo = str(executionTime) + " segundos "
    messagebox.showinfo(title="Tempo de Treinamento NN", message=tempo)

    #Calcula e plota a matriz de confusão do modelo com o auxilo da biblioteca metrics da sklearn
    conf_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_predicted_arr)
    acuracy = "\nAcurácia = " + str(acc) + "\n"
    type = "Rede Neural"
    conf.plot(conf_matrix, acuracy, type)
    return True

def nnClassify(filepath):
    #Lê a imagem a ser classificada
    img = Image.open(filepath)
    #Transforma a imagem em um array e remodela o array para ficar de acordo com a entrada esperada no modelo
    img_arr = np.array(img)
    flat_img_arr = np.reshape(img_arr, (-1, 28, 28))

    #Executa a predição e retorna o provavel valor do digito
    model = tf.keras.models.load_model('models/nn.model')
    prediction = model.predict(flat_img_arr)
    return np.argmax(prediction)