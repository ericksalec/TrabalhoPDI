#Processamento de Imagens - 2021/02 - Erick Sales, Felipe Augusto, Helen Machado, Juan Luiz
import time
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.svm import SVC
from sklearn import metrics
from tkinter import messagebox
import confusion_matrix as conf
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def svmtrain():
    # Marca o inicio do tempo de execução do treinamento
    start = time.time()
    #Fazendo a leitura da base em formato CSV com auxilio da biblioteca pandas
    train_data = pd.read_csv("database/mnist_full.csv", sep=',')
    #train_data = pd.read_csv("database/mnist_small.csv", sep=',')

    #Separando as variáveis X e Y
    #Y fica responsável pela as labels do dados
    y = train_data['label']
    #X fica com os pixels de 1 a 784 das imagens da base
    X = train_data.drop(columns='label')

    #Normalizando os pixels da base
    X = X/255.0
    #Padronizando os pixels da base
    #X_scaled = scale(X)

    #Divisão da base no conjunto de teste e no conjunto de treino aleatoriamente
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.2, random_state=10, stratify=y)

    #Plota um digito da base
    #digit = train_data.iloc[3, 1:]
    #digit.shape
    #digit = digit.values.reshape(28, 28)
    #plt.imshow(digit, cmap='gray')
    #plt.title("Digit Example")

    #Inicio da busca dos possíveis melhores hyperparametros para o modelo:

        #Cria um objeto Kfold com 5 diviões
    #folds = KFold(n_splits=5, shuffle=True, random_state=10)

        #Especifica a faixa dos hyperamatros para ser testada
    #hyper_params = [{'gamma': [1e-2, 1e-3, 1e-4],
                        #'C': [5, 10]}]


        #Define o kernel que vai ser usado no modelo a ser validade
    #model = SVC(kernel="rbf")
    #model = SVC(kernel="poly")
    #model = SVC(kernel="linear")
    #model = SVC(kernel="sigmoid")

        #Usa o GridSearchCV para  fazer a combinação entre os parametros fornecidos
    #model_cv = GridSearchCV(estimator=model,
                                #param_grid = hyper_params,
                                #scoring = 'accuracy',
                                #cv = folds,
                                #verbose = 1,
                                #return_train_score = True)

        #Ajusta o modelo para o conjutno de treino
    #model_cv.fit(X_train, y_train)

        #Salva a combinação de parametros que obteve um melhor resultado
    #best_hyperparams = model_cv.best_params_
    #print("Os melhores hyperparametros encontrados para o problema foram: {0}".format(best_hyperparams))

    #Cria um modelo com base na melhor combinação de parametros obtida anterioemente
    model = SVC(C=5, gamma=0.01, kernel="poly")
    #Ajusta o modelo ao conjunto de treino
    model.fit(X_train, y_train)
    #Usa o modelo criado para prever os resultados do conjunto de teste
    y_pred = model.predict(X_test)

    # Salva o modelo criado para uso posterior
    pickle.dump(model, open("models/svm.sav", 'wb'))

    # Marca o fim do tempo de  treinamento
    end = time.time()
    # Calcula quanto tempo demorou o treinamento e exibe uma msgbox informando esse tempo para o usuário
    executionTime = end - start
    tempo = str(executionTime) + " segundos "
    messagebox.showinfo(title="Tempo de Treinamento NN", message=tempo)

    #Calcula a acurácia do modelo criado comparando os valores das labels previstas com as corretas
    acuracy = "\nAcurácia = " + str(metrics.accuracy_score(y_test, y_pred)) + "\n"
    type = "SVM"

    #Plota da matriz de confusão do modelo
    conf_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
    conf.plot(conf_matrix, acuracy, type)

    return True

def svmClassify(filepath):
    #Carrega a imagem com o auxilio da lib pillow
    img = Image.open(filepath).convert('L')

    #Converte a imagem lida para uma lista
    img_arr = np.array(img)
    flat_img_arr = img_arr.flatten()
    img_list = flat_img_arr.tolist()

    #Le um csv que contem o cabeçalho de separação dos pixels
    cabecalho = pd.read_csv('database/cabecalho.csv', sep=',')
    column_list = list(cabecalho.columns.values)

    #Constroi um novo csv juntando o cabeçalho com os pixels da imagem lida
    hand_df = pd.DataFrame([img_list], columns=column_list)
    hand_df.to_csv('database/hand_check.csv', encoding='utf-8')

    #Le o csv salvo anteriormente e faz a predicao do digito
    image_predict = pd.read_csv("database/hand_check.csv", sep=',', index_col=0)
    svm = pickle.load(open("models/svm.sav", 'rb'))
    predicao = svm.predict(image_predict)
    predicao = predicao.item(0)
    return predicao

