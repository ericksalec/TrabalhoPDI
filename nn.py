import tensorflow as tf

def nntrain():
    #Carrega a base de dados MNIST internamente pela biblioteca tensorflow
    mnist = tf.keras.datasets.mnist
    #Separa os dados da base para treino e para teste
    #X é o dado em si, conjunto de pixels
    #Y é a label do digito, o que ele representa
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #Normalizando os valores dos pixels
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    #Cria o modelo da rede neural
    #O modelo no caso foi o sequencial pela simplicidade
    #Adiciona uma camada de entrada 'flattened' que achata a matriz das imagens 28x28 em um unico vetor de 784
    #Adiciona duas camadas densas
    #Adicona a camada de saida em 10 unidades de saida, um para cada dígito
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    #Compila e otimiza o modelo com base na acurácia
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #Treina o modelo
    model.fit(x_train, y_train, epochs=3)

    #Avalia o modelo criado
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss)
    print(val_acc)

    #Salva o modelo criado para ser usado posteriormente
    model.save('nn.model')



