#Processamento de Imagens - 2021/02 - Erick Sales, Felipe Augusto, Helen Machado, Juan Luiz
import matplotlib.pyplot as plt

def plot(conf_matrix,acuracy,type):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='small')

    plt.xlabel('Dígíto previsto' + acuracy, fontsize=10)
    plt.ylabel('Dígito real', fontsize=10)
    plt.title("Matriz Confusão " + type, fontsize=12)
    plt.show()