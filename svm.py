import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#from sklearn.model_selection import KFold
#from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import confusion_matrix as conf


def svmtrain():
    #Fazendo a leitura da base em formato CSV com auxilio da biblioteca pandas
    #train_data = pd.read_csv("database/mnist_full.csv", sep=',')
    train_data = pd.read_csv("database/mnist_small.csv", sep=',')

    y = train_data['label']
    X = train_data.drop(columns = 'label')
    X = X/255.0

    X_scaled = scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.6, train_size=0.4, random_state=10, stratify=y)


    #Inicio da busca dos possíveis melhores hyperparametros para o modelo baseado:
        #folds = KFold(n_splits=5, shuffle=True, random_state=10)

        #hyper_params = [{'gamma': [1e-2, 1e-3, 1e-4],
        #             'C': [5, 10]}]

        #model = SVC(kernel="rbf")
        #model = SVC(kernel="poly")
        #model = SVC(kernel="linear")

        #model_cv = GridSearchCV(estimator=model,
        #                   param_grid=hyper_params,
        #                    scoring='accuracy',
        #                    cv=folds,
        #                    verbose=1,
        #                   return_train_score=True)

        #model_cv.fit(X_train, y_train)

        #best_hyperparams = model_cv.best_params_

        #print("Os melhores hyperparametros encontrados para o problema foram: {1}".format(best_hyperparams))

    model = SVC(C=10, gamma=0.001, kernel="rbf")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Acurácia", metrics.accuracy_score(y_test, y_pred), "\n")
    print(metrics.confusion_matrix(y_test, y_pred), "\n")

    acuracy = "\nAcurácia = " + str(metrics.accuracy_score(y_test, y_pred)) + "\n"
    type = "SVM"
    conf_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)

    conf.plot(conf_matrix,acuracy,type)

    return True