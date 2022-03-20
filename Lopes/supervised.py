import numpy as np
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.model_selection import train_test_split

def alg_supervis(metodo, data, percnt, folds):
    # vetor para calcular a relevância de cada atributo
    acur = [0]*data.shape[1]
    
    # faz-se a predição de cada atributo tantas vezes igual ao valor do parâmetro folds
    for i in range(folds):
        for j in range(data.shape[1]):
            Y = data.loc[:,data.columns[j]].values
            X = data.drop(data.columns[j], axis=1).values
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=percnt)
            
            y_train=np.asarray(y_train, dtype="|S6")
            y_test=np.asarray(y_test, dtype="|S6")
            clf = metodo
            
            if x_train.size == 0:
                acur[j] += 0
            else: 
                clf.fit(x_train, y_train)
                acur[j] += clf.score(x_test, y_test)
                
    # calculando média de acerto
    acur = [(i/folds) for i in acur]
    resultado = list(zip(data.columns, acur))
    return resultado

def aciona_metodo(grupos, attr_classe, porc_trein, folds, super_alg):
    result = []
    for grupo in grupos:
        data = grupo.drop([attr_classe], axis=1)
        clt = grupo[attr_classe].unique()
        if (super_alg == "PMC"):
            f = alg_supervis(mlp(max_iter=2000), data, porc_trein, folds)
        elif (super_alg == "TREE"):
            f = alg_supervis(tree(max_depth=2000), data, porc_trein, folds)
        result.append((clt, f))
    return result