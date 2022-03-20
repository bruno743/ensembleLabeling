from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

INFOR = 0
PERCENT = 0
VARIATION = 0

def first_stage(df, dfN, X, Y, columnNames):
    print('Method params:')
    global PERCENT
    PERCENT = float(input('Enter the percentual for training (float value): '))
    global VARIATION
    VARIATION = float(input('Enter the value of variation: '))
    values, discretized, infor = discretization(df)
    global INFOR
    INFOR = infor
    groups = clustering(X, Y)
    discretized['target'] = groups

    return discretized

def discretization(df):
    vector_num_faixas = int(input('Enter the range of values: '))
    vector_num_faixas = [vector_num_faixas]*df.shape[1]
    metodo = input('Enter the method:\n1 for EWD;\n2 for EFD.\n')
    if metodo == '2':
        metodo = 'EFD'
    else:
        metodo = 'EWD'
    cluster = df['target']
    data = df.drop(['target'], axis=1) # Deleta a coluna classe
    
    try:
        values = data.to_numpy(dtype =  np.float32)
    except:
        print("Há entradas na bases de dados com valores incorretos")
        #Isso ocorre pois alguma entrada na base possui valor não numerico
        #para algum atributo. Eliminar essa entrada da base pode ser uma 
        #solucao para o problema
        input("Pressione ENTER para sair")
        sys.exit()
    
    ddb = []
    infor = []

    for j in range(0, data.shape[1]):
        if metodo == 'EWD':
            disc_attb = pd.cut(values[:,j],vector_num_faixas[j], labels = False, retbins= True)
        elif metodo == 'EFD':
            disc_attb = pd.qcut(values[:,j], vector_num_faixas[j], labels = False, retbins = True, duplicates = 'drop')
            
        ddb.append(disc_attb[0])
        infor.append((data.columns[j],disc_attb[1]))
    
    ddb = np.asarray(ddb, dtype = 'int32')
    
    for x in range (0, data.shape[1]):
        data.loc[:,data.columns[x]] = [y[x] for y in ddb.T]
    
    data['target'] = cluster
    
    return ddb, data, infor

def clustering(X, Y):
    km = KMeans(n_clusters=len(np.unique(Y)))
    km = km.fit(X)

    '''for cluster in np.unique(km.labels_):
        row_ix = np.where(km.labels_ == cluster)
        plt.scatter(X[row_ix, 0], X[row_ix, 1], s=200)
    
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                  marker='*', s=300,
                  c='r', label='centroid')
    plt.show()'''

    return km.labels_

def final_stage(df, dfN, X, Y, columnNames, baseInformation):
    frames_disc = []
    for index, group in baseInformation.groupby('target'):
        frames_disc.append(group)
    
    global PERCENT
    folds = 10
    
    relevant = start_method(frames_disc, 'target', PERCENT, folds)

    global VARIATION
    global INFOR

    labels = atrib_rotul(frames_disc, 'target', relevant, VARIATION, INFOR)
    acc = [0.]*len(labels)
    for i, group in df.groupby('target'):
        for l in labels:
            p = group.copy()
            for tuple in l[1]:
                p = p[(p[tuple[0]] >= tuple[1]) & (p[tuple[0]] <= tuple[2])]
            print(f'\nLabel {l[0]}: {l[1][0:]}')
            print(f'Accuracy for label {l[0]} in group {i} is: {len(p)} == {round(len(p)/len(group), 3)}')
            if len(p)/len(group) > acc[l[0]]: acc[l[0]] = round(len(p)/len(group), 3)

    return labels, acc

def supervised(metodo, data, percnt, folds):
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

def start_method(grupos, attr_classe, per_trein, folds):
    result = []
    for grupo in grupos:
        data = grupo.drop([attr_classe], axis=1)
        clt = grupo[attr_classe].unique()
        f = supervised(mlp(max_iter=2000), data, per_trein, folds)
        result.append((clt, f))
    
    return result

def atrib_rotul(grupos_disc, attr_classe, infor_attrs, variacao, disc_infor):
    rotulos = []
    # para cada grupo, aciona o método de geração de rótulos
    for grupo in grupos_disc:
        clt = grupo[attr_classe].unique()[0]
        at_info = [i[1] for i in infor_attrs if i[0]==clt][0]
        result_rot = defin_rotul(grupo.drop([attr_classe], axis=1), at_info, disc_infor, variacao)
        rotulos.append((clt, result_rot))
    
    '''print("Rotulos")
    for rotulo in rotulos:
        print(f"Cluster: {rotulo[0]}")
        print(f"Pares Atributo-Intervalo: {rotulo[1]}")'''
    
    return rotulos

def defin_rotul(cluster, at_info, disc_infor, variacao):
    medias = [(i, acuracia*100) for i, acuracia in at_info]
    medias.sort(key=lambda x: x[1], reverse=True)
    minn = medias[0][1] - variacao
    titulos = cluster.columns.values.tolist()
    
    result = []
    # para cada atributo
    for i in range(cluster.shape[1]):
        # se a acuracia do atributo for superior a um valor,
        # será "calculado" o intervalo e o par atributo-intervalo fará parte do rótulo
        if medias[i][1] >= minn: 
            attr = medias[i][0]
            info = [j[1] for j in disc_infor if j[0]==attr][0]
            most_comun_value = cluster[attr].mode()[0]
            try:
                rotulo = (attr, round(info[most_comun_value], 2), round(info[most_comun_value+1], 2))
            except:  
                print("Não foi possivel atribuir rotulo aos clusters")
                #Isso ocorre pois algum atributo foi discretizado de forma 
                #incorreta(provavelmente um atributo tem um mesmo valor 
                #para todas as entradas na base de dados). Eliminar esse 
                #atributo da base pode ser uma solucao para o problema
                input("Pressione ENTER para sair")
                sys.exit()
            result.append(rotulo)
    return result