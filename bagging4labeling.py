from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.cluster import KMeans
import concurrent.futures
import pandas as pd
import numpy as np

def loadDatabase():
    choice = input('Select dataset:\n1 for Iris;\n2 for Breast Cancer;\n3 for Wine;\n' + 
                    '4 for Seeds;\n5 for Glass.\n')
    csv = False
    if choice == '2':
        ds = load_breast_cancer()
    elif choice == '3':
        ds = load_wine()
    elif choice == '4':
        df = pd.read_csv('sementes.csv')
        csv = True
    elif choice == '5':
        df = pd.read_csv('vidros.csv')
        csv = True
    else:
        ds = load_iris()
    
    if not csv:
        df = pd.DataFrame(ds.data)
        df.columns = ds.feature_names
        df['target'] = ds.target
    
    scaler = MinMaxScaler()
    dfNormal = df.drop(['target'], axis=1)
    dfNormal[:] = normalize(scaler.fit_transform(dfNormal))
    dfNormal['target'] = df['target']

    kmeans = KMeans(n_clusters=len(np.unique(df['target'].values))).fit(df.drop(['target'], axis=1).values)

    return df, dfNormal, df.drop(['target'], axis=1).values, df['target'].values, list(df.drop(['target'], axis=1).columns), kmeans


def bagging(database, perSamples, n, choice):
    df, dfN, X, Y, columnNames, kmeans = database
    from Lopes import main as mLopes
    from Lucia import main as mLucia
    from Pertinencia import main as mPertinence

    if choice == 'lopes':
        method = mLopes
    elif choice == 'lucia':
        method = mLucia
    elif choice == 'pertinence':
        method = mPertinence

    baseInformation, nGroups = method.first_stage(df, dfN, X, Y, columnNames, kmeans)
    L = [0]*nGroups # Labels
    A = [0.]*nGroups # Accuracy

    for per in perSamples:
        for i in range(n):
            dfBagg = pd.DataFrame(columns=baseInformation.columns)
            try:
                for index, group in baseInformation.groupby('target'):
                    dfBagg = pd.concat([dfBagg, group.sample(frac=per, replace=True)])
            except:
                dfBagg = pd.concat([dfBagg, baseInformation.sample(frac=per, replace=True)])
            label, acc = method.final_stage(df, dfN, X, Y, columnNames, dfBagg)
            for j in range(len(acc)):
                if acc[j] > A[j]:
                    A[j] = float(acc[j])
                    L[j] = label[j]

    print(f'\n\nLabels for {choice}:\n{L}\n\nAccuracy: {A}\n\n')
    return L, A

db = loadDatabase()
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(bagging, [db, db, db], [[1.], [1.], [1.]],
                    [12, 12, 12], ['lopes', 'lucia', 'pertinence'])
#labels_0, accur_0 = bagging(db, [1.], 1, 'lopes')
#labels_1, accur_1 = bagging(db, [1.], 1, 'lucia')
#labels_2, accur_2 = bagging(db, [1., .8], 5, 'pertinence')
results = list(results)
labels_0, accur_0 = results[0]
labels_1, accur_1 = results[1]
labels_2, accur_2 = results[2]

def bestLabels(labels, accs):
    R = [] # result
    I = [] # index
    for a in tuple(accs):
        I.append(a.index(max(a)))
    
    print(f'Best methods in label:\n')
    for l in tuple(labels):
        print(f'{l[I[0]]}\n')
        R.append(l[I[0]])
        del I[0]

    return R

def comp(G, group, model):
    for m in model:
        for g in G:
            if m[0] not in g: continue
            p = group[(group[m[0]] >= m[1]) & (group[m[0]] <= m[2])]
            if len(p)/len(group) > g[3]:
                G[G.index(g)] = (m[0], m[1], m[2],len(p)/len(group))
    return G

def bestAtts(db, tupl, labels):
    df = db[0]
    L = [] # labels
    for t in tuple(tupl):
        A = [] # accuracys
        for model in t:
            for a in model:
                if (a[0], 0, 0, .0) not in A:
                    A.append((a[0], 0, 0, .0))
        L.append(A)
    
    if min(np.unique(df['target'].values)) > 0:
        for i, group in df.groupby('target'):
            for model_ in labels:
                L[i-1] = comp(L[i-1], group, model_[i-1])
    else:
        for i, group in df.groupby('target'):
            for model_ in labels:
                L[i] = comp(L[i], group, model_[i])
    
    print(f'Best atributes in label:\n')
    for l in L:
        print(f'{l}\n')
    
    return L

#bestAtts(db, zip(labels_0, labels_1, labels_2), [labels_0, labels_1, labels_2])
#bestLabels(zip(labels_0, labels_1, labels_2), zip(accur_0, accur_1, accur_2))