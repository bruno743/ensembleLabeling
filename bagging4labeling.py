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

    print(f'\n\nLabels for {choice}:')
    for l in L: print(l)
    print(f'\nAccuracy: {A}\n')
    return L, A

db = loadDatabase()
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(bagging, [db, db, db], [[1., .8], [1., .8], [1., .8]],
                    [8, 8, 8], ['lopes', 'lucia', 'pertinence'])
#labels_0, accur_0 = bagging(db, [1., .8], 7, 'lopes')
#labels_1, accur_1 = bagging(db, [1., .8], 7, 'lucia')
#labels_2, accur_2 = bagging(db, [1., .8], 7, 'pertinence')
results = list(results)
labels_0, accur_0 = results[0]
labels_1, accur_1 = results[1]
labels_2, accur_2 = results[2]

def bestLabels(labels, accs):
    labels = tuple(labels)
    accs = tuple(accs)
    
    print(f'\nBest methods in label:\n')

    I = []
    for a in accs:
        I.append(max(a))

    df0 = pd.DataFrame(columns=['cluster', 'label', 'acc'])
    df1 = pd.DataFrame(columns=['cluster', 'label', 'acc'])
    count = 0
    for l in labels:
        c, r, a = [], [], []
        for i in range(3):
            c.append(count)             #cluster
            r.append(l[i])              #label
            a.append(accs[count][i])    #accuracy
        df1['cluster'] = c
        df1['label'] = r
        df1['acc'] = a
        df0 = pd.concat([df0, df1])
        count += 1

    df1 = pd.DataFrame(columns=['cluster', 'label', 'acc'])
    df0 = df0.groupby('cluster')
    for index, group in df0:
        g = group.copy()
        g = g[g['acc'] == I[index]]
        if len(g) > 1:
            numAtt = 0
            g_ = g.copy()
            for method in g_.values:
                if len(method[1]) > numAtt:
                    numAtt = len(method[1])
                    d = {'cluster': [method[0]], 'label': [method[1]], 'acc': [method[2]]}
                    g = pd.DataFrame(data=d)
            
        print(f"{g['label'].values[0]}\n")
        df1 = pd.concat([df1, g])

    return df1

def comp(G, group, model):
    for m in model:
        for g in G:
            if m[0] not in g: continue
            p = group[(group[m[0]] >= m[1]) & (group[m[0]] <= m[2])]
            if len(p)/len(group) > g[3]:
                G[G.index(g)] = (m[0], m[1], m[2],len(p)/len(group))
            elif len(p)/len(group) == g[3]:
                if m[2]-m[1] < g[2]-g[1]:
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
    
    print(f'\nBest atributes in label:\n')
    for l in L:
        print(f'{l}\n')
    
    return L

bestAtts(db, zip(labels_0, labels_1, labels_2), [labels_0, labels_1, labels_2])
bestLabels(zip(labels_0, labels_1, labels_2), zip(accur_0, accur_1, accur_2))