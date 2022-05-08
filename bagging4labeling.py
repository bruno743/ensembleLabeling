from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, normalize
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

    return df, dfNormal, df.drop(['target'], axis=1).values, df['target'].values, list(df.drop(['target'], axis=1).columns)


def bagging(database, perSamples, n, choice):
    df, dfN, X, Y, columnNames = database
    from Lopes import main as mLopes
    from Lucia import main as mLucia
    from Pertinencia import main as mPertinence

    if choice == 'lopes':
        method = mLopes
    elif choice == 'lucia':
        method = mLucia
    elif choice == 'pertinence':
        method = mPertinence

    baseInformation, nGroups = method.first_stage(df, dfN, X, Y, columnNames)
    L = [0]*nGroups
    A = [0.]*nGroups
    I = np.unique(Y)

    for per in perSamples:
        for i in range(n):
            #print(f'\n{per*100}% -- Sample {i}')
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

    print(f'\n\nLabels:\n{L}\n\nAccuracy: {A}\n\n')
    return L, A

db = loadDatabase()
labels_0, accur_0 = bagging(db, [1., .8, .65], 3, 'lucia')
labels_1, accur_1 = bagging(db, [1., .8, .65], 2, 'lopes')
labels_2, accur_2 = bagging(db, [1., .8, .65], 3, 'pertinence')

def bestLabels(labels, accs):
    R = []
    I = []
    for a in tuple(accs):
        I.append(a.index(max(a)))
    
    for l in tuple(labels):
        R.append(l[I[0]])
        del I[0]

    print(R)

bestLabels(zip(labels_0, labels_1, labels_2), zip(accur_0, accur_1, accur_2))