from dataclasses import replace
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


def bagging(perSamples, n, choice):
    df, dfN, X, Y, columnNames = loadDatabase()
    from Lopes import main as mLopes
    from Lucia import main as mLucia

    if choice == 'lopes':
        method = mLopes
    elif choice == 'lucia':
        method = mLucia

    baseInformation, nGroups = method.first_stage(df, dfN, X, Y, columnNames)
    L = [0]*nGroups
    A = [0.]*nGroups
    I = np.unique(Y)

    for per in perSamples:
        for i in range(n):
            print(f'\n{per*100}% -- Sample {i}')
            dfBagg = pd.DataFrame(columns=baseInformation.columns)
            for index, group in baseInformation.groupby('target'):
                dfBagg = pd.concat([dfBagg, group.sample(frac=per, replace=True)])
            label, acc = method.final_stage(df, dfN, X, Y, columnNames, dfBagg)
            for j in range(len(acc)):
                if acc[j] > A[j]:
                    A[j] = acc[j]
                    if choice == 'lopes': L[j] = label[j]
                    if choice == 'lucia': L[j] = label[label['Cluster'] == I[j]]

    print(f'\n\nLabels:\n{L}')
    print(f'Accuracy: {A}')

bagging([1., 0.8, 0.65, 0.5], 3, 'lucia')
#bagging([1., 0.8, 0.65, 0.5], 3, 'lopes')