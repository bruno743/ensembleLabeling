import numpy as np

from .part_1 import part_1
from .part_2 import part_2
import pandas as pd

NGROUPS = 0
GS = 0
IGS = 0
DATASET = 0

def first_stage(df, dfN, X, Y, columnNames, kmeans):

    #algorithm = int(input('algorithm: '))
    algorithm = 0
    global NGROUPS
    #NGROUPS = int(input('n groups: '))
    NGROUPS = len(np.unique(Y))
    global GS
    #GS = float(input('gs: '))
    GS = 0.5
    global IGS
    #IGS = float(input('igs: '))
    IGS = 0.0001

    # Init the program
    dataset = []
    for row in X:
        dataset.append([round(float(element), 2) for element in row])
    pertinence_list = part_1(kmeans, dataset, NGROUPS)

    bi = [dataset[i] + pertinence_list[i] for i in range(len(dataset))]

    bi = pd.DataFrame(bi, columns=columnNames+[f'g{i}' for i in range(NGROUPS)])

    global DATASET
    DATASET = dataset
    return bi, NGROUPS

def final_stage(df, dfN, X, Y, columnNames, baseInformation):
    labels = part_2(DATASET, NGROUPS, GS, IGS, columnNames, baseInformation.values.tolist())

    acc = [0.]*len(labels)
    labels_ = [0]*len(labels)
    count = 0
    for i, group in df.groupby('target'):
        for l in labels:
            if l in labels_: continue
            
            p = group.copy()
            p2 = df - p
            for tuple in l:
                p = p[(p[tuple[0]] >= tuple[1]) & (p[tuple[0]] <= tuple[2])]
                p2 = p2[(p2[tuple[0]] >= tuple[1]) & (p2[tuple[0]] <= tuple[2])]

            media = (len(p)/len(group) + (1. - len(p2)/len(df - group)))/2

            if media > acc[count] and l not in labels_:
                acc[count] = round(len(p)/len(group), 3)
                labels_[count] = l
        count+=1

    return labels_, acc