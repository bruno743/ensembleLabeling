import numpy as np

from .fcm import FCM
from .part_1 import part_1
from .part_2 import part_2
import pandas as pd

def mrp(csv_file, num_groups, algorithm):
    dataset = []

    for row in csv_file:
        dataset.append([round(float(element), 2) for element in row])

    if algorithm == 0:  # K-Means
        pertinence_list = part_1(dataset, num_groups)

        return dataset, pertinence_list
    else:  # Fuzzy C-means
        numpy_array = np.array(dataset)
        f_cmeans = FCM(num_groups)
        pertinence_list = f_cmeans.fit(numpy_array)

        return dataset, pertinence_list

NGROUPS = 0
GS = 0
IGS = 0
DATASET = 0

def first_stage(df, dfN, X, Y, columnNames):

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
    dataset, pertinence_list = mrp(X, NGROUPS, algorithm)

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
            p = group.copy()
            p2 = df - p
            for tuple in l:
                p = p[(p[tuple[0]] >= tuple[1]) & (p[tuple[0]] <= tuple[2])]
                p2 = p2[(p2[tuple[0]] >= tuple[1]) & (p2[tuple[0]] <= tuple[2])]
            #print(f'\nLabel {l}')
            #print(f'Accuracy for label {l} in group {i} is: {len(p)} == {round(len(p)/len(group), 3)}')
            media = (len(p)/len(group) + (1. - len(p2)/len(df - group)))/2
            #print(f'\np1: {len(p)/len(group)}\np2: {1. - len(p2)/len(df)}\nmedia: {media}\n')
            if media > acc[count] and l not in labels_:
                acc[count] = round(len(p)/len(group), 3)
                labels_[count] = l
        count+=1

    return labels_, acc