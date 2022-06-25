import sys

import numpy

from . import utils

def part_2(dataset, num_groups, GS, IGS, columnNames, pertinence_array=None):
    num_attrs = len(dataset[0])

    mod = sys.modules[__name__]

    table = pertinence_array
    '''for i in range(len(dataset)):
        table.append(dataset[i] + pertinence_array[i])'''

    counter = 1
    while True:
        for i in range(num_groups):
            setattr(mod, f"group_{i}", [])

        for i in range(num_attrs):
            setattr(mod, f"attr_{i}", [])

        for element in table:
            for value in range(num_groups):
                if element[value + num_attrs] > GS:
                    globals()[f"group_{value}"].append(element)

        _list = []
        for i in range(num_groups):
            for j in range(num_attrs):
                globals()[f"attr_{j}"] = []

            for element in globals()[f"group_{i}"]:
                for value in range(num_attrs):
                    globals()[f"attr_{value}"].append(float(element[value]))

            attrs_array = []
            for j in range(num_attrs):
                if len(globals()[f"attr_{j}"]) > 0:
                    attrs_array.append([min(globals()[f"attr_{j}"]), max(globals()[f"attr_{j}"])])

            _list.append(attrs_array)

        intersection_list = []

        for i in range(num_attrs):
            elements = []
            for j in range(num_groups):
                if len(globals()[f'group_{j}']) > 0:
                    elements.append([round(k, 2) for k in numpy.arange(_list[j][i][0], (_list[j][i][1] + 0.01), 0.01)])

            intersection_list.append(elements)

        unique_value_ranges = []
        count_intersection = 0
        for i in intersection_list:
            if len(utils.verify(i)) > 0:
                count_intersection += 1
            else:
                unique_value_ranges.append(i)

        if count_intersection < num_attrs:
            #print(f"\nIteration => {counter}\n\nGS => {round(GS, 4)}")
            #print(f"\n\n\tFinal labels\n")
            fl = utils.parse_list(intersection_list, columnNames)
            flabels = []
            final_labels = []
            for i in range(len(fl[0])):
                for j in range(len(fl)):
                    flag = 1
                    for k in range(len(fl[0])):
                        if i != k:
                            if (fl[j][k][1] >= fl[j][i][1] and fl[j][k][1] <= fl[j][i][2]) or (fl[j][k][2] >= fl[j][i][1] and fl[j][k][2] <= fl[j][i][2]):
                                flag = 0
                    if flag:
                        flabels.append(fl[j][i])
                final_labels.append(flabels)
                flabels = []
            #print('\n\tSingle range of values\n')
            #utils.parse_list(unique_value_ranges, columnNames, 1)
            return final_labels
        else:
            GS += IGS
            counter += 1
