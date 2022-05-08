def verify(intersection_list):
    has_intersection = {}

    flag = False
    
    for j in intersection_list:
        for k in intersection_list:
            if j != k:
                flag = True
                has_intersection = set.intersection(*map(set, [j, k]))
                if len(has_intersection) > 0:
                    return has_intersection

    if not flag: has_intersection = ['>0' for i in intersection_list]
    return has_intersection


def parse_list(list, columnNames, flag=0):
    final_list = []

    count = 0
    for element_list in list:
        labels = []

        for element in element_list:
            labels.append([element[0], element[-1]])

        final_list.append([(columnNames[count], l[0], l[1]) for l in labels])

        '''if flag != 0:
            print(labels)
        else:
            print(f'Attr_{count} -> {labels}')
        print()'''
        count += 1

    return final_list
