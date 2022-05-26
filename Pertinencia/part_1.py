from sklearn.cluster import KMeans


def part_1(kmeans, dataset, num_groups):
    #kmeans = KMeans(n_clusters=num_groups, random_state=0).fit(dataset)
    distance_array = kmeans.transform(dataset)

    inverse_distance_array = []
    for element in distance_array:
        distance = []
        for value in element:
            distance.append(1 / value)

        total = sum(distance)
        distance.append(total)

        inverse_distance_array.append(distance)

    pertinence_array = []
    for element in inverse_distance_array:
        pertinence = []

        for value in element[:-1]:
            pertinence.append(value / element[-1])

        pertinence_array.append(pertinence)

    return pertinence_array
