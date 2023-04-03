import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from normalize import normalize_data


def main():
    # normalize the data
    my_data = normalize_data()

    # cluster the data
    k = 4
    kmeans_cluster = KMeans(init="random", n_clusters=k, n_init=4, random_state=0)
    kmeans_cluster.fit(my_data)
    centroids = kmeans_cluster.cluster_centers_
    label = kmeans_cluster.fit_predict(my_data)
    unique_labels = np.unique(label)

    # plot the clusters
    plt.figure(figsize=(8, 8))
    for i in unique_labels:
        plt.scatter(my_data[label == i, 0], my_data[label == i, 1], label=i)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=1, color='k', zorder=4)
    plt.legend()

    # plt.show()
    plt.savefig('plot.png')


if __name__ == '__main__':
    main()
