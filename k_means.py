from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from preprocess_data import *

n_clusters = 8


X = get_raw_data_list(DATA_PATH="image_assessment_data.csv")

name_list = get_name_data_list(DATA_PATH="image_assessment_data.csv")

k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=1)
k_means.fit(X)

k_means_labels = k_means.labels_
print('k_means_labels : ', k_means_labels)

k_means_cluster_centers = k_means.cluster_centers_
print('k_means_cluster_centers : ', k_means_cluster_centers)

clusters_data_list = []

for i in range(n_clusters):
    clusters_data = []
    for j in range(len(k_means_labels)):
        if(k_means_labels[j] == i):
            clusters_data.append(name_list[j])
    clusters_data_list.append(clusters_data)

for clusters_data in clusters_data_list:
    print(clusters_data)
    