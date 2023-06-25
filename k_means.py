from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import collections

from preprocess_data import *
from image_utils import *

def k_means_clustering(DATA_PATH, N_CLUSTERS):
    # DATA_PATH = "./data/image_assessment_data_inpaint.csv"
    # N_CLUSTERS = 10

    X = get_raw_data_list_SAMPNet(DATA_PATH, normalized=True)
    X = X[:1000]

    name_list = get_name_data_list(DATA_PATH)

    k_means = KMeans(init="k-means++", n_clusters=N_CLUSTERS, n_init=1)
    k_means.fit(X)

    k_means_labels = k_means.labels_
    print(collections.Counter(k_means_labels))

    k_means_cluster_centers = k_means.cluster_centers_
    print(sum(x ** 2 for x in k_means_cluster_centers[0]))

    show_images_by_label(k_means_labels, name_list, N_CLUSTERS)