from sklearn.cluster import DBSCAN
import numpy as np
import collections

from preprocess_data import *
from image_utils import *

def dbscan_clustering(DATA_PATH, EPS, MIN_SAMPLES):
    # DATA_PATH = "./data/image_assessment_data_inpaint.csv"
    # EPS = 0.7
    # MIN_SAMPLES = 3

    X = get_raw_data_list(DATA_PATH)
    X = X[:1000]

    name_list = get_name_data_list(DATA_PATH)

    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(X)

    dbscan_labels = dbscan.labels_
    print(collections.Counter(dbscan_labels))

    N_CLUSTERS = max(dbscan_labels)

    show_images_by_label(dbscan_labels, name_list, N_CLUSTERS)