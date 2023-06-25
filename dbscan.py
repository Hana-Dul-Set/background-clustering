from sklearn.cluster import DBSCAN
import numpy as np
import collections

from preprocess_data import *
from image_utils import *

def dbscan_clustering(DATA_PATH, EPS, MIN_SAMPLES):
    # DATA_PATH = "./data/image_assessment_data_inpaint.csv"
    # EPS = 0.7
    # MIN_SAMPLES = 3

    X = get_raw_data_list_SAMPNet(DATA_PATH, normalized=True)
    # X = X[:1000]

    name_list = get_name_data_list(DATA_PATH)

    dbscan = DBSCAN(eps=EPS, metric='cosine', min_samples=MIN_SAMPLES).fit(X)

    dbscan_labels = dbscan.labels_
    print(collections.Counter(dbscan_labels))
    # print(collections.Counter(dbscan_labels)[0])
    N_CLUSTERS = max(dbscan_labels) + 1
    print(N_CLUSTERS)
    #show_images_by_label(dbscan_labels, name_list, N_CLUSTERS)
    return collections.Counter(dbscan_labels)[0], N_CLUSTERS

if __name__ == '__main__':
    min_samples = 12 ## 12부터 해야함
    result = []
    try:
        for eps in range(1, 1000, 1):
            zero_label_count, label_cnt = dbscan_clustering('./data/image_assessment_data_inpaint.csv', eps/1000, min_samples)
            if (zero_label_count <= 1000) and (label_cnt > 10):
                result.append(eps/1000)
            print(eps)
    except:
        print(result)
    