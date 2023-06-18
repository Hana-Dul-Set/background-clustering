from sklearn.cluster import DBSCAN
import numpy as np
import collections

from preprocess_data import *

DATA_PATH = "./data/image_assessment_data.csv"
EPS = 0.7
MIN_SAMPLES = 3

X = get_raw_data_list(DATA_PATH)
X = X[:1000]

name_list = get_name_data_list(DATA_PATH)

dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(X)

dbscan_labels = dbscan.labels_

print(dbscan_labels)
print(collections.Counter(dbscan_labels))