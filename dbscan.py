from sklearn.cluster import DBSCAN

from preprocess_data import *

EPS = 0.5
MIN_SAMPLES = 4

X = get_raw_data_list(DATA_PATH="image_assessment_data.csv")

X = X[:2]

name_list = get_name_data_list(DATA_PATH="image_assessment_data.csv")

dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(X)

dbscan_labels = dbscan.labels_
print(dbscan_labels)