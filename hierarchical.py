from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import collections

from preprocess_data import *

DATA_PATH = "./data/image_assessment_data.csv"
N_CLUSTERS = 8

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

X = get_raw_data_list(DATA_PATH)
X = X[:1000]

name_list = get_name_data_list(DATA_PATH)

hierarch = AgglomerativeClustering(n_clusters=N_CLUSTERS).fit(X)

hierarch_labels = hierarch.labels_
print(hierarch_labels)
print(collections.Counter(hierarch_labels))

"""
plot_dendrogram(hierarch, truncate_mode=None)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
"""