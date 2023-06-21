from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import collections

from preprocess_data import *
from image_utils import *

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

def hierachy_clustering(DATA_PATH, N_CLUSTERS):
    # DATA_PATH = "./data/image_assessment_data_inpaint.csv"
    # N_CLUSTERS = 8
    X = get_raw_data_list(DATA_PATH)
    X = X[:1000]

    name_list = get_name_data_list(DATA_PATH)

    hierarch = AgglomerativeClustering(n_clusters=N_CLUSTERS).fit(X)

    # dendrogram
    # hierarch = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(X)

    hierarch_labels = hierarch.labels_
    print(collections.Counter(hierarch_labels))

    hierarch_labels = hierarch_labels[1:]
    show_images_by_label(hierarch_labels, name_list, N_CLUSTERS)

    # dendrogram
    """
    plot_dendrogram(hierarch, truncate_mode=None)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    """