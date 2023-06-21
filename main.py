import sys

from k_means import *
from dbscan import *
from hierarchical import *

DATA_PATH = "./data/image_assessment_data_inpaint.csv"

def main(argv):
    clustering_method = argv[1]
    
    if(clustering_method == '-k'):
        cluster_cnt = int(argv[2])
        k_means_clustering(DATA_PATH, cluster_cnt)
    
    if(clustering_method == '-d'):
        eps = float(argv[2])
        min_samples = int(argv[3])
        dbscan_clustering(DATA_PATH, eps, min_samples)
    
    if(clustering_method == '-h'):
        cluster_cnt = int(argv[2])
        hierachy_clustering(DATA_PATH, cluster_cnt)


if __name__ =='__main__':
    main(sys.argv)