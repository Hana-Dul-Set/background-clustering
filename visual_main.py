import sys

from cluster_viewer import *
from nearest_image_viewer import *
from hog import *

DATA_PATH = './data/csv/' + get_config_string() + ';proportion' + '.csv'
IMAGE_PATH = './data/image/bg-20k-train'
JSON_PATH = "./data/result/normalized"

def main(argv):
    clustering_method = argv[1]
    
    if(clustering_method == '-c'):
        cluster_viewer(IMAGE_PATH, JSON_PATH)
    
    if(clustering_method == '-n'):
        nearest_image_viewer(IMAGE_PATH, DATA_PATH)
    
if __name__ =='__main__':
    main(sys.argv)