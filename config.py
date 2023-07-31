import cv2, os

class Config:
    def __init__(self):

        self.hog_config = {
            'Image_resize': (128, 128), # (width, height)
            'Image_convert': cv2.COLOR_BGR2GRAY, # cv2.COLOR_BGR2GRAY | cv2.COLOR_BGR2RGB | cv2.COLOR_BGR2HSV
            'Cell_size': (16, 16), # (width, height),
            'Block_size': (1, 1), # (multiple_width, multiple_height),
            'Magnitude_threshold': 151, # minimum magnitude limit,
            'n_bins': 9, # n_bins
        }

        self.cluster_config = {
            'hog_weight': 50
        }

        self.IMAGE_PATH = './data/image/inpaint_results_filtered_selected'
        self.CSV_PATH = './data/0722csv'
        # self.DATA_PATH = os.path.join(self.CSV_PATH, self.get_hog_config_string() + ';proportion' + '.csv')
        # self.DATA_PATH = './data/resnet50/resnet50.csv'
        self.DATA_PATH = os.path.join(self.CSV_PATH, 'resnet50' + self.get_hog_config_string() + '.csv')

        self.JSON_PATH = './data/result/normalized/0722'
        

        # DATA_PATH = "./data/csv/image_assessment_data_inpaint.csv"
        # DATA_PATH = "./data/csv/line_detection_data_inpaint.csv"
        # DATA_PATH = "./data/csv/image_assessment_data_landscape_data.csv"
        # DATA_PATH = "./data/csv/image_assessment_data_bg-20k-train.csv"
        # DATA_PATH = "./data/csv/temp_vector.csv"
        # DATA_PATH = './data/csv/hog_bg-20k-train_gray_gausblur(sigma=5).csv'
        # DATA_PATH = './data/csv/hog_bg-20k-train_hsv.csv'
        # DATA_PATH = './data/csv/hog(cellSize=128x128)_bg-20k-train_gray.csv'
        # DATA_PATH = './data/csv/' + get_config_string() + ';proportion' + '.csv'
        # DATA_PATH = './data/0713csv/' + get_config_string_overlapping() + '.csv'

        # IMAGE_PATH = './data/image/inpaint_selected_results'
        # IMAGE_PATH = './data/image/line_detected_inpaint'
        # IMAGE_PATH = './data/image/landscape_data'
        # IMAGE_PATH = './data/image/temp'

        
    def get_hog_config_string(self):
        hog_config_string = str(self.hog_config).replace(' ', '')
        return hog_config_string
    def get_cluster_config_string(self):
        cluster_config_string = str(self.cluster_config).replace(' ', '')
        return cluster_config_string
    
config = Config()