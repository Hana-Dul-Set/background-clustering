import matplotlib.pyplot as plt
import numpy as np
import os

from preprocess_data import *

IMAGE_PATH = './data/inpaint_selected_results'

def show_10_images(image_filenames):

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    for ax, filename in zip(axes.flatten(), image_filenames):
        image = plt.imread(os.path.join(IMAGE_PATH, filename))  
        ax.imshow(image)              
        ax.axis('off')

    plt.tight_layout()

    plt.show()

def show_images_by_label(label_list, name_list, label_cnt):
    data_by_label_list = get_elements_list_by_label(label_list, name_list, label_cnt)

    for one_label_data_list in data_by_label_list:
        random_elements = get_random_elements(one_label_data_list, 10)
        show_10_images(random_elements)