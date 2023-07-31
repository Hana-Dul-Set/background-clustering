import cv2
import numpy as np
import os
import time
import math
import heapq

from config import config

hog_config = config.hog_config

def get_coordinate_of_vector(length, degree):
    radians = np.radians(degree)
    x = length * np.cos(radians)
    y = length * np.sin(radians)
    return [x, y]

def get_gradient(image):

    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_x = gradient_x / int(np.max(np.abs(gradient_x)) if np.max(np.abs(gradient_x)) != 0 else 1) * 255
    gradient_y = gradient_y / int(np.max(np.abs(gradient_y)) if np.max(np.abs(gradient_y)) != 0 else 1) * 255

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_orientation = np.degrees(np.arctan2(gradient_y, gradient_x)) % 180
    

    for x in range(gradient_magnitude.shape[0]):
        for y in range(gradient_magnitude.shape[1]):
            if gradient_magnitude[x][y] < hog_config['Magnitude_threshold']:
                gradient_magnitude[x][y] = 0
            # else:
            #     gradient_magnitude[x][y] = hog_config['Magnitude_threshold']

    return gradient_magnitude, gradient_orientation

def get_max_values(target_list, rank):
    rank_list = heapq.nlargest(rank, target_list)
    if 0 in rank_list:
        rank_list.remove(0)
    if len(rank_list) == 0:
        result = [0 for x in target_list]
    else:
        result = [x if x in rank_list else 0 for x in target_list]
    return result    

def get_histogram(magnitude, orientation, mag_threshold, n_bins, summation_vector=False):
    max_degree = 180
    degree_axis = list(range(0, max_degree, int(180 / n_bins)))
    histogram = [0] * len(degree_axis)
    cell_size = magnitude.shape
    diff = 180/n_bins
    for x in range(cell_size[0]):
        for y in range(cell_size[1]):
            if magnitude[x][y] < mag_threshold:
                continue
            index = int(orientation[x][y]//diff)
            deg = index * diff
            histogram[index] += magnitude[x][y] * (1-(orientation[x][y]-deg)/diff)

            index += 1
            if index == n_bins:
                index = 0
            histogram[index] += magnitude[x][y] * ((orientation[x][y]-deg)/diff)
    if summation_vector: 
        temp = histogram
        sum_x = 0
        sum_y = 0
        for i, degree in enumerate(degree_axis):
            vector_sum = get_coordinate_of_vector(temp[i], degree)
            sum_x += vector_sum[0]
            sum_y += vector_sum[1]

        return np.array([sum_x, sum_y])
    
    histogram = get_max_values(histogram, 3)
    return np.array(histogram)

def get_histogram_map(image):
    image = cv2.resize(image, hog_config['Image_resize'])

    image = cv2.cvtColor(image, hog_config['Image_convert'])
    magnitude, orientation = get_gradient(image)
    """
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    image = saliencyMap
    """
    cell_size = hog_config['Cell_size']
    mag_threshold = hog_config['Magnitude_threshold']
    n_bins = hog_config['n_bins']

    histogram_map = np.zeros((hog_config['Image_resize'][0]//cell_size[0], hog_config['Image_resize'][1]//cell_size[1], n_bins))
    for x in range(0, image.shape[0], cell_size[0]):
        for y in range(0, image.shape[1], cell_size[1]):
            x_start = x
            y_start = y
            x_end = x + cell_size[0]
            y_end = y + cell_size[1]
            
            cell_magnitude, cell_orientation = magnitude[x_start:x_end, y_start:y_end], orientation[x_start:x_end, y_start:y_end]
            histogram = get_histogram(cell_magnitude, cell_orientation, mag_threshold, n_bins)
            histogram_map[x//cell_size[0]][y//cell_size[1]] = histogram
    return np.array(histogram_map)

def get_hog(IMAGE_PATH, image_name):
    image = cv2.imread(os.path.join(IMAGE_PATH, image_name))
    histogram_map = get_histogram_map(image)
    hog = np.array([])
    map_size = histogram_map.shape
    block_size = hog_config['Block_size']
    for x in range(0, map_size[0]-block_size[0]+1):
        for y in range(0, map_size[1]-block_size[1]+1):
            histogram_vector = np.array([])
            for bx in range(x, x + block_size[0]):
                for by in range(y, y + block_size[1]):
                    histogram_vector = np.concatenate((histogram_vector, histogram_map[bx][by]))
            norm = np.linalg.norm(histogram_vector)
            if norm == 0:
                norm = 1
            norm_vector = histogram_vector / norm
            hog = np.concatenate((hog, norm_vector))
    return list(hog)

def get_hog_sum_vector(IMAGE_PATH, image_name):
    image = cv2.imread(os.path.join(IMAGE_PATH, image_name))
    image = cv2.resize(image, hog_config['Image_resize'])

    image = cv2.cvtColor(image, hog_config['Image_convert'])

    cell_size = hog_config['Cell_size']
    mag_threshold = hog_config['Magnitude_threshold']
    n_bins = hog_config['n_bins']

    histogram_vector = []
    for x in range(0, image.shape[0], cell_size[0]):
        for y in range(0, image.shape[1], cell_size[1]):
            x_start = x
            y_start = y
            x_end = x + cell_size[0]
            y_end = y + cell_size[1]
            cell = image[x_start:x_end, y_start:y_end]
            magnitude, orientation = get_gradient(image=cell)
            histogram = get_histogram(magnitude, orientation, mag_threshold, n_bins, summation_vector=True)
            histogram_vector += list(histogram)
        
    return np.array(histogram_vector)
    
if __name__ == '__main__':
    # print(get_hog('./data/image/bg-20k-train', 'h_0e805e0e.jpg'))
    # show_magnitude('./data/image/temp', 'white.jpg')
    # show_vector('./data/image/temp', 'h_3f507d02.jpg')
    # show_magnitude('./data/image/temp', 'h_3f507d02.jpg')
    # get_hog('./data/image/temp', 'h_3f507d02.jpg')
    # show_vector_by_cell('./data/image/temp', 'h_3f507d02.jpg')
    cv2.waitKey(0)
    cv2.destroyAllWindows()