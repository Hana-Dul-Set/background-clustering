import numpy as np
import random
import math
import cv2
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

from hog import *


def get_normalized_vector(row):
    vector = eval(row)
    norm = math.sqrt(sum(x ** 2 for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]

def get_middle_point(vector):
    x1, x2 = vector[0], vector[2]
    y1, y2 = vector[1], vector[3]
    return (x1+x2)/2, (y1+y2)/2

def get_gradient(vector):
    x1, x2 = vector[0], vector[2]
    y1, y2 = vector[1], vector[3]
    return (y2-y1)/(x2-x1)

def get_raw_data_list_SAMPNet(DATA_PATH, normalized=False):
    ret = []
    convert_row = eval
    if(normalized):
        convert_row = get_normalized_vector
    with open(DATA_PATH, "r") as file:
        data_list = file.readlines()
        for row in data_list:
            row = row.split('/')[3]
            ret.append((convert_row(row)))
    return np.array(ret)

def append_midpoint_and_gradient(raw_data, line_list):
    line_list.sort()
    for line in line_list:
        midx, midy = get_middle_point(line)
        raw_data.append(midx)
        raw_data.append(midy)
        raw_data.append(get_gradient(line))
    return raw_data

def get_raw_data_list_DRM(DATA_PATH, normalized=False):
    ret = []
    convert_row = eval
    if(normalized):
        convert_row = get_normalized_vector
    with open(DATA_PATH, "r") as file:
        data_list = file.readlines()
        for row in data_list:
            row = row.split('/')
            out_dict = eval(row[1])['out']
            pri = out_dict['pri'][0]
            mul = out_dict['mul'][0]
            mul.remove(pri)
            raw_data = []
            append_midpoint_and_gradient(raw_data, [pri])
            append_midpoint_and_gradient(raw_data, mul)
            ret.append(raw_data)
    max_len = max(len(x) for x in ret)
    for raw_data in ret:
        if len(raw_data) < max_len:
            for cnt in range(max_len - len(raw_data)):
                raw_data.append(0)
    return ret

def get_name_data_list(DATA_PATH):
    ret = []
    with open(DATA_PATH, "r") as file:
        data_list = file.readlines()
        for row in data_list:
            row = row.split('/')[0]
            ret.append(row)
    return ret

def get_random_elements(array, random_element_cnt):
    if(len(array) < random_element_cnt):
        random_element_cnt = len(array)
    random_indices = random.sample(range(len(array)), random_element_cnt)
    random_elements = [array[i] for i in random_indices]

    return random_elements

def get_elements_list_by_label(label_list, name_list, label_cnt):
    data_by_label_list = []
    for i in range(label_cnt):
        one_label_list = []
        for j in range(len(label_list)):
            if(label_list[j] == i):
                one_label_list.append(name_list[j])
        data_by_label_list.append(one_label_list)
    return data_by_label_list

def get_dict_of_elements_list_by_label(label_list, name_list, label_cnt):
    data_by_label_list = {}
    for i in range(label_cnt):
        one_label_list = []
        for j in range(len(label_list)):
            if(label_list[j] == i):
                one_label_list.append(name_list[j])
        data_by_label_list[str(i)] = one_label_list
    return data_by_label_list

def get_vector_from_image(IMAGE_PATH, img_name):

    # Load the image
    image = cv2.imread(os.path.join(IMAGE_PATH, img_name))
    print(os.path.join(IMAGE_PATH, img_name))
    print(image.shape)
    image = cv2.resize(image, dsize=(100, 100))

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Create a saliency object using the SALIENCY_OTSU algorithm
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    # Calculate the saliency map
    success, gray = saliency.computeSaliency(gray)

    # Flatten the image into a 1D vector
    flattened_image = gray.reshape(-1)

    # Print the shape and the flattened vector
    print("Flattened image shape:", flattened_image.shape)
    print("Flattened image vector:", flattened_image)

    return flattened_image.tolist()

def get_raw_data_list_temp_vector(DATA_PATH, normalized=False):
    ret = []
    convert_row = eval
    if(normalized):
        convert_row = get_normalized_vector
    with open(DATA_PATH, "r") as file:
        data_list = file.readlines()
        for row in data_list:
            row = row.split('/')[1]
            ret.append((convert_row(row)))
    return np.array(ret)

def get_distance(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    vector1 = vector1.reshape(1, -1)
    vector2 = vector2.reshape(1, -1)
    similarity = cosine_similarity(vector1, vector2)
    
    # similarity = euclidean(vector1, vector2)
    return similarity

def get_raw_data_list_hog(DATA_PATH, normalized=False):
    ret = []
    convert_row = eval
    if(normalized):
        convert_row = get_normalized_vector
    with open(DATA_PATH, "r") as file:
        data_list = file.readlines()
        for row in data_list:
            row = row.split('/')[1]
            ret.append((convert_row(row)))
    return np.array(ret)    

def get_nearest_images(target_image_name, raw_data_list, name_list, nearest_image_cnt):
    target_image_data = raw_data_list[name_list.index(target_image_name)]
    data_and_name_list = []
    for i in range(len(name_list)):
        if name_list[i] == target_image_name:
            continue
        distance = euclidean(target_image_data, raw_data_list[i])
        data_and_name_list.append([distance, name_list[i], raw_data_list[i]])
    data_and_name_list.sort()
    return data_and_name_list[:nearest_image_cnt]

if __name__ == '__main__':
    image_list = os.listdir('./data/image/bg-20k-train')
    if '.DS_Store' in image_list:
        image_list.remove('.DS_Store')
    hog_list = []
    print(image_list)
    filename = get_config_string() + ';proportion' + '.csv'
    if os.path.exists('./data/csv/' + filename):
        print("There exists file!")
    else:
        for img in tqdm(image_list):
            with open('./data/csv/' + filename, 'a') as f:
                f.writelines(img + '/' + str(get_hog("./data/image/bg-20k-train", img)) + '\n')

    