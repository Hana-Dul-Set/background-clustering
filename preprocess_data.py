import numpy as np
import random
import math

def normalized_vector(vector):
    norm = math.sqrt(sum(x ** 2 for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]

def get_raw_data_list(DATA_PATH):
    ret = []
    with open(DATA_PATH, "r") as file:
        data_list = file.readlines()
        for row in data_list:
            row = row.split('/')[3]
            ret.append((eval(row)))
    return np.array(ret)

def get_normalized_raw_data_list(DATA_PATH):
    ret = []
    with open(DATA_PATH, "r") as file:
        data_list = file.readlines()
        for row in data_list:
            row = row.split('/')[3]
            ret.append(normalized_vector(np.array(eval(row))))
    return np.array(ret)

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