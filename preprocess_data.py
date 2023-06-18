import numpy as np

def get_raw_data_list(DATA_PATH):
    ret = []
    with open(DATA_PATH, "r") as file:
        data_list = file.readlines()
        for row in data_list:
            row = row.split('/')[3]
            ret.append(eval(row))
    return np.array(ret)

def get_name_data_list(DATA_PATH):
    ret = []
    with open(DATA_PATH, "r") as file:
        data_list = file.readlines()
        for row in data_list:
            row = row.split('/')[0]
            ret.append(row)
    return ret


if __name__ == "__main__":
    print(get_raw_data_list("image_assessment_data.csv"))
