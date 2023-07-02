import json
import os
import datetime

def get_json_from_dict(input_dict):
    res_json = json.dumps(input_dict)
    return res_json

def save_json(FILE_PATH, filename, json_data):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S;")
    filename = now + filename + '.json'
    with open(os.path.join(FILE_PATH, filename), 'w') as f:
        json.dump(json_data, f, indent=4)

def get_dict_from_json(JSON_PATH):
    with open(JSON_PATH, 'r') as json_file:
        json_dict = json.load(json_file)
    return json_dict