# Support IO file
import os
import json
import numpy as np


def read_txt(direct_path):
    """
        @brief      read file.txt
        @param      direct_path:          file directory
        @retval     read data
    """
    
    with open(direct_path, encoding='utf-8') as txt_file:
        data = txt_file.read()
    return data

def json2dict(direct_path):
    """
        @brief      read file.json
        @param      direct_path:          file directory
        @retval     read data
    """
    
    with open(direct_path) as json_file:
        data = json.load(json_file)
    return data

def dict2json(direct_path, data):
    """
        @brief      export dictionary data to json file
        @param      direct_path:          export file directory
    """
    
    with open(direct_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        print('Your file has been created, \ncheck path {direct_path}'\
            .format(direct_path = direct_path))
        
def save_numpy(path, data, subset, mag): 
    x_path = os.path.join(path, 'numpy-data', subset + '_' + mag + '_x.npy')
    with open(x_path, 'wb') as f:
        np.save(f, data[0])
    y_path = os.path.join(path, 'numpy-data', subset + '_' + mag + '_y.npy')
    with open(y_path, 'wb') as f:
        np.save(f, data[1])
        
def get_numpy_data(direct_path, subset, mag):
    train_x_path = os.path.join(direct_path, subset + '_' + mag + '_x.npy')
    train_x = np.load(train_x_path)
    train_y_path = os.path.join(direct_path, subset + '_' + mag + '_y.npy')
    train_y = np.load(train_y_path)
    return train_x, train_y

