import sys
import os
import re
from CreateData import *

def main():
    first_index = int(sys.argv[1])
    try:
        last_index = int(sys.argv[2])
    except IndexError:
        last_index = None
    # first_index = int(input('Please type first index: '))
    # last_index = None
    print('\nWelcom to ClearErrorData.py')
    path = os.path.dirname(os.path.abspath(__file__))
    created_data_folder_path = os.path.join(path, 'data\positive') 
    clear_file_by_name_index(created_data_folder_path, first_index, last_index)
    
def clear_file_by_name_index(direct_path, first_index, last_index):
    all_filenames = os.listdir(direct_path)
    if last_index is None:
        last_index = get_count(direct_path) + 1
    del_index = range(first_index, last_index)
    for filename in all_filenames:
        cur_index = int(re.findall(r'\d+', filename)[0])
        if cur_index in del_index:
            os.remove(os.path.join(direct_path, filename))
            
if __name__ == "__main__": 
    main()