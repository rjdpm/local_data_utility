import os, json
import sys, csv
import gzip, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

import torch

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore", module="mordred")

__all__ = [
    'create_file',
    'create_folder',
    'save_images',
    'list2json',
    'dict2json',
    'savedict2json',
    'dict_to_csv',
    'add_results2df',
    'save_list2json',
    'save2pickle',
    'save_dict_pickle',
    'write_csv_columnwise',
    'write_csv_rowwise',
    'writer_dict_csv',
]

def create_file(folder_path, file_name = ''):
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    if file_name:
        file_path = os.path.join(folder_path, file_name)
    else:
        file_path = folder_path
    open(file_path, 'a').close()
    
    return file_path

def create_folder(folder_name: str) -> str:
    
    if len(folder_name):
        if not os.path.isdir(folder_name):
            print(f'Creating Folder:{folder_name}')
            os.makedirs(folder_name)
        
    return folder_name

def save_images(img, name_list, save_path):

    save_path = create_folder(save_path)
    drug_iter = iter(name_list)
    for im in img:
        drug_name = next(drug_iter)
        im.save(save_path + drug_name + '.png')

    return None

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
def list2json(input_list, filename='untitled', filepath='./'):
    """
    Saves a list of serializable Python objects to a JSON file.

    Parameters:
        input_list (list): A list of Python objects (e.g., dicts, tuples, arrays).
        filename (str): The name of the output file without extension.
        filepath (str): Directory path where the file will be saved.

    Returns:
        None
    """
    os.makedirs(filepath, exist_ok=True)
    full_path = os.path.join(filepath, f"{filename}.json")

    with open(full_path, 'w') as json_file:
        json.dump(input_list, json_file, indent=4, cls=NumpyJSONEncoder)
    print(f'File saved to: {full_path}')

    return None

def dict2json(dict_, filepath='./'):
    
    # with open(f'{filepath}/{filename}.json', 'w') as json_file:
    #     json.dump(json_str, json_file, indent=4)#, cls=NumpyJSONEncoder)
    json_str = json.dumps(dict_, indent=4)
    with open(filepath, "w") as f:
        f.write(json_str)
    print(f'JSON file saved in: {filepath}')
    
def savedict2json(data: dict, path: str, to_sort: bool = False):
    """Save a dictionary to a readable JSON file, converting NumPy, Torch, sets, tuples, etc."""
    
    def convert(obj):
        # numpy array
        if isinstance(obj, np.ndarray):
            return sorted(obj.tolist()) if to_sort else obj.tolist()

        # numpy scalar
        if isinstance(obj, np.generic):
            return obj.item()

        # torch tensor
        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
            return sorted(arr.tolist()) if to_sort else arr.tolist()

        # set → list
        if isinstance(obj, set):
            lst = list(obj)
            return sorted(lst) if to_sort else lst

        # tuple → list
        if isinstance(obj, tuple):
            return [convert(x) for x in obj]

        # list → recursively convert elements
        if isinstance(obj, list):
            return [convert(x) for x in obj]

        # dict → recursively convert values
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}

        return obj

    serializable_data = convert(data)

    with open(path, 'w') as f:
        json.dump(serializable_data, f, indent=4)

def dict_to_csv(dictionary: dict, header: list, filepath: str, filename: str):
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    filepath = ''.join([filepath, filename, '.csv'])

    with open(filepath, mode ='w')as file:
        writer = csv.writer(file)
        writer.writerow(header) #['string', 'Names']

        for row in zip(keys, values):
            writer.writerow(row)
            
def add_results2df(results: OrderedDict,
                   results_savename: str='./Untitled.csv'):
    
    if isinstance(results, OrderedDict):
        results = pd.DataFrame([results])
    if not os.path.isfile(results_savename):
        results.to_csv(results_savename, index = False)
    else:
        df = pd.read_csv(results_savename)
        df = pd.concat([df, results], ignore_index=True)
        df.to_csv(results_savename, index=False)

def save_list2json(file_: list,
                   filepath: str
                   ) -> list:
    
    with open(f'{filepath}', 'w') as json_file:
            json.dump([s for s in file_], json_file, indent=4)
            
    return file_

def save2pickle(file_: object, filepath: str, compress: bool = False):
    """
    Save Python object to a pickle file.
    If compress=True, saves in gzip format (regardless of extension).
    """
    _ = create_folder(filepath[:-len(filepath.split('/')[-1])])
    if compress or filepath.endswith(".gz"):
        with gzip.open(filepath, "wb") as fp:
            pickle.dump(file_, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(filepath, "wb") as fp:
            pickle.dump(file_, fp, protocol=pickle.HIGHEST_PROTOCOL)

def save_dict_pickle(dict_, save_filename='temp_save_filename.pkl', protocol=4):

    # create folder (if not) 
    _ = create_folder(save_filename[:-len(save_filename.split('/')[-1])])
    with open(save_filename,'wb') as f:
        pickle.dump(dict_, f, protocol=protocol)

def write_csv_columnwise(file_path, file_name, headers = [], column_list = []):
    
    '''
    - headers : This must be the list of strings of the filednames.
    - column_list : This must be list of lists. First element of column_list should be the first column of the csv file.
       
    - **csv file will contain number of rows = min of column lengths of the given column list.
    '''

    if len(headers) == len(column_list):
        dicts = {}
        save_path = create_file(file_path, file_name)
        min_len = sorted(len(column_list[i]) for i in range(len(column_list)))[0]
        
        with open(save_path, mode ='w') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            
            for j in range(min_len):
                for i in range(len(column_list)):
                    dicts[headers[i]] = column_list[i][j]
                writer.writerow(dicts)

    else:
        raise AttributeError(f'header length = {len(headers)} not equal length of column list = {len(column_list)}')
    
    file_ = pd.read_csv(save_path)
    return file_

def write_csv_rowwise(file_path, file_name, rows: list, headers = []):


    '''
    - rows: This must be list of lists. For example: first element of rows should be the first row of the csv file.
    - headers: This must be the list of strings of the filednames.
    '''

    file_path = create_file(file_path, file_name)

    with open(file_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for row in rows:
            writer.writerow(row)

    file_ = pd.read_csv(file_path)

    return file_

def writer_dict_csv(*dicts, headers, file_path, file_name):

    file_path = create_file(file_path, file_name)

    with open(file_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for d in dicts:
            writer.writerow(d)

    file_ = pd.read_csv(file_path)

    return file_