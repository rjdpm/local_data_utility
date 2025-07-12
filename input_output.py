"""
-----------------------------------------------------------------------------
File read, write helpers
-----------------------------------------------------------------------------
AUTHOR: Soumitra Samanta (soumitramath39@gmail.com)
-----------------------------------------------------------------------------
"""

import os
import pickle
import pandas as pd
from tqdm import tqdm
from typing import Union, Tuple, Dict, List, Any

__all__ = [
    'create_folder',
    'save_dict_csv_pandas', 
    'save_dict_xlsx_pandas',
    'save_dict_pickle',
    'load_dict_pickle', 
    'save_list_pickle',
    'load_list_pickle',
    'read_csv_file_to_dict',
    'read_xlsx_file_to_dict',
    'line_count_csv_file',
    'save_value_to_txt',
    'save_list_to_txt',
    'read_parquet_file_to_dict_pandas',
    'save_dict_parquet_pandas',
    
]


def create_folder(folder_name: str) -> None:
    """Create folder if not exist"""
    
    if len(folder_name):
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
        
    return folder_name


def save_dict_csv_pandas(
    dict_name: Dict[str, List], 
    save_filename: str = 'temp_save_filename.csv',
    index=False,
) -> None:
    """Save a dictionary into a csv file"""
    
    # create folder (if not) 
    _ = create_folder(save_filename[:-len(save_filename.split('/')[-1])])
    pd.DataFrame.from_dict(dict_name).to_csv(save_filename, index=index)

    
def save_dict_xlsx_pandas(
    dict_name: Dict[str, List], 
    save_filename: str = 'temp_save_filename.xlsx',
    index=False,
) -> None:
    """Save a dictionary into a xlsx file"""
    
    # create folder (if not) 
    _ = create_folder(save_filename[:-len(save_filename.split('/')[-1])])
    pd.DataFrame.from_dict(dict_name).to_excel(save_filename, index=index)

    
def save_dict_pickle(
    dict_name: Dict[str, List], 
    save_filename: str = 'temp_save_filename.pkl', 
    protocol: int = 4
) -> None:
    """Save a dictionary into a pkl file"""
    
    # create folder (if not) 
    _ = create_folder(save_filename[:-len(save_filename.split('/')[-1])])
    with open(save_filename,'wb') as f:
        pickle.dump(dict_name, f, protocol=protocol)
        
        
def load_dict_pickle(input_filename: str) -> Dict[str, List]:
    """Load data from a .pkl file"""
    
    with open(input_filename,'rb') as f:
        dict_name = pickle.load(f)
    
    return dict_name


def save_list_pickle(
    list_name: List, 
    save_filename: str = 'temp_save_filename.pkl', 
) -> None:
    """Save a dictionary into a pkl file"""
    
    # create folder (if not) 
    _ = create_folder(save_filename[:-len(save_filename.split('/')[-1])])
    with open(save_filename,'wb') as f:
        pickle.dump(list_name, f)
        
        
def load_list_pickle(
    input_filename: str, 
    verbose: bool = True
) -> List:
    """Load data from a .pkl file"""
    
    if verbose:
        print('='*70)
        print('Reading data from "{}"' .format(input_filename))
        print('-'*70)
        
    with open(input_filename,'rb') as f:
        list_name = pickle.load(f)
    if verbose:
        print('Done!')
        print('='*70)
        
    return list_name


def read_csv_file_to_dict(
    filename: str,
    verbose: bool = True
)->Dict:
    """Read a csv file into a dictionary"""
    
    if verbose:
        print('='*70)
        print('Reading data from "{}"' .format(filename))
        print('-'*70)
        
    df_data = pd.read_csv(filename)
    dict_data = {}
    for c in df_data.keys():
        dict_data[c] = df_data[c].tolist()
    if verbose:
        print('Done!')
        print('='*70)
        
    return dict_data

def read_xlsx_file_to_dict(
    filename: str,
    verbose: bool = True
)->Dict:
    """Read a xlsx file into a dictionary"""
    
    if verbose:
        print('='*70)
        print('Reading data from "{}"' .format(filename))
        print('-'*70)
        
    df_data = pd.read_excel(filename)
    dict_data = {}
    for c in df_data.keys():
        dict_data[c] = df_data[c].tolist()
    if verbose:
        print('Done!')
        print('='*70)
        
    return dict_data
    


def line_count_csv_file(
    filename: str, 
    chunksize: int = 1000, 
) -> int:
    """Count number of lines in csv file"""
    
    print('Counting number of data point in: "{}"' .format(filename))
    print('-'*70)
    count = 0
    for chunk in tqdm(pd.read_csv(filename, usecols=[0], chunksize=chunksize)):
        count += len(chunk)
            
    return count


    
def save_value_to_txt(
    value: Any, 
    save_filename: str = 'temp_save_filename.txt',
) -> None:
    """Save a value into a txt file"""
    
    # create folder (if not) 
    _ = create_folder(save_filename[:-len(save_filename.split('/')[-1])])
    with open(save_filename, 'w') as fn:
        fn.write(str(value))
        
        
def save_list_to_txt(
    list_values: List[Any], 
    save_filename: str = 'temp_save_filename.txt',
) -> None:
    """Save a list into a txt file"""
    
    # create folder (if not) 
    _ = create_folder(save_filename[:-len(save_filename.split('/')[-1])])
    with open(save_filename, 'w') as fn:
        for v in list_values:
            fn.write(str(v) + '\n')
            
            
def read_parquet_file_to_dict_pandas(
    filename: str,
    verbose: bool = True
)->Dict:
    """Read a parquet file into a dictionary"""
    
    if verbose:
        print('='*70)
        print('Reading data from "{}"' .format(filename))
        print('-'*70)
    df_data = pd.read_parquet(filename)
    dict_data = {}
    for c in df_data.keys():
        dict_data[c] = df_data[c].tolist()
    if verbose:
        print('Done!')
        print('='*70)
        
    return dict_data

def save_dict_parquet_pandas(
    dict_name: Dict[str, List], 
    save_filename: str = 'temp_save_filename.parquet',
    index=False,
) -> None:
    """Save a dictionary into a csv file"""
    
    # create folder (if not) 
    _ = create_folder(save_filename[:-len(save_filename.split('/')[-1])])
    pd.DataFrame.from_dict(dict_name).to_parquet(save_filename, index=index)
