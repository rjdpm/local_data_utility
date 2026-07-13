import re
import os
import ast
import sys
import csv
import umap
import gzip
import pytz
import shap
import copy
import json
import math
import time
import pickle
import random
import shutil
import inspect
import sklearn
import warnings
import tempfile
import configparser
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pympler import asizeof
import matplotlib
# matplotlib.use('TkAgg')
from umap.umap_ import UMAP
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
from contextlib import contextmanager
from ordered_set import OrderedSet
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from typing import Any, List, Tuple, Union, Callable, Optional, Dict
from PIL import Image, ImageDraw, ImageFont

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import label_binarize
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, TruncatedSVD
from sklearn.svm import SVC, SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report,
                             roc_curve, auc, precision_recall_curve,
                             pairwise_distances
                             )


import torch
import torch.nn as nn
from IPython.display import display
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import DataLoader as PyGDataLoader
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Disable all logs from the 'rdApp' logger (the main source of RDKit messages)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
# Disable all logs from mordred library
import logging
logging.getLogger('mordred').setLevel(logging.CRITICAL)


plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['font.style'] = 'italic'

__all__ = [
    'show_img',
    'get_dir',
    'print_results',
    'print_results_all',
    'list_diff',
    'save_images',
    'matrix2onehot_encode',
    'datetime_now',
    'round_up',
    'get_scale_power',
    'execution_time',
    
    'parse_any',
    'check_tensor',
    'Normalize',
    'fix_seed',
    'auto_repr',
    'get_func_input_names',
    'suppress_output',
    'python_object_size',
    'get_own_methods',
    'get_own_methods_with_docstrings',
    'collect_class_definitions',
    
    'create_file',
    'create_temp_config',
    'create_folder',
    'attach_ids',
    'list2json',
    'dict2json',
    'load_json',
    'save_list2json',
    'save2pickle',
    'load_from_pickle',
    'read_data_from_excel',
    'dict_to_csv',
    'savedict2json',
    'add_results2df',
    'read_csv2list',
    'read_csv2array',
    'read_csv2dict',
    'write_csv_columnwise',
    'write_csv_rowwise',
    'writer_dict_csv',
    'save_dict_csv_pandas',
    'save_dict_pickle',
    'load_dict_pickle',
    'line_count_csv_file',
    
    'data_partition_random_indices',
    'data_split_random_df',
    'selective_range_data_sampling',
    'selective_range_data_split',
    'data_split_nfold',
    'classify_columns',
    
    'exclude_strings',
    'str2chars',
    'liststr2chars',
    'padding',
    'liststring2max_len',
    'get_tokens',
    'string2onehot',
    'liststring2onehot',
    'onehot2string',
    'bool_onehot_encodings2string',
    'float_onehot_encodings2string',
    
    'PCA_fit_transform',
    'KMeans_fit_predict',
    'DBScan_fit_predict',
    
    'get_colors',
    'apply_reductions',
    'plot_reductions',
    'plot_projection_grid_seaborn',
    'plot_3d',
    'plot_2d',
    'collage_from_dict',
    'customised_plot',
    
    'convex_combination',
    'convex_PCA',
    'convex_inv_PCA',
    
    'euc_dist',
    'hyperspherical_distance',
    'pairwise_hyperspherical_distance',
    'near_points',
    
    'elementary_ops_interchange_rows',
    'elementary_ops_scale_row',
    'elementary_ops_change_row',
    'elementary_ops_interchange_rows_mult',
    'elementary_ops_scale_row_mult',
    'elementary_ops_change_row_mult',
    'elementary_ops_interchange_cols',
    'elementary_ops_scale_col',
    'elementary_ops_change_col',
    'sweep_out_row',
    'sweep_out_col',
    'pivoting',
    'row_reduced_echelon_form',
    
    'clamp',
    'surrounding_coordinates',
    'bytesarray2pyfile',
    'pyfile2bytesarray',
    'get_sub',   
    
    'make_kfold_indices',
    'df2cleaned_df',
    'merge_list_columns_by_key',
    'remove_highly_correlated_columns',
    'hierarchical_feature_selection',
    'data_stat',
    'find_outliers_iqr',
    'get_outliers',
    
    'parity_plot_with_folds_and_percent',
    'parity_plot_with_folds',
    'bland_altman_plot',
    'plot_distribution_compair',
    'plot_multiple_distribution',
    'plot_multiple_hist',
    'plot_hist_compair',
    'dist_hist_comparison',
    'plot_stacked_bars',
    
    'percentage_within_percent_error',
    'percentage_within_fold_change',
    'geometric_mean_fold_error',
    'min_max',
    'regression_test_metrics',
    'test_rf',
    'subset_loader',
    'count_linear_layers',
    'normalize_data_labels',
    
    'ClassificationResultAnalyzer',
    'TrendAnalyser',
    'Any', 'List', 'Tuple', 'Union', 'Callable', 'Optional', 'Dict',
]
    
    
def show_img(img):

    [display(x) for x in img]

    return None

def get_dir(path):
    return '/'.join(path.split('/')[:-1])
     

# def print_results(train_results, test_results, val_results):
    
#     print('='*70)
#     print(f"{'& Data':7} & {'R2':7} & {'RMSE':7} & {'MAE':7} & {'PCC':7} & {'GMFE':7} & {'Fold-2':7} & {'Fold-3':7} & {'Fold-5':7}")
#     print('-'*70)
#     for k, results in {'& Train': train_results, '& Test': test_results, '& Val': val_results}.items():
#         print(f"{k:7} & "
#             f"{str(results['r2']):7} & "
#             f"{str(results['rmse']):7} & "
#             f"{str(results['mae']):7} & "
#             f"{str(results['PCC']):7} & "
#             f"{str(results['GMFE']):7} & "
#             f"{str(results['fold2']):7} & "
#             f"{str(results['fold3']):7} & "
#             # f"{str(results['fold5']):7} {r"\\"}")
#             f"{str(results['fold5']):7} \\\\")
#     print('='*70)
    
def print_results(train_results, test_results, val_results):
    
    print('='*70)
    print(f"{'& Data':7} & {'R2':7} & {'RMSE':7} & {'MAE':7} & {'PCC':7} & {'GMFE':7} & {'Fold-2':7} & {'Fold-3':7} & {'Fold-5':7}")
    print('-'*70)
    for k, results in {'& Train': train_results, '& Test': test_results, '& Val': val_results}.items():
        print(f"{k:7} & "
            f"{results['r2']:7.2f} & "
            f"{results['rmse']:7.2f} & "
            f"{results['mae']:7.2f} & "
            f"{results['PCC']:7.2f} & "
            f"{results['GMFE']:7.2f} & "
            f"{results['fold2']:7.2f} & "
            f"{results['fold3']:7.2f} & "
            # f"{results['fold5']:7.2f} {r"\\"}")
            f"{results['fold5']:7.2f} \\\\")
    print('='*70)
    
    
# def print_results_all(results_dict):
    
#     print('='*70)
#     print(f"{'& Data':7} & {'R2':7} & {'RMSE':7} & {'MAE':7} & {'PCC':7} & {'GMFE':7} & {'Fold-2':7} & {'Fold-3':7} & {'Fold-5':7}")
#     print('-'*70)
#     for k, results in results_dict.items():
#         print(f"& {k:7} & "
#             f"{str(results['r2']):7.2f} & "
#             f"{str(results['rmse']):7.2f} & "
#             f"{str(results['mae']):7.2f} & "
#             f"{str(results['PCC']):7.2f} & "
#             f"{str(results['GMFE']):7.2f} & "
#             f"{str(results['fold2']):7.2f} & "
#             f"{str(results['fold3']):7.2f} & "
#             # f"{str(results['fold5']):7.2f} {r"\\"}")
#             f"{str(results['fold5']):7.2f} \\\\")
#     print('='*70)

def print_results_all(results_dict, precision=2):

    if not results_dict:
        print("No results to display.")
        return

    # Collect all metric keys dynamically
    all_metrics = set()
    for results in results_dict.values():
        all_metrics.update(results.keys())

    # Preserve a sensible order (optional priority)
    preferred_order = ['r2', 'rmse', 'mae', 'PCC', 'GMFE']
    remaining = sorted([m for m in all_metrics if m not in preferred_order])
    metrics = preferred_order + remaining

    # Filter only those actually present
    metrics = [m for m in metrics if m in all_metrics]

    # Header
    header = ["Data"] + metrics
    col_width = max(7, max(len(h) for h in header))

    print("=" * (len(header) * (col_width + 3)))

    # Print header row
    header_row = " & ".join(f"{h:>{col_width}}" for h in header)
    print(header_row)

    print("-" * (len(header) * (col_width + 3)))

    # Print rows
    for k, results in results_dict.items():
        row = [k]

        for m in metrics:
            val = results.get(m, None)
            if val is None:
                row.append("NA")
            else:
                row.append(f"{val:.{precision}f}")

        print(" & ".join(f"{r:>{col_width}}" for r in row) + " \\\\")

    print("=" * (len(header) * (col_width + 3)))


def list_diff(list1, list2):
    
   temp =  [item for item in list1 if item not in list2]
   
   return temp

# Given drug images it will save the images in the given path with their corresponding names :-
def save_images(img, name_list, save_path):

    save_path = create_folder(save_path)
    drug_iter = iter(name_list)
    for im in img:
        drug_name = next(drug_iter)
        im.save(save_path + drug_name + '.png')

    return None

def matrix2onehot_encode(matrix):

    '''Matrix should be 3-dimesional matrix'''

    if torch.is_tensor(matrix):
        matrix = matrix.detach().numpy()

    idx = np.argmax(matrix, axis = 2)
    one_hot_enc = np.zeros_like(matrix)
    for k in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            one_hot_enc[k, j, idx[k, j]] = 1

    return one_hot_enc

def datetime_now(path: bool = True):
    """
    Returns current datetime in Asia/Kolkata timezone.

    Returns:
        dt   : full formatted datetime string
        date : YYYY_MM_DD
        time : HH_MM_SS
    """
    ist = pytz.timezone("Asia/Kolkata")

    # single timestamp (IMPORTANT)
    now = datetime.now(ist)
    dt = now.strftime("%Y%m%d_%H_%M_%S" if path else "%Y/%m/%d [%H:%M:%S]")
    date = now.strftime("%Y_%m_%d")
    time = now.strftime("%H_%M_%S")

    return dt, date, time

def execution_time(func, *args):

    tic = time.time()
    func_value = func(*args)#Calculate function value
    toc = time.time()

    return (toc - tic), func_value

def round_up(num: float = 1.987,
             digit: int = 2
             ) -> float:
    
    if len(str(num).split('.')[-1]) >= digit:
        dec = 10**digit
        temp = math.floor(num * dec) / dec
    else:
        temp = num
    
    return temp

def get_scale_power(value: float) -> int:
    """
    Returns the power of 10 scale a value falls into.
    - For example, 0.1 to 0.9 => -1, 1 to 9 => 0, 10 to 99 => 1, etc.
    - Handles negative values and 0 appropriately.

    Parameters:
        value (float): The input value.

    Returns:
        int: The scale as a power of 10.
    """
    if value == 0:
        return -1e15#float('-inf')  # Logarithmically undefined scale
    abs_value = abs(value)
    log_value = math.log10(abs_value)
    powerof10 = int(math.floor(log_value))
    
    return powerof10

def parse_any(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return str(value)

def check_tensor(t, name, max_print=10):
    nan_idx = torch.isnan(t).nonzero(as_tuple=False)
    inf_idx = torch.isinf(t).nonzero(as_tuple=False)

    if nan_idx.numel() > 0:
        print(f"[NaN] detected in {name} "
            f"(count={nan_idx.shape[0]})")
        print(" NaN coordinates (first few):")
        for idx in nan_idx[:max_print]:
            print("  ", idx.tolist())

    if inf_idx.numel() > 0:
        print(f"[Inf] detected in {name} "
            f"(count={inf_idx.shape[0]})")
        print(" Inf coordinates (first few):")
        for idx in inf_idx[:max_print]:
            print("  ", idx.tolist())

    try:
        min_val = t.min().item()
        max_val = t.max().item()
        print(f"{name}: min={min_val:.3e}, max={max_val:.3e}")
    except RuntimeError:
        # min/max fail if tensor is all-NaN
        print(f"{name}: min/max undefined (all NaN?)")
        
# Function to Normalize an array :-
def Normalize(array):
    
    mean, std = np.mean(array), np.std(array)
    result = (array - mean)/std
    
    return result
        
def get_func_input_names(func):
    
    sig = inspect.signature(func)
    param_list =  [param.name for param in sig.parameters.values()
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)]
    
    return param_list

def get_own_methods(child_cls):
    """
    Retrieve methods defined directly within a given class and distinguish them 
    from methods inherited from parent classes.

    Parameters:
        child_cls (type): The class object to inspect.

    Returns:
        Tuple[
            List[Tuple[str, Callable]],  # Methods defined in the class itself
            List[Tuple[str, Callable]]   # Methods inherited from parent classes
            ]: 
            A tuple containing two lists:
            - The first list consists of (method_name, method_object) tuples 
              for methods defined directly in the class.
            - The second list contains methods inherited from any superclass.
    """
    own_method_names = set(child_cls.__dict__)
    all_methods = inspect.getmembers(child_cls, predicate=inspect.isfunction)
    
    own_methods = [(name, method) for name, method in all_methods if name in own_method_names]
    inherited_methods = [(name, method) for name, method in all_methods if name not in own_method_names]
    
    return own_methods, inherited_methods

def get_own_methods_with_docstrings(class_):
    """
    Extracts the docstrings of all methods defined directly within the given class.

    This function inspects the class and returns a dictionary mapping the names of 
    methods that are explicitly defined in the class (i.e., not inherited) to their 
    respective docstrings.

    Parameters:
        class_ (type): The class object to inspect.

    Returns:
        dict: A dictionary where keys are method names (str) and values are 
              their corresponding docstrings (str). If a method lacks a docstring, 
              the value will be the placeholder string "(No docstring provided)".
    """
    own_method_names = set(class_.__dict__)
    method_docs = {}
    
    for name, method in inspect.getmembers(class_, predicate=inspect.isfunction):
        if name in own_method_names:
            doc = inspect.getdoc(method)
            method_docs[name] = doc or "(No docstring provided)"
    
    return method_docs


def fix_seed(np_seed=30, torch_seed=30, random_seed=30):
    
    random.seed(random_seed)
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
def auto_repr(self):
    """
    Generalized __repr__ for classes.
    Constructs a multi-line string showing the class name and __init__ arguments with their current values.
    """
    cls_name = self.__class__.__name__
    try:
        sig = inspect.signature(self.__init__)
        args = [p for p in sig.parameters if p != "self"]
    except Exception:
        args = list(vars(self).keys())

    body = ",\n  ".join(f"{k}={repr(getattr(self, k, None))}" for k in args)
    
    return f"{cls_name}(\n  {body}\n)"

def collect_class_definitions(self):
    """
    Collect source code for all classes in the inheritance chain.
    """
    definitions = []
    for cls in self.__class__.__mro__[::-1]:  # base to top
        if cls in (object, nn.Module):
            continue
        definitions.append(inspect.getsource(cls))
        
    return "\n".join(definitions)


def python_object_size(obj: Union[str, Any], use_deep_size: bool = True) -> str:
    """
    Get the size of a file or Python object in a human-readable format.

    Parameters:
        obj (str or Any): Path to the file or any Python object.
        use_deep_size (bool): Whether to use deep size estimation for objects.

    Returns:
        str: Size of the object in KB, MB, or GB.
    """
    if isinstance(obj, str) and os.path.isfile(obj):
        size = os.path.getsize(obj)
    else:
        size = asizeof.asizeof(obj) if use_deep_size else sys.getsizeof(obj)

    if size < 1024:
        return f"{size} bytes"
    elif size < 1024**2:
        return f"{size / 1024:.2f} KB"
    elif size < 1024**3:
        return f"{size / 1024**2:.2f} MB"
    else:
        return f"{size / 1024**3:.2f} GB"

@contextmanager
def suppress_output():
    """Context manager to suppress prints in a block of code."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def create_folder(folder_name: str) -> str:
    
    if len(folder_name):
        if not os.path.isdir(folder_name):
            print(f'Creating Folder:{folder_name}')
            os.makedirs(folder_name)
        
    return folder_name

def attach_ids(df, prefix, id_col='UNIQUE_ID'):
    df = df.copy()
    width = get_scale_power(len(df)) + 3
    if id_col not in df.columns:
        df.insert(0, id_col, 'NA')
    df[id_col] = [f"{prefix}{i:0{width}d}" for i in range(1, len(df) + 1)]
    return df

def create_file(folder_path, file_name = ''):
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    if file_name:
        file_path = os.path.join(folder_path, file_name)
    else:
        file_path = folder_path
    
    open(file_path, 'a').close()
    
    return file_path

def create_temp_config(config_path, section, **kwargs):
    temp_config_path = tempfile.mktemp(suffix='.ini')
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    if not config.has_section(section):
        config.add_section(section)
    
    for key, value in kwargs.items():
        config.set(section, key, value)
    
    with open(temp_config_path, 'w') as configfile:
        config.write(configfile)
    
    return temp_config_path

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

# Function to load a json file :-
def load_json(path):

    with open(path, 'r') as fp:
        data = json.loads(fp.read())

    return data

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
        
# def savedict2json(data: dict, path: str, to_sort:bool=False):
    # """Save a dictionary to a readable JSON file, converting sets to lists."""
#     def convert(obj):
#         if isinstance(obj, (set, np.ndarray)):
#             return sorted(list(obj)) if to_sort else list(obj)
#         if isinstance(obj, torch.Tensor):
#             return sorted(list(obj.numpy())) if to_sort else list(obj.numpy())
#         if isinstance(obj, dict):
#             return {k: convert(v) for k, v in obj.items()}
        
#         return obj

#     serializable_data = {k: convert(v) for k, v in data.items()}

#     with open(path, 'w') as f:
#         json.dump(serializable_data, f, indent=4)

# Function to change a dictionary object to a csv file :-
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


def load_from_pickle(filepath: str):
    """
    Load Python object from a pickle file.
    Automatically detects gzip compression (even without .gz extension).
    """
    with open(filepath, "rb") as f:
        magic = f.read(2)
    
    if magic == b"\x1f\x8b":  # gzip magic number
        with gzip.open(filepath, "rb") as fp:
            file_ = pickle.load(fp)
    else:
        with open(filepath, "rb") as fp:
            file_ = pickle.load(fp)
    return file_


def read_data_from_excel(file_path: str, sheet_num: int = None):
    
    # To read all sheets into a dictionary of DataFrames
    sheets_dict = pd.read_excel(file_path, sheet_name=None)
    sheets = []
    # dataframes = []
    # Access each sheet by name
    for sheet_name in sheets_dict.keys():
        sheets.append(sheet_name)
        # dataframes.append(df)  # Display the first few rows of each sheet
    
    if not isinstance(sheet_num, int): 
        print(f'Keys: {list(sheets_dict.keys())}')
        return sheets_dict
    else:   
         return sheets_dict[sheets[sheet_num]]
    

def save_dict_csv_pandas(dict_name, save_filename='temp_save_filename.csv'):

    # create folder (if not) 
    _ = create_folder(save_filename[:-len(save_filename.split('/')[-1])])
    pd.DataFrame.from_dict(dict_name).to_csv(save_filename, index=False)



def save_dict_pickle(dict_, save_filename='temp_save_filename.pkl', protocol=4):

    # create folder (if not) 
    _ = create_folder(save_filename[:-len(save_filename.split('/')[-1])])
    with open(save_filename,'wb') as f:
        pickle.dump(dict_, f, protocol=protocol)



def load_dict_pickle(input_filename):
    
    with open(input_filename,'rb') as f:
        dict_name = pickle.load(f)

    return dict_name



def line_count_csv_file(filename, chunksize=1000):

    print('Counting number of data point in: "{}"' .format(filename))
    count = 0
    for chunk in tqdm(pd.read_csv(filename, usecols=[0], chunksize=chunksize)):
        count += len(chunk)

    return count

def read_csv2list(file_path):
    
    '''
    - Reads a .csv file in the given filepath and returns it in a list format.
    '''

    with open(file_path, mode ='r')as file:
        csv_reader = csv.reader(file)

        headers = next(csv_reader)
        rows = []
        # displaying the contents of the CSV file
        for line in csv_reader:
            rows.append(line)

    return rows, headers



def read_csv2array(file_path):

    file_ = pd.read_csv(file_path)
    file_ = np.array(file_)

    return file_


def read_csv2dict(file_path):

    read_dict = {}

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames
        i=0
        for row in reader:
            read_dict[i] = row
            i+=1

    return read_dict, headers



def write_csv_columnwise(file_path, file_name, headers = [], column_list = []):
    
    '''
    - headers : This must be the list of strings of the filednames.
    - column_list : This must be list of lists. First element of column_list should be the first column of the csv file.
       
    - **csv file will contain number of rows = min of column lengths of the given column list.
    '''

    if len(headers) == len(column_list):

        save_path = create_file(file_path, file_name)

        min_len = sorted(len(column_list[i]) for i in range(len(column_list)))[0]
        dicts = {}
        
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
       
# dataset partion indices
def data_partition_random_indices(num_data_points, train_ratio=0.5, val_ratio=0.2, test_ratio=0.3):
    
    num_train_data_points = int(train_ratio*num_data_points)
    num_val_data_points = int(val_ratio*num_data_points)
    
    idx = np.random.permutation(num_data_points)
    idx_train = idx[:num_train_data_points]
    idx_val = idx[num_train_data_points:num_train_data_points+num_val_data_points]
    idx_test = idx[num_train_data_points+num_val_data_points:]

    return idx_train, idx_val, idx_test

def data_split_random_df(df,
                         ratios=(0.7, 0.2, 0.1),
                         labels=('Tr', 'Te', 'Val'),
                         seed=40,
                         data_split_col_name='Data_Split',
                         index=1
                         ):
    """
    Randomly split a DataFrame into n partitions based on given ratios.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    ratios : tuple or list of floats
        Fractions for each split (should sum to 1.0).
    labels : list of str, optional
        Names for each split. If None, defaults to ['Split_1', 'Split_2', ...].
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with an added 'Data_Split' column indicating the partition.
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Ratios must sum to 1.0")

    n = len(ratios)
    if labels is None:
        labels = [f"Split_{i+1}" for i in range(n)]
    elif len(labels) != n:
        raise ValueError("Length of labels must match number of ratios")

    df = df.copy()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))

    sizes = (np.array(ratios) * len(df)).astype(int)
    sizes[-1] = len(df) - sizes[:-1].sum()  # fix rounding

    df.insert(index, data_split_col_name, '')

    start = 0
    for size, label in zip(sizes, labels):
        end = start + size
        split_idx = perm[start:end]
        df.loc[df.index[split_idx], data_split_col_name] = label
        start = end

    return df

# def data_split_random_df(df, ratio=(0.5, 0.3, 0.2)):
    
#     df['Data_Split'] = 'Other'
#     idx = np.random.permutation(np.arange(len(df)))
#     num_train_data, num_test_data, num_val_data = int(len(idx)*ratio[0]), int(len(idx)*ratio[1]), int(len(idx)*ratio[2])
#     train_idx, test_idx, val_idx = idx[:num_train_data], idx[num_train_data:num_train_data+num_test_data], idx[num_train_data+num_test_data:]
#     df.loc[df.index.isin(train_idx), 'Data_Split'] = 'Tr'
#     df.loc[df.index.isin(test_idx), 'Data_Split'] = 'Te'
#     df.loc[df.index.isin(val_idx), 'Data_Split'] = 'Val'
    
#     return df


def selective_range_data_sampling(df, ticks: list, column='logp', frac=0.7, random_state=1):
    """
    Sample data from ranges defined by `ticks` in the given column and create a summary DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    ticks (list): The list of tick values defining the ranges.
    frac (float): The fraction of each range to sample.
    random_state (int): The random state for reproducibility.

    Returns:
    pd.DataFrame: A DataFrame containing the sampled data.
    pd.DataFrame: A summary DataFrame with range names and counts.
    """
    df_list = []
    counts = []

    # First range: less than the first tick
    df_first = df[df[column] < ticks[0]]
    counts.append(len(df_first))
    df_first = df_first.sample(frac=frac, random_state=random_state)
    df_list.append(df_first)

    # Middle ranges: between ticks[i] and ticks[i+1]
    for i in range(len(ticks) - 1):
        df_range = df[(ticks[i] < df[column]) & (df[column] <= ticks[i + 1])]
        counts.append(len(df_range))
        df_range = df_range.sample(frac=frac, random_state=random_state)
        df_list.append(df_range)

    # Last range: greater than the last tick
    df_last = df[df[column] > ticks[-1]]
    counts.append(len(df_last))
    df_last = df_last.sample(frac=frac, random_state=random_state)
    df_list.append(df_last)

    # Create range labels
    range_labels = [f"x<{ticks[0]}"] + [f"{ticks[i]}<x<={ticks[i + 1]}" for i in range(len(ticks) - 1)] + [f"{ticks[-1]}<x"]

    # Calculate total count
    total_count = sum(counts)

    # Prepare summary DataFrame
    dict_summary = {
        'range': range_labels + ['Total'],
        column: counts + [total_count]
    }
    summary_df = pd.DataFrame(dict_summary)

    # Concatenate all sampled DataFrames
    mod_df = pd.concat(df_list, axis=0, ignore_index=True)

    return mod_df, summary_df
    
    
def selective_range_data_split(df, ticks, column='logp', ratio=(0.5, 0.3, 0.2), random_state=1, col_loc=2, data_split_col_name = 'Data_Split'):
    
    """
    Splits the data into train, val, and test sets from defined value ranges in a specified column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    ticks (list): List of tick values defining the range boundaries.
    column (str): The column to apply the range filtering on.
    ratio (tuple): Fraction of each range to assign to train, test and val data.
    random_state (int): Random seed for reproducibility.

    Returns:
    pd.DataFrame: Splitted Dataframe
    """
    
    train_frac, test_frac, val_frac = ratio
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    def split_indices(idx_range):
        
        train_idx, val_idx, test_idx = [], [], []
        if len(idx_range):
            train_idx, temp_idx = train_test_split(idx_range, test_size=1 - train_frac, random_state=random_state)
            val_idx, test_idx = train_test_split(temp_idx, test_size=test_frac / (val_frac + test_frac), random_state=random_state)
        return train_idx, val_idx, test_idx

    train_idx_all, val_idx_all, test_idx_all = [], [], []

    idx_first = df[df[column] < ticks[0]].index
    train, val, test = split_indices(idx_first)
    train_idx_all += list(train)
    val_idx_all += list(val)
    test_idx_all += list(test)

    for i in range(len(ticks) - 1):
        idx_mid = df[(ticks[i] < df[column]) & (df[column] <= ticks[i + 1])].index
        train, val, test = split_indices(idx_mid)
        train_idx_all += list(train)
        val_idx_all += list(val)
        test_idx_all += list(test)

    idx_last = df[df[column] > ticks[-1]].index
    train, val, test = split_indices(idx_last)
    train_idx_all += list(train)
    val_idx_all += list(val)
    test_idx_all += list(test)
    
    if data_split_col_name not in df.columns:
        df.insert(col_loc, data_split_col_name, ['Other']*len(df))
    
    df.loc[df.index.isin(train_idx_all), data_split_col_name] = 'Tr'
    df.loc[df.index.isin(test_idx_all), data_split_col_name] = 'Te'
    df.loc[df.index.isin(val_idx_all), data_split_col_name] = 'Val'

    return df

def data_split_nfold(df: pd.DataFrame,
                     n_folds: List[int]|int,
                     data_split_col_idx = 3,
                     data_split_col_name: str = 'Data_Split',
                     ):
    
    df = df.copy()
    n_data = len(df)
    
    if isinstance(n_folds, int):
        n_folds = [n_folds]
    for n_fold in n_folds:
        random_idx = np.random.permutation(np.arange(n_data))
        n_fold_datapts = n_data//n_fold
        colm = f'{data_split_col_name}_CV{(n_fold)}'
        df.insert(data_split_col_idx, colm, 'CV_1')

        for i in range(n_fold):
            idx = random_idx[i*n_fold_datapts: (i+1)*n_fold_datapts]
            mask = df.index.isin(idx)
            df.loc[mask, colm] = f'CV_{(i+1)}'
            
    return df


def classify_columns(df: pd.DataFrame, cat_threshold: int = 20) -> pd.Series:
    """
    Classifies columns in a DataFrame as 'numeric', 'categorical', or 'non-numeric'.

    Heuristic:
        - Integer columns with few unique values are classified as 'categorical'.
        - Float or high-cardinality numeric columns are classified as 'numeric'.
        - Object or string columns with low unique values are 'categorical', else 'non-numeric'.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cat_threshold (int): Maximum number of unique values to consider a column categorical.

    Returns:
        pd.Series: A mapping of column names to their inferred types.
    """
    col_types = {}

    for col in df.columns:
        series = df[col]
        unique_vals = series.nunique(dropna=True)

        if series.isnull().all():
            col_types[col] = 'unknown'  # or 'non-informative'
            continue

        if pd.api.types.is_numeric_dtype(series):
            if unique_vals <= cat_threshold:
                col_types[col] = 'categorical'
            else:
                col_types[col] = 'numeric'
        elif pd.api.types.is_bool_dtype(series):
            col_types[col] = 'categorical'
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_types[col] = 'datetime'
        elif unique_vals <= cat_threshold:
            col_types[col] = 'categorical'
        else:
            col_types[col] = 'non-numeric'

    return pd.Series(col_types, name='inferred_type')


# Given a set of strings and a set of characters it will return all the strings that doesn't contains the characters.
def exclude_strings(all_strings = [], exclude_char_list = []):
    
    '''
    Input:
        - all_strings : All the strings in a list format
        - exclude_char_list : List of characters.
    Output:
        - all_strings : List of strings.
        
        - **Those strings contains one of the characters from <exclude_char_list> will be eleminated from the list.
    '''

    for ex in exclude_char_list:
        all_strings = [string for string in all_strings if ex not in string]
    all_strings = list(set(all_strings))

    return all_strings

def str2chars(string: str) -> set:
    
    '''
    Input: A string
    Output: Set of unique characters in the string
    '''
    
    # Remove any whitespace and convert to a set of characters
    chars = set(string.replace(' ', ''))
    
    # Remove any non-ASCII characters
    chars = {c for c in chars if ord(c) < 128}
    
    return chars

def liststr2chars(string_list: List[str]) -> List[str]:
    
    '''
    Input: List of string
    Outpur: List of unique characters form all the string
    '''
    
    all_chars = set()
    for string in tqdm(string_list):
        chars = str2chars(string)
        all_chars = all_chars | chars
        
    return sorted(list(all_chars), key = len, reverse=True)


def padding(string: str,
            max_string_len: int = 250,
            padding: str = ' '
            ) -> str:
        
    if len(string) <= max_string_len:
        if padding:
            string = string + padding * (max_string_len - len(string))
        else:
            raise ValueError('Define your padding character: "{}"'.format(padding))
        
    return string


def liststring2max_len(string_list: Any,
                       char_list: Any
                      ) -> tuple[int, str]:
    
    '''
    Input: A list of string
    Output: The Maximum length of the string
    '''
    
    # Create a regex pattern for the vocabulary
    char_list = sorted(char_list, key=len, reverse=True)
    vocab_pattern = '|'.join(map(re.escape, char_list))
    
    max_len = 0
    for string in tqdm(string_list):
        tokens = re.findall(vocab_pattern, string)
        string_len = len(tokens)
        
        if string_len >= max_len:
            max_len = string_len
            max_len_string = string
            
    return max_len, max_len_string


def get_tokens(str: str,
               char_list: list
               ) -> list:
    
    # Precompile the regex for faster repeated usage
    vocab_pattern = '|'.join(map(re.escape, char_list))
    vocab_regex = re.compile(vocab_pattern)
    
    # Tokenize the string string
    tokens = vocab_regex.findall(str)
    
    return tokens


def string2onehot(string: str,
               char_list: list,
               max_str_len: int
               ) -> tuple[np.ndarray[np.bool_], list[str]]:
    
    '''
    Input: 
        - A string
        - Unique Characters List
        - Maximum length of the string
        
    Output: 
        - One Hot Encoding of that string
        - All the tokens in the sorted form
        - Unique tokens
        - Length of the string w.r.t the tokens
    '''
    
    # Tokenize the string string
    tokens = get_tokens(string=string, char_list=char_list)
    
    # Pre-allocate one-hot encoding array
    one_hot = np.zeros((max_str_len, len(char_list) + 1), dtype=bool)
    
    # Create a lookup table for one-hot encoding of each character
    token_to_index = {token: idx for idx, token in enumerate(char_list)}
    
    # Process tokens and fill in the one-hot matrix
    for i, token in enumerate(tokens):
        if token in token_to_index:
            one_hot[i, token_to_index[token]] = True #1
        else:
            # If token is not in vocabulary, ValueError will be raised
            raise ValueError(f'Bad string Error. Token - {token} not in the Character List: {char_list}.')
        
    # Fill remaining rows with padding (last column set to 1)
    if len(tokens) < max_str_len:
        one_hot[len(tokens):, -1] = True #1
    
    # Unique tokens in the string string
    unique_tokens = sorted(set(tokens), key=len, reverse=True)
    
    # Length of string with respect to tokens
    string = len(tokens)
    
    return one_hot, tokens, unique_tokens, string


def liststring2onehot(string_list: List[str],
                   char_list: List[str],
                   max_str_len: int
                   ) -> np.ndarray[np.bool_]:
    
    '''
    Input:
        - List of string
        - Unique Character List
        - Maximum length of the string
        
    Output: An Array containing all the one hot encodings of the string
    '''
    
    onehot_encodings = np.zeros((len(string_list), max_str_len, len(char_list)+1), dtype=bool)
    for i in tqdm(range(len(string_list))):
        onehot, _ = string2onehot(string_list[i], char_list, max_str_len)
        onehot_encodings[i] = onehot
        
    return onehot_encodings



def onehot2string(onehot_mat: np.ndarray,
                  vocabulary: List[str],
                  ) -> str:
    
    string = ''
    for i in range(len(onehot_mat)):
        char_idx_x = np.where(onehot_mat[i] == 1)
        if np.size(char_idx_x[0]) != 0:
            string += vocabulary[char_idx_x[0][0]] if char_idx_x[0][0] < len(vocabulary) else ''
        
    return string


def bool_onehot_encodings2string(onehot_encodings: np.ndarray[np.bool_],
                            char_list: List[str]):
    
    string_list = [None]*len(onehot_encodings)
    for i in range(len(onehot_encodings)):
        string_list[i] = onehot2string(onehot_encodings[i], char_list)
        
    return string_list

def float_onehot_encodings2string(onehot_encodings: np.ndarray[np.bool_],
                            char_list: List[str]):
    
    string_list = [None]*len(onehot_encodings)
    indices = np.argmax(onehot_encodings, axis=-1)
    shape = onehot_encodings.shape
    temp = np.zeros((shape), dtype=bool)
    for i in range(shape[0]):
        temp[i, range(shape[1]), indices[i]] = True
    
    string_list = bool_onehot_encodings2string(temp, char_list)
        
    return string_list


def PCA_fit_transform(matrix, n_components = 2):

    PCA = sklearn.decomposition.PCA(n_components = n_components)
    reduced_matrix = PCA.fit_transform(matrix)

    return  reduced_matrix


def KMeans_fit_predict(matrix, n_clusters=5):

    clusters_dict = {}

    KMeans = sklearn.cluster.KMeans(n_clusters = n_clusters)
    labels = KMeans.fit_predict(matrix)
    for i in range(n_clusters):
        clusters_dict[i] = matrix[labels == i]
        
    return clusters_dict, labels, KMeans


def DBScan_fit_predict(matrix, eps=10, min_samples=5):
    """
    DBSCAN clustering with post-hoc medoid computation.

    Returns
    -------
    clusters_dict : dict
        cluster_label -> points (noise included as label -1)
    labels : np.ndarray
        DBSCAN labels
    cluster_medoids : dict
        cluster_label -> medoid (noise excluded)
    DBSCAN:
        dbscan object
    """

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(matrix)

    # Store clusters
    clusters_dict = {}
    cluster_medoids = {}
    
    for lab in np.unique(labels):
        points = matrix[labels == lab]
        clusters_dict[lab] = points

        D = pairwise_distances(points)
        medoid_idx = np.argmin(D.sum(axis=1))
        cluster_medoids[lab] = points[medoid_idx]

    return clusters_dict, labels, cluster_medoids, dbscan

def get_colors(colours = ['red', 'blue', 'green', 'violet', 'pink', 'orange', 'gray', 'yellow']):
    
    # Define default color list (excluding hard-to-see ones)
    base_colors = colours
    base_colors += list(mcolors.TABLEAU_COLORS)
    base_colors += list(mcolors.XKCD_COLORS)
    base_colors += list(mcolors.CSS4_COLORS)
    excluded = {'white', 'snow', 'ghostwhite', 'ivory'}
    base_colors = list(OrderedDict.fromkeys([c for c in base_colors if c not in excluded]))

    # Setup color iterator
    if isinstance(colours, list):
        color_iter = iter(colours + base_colors)
    else:
        color_iter = iter(['black'] * 100)
        
    return color_iter


def apply_reductions(df: pd.DataFrame | np.ndarray,
                    value_col: str | int | None = None,
                    proj_names: List[str] = ["PCA", "t-SNE", "UMAP", "Isomap", "FactorAnalysis", "MDS", "TruncatedSVD", "KernelPCA"],
                    use_lda: bool = False,
                    normalize: bool = False,
                    ) -> Tuple[Dict[str, np.ndarray], Dict[str, object], np.ndarray]:
    """
    Apply multiple dimensionality reduction techniques (2D projection).

    - Args:
        - df (pd.DataFrame): Input dataframe with features + value column.
        - proj_names (list): Required projections.
        - value_col (str): Column used for coloring (can be discrete or continuous).
        - use_lda (bool): Whether to include LDA (requires discrete labels).
    
    - Returns: 
        - Output: (projections, trained, values)
        - projections: dict {method -> (n,2) array}
        - trained_reducers: dict {method -> fitted reducer object}
        - values: colour/label vector (or None)
    """

    # --- Extract X and values ---
    if value_col is not None:
        if isinstance(df, pd.DataFrame):
            values = df[value_col].to_numpy()
            X = df.drop(columns=[value_col]).to_numpy()
        else:
            values = df[:, value_col]
            X = np.delete(df, value_col, axis=1)
    else:
        values = None
        X = df.to_numpy() if isinstance(df, pd.DataFrame) else df

    reducers = {
            "PCA": PCA(n_components=2, random_state=42),
            "UMAP": UMAP(n_neighbors=10, min_dist=1., n_components=2, metric="cosine", random_state=42),
            "Isomap": Isomap(n_components=2),
            "FactorAnalysis": FactorAnalysis(n_components=2, random_state=42),
            "MDS": MDS(n_components=2, random_state=42, n_init=1, max_iter=300),
            "t-SNE": TSNE(n_components=2, random_state=42, init="pca"),
            "TruncatedSVD": TruncatedSVD(n_components=2, random_state=42),
            "KernelPCA": KernelPCA(n_components=2, kernel="rbf", random_state=42),
        }

    if use_lda and values is not None:
        reducers["LDA"] = LDA(n_components=2)

    projections = {}
    trained = {}

    for name in proj_names:
        if name not in reducers:
            continue

        try:
            reducer = reducers[name]
            X = StandardScaler().fit_transform(X) if normalize else X
            if name == "LDA":
                Z = reducer.fit_transform(X, values)
            else:
                Z = reducer.fit_transform(X)

            projections[name] = Z
            trained[name] = reducer

        except Exception as e:
            print(f"⚠️ {name} failed: {e}")

    return projections, trained, values


def plot_reductions(results: dict,
                    values: dict,
                    ncols: int = 3,
                    figsize=(16, 12),
                    cmap="viridis",
                    cbar_name="Value"):
    """
    Plot 2D projections obtained from multiple dimensionality reduction
    methods.

    Parameters
    ----------
    results : dict
        Dictionary mapping projection names to 2D coordinates:
        {method_name: ndarray of shape (n_samples_i, 2)}.

    values : dict
        Dictionary mapping projection names to label/value vectors:
        {method_name: ndarray of shape (n_samples_i,)}.

        Each projection may contain a different number of samples.

    ncols : int, default=3
        Number of subplot columns.

    figsize : tuple, default=(16,12)
        Figure size.

    cmap : str, default="viridis"
        Colormap for continuous labels.

    cbar_name : str, default="Value"
        Colorbar label.

    Returns
    -------
    None
    """

    sns.set_style("whitegrid")

    values = {k: np.asarray(v) for k, v in values.items()}

    # Validate inputs
    if set(results.keys()) != set(values.keys()):
        raise ValueError("results and values must have identical keys.")

    for key in results:
        if len(results[key]) != len(values[key]):
            raise ValueError(f"{key}: number of samples in results and values do not match.")

    n_methods = len(results)
    nrows = int(np.ceil(n_methods / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.ravel(axes)

    # Global normalization for continuous labels
    continuous_arrays = [v for v in values.values() if np.issubdtype(v.dtype, np.number) and len(np.unique(v)) >= 15]

    if len(continuous_arrays) > 0:
        all_vals = np.concatenate(continuous_arrays)
        global_norm = plt.Normalize(all_vals.min(), all_vals.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=global_norm)
        sm.set_array([])
    else:
        global_norm = None

    for ax, (name, proj) in zip(axes, results.items()):

        y = values[name]
        is_discrete = (y.dtype == object
                       or pd.api.types.is_categorical_dtype(y)
                       or len(np.unique(y)) < 15
                       )

        if is_discrete:
            unique_vals = np.unique(y)
            palette = sns.color_palette("tab10", len(unique_vals))
            lut = dict(zip(unique_vals, palette))
            colors = pd.Series(y).map(lut)
            ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=20, alpha=0.8)
        else:
            ax.scatter(proj[:, 0], proj[:, 1], c=y, cmap=cmap, norm=global_norm, s=20, alpha=0.8)

        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colorbar (continuous case)
    if global_norm is not None:
        cbar = fig.colorbar(sm, ax=axes[:n_methods], fraction=0.02, pad=0.02,)
        cbar.set_label(cbar_name, fontsize=11)

    # Hide unused axes
    for ax in axes[n_methods:]:
        ax.axis("off")

    plt.show()

def plot_projection_grid_seaborn(
    matrices,
    labels,
    rep_names,
    cmap="viridis",
    highlight_index=None,
    normalise=True,
    show_xticks=False,
    show_yticks=False,
    show_grid=False,
    figsize=None,
    cbar_label='Fraction unbound value',
    suptitle = "2D Projections of Different Representations",
    proj_names=["PCA", "t-SNE", "UMAP", "Isomap",
                "FactorAnalysis", "MDS",
                "TruncatedSVD", "KernelPCA"],
):
    """
    Visualize multiple feature representations using a grid of 2D dimensionality
    reduction projections.

    Each row corresponds to one feature representation, while each column
    corresponds to a dimensionality reduction technique. Every feature matrix is
    standardized independently before projection. Each representation may contain
    a different number of samples.

    Parameters
    ----------
    matrices : list of ndarray
        List of feature matrices of shape (n_samples_i, n_features_i).

    labels : list of array-like
        List of label vectors, one for each representation. Each label vector
        must have the same length as the corresponding feature matrix.

    rep_names : list of str
        Names of the feature representations.

    highlight_index : int, optional
        Sample index to highlight. A point is highlighted only if the index
        exists for that representation.

    proj_names : list of str or "all", optional
        Projection methods to visualize.

    Returns
    -------
    None
    """

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)

    matrices = [np.asarray(X) for X in matrices]
    labels = [np.asarray(y) for y in labels]

    if not (len(matrices) == len(labels) == len(rep_names)):
        raise ValueError("matrices, labels and rep_names must have the same length.")

    for X, y in zip(matrices, labels):
        if len(X) != len(y):
            raise ValueError("Each label vector must have the same number of samples as its corresponding feature matrix.")

    n_reps = len(matrices)
    if proj_names == "all":
        proj_names = ["PCA", "t-SNE", "UMAP", "Isomap", "FactorAnalysis", "MDS", "TruncatedSVD", "KernelPCA"]

    n_col = len(proj_names)
    fig, axes = plt.subplots(n_reps, 
                             n_col,
                             figsize=(4 * n_col, 3.3 * n_reps) if figsize is None else figsize,
                             squeeze=False,
                             )
    continuous_rows = []

    for i, (X, y, rep_name) in enumerate(zip(matrices, labels, rep_names)):

        Xs = StandardScaler().fit_transform(X) if normalise else X
        reducers = {
            "PCA": PCA(n_components=2, random_state=42),
            "t-SNE": TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42,),
            "UMAP": umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, ),
            "Isomap": Isomap(n_components=2, n_neighbors=15),
            "FactorAnalysis": FactorAnalysis(n_components=2, random_state=42,),
            "MDS": MDS(n_components=2, random_state=42, n_init=1, max_iter=300,),
            "TruncatedSVD": TruncatedSVD(n_components=2, random_state=42,),
            "KernelPCA": KernelPCA(n_components=2, kernel="rbf", random_state=42,),
            }

        is_continuous = np.issubdtype(y.dtype, np.floating)

        if is_continuous:
            continuous_rows.append(y)

        for j, name in enumerate(proj_names):

            Z = reducers[name].fit_transform(Xs)
            ax = axes[i, j]

            if is_continuous:
                ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap=cmap, s=35, alpha=0.85)
            else:
                sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y, palette="tab10", ax=ax, s=35, alpha=0.85, linewidth=0, legend=False,)

            if (highlight_index is not None and highlight_index < len(Z)):
                ax.scatter(Z[highlight_index, 0], Z[highlight_index, 1], color="red", s=20, zorder=5)

            if i == 0:
                ax.set_title(name, fontsize=20, fontweight="bold")

            if j == 0:
                ax.set_ylabel(rep_name, fontsize=20, fontweight="bold",)
            else:
                ax.set_ylabel("")

            # ax.set_xlabel("")
            # ax.set_xticks([])
            # ax.set_yticks([])

            ax.set_xlabel("")
            if not show_xticks:
                ax.set_xticks([])

            if not show_yticks:
                ax.set_yticks([])
            ax.grid(show_grid)

    if len(continuous_rows) > 0:

        all_values = np.concatenate(continuous_rows)
        norm = plt.Normalize(all_values.min(), all_values.max(),)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cax = fig.add_axes([1.01, 0.12, 0.018, 0.76])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(cbar_label, fontsize=22, fontweight="bold",)
        cbar.ax.tick_params(labelsize=20)

    plt.suptitle(suptitle, fontsize=25, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_3d(
    matrices=None,
    colours=None,
    title='3D Scatter Plot',
    x_label='X',
    y_label='Y',
    z_label='Z',
    fig_size=(8, 8),
    view_init=(20, 20),
    titlefontsize=30,
    names=None,  # list of strings
    X=None, Y=None, Z=None  # coords for annotation
):
    """
    Plot 1D, 2D, or 3D matrices in a 3D scatter plot.

    Args:
        matrices (list of np.ndarray): Each matrix can be 1D, 2D, or 3D in shape.
        colours (list of str or True/False): List of color names or True/False for default behavior.
        title (str): Title of the plot.
        x_label, y_label, z_label (str): Axis labels.
        fig_size (tuple): Size of the figure.
        view_init (tuple): Elevation and azimuthal viewing angles.
        names (list of str): Annotation text labels.
        X, Y, Z (list or np.ndarray): Coordinates for annotation points.
    """
    if matrices is None:
        raise ValueError("You must provide a list of matrices.")
    if not isinstance(matrices, list):
        matrices = [matrices]

    # Define default color list (excluding hard-to-see ones)
    base_colors = ['red', 'blue', 'green', 'violet', 'pink', 'orange', 'gray', 'yellow']
    base_colors += list(mcolors.TABLEAU_COLORS)
    base_colors += list(mcolors.XKCD_COLORS)
    base_colors += list(mcolors.CSS4_COLORS)
    excluded = {'white', 'snow', 'ghostwhite', 'ivory'}
    base_colors = list(OrderedDict.fromkeys([c for c in base_colors if c not in excluded]))

    # Setup color iterator
    if isinstance(colours, list):
        color_iter = iter(colours + base_colors)
    elif colours is True:
        color_iter = iter(base_colors)
    else:
        color_iter = iter(['black'] * len(matrices))

    # Prepare figure and axes
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    ax.grid()

    # Plot each matrix
    for idx, mat in enumerate(matrices):
        mat = np.asarray(mat)
        if mat.ndim == 1:
            x = mat
            y = np.zeros_like(mat)
            z = np.zeros_like(mat)
        elif mat.ndim == 2:
            if mat.shape[1] == 3:
                x, y, z = mat[:, 0], mat[:, 1], mat[:, 2]
            elif mat.shape[1] == 2:
                x, y = mat[:, 0], mat[:, 1]
                z = np.zeros_like(x)
            elif mat.shape[1] == 1:
                x = mat[:, 0]
                y = np.zeros_like(x)
                z = np.zeros_like(x)
            else:
                print(f"Skipping matrix index {idx}: unsupported shape {mat.shape}")
                continue
        else:
            print(f"Skipping matrix index {idx}: not a 1D or 2D matrix.")
            continue

        ax.scatter(x, y, z, color=next(color_iter), label=f"Set {idx+1}")

    # Add labels if provided
    if names and X is not None and Y is not None and Z is not None:
        for name, x, y, z in zip(names, X, Y, Z):
            ax.text(x, y, z, name)

    # Set labels and view
    labelsfontsize = titlefontsize - 5
    labelpad=30
    ax.set_title(title, fontsize=titlefontsize, color='darkgreen', fontweight='bold')
    ax.set_xlabel(x_label, labelpad=labelpad, fontsize=labelsfontsize)
    ax.set_ylabel(y_label, labelpad=labelpad, fontsize=labelsfontsize)
    ax.set_zlabel(z_label, labelpad=labelpad, fontsize=labelsfontsize)
    
    tickfontsize = titlefontsize - 8
    tickpad = 8  # increase if fonts are large
    ax.tick_params(axis='x', labelsize=tickfontsize, pad=tickpad)
    ax.tick_params(axis='y', labelsize=tickfontsize, pad=tickpad)
    ax.tick_params(axis='z', labelsize=tickfontsize, pad=tickpad + 4)
    ax.view_init(elev=view_init[0], azim=view_init[1])
    plt.tight_layout()
    plt.show()

def fig_to_image(fig, dpi=100):
    import io
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)

    img = Image.open(buf)

    # --- SAFETY FIX ---
    # Ensure all metadata values are strings
    if hasattr(img, "info") and isinstance(img.info, dict):
        img.info = {k: str(v) for k, v in img.info.items()}

    return img


def plot_2d(matrices=None,
            colours=True,
            title='2D Scatter Plot',
            x_label='X',
            y_label='Y',
            plot_type='scatter', # 'continuous'
            fig_size=(8, 8),
            names=None,
            X=None,
            Y=None,
            scatter_labels = False,
            title_fontsize=25,
            linewidth=2,
            markers = ['o', '^', 'd', 's', 'p', 'X', 'v', '<', '>', 'h', '+', '*', 'x'],
            xticks = False,
            yticks = False,
            xtickslabels = False,
            ytickslabels = False,
            x_lim = None,
            y_lim = None,
            legend_loc = None,#'upper center',
            plot_legends = True,
            savepath='',
            return_img=False,
            ):
    """
    Plot 1D or 2D matrices as 2D scatter plots.

    Args:
        matrices (list of np.ndarray): List of 1D or 2D arrays.
        colours (list of str or bool): List of color strings or True for default, False for none.
        title (str): Title of the plot.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        fig_size (tuple): Size of the figure.
        names (list of str or None): Labels for annotations.
        X (np.ndarray): X-coordinates for annotations (if different from matrices).
        Y (np.ndarray): Y-coordinates for annotations (if different from matrices).
    """
    if matrices is None:
        raise ValueError("The 'matrices' argument must be provided and non-empty.")
    if not isinstance(matrices, list):
        matrices = [matrices]
        
    markers = markers*int(np.ceil(len(matrices)/len(markers)))
    markers = iter(markers)

    # Prepare color palette
    base_colors = list(OrderedDict.fromkeys(
        ['red', 'blue', 'green', 'maroon', 'magenta', 'darkorchid', 'darkorange', 'darkmagenta', 'brown', 'pink', 'gray', 'yellow'] +
        list(mcolors.TABLEAU_COLORS) +
        list(mcolors.XKCD_COLORS) +
        list(mcolors.CSS4_COLORS)
    ))
    excluded = {'white', 'snow', 'ghostwhite', 'ivory'}
    base_colors = [c for c in base_colors if c not in excluded]

    # Color iterator logic
    if isinstance(colours, list):
        color_iter = iter(colours)
    elif colours is True:
        color_iter = iter(base_colors)
    else:
        color_iter = iter(['black'] * len(matrices))

    # Set up plot
    fig = plt.figure(figsize=fig_size)
    plt.grid(color='gray', linestyle='--', linewidth=0.8)
    plt.title(title, fontsize=title_fontsize, fontweight='bold', color='darkgreen')

    # Plot each matrix
    for idx, matrix in enumerate(matrices):
        matrix = np.asarray(matrix)
        if matrix.ndim == 1:
            x_vals = matrix
            y_vals = np.zeros_like(matrix)
        elif matrix.ndim == 2:
            if matrix.shape[1] == 2:
                x_vals = matrix[:, 0]
                y_vals = matrix[:, 1]
            elif matrix.shape[1] == 1:
                x_vals = matrix[:, 0]
                y_vals = np.zeros_like(x_vals)
            else:
                print(f"Skipping matrix at index {idx}: has unsupported shape {matrix.shape}")
                continue
        else:
            print(f"Skipping matrix at index {idx}: not 1D or 2D")
            continue
        
        if plot_type == 'scatter':
            if scatter_labels:
                plt.scatter(x_vals, y_vals, color=next(color_iter), label=scatter_labels[idx], s = title_fontsize*5, marker=next(markers), alpha=1.0)
            else:
                plt.scatter(x_vals, y_vals, color=next(color_iter), label=f"Set {idx+1}", s = title_fontsize*5, marker=next(markers), alpha=1.0)
        else:
            if scatter_labels:
                plt.plot(x_vals, y_vals, color=next(color_iter), label=scatter_labels[idx], linestyle='-', linewidth=linewidth, marker=next(markers), markersize = 5*linewidth)
            else:
                plt.plot(x_vals, y_vals, color=next(color_iter), label=f"Set {idx+1}", linestyle='-', linewidth=linewidth, marker=next(markers), markersize = 5*linewidth)

    # xmin_, xmax_ = plt.xlim()
    # x_ = np.linspace(xmin_, xmax_, 200)
    # # ±0.5 offset lines
    # plt.plot(x_, x_ + 0.5, color="green", linestyle="--", linewidth=2, label="±0.5 interval")
    # plt.plot(x_, x_ - 0.5, color="green", linestyle="--", linewidth=2)

    # Annotations (if applicable)
    if names:
        if X is None or Y is None:
            # Try to infer from the first valid matrix
            for m in matrices:
                m = np.asarray(m)
                if m.ndim == 2 and m.shape[1] == 2:
                    X, Y = m[:, 0], m[:, 1]
                    break
                elif m.ndim == 1:
                    X = m
                    Y = np.zeros_like(m)
                    break
        if X is not None and Y is not None:
            for name, x, y in zip(names, X, Y):
                plt.annotate(name, (x, y), textcoords="offset points", xytext=(0, -10), ha='center')

    if isinstance(xticks, (np.ndarray, list, tuple)):
        xticks = list(xticks)
        if isinstance(xtickslabels, (np.ndarray, list, tuple)):
            plt.xticks(xticks, xtickslabels if len(xticks) == len(xtickslabels) else xticks)
        else:
            plt.xticks(xticks, xticks)
    if isinstance(yticks, (np.ndarray, list, tuple)):
        yticks = list(yticks)
        if isinstance(ytickslabels, (np.ndarray, list, tuple)):
            plt.yticks(yticks, ytickslabels if len(yticks) == len(ytickslabels) else yticks)
        else:
            plt.yticks(yticks, yticks)

    # Axes labels
    plt.xlabel(x_label, labelpad=10, fontsize=title_fontsize-3)
    plt.ylabel(y_label, labelpad=10, fontsize=title_fontsize-3)
    plt.xticks(fontsize=title_fontsize-5)
    plt.yticks(fontsize=title_fontsize-5)

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    
    if plot_legends:
        plt.legend(fontsize=title_fontsize-8, loc=legend_loc, frameon=True)
    plt.tight_layout()
    
    if savepath:
        create_folder(get_dir(savepath))
        plt.savefig(savepath,  dpi=300)
        plt.close()
        print(f'Image saved in: {savepath}')
    elif return_img:
        return fig_to_image(fig, dpi=300)
    else:
        plt.show()


def collage_from_dict(img_dict, save_path=None, cols=3, padding=10,
                      bg_color=(255,255,255), title=None, title_height=80):

    names = list(img_dict.keys())
    imgs = list(img_dict.values())

    n = len(imgs)
    rows = math.ceil(n / cols)

    # assume same image size
    w, h = imgs[0].size

    # compute width first (needed for font scaling)
    collage_w = cols * w + padding * (cols + 1)

    # ---- Title handling ----
    y_offset = 0
    if title:
        font_size = collage_w // 80  # BIG title

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except:
            raise RuntimeError("Install DejaVuSans-Bold.ttf on your system")

        # adjust title height based on font
        bbox = (0, 0, 0, 0)
        while True:
            bbox = ImageDraw.Draw(Image.new("RGB", (10,10))).textbbox((0, 0), title, font=font)
            text_w = bbox[2] - bbox[0]

            if text_w <= collage_w * 0.95:
                break

            font_size -= 2
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)

        text_h = bbox[3] - bbox[1]
        title_height = max(title_height, text_h + 20)

        y_offset = title_height
    else:
        font = None

    # ---- Final canvas size ----
    collage_h = rows * h + padding * (rows + 1) + y_offset

    collage = Image.new("RGB", (collage_w, collage_h), bg_color)
    draw = ImageDraw.Draw(collage)

    # ---- Draw title ----
    if title:
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        x_text = (collage_w - text_w) // 2
        y_text = (title_height - text_h) // 2

        draw.text((x_text, y_text), title, fill=(0, 0, 0), font=font)

        # separator line
        draw.line((0, title_height, collage_w, title_height), fill=(0,0,0), width=2)

    # ---- Paste images ----
    for i, (name, img) in enumerate(img_dict.items()):
        r = i // cols
        c = i % cols

        x = padding + c * (w + padding)
        y = padding + r * (h + padding) + y_offset

        collage.paste(img, (x, y))

        draw.text((x + 5, y + 5), name, fill=(0,0,0))

    if save_path:
        collage.save(save_path)

    return collage
 
# def collage_from_dict(img_dict, save_path=None, cols=3, padding=10, bg_color=(255,255,255)):
#     """
#     Create a collage from a dictionary of PIL images.

#     Parameters
#     ----------
#     img_dict : dict[str, PIL.Image]
#     save_path : str
#     cols : int
#         number of columns in collage
#     padding : int
#         space between images
#     bg_color : tuple
#         background color
#     """

#     names = list(img_dict.keys())
#     imgs = list(img_dict.values())

#     n = len(imgs)
#     rows = math.ceil(n / cols)

#     # assume same image size
#     w, h = imgs[0].size

#     collage_w = cols * w + padding * (cols + 1)
#     collage_h = rows * h + padding * (rows + 1)

#     collage = Image.new("RGB", (collage_w, collage_h), bg_color)

#     draw = ImageDraw.Draw(collage)

#     for i, (name, img) in enumerate(img_dict.items()):

#         r = i // cols
#         c = i % cols

#         x = padding + c * (w + padding)
#         y = padding + r * (h + padding)

#         collage.paste(img, (x, y))

#         # optional label
#         draw.text((x + 5, y + 5), name, fill=(0,0,0))
#     if save_path is not None:
#         collage.save(save_path)

#     return collage 

def customised_plot(matrix, PCA = True, Kmeans = False, plot = True, view_init = [30, 30]):

    clusters_dict = {}
    reduced_matrix = np.array([])
    labels = []

    if PCA:
        reduced_matrix = PCA_fit_transform(matrix)
        matrix = reduced_matrix

    if Kmeans:
        clusters_dict, labels = KMeans_fit_predict(matrix)

    if plot:
        if matrix.shape[1] == 3:
            if Kmeans:
                plot_3d(*clusters_dict.values())

            else:
                plot_3d(matrix)

        elif matrix.shape[1] == 2:
            if Kmeans:
                plot_2d(*clusters_dict.values())

            else:
                plot_2d(matrix)

        else:
            raise ValueError(f'Points should be in R^2 or R^3 but points got are in  R^{matrix.shape[1]}')


    return reduced_matrix, labels, clusters_dict


def convex_combination(X, Y, no_of_points = 1000, type_ = 'numpy', space = 'euclidean'):

    t = np.linspace(0, 1, no_of_points)
    # t = np.random.uniform(0, 1, no_of_points)

    if space == 'euclidean':
        z = np.array([(t[i]*X + (1-t[i])*Y) for i in range(len(t))])

    elif space == 'hypersphere':
        theta = np.arccos(np.dot(X, Y.T))
        z = np.array([(np.sin(t[i]*theta)*X + np.sin((1-t[i])*theta)*Y)/np.sin(theta)
                      for i in range(len(t))])

    else:
        raise ValueError(f'space = {space} is not defined')

    if type_ == 'torch':
        z = torch.from_numpy(z).type(torch.float)

    return z


def convex_PCA(X, Y, no_of_points = 1000, type_ = 'numpy', space = 'euclidean', plot = False):

        z = convex_combination(X, Y, no_of_points, type_, space)
        reduced_matrix, PCA_model = PCA_fit_transform(z)

        if plot:
            if reduced_matrix.shape[1] == 2:
                plot_2d(reduced_matrix)

            elif reduced_matrix.shape[1] == 3:
                plot_3d(reduced_matrix)

            else:
                raise ValueError(f'Dimension of reduced matrix {reduced_matrix.shape} is greater gretter than 3')

        return z, reduced_matrix, PCA_model


def convex_inv_PCA(PCA_model, X, Y, no_of_points = 1000, type_ = 'numpy', space = 'euclidean'):

    z = convex_combination(X, Y, no_of_points, type_ , space)
    inv_pca_z = PCA_model.inverse_transform(z)

    return inv_pca_z


def euc_dist(X, Y, metric = 'euclidean'):
        
    dist = sklearn.metrics.pairwise_distances(X, Y, metric = metric)

    return dist


def hyperspherical_distance(X, Y, r = 1, type_ = 'cos'):
        
    if type_ == 'cos':
        dot_product = np.dot(X, Y)/(np.linalg.norm(X)*np.linalg.norm(Y))
        dot_product = round(dot_product[0], 4)
        result = r*np.arccos(dot_product/r**2)

    elif type_ == 'sin':
        euc_dist = np.linalg.norm(X - Y)
        euc_dist = round(euc_dist, 4)
        result = 2*r*np.arcsin(euc_dist/(2*r))

    else:
        raise ValueError(f'Hyperspherical distance is not defined for type-{type_}.')

    return result

def pairwise_hyperspherical_distance(X, Y, r = 1, type_ = 'cos'):
    
    dim_X = np.array(X).shape[0]
    dim_Y = np.array(Y).shape[0]
    pairwise_distances = np.array([hyperspherical_distance(X[i].reshape(1, -1), Y[j]) for i in range(dim_X) for j in range(dim_Y)]).reshape(dim_X, -1)
    
    return pairwise_distances

def near_points(vector: np.ndarray, matrix: np.ndarray, distance = 'euclidean', no_of_points = 1):
        
    vector = np.array(vector)
    matrix = np.array(matrix)
    vector = vector.reshape(1, -1)


    pairwise_distances = sklearn.metrics.pairwise_distances(vector, matrix, metric=distance)
    sort_idx = np.argsort(pairwise_distances)

    closest_vectors = matrix[sort_idx[0, :no_of_points]]

    return closest_vectors, sort_idx



def elementary_ops_interchange_rows(
    A: np.array, 
    ith_row: int, 
    jth_row: int
)->np.array:
    """
    Elementary row operation: interchange i-th and j-th rows
    
     Inputs:
        - A: Given matrix
        - ith_row: i-th row
        - jth_row: j-th row
        
    Output:
        - A: Resultant rows interchanged matrix
    """
    
    
    for i in range(A.shape[1]):
        temp = A[ith_row][i]
        A[ith_row][i] = A[jth_row][i]
        A[jth_row][i] = temp
    
    return A



def elementary_ops_scale_row(
    A: np.array, 
    ith_row: int, 
    scalar_val: float
)->np.array:
    """
    Elementary row operation: Scaling i-th row  with the scalar_val value
    
     Inputs:
        - A: Given matrix
        - ith_row: row want to scale
        - scalar_val: scalar value
        
    Output:
        - A: Resultant row scaling matrix
    """
    
    
    for i in range(A.shape[1]):
        A[ith_row][i] *= scalar_val
        
    A = A.astype(np.float64)
        
    return A


def elementary_ops_change_row(
    A: np.array, 
    ith_row: int, 
    jth_row: int,
    scalar_val: float
)->np.array:
    """
    Elementary row operation: Change i-th based on i<-i + scalar_val*j
    
     Inputs:
        - A: Given matrix
        - ith_row: i-th row
        - jth_row: j-th row
        - scalar_val: scalar value
        
    Output:
        - A: Resultant row updated matrix
    """
    
    for i in range(A.shape[1]):
        A[ith_row][i] += scalar_val*A[jth_row][i]
    
    return A


def elementary_ops_interchange_rows_mult(
    A: np.array, 
    ith_row: int, 
    jth_row: int
)->tuple:
    """
    Elementary row operation: interchange i-th and j-th rows
    
     Inputs:
        - A: Given matrix
        - ith_row: i-th row
        - jth_row: j-th row
        
    Output:
        - A: Resultant rows interchanged matrix
    """
    I = np.identity(A.shape[0])
    B = elementary_ops_interchange_rows(I, ith_row, jth_row)
    A = np.matmul(B, A)
    
    return A, I

def elementary_ops_scale_row_mult(
    A: np.array, 
    ith_row: int, 
    scalar_val: float
)->tuple:
    """
    Elementary row operation: Scaling i-th row  with the scalar_val value
    
     Inputs:
        - A: Given matrix
        - ith_row: row want to scale
        - scalar_val: scalar value
        
    Output:
        - A: Resultant row scaling matrix
    """
    
    I = np.identity(A.shape[0])
    B = elementary_ops_scale_row(I, ith_row, scalar_val)
    A = np.matmul(B, A)
               
    return A, I


def elementary_ops_change_row_mult(
    A: np.array, 
    ith_row: int, 
    jth_row: int,
    scalar_val: float
)->tuple:
    """
    Elementary row operation: Change i-th based on i<-i + scalar_val*j
    
     Inputs:
        - A: Given matrix
        - ith_row: i-th row
        - jth_row: j-th row
        - scalar_val: scalar value
        
    Output:
        - A: Resultant row updated matrix
    """
    
    I = np.identity(A.shape[0])
    B = elementary_ops_change_row(I, ith_row, jth_row, scalar_val)
    A = np.matmul(B, A)
        
    return A, I


def elementary_ops_interchange_cols(
    A: np.array, 
    ith_col: int, 
    jth_col: int
)->np.array:
    """
    Elementary col operation: interchange i-th and j-th cols
    
     Inputs:
        - A: Given matrix
        - ith_col: i-th col
        - jth_col: j-th col
        
    Output:
        - A: Resultant cols interchhanged matrix
    """
    
    for i in range(A.shape[0]):
        temp = A[i][ith_col]
        A[i][ith_col] = A[i][jth_col]
        A[i][jth_col] = temp
    
    return A


def elementary_ops_scale_col(
    A: np.array, 
    ith_col: int, 
    scalar_val: float
)->np.array:
    """
    Elementary col operation: Scaling i-th col  with the scalar_val value
    
     Inputs:
        - A: Given matrix
        - ith_col: col want to scale
        - scalar_val: scalar value
        
    Output:
        - A: Resultant col scaling matrix
    """
    
    for i in range(A.shape[0]):
        A[i][ith_col] *= scalar_val
            
    return A

def elementary_ops_change_col(
    A: np.array, 
    ith_col: int, 
    jth_col: int,
    scalar_val: float
)->np.array:
    """
    Elementary col operation: Change i-th col based on i<-i + scalar_val*j
    
     Inputs:
        - A: Given matrix
        - ith_col: i-th col
        - jth_col: j-th col
        - scalar_val: scalar value
        
    Output:
        - A: Resultant col updated matrix
    """
    
    for i in range(A.shape[0]):
        A[i][ith_col] += scalar_val*A[i][jth_col]
               
    return A


def sweep_out_row(
    A: np.array,
    ith_row: int,
    pivot_element: tuple
)->np.array:
    """
    Sweep out a i-th row based on the given pivot element
    
    Inputs:
        - A: Given matrix
        - ith_row: row to sweep out
        - pivot_element pivot element
        
    Output:
        - A: Resultant row sweep out matrix
    """
    
    if A[pivot_element[0], pivot_element[1]] != 0:
        A = elementary_ops_scale_col(A, pivot_element[1], 1/A[pivot_element[0], pivot_element[1]])
        for j in range(A.shape[1]):
            if(j == pivot_element[1]):
                continue
            else:
                A = elementary_ops_change_col(A, j, pivot_element[1], -A[pivot_element[0]][j])

    return A
    
    
def sweep_out_col(
    A: np.array,
    ith_col: int,
    pivot_element: tuple
)->np.array:
    """
    Sweep out a i-th column based on the given pivot element
    
    Inputs:
        - A: Given matrix
        - ith_col: column to sweep out
        - pivot_element pivot element
        
    Output:
        - A: Resultant column sweep out matrix
    """
    
    if A[pivot_element[0], pivot_element[1]] != 0:
        A = elementary_ops_scale_row(A, pivot_element[0], 1/A[pivot_element[0], pivot_element[1]]).astype(float)
        for j in range(A.shape[0]):
            if(j == pivot_element[0]):
                continue
            else:
                A = elementary_ops_change_row(A, j, pivot_element[0], -A[j][pivot_element[1]]).astype(float)
    
    return A



def pivoting(A: np.array,
             position:tuple
            )->tuple:
    
    """Returns the pivot position and the pivot element of a matrix after a given position.
    
    Inputs:
        - A: Given matrix
        
    Output:
        - pivot_position: The pivot position i.e. non-zero value with maximum absolute value in a column.
        - pivot_element: Pivot element of that position.
    """
    
    A = A.astype(np.float64)
    pivot_idx = np.argmax(abs(A[position[0]:, position[1]:]), axis = 0)[0]
    pivot_element = A[position[0]+pivot_idx, position[1]]
    
    pivot_position = (position[0]+pivot_idx, position[1])
    
    if pivot_element != 0:
        pivot_position = pivot_position
        
    else:
        pivot_position, pivot_element = pivoting(A, (position[0]+1, position[1]+1))  # Recursion Function

    return pivot_position, pivot_element


def row_reduced_echelon_form(
    A: np.array
)-> np.array:
    """Reduction to echelon form of a matrix.
    
    Inputs:
        - A: Given matrix
        
    Output:
        - A: Echelon form of A
    """
    
    min_dim = min(A.shape[0], A.shape[1])
    position = (0, 0)
    
    for i in range(min_dim):
        if position[0]<A.shape[0] and position[1]<A.shape[1]:
#             print(A, '\n', '-'*75)
            pivot_position, pivot = pivoting(A, position)
            A = elementary_ops_interchange_rows(A, pivot_position[0], i)
            pivot_position = (i, pivot_position[1])
            A = sweep_out_col(A, pivot_position[1], pivot_position)
            position = (pivot_position[0]+1, pivot_position[1]+1)
        
    return A


def clamp(value, min_value=0, max_value=1):
    
    return max(min(value, max_value), min_value)


def surrounding_coordinates(position):
    
    i, j = position[0], position[1]
    right = (i, clamp(j+1, max_value=3))
    left = (i, clamp(j-1))
    up = (clamp(i-1), j)
    down = (clamp(i+1, max_value=2), j)
    
    return right, left, up, down

def bytesarray2pyfile(pickled_file, save_filepath='./', save_filename='bytesarray2pyfile'):
    
    save_filepath = create_folder(save_filepath)
    abs_path = ''.join([save_filepath, save_filename, '.py'])
    unpickled_object = pickle.loads(pickled_file)
    
    with open(abs_path , 'wb') as fp:
        fp.write(unpickled_object)
    
    print('File saved in \n {}'.format(abs_path))  
    return None

def pyfile2bytesarray(filepath, filename, save_filepath='./', save_filename='pyfile2bytesarray'):
    
    file = ''.join([filepath, filename])
    save_filepath = create_folder(save_filepath)
    save_filepath = ''.join([save_filepath, save_filename])
    
    with open(file , 'rb') as fp:
        file = fp.read()
        bytes_file = pickle.dumps(file)
    with open(save_filepath, 'wb') as fp:
        pickle.dump(bytes_file, save_filepath)
    
    return bytes_file


# function to convert to subscript 
def get_sub(x): 
	normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
	sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
	res = x.maketrans(''.join(normal), ''.join(sub_s)) 
	return x.translate(res) 

# display subscript 
# print('H{}SO{}'.format(get_sub('2'),get_sub('4'))) #H₂SO₄ 


# function to convert to superscript 
def get_super(x): 
	normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
	super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
	res = x.maketrans(''.join(normal), ''.join(super_s)) 
	return x.translate(res) 

# display superscipt 
# print('e{}'.format(get_super('x'))) #ᴳᵉᵉᵏˢᶠᵒʳᴳᵉᵉᵏˢ 

# Delete a folder with a fix name from every place in a given path
def del_folder(path, folder_name = 'abc'):
    i = 0
    for root, dir, files in os.walk(path):
        for name in dir:
            if name == folder_name:
                folder_path = os.path.join(root, name)
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    print(f'Folder: {folder_path} not deleted:{e}')
                else:
                    i += 1
                    print(f'Folder: {folder_path} deleted.')
    print(f'Deletion Complete:\n {i} folders deleted')
    

def make_kfold_indices(n, k, shuffle=True, seed=42):
    """
    Generate balanced k-fold indices for arbitrary-sized data.

    Parameters
    ----------
    data : list or array-like
        Your dataset (only its length is used).
    k : int
        Number of folds.
    shuffle : bool
        Whether to shuffle before splitting.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    folds : list of numpy arrays
        folds[i] contains the indices for validation fold i.
    """

    if isinstance(n, list):
        indices = n
        n = len(n)
    else:
        indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    # Balanced fold sizes
    base = n // k
    remainder = n % k

    fold_sizes = [base + 1 if i < remainder else base for i in range(k)]

    folds = {}
    start = 0
    for i, size in enumerate(fold_sizes):
        folds[f'CV-{i+1}'] = indices[start:start + size]
        start += size

    return folds

def df2cleaned_df(df: pd.DataFrame,
                  key_col: str,
                  target_col: str,
                  key_col_rename: str = 'Processed_Key',
                  target_col_rename: str = 'target_values_list',
                  mean_col_name: str = 'target_mean',
                  std_col_name: str = 'target_std',
                  key_preprocessor: Optional[Callable] = None,
                  feature_extractor: Optional[Callable] = None,
                  data_split: bool = True,
                  train_idx: Union[list, tuple, np.ndarray] = [],
                  val_idx: Union[list, tuple, np.ndarray] = [],
                  test_idx: Union[list, tuple, np.ndarray] = []
                  ) -> pd.DataFrame:
    """
    General-purpose dataframe cleaner with key preprocessing, aggregation, feature extraction, and optional data split.

    Args:
        df (pd.DataFrame): Input dataframe with at least key_col and target_col.
        key_col (str): Column to group by (e.g., 'SMILES', 'UserID').
        target_col (str): Column to aggregate (e.g., 'score').
        key_col_rename (str): New column name for processed key values.
        target_col_rename (str): New column name for aggregated target list.
        mean_col_name (str): Column name for mean of aggregated target.
        std_col_name (str): Column name for std of aggregated target.
        key_preprocessor (Callable, optional): Function to transform key_col values (e.g., canonicalize SMILES).
        feature_extractor (Callable, optional): Function to extract features from the processed key_col.
        data_split (bool): Whether to split into train/val/test sets.
        train_idx, val_idx, test_idx: Index lists for data splitting.

    Returns:
        pd.DataFrame: Cleaned, featured, and optionally split dataframe.
    """

    # Step 1: Preprocess key column if needed
    if key_preprocessor is not None:
        df[key_col_rename] = df[key_col].apply(key_preprocessor)
    else:
        key_col_rename = key_col  # Just group by original key if no preprocessor
        df[key_col_rename] = df[key_col]

    # Step 2: Build mapping from processed key back to original key
    key_mapping = dict(zip(df[key_col_rename], df[key_col]))

    # Step 3: Group by processed key and aggregate target
    grouped_dict = df.groupby(key_col_rename)[target_col].apply(list).to_dict()
    grouped_df = pd.DataFrame(list(grouped_dict.items()), columns=[key_col_rename, target_col_rename])

    # Step 4: Restore original key from mapping
    grouped_df.insert(0, key_col, grouped_df[key_col_rename].map(key_mapping))

    # Step 5: Compute statistics
    grouped_df[mean_col_name] = grouped_df[target_col_rename].apply(lambda x: np.mean(x) if x else np.nan)
    grouped_df[std_col_name] = grouped_df[target_col_rename].apply(lambda x: np.std(x) if x else np.nan)

    # Step 6: Extract features from processed key
    if feature_extractor is not None:
        features = grouped_df[key_col_rename].apply(feature_extractor).tolist()
        features_df = pd.DataFrame(features)
        features_df = features_df.select_dtypes(include=[np.number])
        grouped_df = pd.concat([grouped_df, features_df], axis=1)

    # Step 7: Optional data split
    if data_split:
        grouped_df.insert(2, 'Data_Split', ['Other'] * len(grouped_df))

        if not (len(train_idx) and len(val_idx) and len(test_idx)):
            print('Splitting dataset randomly as index lists were not provided.')
            idx = np.random.permutation(len(grouped_df))
            n = len(idx)
            train_idx, val_idx, test_idx = idx[:n//2], idx[n//2:n*4//5], idx[n*4//5:]

        grouped_df.loc[grouped_df.index.isin(train_idx), 'Data_Split'] = 'Tr'
        grouped_df.loc[grouped_df.index.isin(val_idx), 'Data_Split'] = 'Val'
        grouped_df.loc[grouped_df.index.isin(test_idx), 'Data_Split'] = 'Te'

    return grouped_df


def merge_list_columns_by_key(target_df: pd.DataFrame,
                               reference_df: pd.DataFrame,
                               key_column: str,
                               list_column: str,
                               mean_column: Optional[str] = None,
                               std_column: Optional[str] = None,
                               parse_strings_to_lists: bool = True,
                               add_uncommon: bool = False,
                               ) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    Merge list-type column values in `target_df` using matching keys from `reference_df`.

    Args:
        target_df (pd.DataFrame): The DataFrame to be updated.
        reference_df (pd.DataFrame): The reference DataFrame providing list values.
        key_column (str): Column name used to match entries between the two DataFrames.
        list_column (str): Name of the column that holds list-type values to merge.
        mean_column (str, optional): If provided, updates this column with the new mean of the merged list.
        std_column (str, optional): If provided, updates this column with the new standard deviation.
        parse_strings_to_lists (bool): Whether to convert string-represented lists to Python lists using `ast.literal_eval`.

    Returns:
        Tuple[pd.DataFrame, Dict[int, int]]:
            - A modified copy of `target_df` with merged list entries and optionally updated statistics.
            - A dictionary mapping row indices in `target_df` to matched row indices in `reference_df`.
    """

    updated_df = copy.deepcopy(target_df)

    # Identify common keys and build index maps
    shared_keys = set(updated_df[key_column]) & set(reference_df[key_column])
    target_idx_map = {k: i for i, k in updated_df[key_column].items() if k in shared_keys}
    reference_idx_map = {k: i for i, k in reference_df[key_column].items() if k in shared_keys}
    row_mapping = {target_idx_map[k]: reference_idx_map[k] for k in shared_keys}

    # Safely convert list strings to actual lists if needed
    if parse_strings_to_lists:
        updated_df[list_column] = updated_df[list_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        reference_df[list_column] = reference_df[list_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Merge list values from reference into target
    for target_idx, ref_idx in row_mapping.items():
        updated_df.at[target_idx, list_column].extend(reference_df.at[ref_idx, list_column])

    # Compute statistics if specified
    if mean_column:
        updated_df[mean_column] = updated_df[list_column].apply(lambda x: np.mean(x) if isinstance(x, list) and x else np.nan)
    if std_column:
        updated_df[std_column] = updated_df[list_column].apply(lambda x: np.std(x) if isinstance(x, list) and x else np.nan)
        
    if add_uncommon:
        # Add rows from reference_df that are not in target_df
        uncommon_rows = reference_df[~reference_df[key_column].isin(updated_df[key_column])]
        updated_df = pd.concat([updated_df, uncommon_rows], ignore_index=True)

    return updated_df, row_mapping


def remove_highly_correlated_columns(df, threshold=0.9):
    """
    Removes columns with correlation higher than a specified threshold.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Correlation threshold for removing columns.

    Returns:
        pd.DataFrame: DataFrame with highly correlated columns removed.
    """
    df_ = df.copy()
    num_df = df_.select_dtypes(include=[np.number])
    corr_matrix = num_df.corr().abs()
    # Select the upper triangle of the correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find columns to drop (highly correlated)
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    # Drop correlated numerical columns, keep categorical columns
    reduced_df = df_.drop(columns=to_drop)

    return reduced_df


def hierarchical_feature_selection(X_train: pd.DataFrame,
                                   y_train: pd.DataFrame,
                                   X_val: pd.DataFrame,
                                   y_val: pd.DataFrame,
                                   threshold: float = 0.001,
                                   col_drop_threshold: float = 0.95,
                                   n_estimators: int = 500,
                                   random_state: int = 42,
                                   criterion: str = 'squared_error',
                                   max_depth: int | None = None,
                                   min_samples_split: int = 2,
                                   min_samples_leaf: int = 1,
                                   min_weight_fraction_leaf: float = 0,
                                   max_features: str = 'sqrt',
                                   max_leaf_nodes: int | None = None,
                                   min_impurity_decrease: float = 0,
                                   bootstrap: bool = True
                                   ) -> List[str]:
    ''' Output:  
            feature_names: the selected features list
    '''
    
    X_train_ = X_train.copy()
    y_train_ = y_train.copy()
    X_val_ = X_val.copy()
    y_val_ = y_val.copy()
    
    X_train_ = remove_highly_correlated_columns(X_train_, col_drop_threshold)
    X_val_ = X_val_[X_train_.columns]
    
    # Train Random Forest model
    rf = RandomForestRegressor(n_estimators=n_estimators,
                               random_state=random_state,
                               criterion=criterion,
                               max_depth=max_depth,
                               min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf,
                               min_weight_fraction_leaf=min_weight_fraction_leaf,
                               max_features=max_features,
                               min_impurity_decrease = min_impurity_decrease,
                               max_leaf_nodes=max_leaf_nodes,
                               bootstrap=bootstrap
                               )
    print('Fitting the model at iteration = 0')
    rf.fit(X_train_, y_train_)
    
    # Initial Model Performance
    y_pred = rf.predict(X_val_)
    print("Initial RMSE:", root_mean_squared_error(y_val_, y_pred))
    
    # Feature importance
    feature_importances = rf.feature_importances_
    feature_names = X_train_.columns
    
    # Hierarchical selection
    iteration = 0
    while len(feature_names) > 1:
        iteration += 1
        print(f"Iteration {iteration} - Number of features: {len(feature_names)}")
        
        # Sort features by importance
        # sorted_idx = np.argsort(feature_importances)
        
        # # Plot feature importance
        # plt.barh(range(len(feature_importances)), feature_importances[sorted_idx])
        # plt.yticks(range(len(feature_importances)), feature_names[sorted_idx])
        # plt.show()

        # Filter features by threshold or by dropping least important ones
        mask = feature_importances > threshold
        if sum(mask) == len(feature_names):
            print("No features below the threshold, stopping.")
            break
        feature_names = feature_names[mask]
        X_train_, X_val_ = X_train_[feature_names], X_val_[feature_names]
        
        # Retrain the model with the reduced features
        rf = RandomForestRegressor(n_estimators=n_estimators,
                                   random_state=random_state,
                                   criterion=criterion,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   min_weight_fraction_leaf=min_weight_fraction_leaf,
                                   max_features=max_features,
                                   min_impurity_decrease = min_impurity_decrease,
                                   max_leaf_nodes=max_leaf_nodes,
                                   bootstrap=bootstrap
                                   )
        rf.fit(X_train_, y_train_)
        y_pred = rf.predict(X_val_)
        print("Reduced RMSE:", root_mean_squared_error(y_val_, y_pred))
        
        # Update feature importance for the remaining features
        feature_importances = rf.feature_importances_

    return feature_names


def data_stat(data_list: List[float]) -> Tuple[float, float, float, float, float, float]:
    
    data = np.array(data_list)
    min_value = min(data)
    max_value = max(data)
    mean_value = np.mean(data)
    median_value = np.median(data)
    std_value = data.std()
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    # Approximate mode using histogram bin with the highest count
    counts, bins = np.histogram(data, bins=30)
    mode_value = bins[np.argmax(counts)]
    
    return {'Min': min_value, 'Max': max_value, 'Mean': mean_value, 'Median': median_value, 'Mode': mode_value,
            'Std': std_value, 'Q1': Q1, 'Q3': Q3, 'IQR': IQR, 'Lower': lower, 'Upper': upper}

def find_outliers_iqr(df, column):
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    return df[(df[column] < lower) | (df[column] > upper)]


def get_outliers(df, columns=None, multiplier=1.5, verbose=False):
    """
    Identify outliers and non-outlier ranges for specific numeric columns 
    in a DataFrame using the Interquartile Range (IQR) method.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns : list, optional
        List of column names to analyze. If None, all numeric columns are used.
    multiplier : float, optional (default=1.5)
        The multiplier for the IQR to determine outlier thresholds.
        (1.5 for mild, 3 for extreme outliers.)
    verbose : bool, optional (default=True)
        If True, prints concise summary info.

    Returns
    -------
    summary_df : pandas.DataFrame
        DataFrame summarizing bounds, IQR, and outlier counts for each column.
    outliers_dict : dict
        Dictionary containing lists of outlier values and ranges per column.
    cleaned_df : pandas.DataFrame
        Copy of the input dataframe with outliers replaced by NaN (for optional cleaning).
    """

    # Determine which columns to use
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()[:5]
    else:
        columns = [col for col in columns if col in df.columns]

    if not columns:
        raise ValueError("No valid numeric columns found for analysis.")

    outlier_info = {}

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        mask_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_values = df.loc[mask_outliers, col].tolist()

        outlier_info[col] = {
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "num_outliers": mask_outliers.sum(),
            "outlier_indices": df.index[mask_outliers].tolist(),
            "outlier_values": outlier_values,
            "non_outlier_range": (lower_bound, upper_bound)
        }

    summary_df = pd.DataFrame(outlier_info).T
    cleaned_df = df.copy()

    # Replace outliers with NaN in the chosen columns only
    for col in columns:
        lb, ub = summary_df.loc[col, ['lower_bound', 'upper_bound']]
        cleaned_df.loc[(cleaned_df[col] < lb) | (cleaned_df[col] > ub), col] = pd.NA

    if verbose:
        print("\n=== Outlier Summary (IQR Method) ===")
        print(summary_df[['lower_bound', 'upper_bound', 'num_outliers']])

    return summary_df, outlier_info, cleaned_df.dropna(how='any')

def bland_altman_plot(y_true, y_pred, title='', pad_frac=0.05,
                      x_lims=(-10, 10), y_lims=(-10, 10), return_stats=False, color='red'):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mean_values = (y_true + y_pred) / 2
    diff_values = y_true - y_pred

    # Mean and standard deviation of differences
    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values, ddof=1)

    # Limits of Agreement (LOA)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    # Calculate percentage of points within LOA
    within_loa = np.sum((diff_values >= loa_lower) & (diff_values <= loa_upper))
    total_points = len(diff_values)
    percent_within_loa = (within_loa / total_points) * 100
    
    # ---------- axis limits (data-driven) ----------
    def padded_limits(values, frac):
        vmin, vmax = np.min(values), np.max(values)
        span = vmax - vmin
        if span == 0:
            span = abs(vmin) if vmin != 0 else 1.0
        pad = frac * span
        return vmin - pad, vmax + pad

    x_lims = padded_limits(mean_values, pad_frac)
    y_lims = padded_limits(np.r_[diff_values, loa_upper, loa_lower], pad_frac)


    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_values, diff_values, color=color, alpha=0.6, label=title)
    plt.axhline(mean_diff, color='black', linestyle='-')
    plt.axhline(loa_upper, color='blue', linestyle='-')
    plt.axhline(loa_lower, color='blue', linestyle='-')
    plt.xlabel('Mean of True and Predicted Values', fontsize=20)
    plt.ylabel('Difference (True - Predicted)', fontsize=20)
    plt.xlim(x_lims)
    plt.ylim(y_lims)

    # Smart ticks
    # num_xticks = int(((x_lims[1] - x_lims[0]) * 2 + 1))
    # # num_yticks = int(((y_lims[1] - y_lims[0]) * 2 + 1))
    # plt.xticks(np.round(np.linspace(x_lims[0], x_lims[1], num_xticks), 2), fontsize=15)
    # # plt.yticks(np.round(np.linspace(y_lims[0], y_lims[1], num_yticks), 2), fontsize=15)
    # plt.yticks([-2.5, -2, -1.5, -1, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    # Annotations for mean and LOA lines
    x_text_pos = x_lims[1]

    plt.text(x_text_pos, loa_upper + 0.1, f'+1.96 SD ({loa_upper:.2f})', fontsize=14, color='blue', ha='right', fontweight='bold')
    plt.text(x_text_pos, mean_diff + 0.1, f'Mean ({mean_diff:.2f})', fontsize=14, color='black', ha='right', fontweight='bold')
    plt.text(x_text_pos, loa_lower + 0.1, f'-1.96 SD ({loa_lower:.2f})', fontsize=14, color='blue', ha='right', fontweight='bold')

    # Vertical double-arrow between LOA lines
    x_arrow = x_lims[0] + 0.3  # Slightly to the right of y-axis
    plt.annotate(
        '', 
        xy=(x_arrow, loa_upper), 
        xytext=(x_arrow, loa_lower), 
        arrowprops=dict(arrowstyle='<->', color='purple', lw=2)
    )

    # Add percentage text next to arrow
    plt.text(x_arrow - 0.1, (loa_upper + loa_lower) / 2+0.8, f'Within: {percent_within_loa:.1f}%', 
             va='top', ha='right', fontsize=13, color='purple', fontweight='bold', rotation=90)

    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()

    if return_stats:
        return mean_diff, std_diff, loa_upper, loa_lower, percent_within_loa
    
    
def parity_plot_with_folds(
    y_true,
    y_pred,
    title='',
    color='red',
    alpha=0.6,
    pad_frac=0.05,
    return_stats=False,
    figsize = (12, 6),
    title_fontsize=20,
    folds = [2, 3],
    base_colours = ['blue', 'green', 'violet', 'pink', 'orange', 'gray', 'yellow'],
):
    """
    Parity (x vs y) plot with y=x, 2-fold and 3-fold error bands.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted values.
    title : str
        Legend label.
    color : str
        Scatter color.
    alpha : float
        Scatter transparency.
    pad_frac : float
        Fractional padding for axis limits.
    return_stats : bool
        If True, returns fold-accuracy percentages.

    Returns
    -------
    dict (optional)
        Percent of points within 2-fold and 3-fold ranges.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    # ---------- axis limits ----------
    all_vals = np.concatenate([y_true, y_pred])
    vmin, vmax = np.min(all_vals), np.max(all_vals)
    span = vmax - vmin if vmax > vmin else max(abs(vmin), 1.0)
    pad = pad_frac * span

    xlims = (vmin - pad, vmax + pad)
    ylims = xlims

    # ---------- reference lines ----------
    x = np.linspace(*xlims, 500)

    # ---------- plot ----------
    plt.figure(figsize=figsize)

    plt.scatter(y_true, y_pred, color=color, alpha=alpha, label=title)

    # Perfect fit
    plt.plot(x, x, 'k-', lw=2, label='y = x')

    # 2-fold
    # plt.plot(x, 2 * x, 'b--', lw=1.5, label='2-fold')
    # plt.plot(x, 0.5 * x, 'b--', lw=1.5)

    # # 3-fold
    # plt.plot(x, 3 * x, 'g-.', lw=1.5, label='3-fold')
    # plt.plot(x, x / 3, 'g-.', lw=1.5)
    
    all_colors = get_colors(colours=base_colours)
    for f in folds:
        C = next(all_colors)
        plt.plot(x, f * x, '-.', lw=1.5, label=f'{f}-fold', color=C)
        plt.plot(x, x / f, '-.', lw=1.5, color=C)

    plt.xlim(xlims)
    plt.ylim(ylims)

    plt.xlabel('True Values', fontsize=title_fontsize-3)
    plt.ylabel('Predicted Values', fontsize=title_fontsize-3)
    plt.xticks(fontsize = title_fontsize-4)
    plt.yticks(fontsize = title_fontsize-4)

    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=title_fontsize-8)
    plt.tight_layout()
    plt.show()

    # ---------- fold statistics ----------
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.maximum(y_pred / y_true, y_true / y_pred)

    within_2fold = np.mean(ratio <= 2) * 100
    within_3fold = np.mean(ratio <= 3) * 100

    if return_stats:
        return {
            "within_2fold_percent": within_2fold,
            "within_3fold_percent": within_3fold
        }
        
def parity_plot_with_folds_and_percent(
    y_true,
    y_pred,
    title='',
    color='red',
    alpha=0.6,
    pad_frac=0.05,
    percent_errors=(0.10, 0.20),
    base_colours = ['blue', 'green', 'violet', 'pink', 'orange', 'gray', 'yellow'],
    return_stats=False,
    folds = [2, 3],
    figsize=(7, 7)
):
    """
    Parity (x vs y) plot with:
      - y = x
      - 2-fold, 3-fold bands
      - ±percentage error bands

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted values.
    percent_errors : tuple
        Percentage errors as fractions (e.g., 0.10 = 10%).
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    # ---------- axis limits ----------
    all_vals = np.concatenate([y_true, y_pred])
    vmin, vmax = np.min(all_vals), np.max(all_vals)
    span = vmax - vmin if vmax > vmin else max(abs(vmin), 1.0)
    pad = pad_frac * span

    xlims = (vmin - pad, vmax + pad)
    ylims = xlims

    x = np.linspace(*xlims, 600)

    # ---------- plot ----------
    plt.figure(figsize=figsize)

    plt.scatter(y_true, y_pred, color=color, alpha=alpha, label=title)

    # Perfect fit
    plt.plot(x, x, 'k-', lw=2, label='y = x')

    # Fold-error bands
    plt.plot(x, 2 * x, 'b--', lw=1.4, label='2-fold')
    plt.plot(x, 0.5 * x, 'b--', lw=1.4)

    plt.plot(x, 3 * x, 'g--', lw=1.4, label='3-fold')
    plt.plot(x, x / 3, 'g--', lw=1.4)
    
    all_colors = get_colors(colours=base_colours)
    for f in folds:
        C = next(all_colors)
        plt.plot(x, f * x, '-.', lw=1.5, label=f'{f}-fold', color=C)
        plt.plot(x, x / f, '-.', lw=1.5, color=C)

    # Percentage-error bands
    for p in percent_errors:
        plt.plot(x, (1 + p)*x, color='orange', ls=':', lw=1.6,
                 label=f'±{int(p*100)}%' if p == percent_errors[0] else None)
        plt.plot(x, (1 - p)*x, color='orange', ls=':', lw=1.6)

    plt.xlim(xlims)
    plt.ylim(ylims)

    plt.xlabel('True Values', fontsize=16)
    plt.ylabel('Predicted Values', fontsize=16)

    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # ---------- statistics ----------
    stats = {}

    # Fold accuracy
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.maximum(y_pred / y_true, y_true / y_pred)

    stats["within_2fold_percent"] = np.mean(ratio <= 2) * 100
    stats["within_3fold_percent"] = np.mean(ratio <= 3) * 100

    # Percentage accuracy
    for p in percent_errors:
        mask = np.abs(y_pred - y_true) <= p * np.abs(y_true)
        stats[f"within_{int(p*100)}pct_percent"] = np.mean(mask) * 100

    if return_stats:
        return stats



def plot_distribution_compair(data_list1: List[np.ndarray | list],
                              data_list2: List[np.ndarray | list],
                              subplot_title: List[str] = None,
                              alpha: float = 0.4,
                              figsize: tuple = (12, 12),
                              suptitle: str = 'Suptitle',
                              savepath: str = '',
                              x_lims: tuple | None = None,
                              ax1_label: str = 'Data - 1',
                              ax2_label: str = 'Data - 2',
                              ax_label_fontsize: int = 10
                              ) -> None:
    
    """
    Generates a figure with side-by-side Kernel Density Estimate (KDE) plots 
    for comparing the distributions of paired data sets.

    The function plots the distribution from `data_list1` (e.g., 'Train Data') 
    against the corresponding distribution from `data_list2` (e.g., 'Independent Data') 
    for each pair of data arrays/lists. Each subplot includes vertical lines 
    for mean, median, mode, min, and max values.

    This function requires `matplotlib.pyplot` (as `plt`), `seaborn` (as `sns`), 
    and a custom `data_stat` function (which must return: min, max, mean, median, 
    mode, and one ignored value).

    Args:
        data_list1 (List[np.ndarray | list]): A list of data arrays/lists for the first set (e.g., training data).
        data_list2 (List[np.ndarray | list]): A list of data arrays/lists for the second set (e.g., independent data), 
                                                must have the same length as `data_list1`.
        subplot_title (List[str]): A list of titles for each pair of subplots (e.g., feature names), 
                                   must have the same length as `data_list1`.
        alpha (float, optional): Transparency level for the KDE fill. Defaults to 0.4.
        figsize (tuple, optional): Figure size (width, height). Defaults to (12, 12).
        suptitle (str, optional): Main title for the entire figure. Defaults to 'Suptitle'.
        savepath (str, optional): File path to save the figure. If empty, the plot is displayed. Defaults to ''.
        x_lims (tuple | None, optional): Tuple (min_x, max_x) to manually set the x-axis limits for all subplots. 
                                         If None, limits are auto-calculated based on data. Defaults to None.
        ax1_label (str, optional): Label for the first column's data (e.g., 'Train Data'). Defaults to 'Train Data'.
        ax2_label (str, optional): Label for the second column's data (e.g., 'Independent Data'). Defaults to 'Independent Data'.
        ax_label_fontsize (int, optional): Base font size for axis labels and legend. Titles are larger. Defaults to 10.

    Raises:
        AssertionError: If `data_list1`, `data_list2`, and `subplot_title` do not have the same length. 
                        (Note: While not explicitly coded, this is a necessary pre-condition).
    
    Returns:
        None: Displays or saves the plot.
    """
    if subplot_title is None:
        subplot_title = [f'Type - {i+1}' for i in range(len(data_list1))]
    else:
        assert len(data_list1) == len(data_list2) == len(subplot_title), "Titles must match number of datasets."

    fig, axes = plt.subplots(len(data_list1), 2, figsize=figsize)
    
    # # Convert axes to 2D array format if len(data_list1) is 1
    if len(data_list1) == 1:
        axes = np.array([axes])

    for i, data in enumerate(data_list1):
        
        _data_stat1 = data_stat(data_list1[i])
        min_value1, max_value1, mean_value1, median_value1, mode_value1 = _data_stat1['Min'], _data_stat1['Max'], _data_stat1['Mean'], _data_stat1['Median'], _data_stat1['Mode']
        _data_stat2 = data_stat(data_list2[i])
        min_value2, max_value2, mean_value2, median_value2, mode_value2 = _data_stat2['Min'], _data_stat2['Max'], _data_stat2['Mean'], _data_stat2['Median'], _data_stat2['Mode']

        if x_lims:
            min_x, max_x = x_lims[0], x_lims[1]
            
        else:
            # Determine the x-limits for both data_list1 and data_list2
            min_x = min(min(data_list1[i]), min(data_list2[i]))
            max_x = max(max(data_list1[i]), max(data_list2[i]))
            
            # Determine the x-limits for both data_list1 and data_list2
            mean_min_x = max(np.array(data_list1[i]).mean(), np.array(data_list2[i]).mean())
            mean_max_x = max(np.array(data_list1[i]).mean(), np.array(data_list2[i]).mean())
            
            min_x = min_x+mean_min_x if mean_min_x<0 else min_x-mean_min_x
            max_x = max_x+mean_max_x if mean_max_x>0 else max_x-mean_max_x
        
        # KDE plot (on the left column for train data)
        sns.kdeplot(data, ax=axes[i, 0], alpha=alpha, color='orange', fill=True)
        axes[i, 0].set_title(f'{ax1_label}: {subplot_title[i]}\n', fontsize=ax_label_fontsize+5)
        axes[i, 0].set_xlabel(f'{subplot_title[i]}\n', fontsize=ax_label_fontsize)
        axes[i, 0].set_ylabel('Density', fontsize=ax_label_fontsize)
        axes[i, 0].set_xlim(min_x, max_x)  # Set x-limits to the calculated min and max
        
        axes[i, 0].axvline(mean_value1, color='r', linestyle='-.', label=f'Mean: {mean_value1:.2f}')
        axes[i, 0].axvline(median_value1, color='g', linestyle='-.', label=f'Median: {median_value1:.2f}')
        axes[i, 0].axvline(mode_value1, color='b', linestyle='-.', label=f'Mode: {mode_value1:.2f}')
        axes[i, 0].axvline(min_value1, color='cyan', linestyle='-.', label=f'Min: {min_value1:.2f}')
        axes[i, 0].axvline(max_value1, color='violet', linestyle='-.', label=f'Max: {max_value1:.2f}')
        axes[i, 0].legend(fontsize=ax_label_fontsize)
        axes[i, 0].grid()

        # KDE plot (on the right column for rr data)
        sns.kdeplot(data_list2[i], ax=axes[i, 1], alpha=alpha, color='orange', fill=True)
        axes[i, 1].set_title(f'{ax2_label}: {subplot_title[i]}\n', fontsize=ax_label_fontsize+5)
        axes[i, 1].set_xlabel(f'{subplot_title[i]}\n', fontsize=ax_label_fontsize)
        axes[i, 1].set_ylabel('Density', fontsize=ax_label_fontsize)
        axes[i, 1].set_xlim(min_x, max_x)  # Set x-limits to the same min and max
        
        axes[i, 1].axvline(mean_value2, color='r', linestyle='-.', label=f'Mean: {mean_value2:.2f}')
        axes[i, 1].axvline(median_value2, color='g', linestyle='-.', label=f'Median: {median_value2:.2f}')
        axes[i, 1].axvline(mode_value2, color='b', linestyle='-.', label=f'Mode: {mode_value2:.2f}')
        axes[i, 1].axvline(min_value2, color='cyan', linestyle='-.', label=f'Min: {min_value2:.2f}')
        axes[i, 1].axvline(max_value2, color='violet', linestyle='-.', label=f'Max: {max_value2:.2f}')
        axes[i, 1].legend(fontsize=ax_label_fontsize)
        axes[i, 1].grid()
        
    plt.suptitle(suptitle, fontsize=ax_label_fontsize+10, fontweight='bold')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath,  dpi=100)
        plt.close()
    else:
        plt.show()
        
        
def plot_multiple_distribution(data_list1, n_cols=0, figsize=None, subplot_title=None,
                    suptitle="Density", x_label="Property", subplot_type='Dataset',
                    ax_label_fontsize=12, alpha=0.6, x_lims=None, savepath=None, log_scale=False
                    ) -> None:
    
    """
    Generates a figure with multiple Kernel Density Estimate (KDE) plots 
    for visualizing the distribution of several datasets.

    Each dataset in `data_list1` is plotted in a separate subplot, along with 
    vertical lines indicating key statistical measures (mean, median, mode, min, max).
    The function aims to standardize the x-axis limits across all subplots 
    for easier visual comparison.

    This function requires `matplotlib.pyplot` (as `plt`), `numpy` (as `np`), 
    `seaborn` (as `sns`), and a custom `data_stat` function (which must 
    return: min, max, mean, median, mode, and one ignored value).

    Args:
        data_list1 (List[np.ndarray | list]): A list of data arrays/lists, where each element 
                                              will be plotted as a separate distribution.
        n_cols (int, optional): The number of columns for the subplot grid. 
                                If 0, it's calculated as the square root of the number of datasets. Defaults to 0.
        figsize (tuple | None, optional): Figure size (width, height). If None, it's auto-calculated 
                                          based on the number of rows and columns. Defaults to None.
        subplot_title (List[str] | None, optional): A list of titles for each subplot (e.g., feature names). 
                                                    Must match the length of `data_list1`. Defaults to None (empty titles).
        suptitle (str, optional): Main title for the entire figure. Defaults to 'Density'.
        x_label (str, optional): Label for the x-axis of all subplots. Defaults to 'Property'.
        subplot_type (str, optional): Prefix for the subplot title (e.g., 'Dataset: Title'). Defaults to 'Dataset'.
        ax_label_fontsize (int, optional): Base font size for axis labels and legend. Titles are larger. Defaults to 12.
        bins (int, optional): Number of bins for potential histogram (though only KDE is currently plotted). Defaults to 30.
        alpha (float, optional): Transparency level for the KDE fill. Defaults to 0.6.
        kde (bool, optional): Placeholder for enabling/disabling KDE (currently always plots KDE). Defaults to True.
        x_lims (tuple | None, optional): Tuple (min_x, max_x) to manually set the x-axis limits for all subplots. 
                                         If None, limits are auto-calculated from the *first* dataset in `data_list1`. 
                                         NOTE: The limit calculation logic in the function is currently flawed as 
                                         it only uses `data_list1[0]` to determine the global limits. Defaults to None.
        savepath (str | None, optional): File path to save the figure. If None, the plot is displayed. Defaults to None.
        log_scale (bool): Plot the KDE curve in log scale or not.

    Raises:
        AssertionError: If `subplot_title` is provided but its length does not match `data_list1`.
    
    Returns:
        None: Displays or saves the plot.
    """

    if not n_cols:
        n_cols = int(np.sqrt(len(data_list1)))
    # --- Figure size ---
    n_datasets = len(data_list1)
    rows = int(np.ceil(n_datasets / n_cols))
    cols = n_cols
    if figsize is None:
        figsize = (6 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    # --- Titles handling ---
    if subplot_title is None:
        subplot_title = [f'Type - {i+1}' for i in range(len(data_list1))]
    else:
        assert len(data_list1) == len(subplot_title), "Titles must match number of datasets."

    plt.suptitle(suptitle, fontsize=ax_label_fontsize + 10, fontweight='bold')

    # --- Global x-limits ---
    if x_lims:
            min_x, max_x = x_lims[0], x_lims[1]
    else:
        # Determine the x-limits for both data_list1 and data_list2
        min_x = min(min(data_list1[0]), min(data_list1[0]))
        max_x = max(max(data_list1[0]), max(data_list1[0]))
        
        # Determine the x-limits for both data_list1 and data_list1
        mean_min_x = max(np.array(data_list1[0]).mean(), np.array(data_list1[0]).mean())
        mean_max_x = max(np.array(data_list1[0]).mean(), np.array(data_list1[0]).mean())
        
        min_x = min_x+mean_min_x if mean_min_x<0 else min_x-mean_min_x
        max_x = max_x+mean_max_x if mean_max_x>0 else max_x-mean_max_x

    # --- Loop over datasets ---
    for idx, data in enumerate(data_list1):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]

        # Compute stats (replace with your function)
        _data_stat = data_stat(data)
        min_value1, max_value1, mean_value1, median_value1, mode_value1 = _data_stat['Min'], _data_stat['Max'], _data_stat['Mean'], _data_stat['Median'], _data_stat['Mode']
        sns.kdeplot(data, ax=ax, color='green', linewidth=1, fill=True, edgecolor="black", alpha=alpha, log_scale=log_scale)

        ax.set_title(f'{subplot_type}: {subplot_title[idx]}', fontsize=ax_label_fontsize + 3)
        ax.set_xlabel(x_label, fontsize=ax_label_fontsize)
        ax.set_ylabel('Density', fontsize=ax_label_fontsize)

        # Vertical lines
        ax.axvline(mean_value1, color='r', linestyle='-.', label=f'Mean: {mean_value1:.2f}')
        ax.axvline(median_value1, color='g', linestyle='-.', label=f'Median: {median_value1:.2f}')
        ax.axvline(mode_value1, color='b', linestyle='-.', label=f'Mode: {mode_value1:.2f}')
        ax.axvline(min_value1, color='cyan', linestyle='-.', label=f'Min: {min_value1:.2f}')
        ax.axvline(max_value1, color='purple', linestyle='-.', label=f'Max: {max_value1:.2f}')

        ax.legend(loc='upper right', fontsize=ax_label_fontsize-1)
        ax.grid()
        ax.set_xlim(min_x, max_x)

    # --- Remove empty axes ---
    for idx in range(n_datasets, rows * cols):
        r, c = divmod(idx, n_cols)
        fig.delaxes(axes[r, c])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    
    
def plot_hist_compair(data_list1: List[np.ndarray | list],
                      data_list2: List[np.ndarray | list],
                      subplot_title: List[str] = None,
                      bins: int = 30,
                      alpha: float = 0.4,
                      figsize: tuple = (12, 12),
                      suptitle: str = '',
                      savepath: str = '',
                      ax1_label: str = 'Data - 1',
                      ax2_label: str = 'Data - 2',
                      x_lims: tuple | None = None,
                      x_label: str = 'X - axis',
                      ax_label_fontsize: int = 10,
                      kde=True,
                      stat='density'
                      ):
    """
    Generates a figure with side-by-side Histograms for comparing the frequency 
    distributions of paired data sets.

    The function plots the histogram from `data_list1` (e.g., 'Train Data') 
    against the corresponding histogram from `data_list2` (e.g., 'Independent Data') 
    for each pair of data arrays/lists. Each subplot includes vertical lines 
    for mean, median, mode, min, and max values. Optionally, a Kernel Density Estimate (KDE) 
    can be overlaid on the histograms.

    This function requires `matplotlib.pyplot` (as `plt`), `numpy` (as `np`), 
    `seaborn` (as `sns` if `kde=True`), and a custom `data_stat` function 
    (which must return: min, max, mean, median, mode, and one ignored value).

    Args:
        data_list1 (List[np.ndarray | list]): A list of data arrays/lists for the first set (e.g., training data).
        data_list2 (List[np.ndarray | list]): A list of data arrays/lists for the second set (e.g., independent data), 
                                              must have the same length as `data_list1`.
        subplot_title (List[str]): A list of titles for each pair of subplots (e.g., feature names), 
                                   must have the same length as `data_list1`.
        bins (int, optional): The number of bins to use for the histogram. Defaults to 30.
        alpha (float, optional): Transparency level for the histogram bars. Defaults to 0.4.
        figsize (tuple, optional): Figure size (width, height). Defaults to (12, 12).
        suptitle (str, optional): Main title for the entire figure. Defaults to 'Suptitle'.
        savepath (str, optional): File path to save the figure. If empty, the plot is displayed. Defaults to ''.
        ax1_label (str, optional): Label for the first column's data (e.g., 'Train Data'). Defaults to 'Train Data'.
        ax2_label (str, optional): Label for the second column's data (e.g., 'Independent Data'). Defaults to 'Independent Data'.
        x_lims (tuple | None, optional): Tuple (min_x, max_x) to manually set the x-axis limits for all subplots. 
                                         If None, limits are auto-calculated based on both paired data sets. Defaults to None.
        ax_label_fontsize (int, optional): Base font size for axis labels and legend. Titles are larger. Defaults to 10.
        kde (bool, optional): If True, overlays a Kernel Density Estimate (KDE) plot on the histogram. Defaults to False.

    Raises:
        AssertionError: If `data_list1`, `data_list2`, and `subplot_title` do not have the same length. 
                        (Note: This is a necessary pre-condition for the loops to work correctly).
    
    Returns:
        None: Displays or saves the plot.
    """

    kde = False if stat == 'probability' else kde
    if subplot_title is None:
        subplot_title = [f'Type - {i+1}' for i in range(len(data_list1))]
    else:
        assert len(data_list1) == len(data_list2) == len(subplot_title), "Titles must match number of datasets."
        
    fig, axes = plt.subplots(len(data_list1), 2, figsize=figsize)
    
    # # Convert axes to 2D array format if len(data_list1) is 1
    if len(data_list1) == 1:
        axes = np.array([axes])

    plt.suptitle(suptitle, fontsize=ax_label_fontsize+10, fontweight='bold')
    for i, _ in enumerate(data_list1):
        
        _data_stat1 = data_stat(data_list1[i])
        _data_stat2 = data_stat(data_list2[i])
        min_value1, max_value1, mean_value1, median_value1, mode_value1 = _data_stat1['Min'], _data_stat1['Max'], _data_stat1['Mean'], _data_stat1['Median'], _data_stat1['Mode']
        min_value2, max_value2, mean_value2, median_value2, mode_value2 = _data_stat2['Min'], _data_stat2['Max'], _data_stat2['Mean'], _data_stat2['Median'], _data_stat2['Mode']

        if x_lims:
            min_x, max_x = x_lims[0], x_lims[1]
            
        else:
            # Determine the x-limits for both data_list1 and data_list2
            min_x = min(min(data_list1[i]), min(data_list2[i]))
            max_x = max(max(data_list1[i]), max(data_list2[i]))
            
            # Determine the x-limits for both data_list1 and data_list2
            mean_min_x = max(np.array(data_list1[i]).mean(), np.array(data_list2[i]).mean())
            mean_max_x = max(np.array(data_list1[i]).mean(), np.array(data_list2[i]).mean())
            
            min_x = min_x+mean_min_x if mean_min_x<0 else min_x-mean_min_x
            max_x = max_x+mean_max_x if mean_max_x>0 else max_x-mean_max_x
        
        # Bar plot (on the left column)
        # axes[i, 0].hist(data_list1[i], bins=bins, alpha=alpha, color='orange', edgecolor='black')
        sns.histplot(data_list1[i], ax=axes[i, 0], color='green', bins=bins, stat=stat,
                     edgecolor='black', linewidth=1, alpha=alpha)
        if kde:
            sns.kdeplot(data_list1[i], ax=axes[i, 0], color='red', linewidth=1)
            
        axes[i, 0].set_title(f'{ax1_label}: {subplot_title[i]}\n', fontsize=ax_label_fontsize+5)
        axes[i, 0].set_xlabel(f'{x_label}\n', fontsize=ax_label_fontsize)
        axes[i, 0].set_ylabel(stat.capitalize(), fontsize=ax_label_fontsize)
        axes[i, 0].set_xlim(min_x, max_x)  # Set x-limits to the calculated min and max
        
        axes[i, 0].axvline(mean_value1, color='r', linestyle='-.', label=f'Mean: {mean_value1:.2f}')
        axes[i, 0].axvline(median_value1, color='g', linestyle='-.', label=f'Median: {median_value1:.2f}')
        axes[i, 0].axvline(mode_value1, color='b', linestyle='-.', label=f'Mode: {mode_value1:.2f}')
        axes[i, 0].axvline(min_value1, color='cyan', linestyle='-.', label=f'Min: {min_value1:.2f}')
        axes[i, 0].axvline(max_value1, color='violet', linestyle='-.', label=f'Max: {max_value1:.2f}')
        axes[i, 0].legend(loc='upper right', fontsize=ax_label_fontsize)  # Add legend to bar plot
        axes[i, 0].grid()
        # if kde:
        #     sns.kdeplot(data_list1[i], ax=axes[i, 0], color='red', linewidth=1, alpha=alpha)

        # Bar plot (on the left column)
        # axes[i, 1].hist(data_list2[i], bins=bins, alpha=alpha, color='orange', edgecolor='black')
        sns.histplot(data_list2[i], ax=axes[i, 1], color='green', bins=bins, stat=stat,
                     edgecolor='black', linewidth=1, alpha=alpha)
        if kde:
            sns.kdeplot(data_list2[i], ax=axes[i, 1], color='red', linewidth=1)
        axes[i, 1].set_title(f'{ax2_label}: {subplot_title[i]}\n', fontsize=ax_label_fontsize+5)
        axes[i, 1].set_xlabel(f'{x_label}\n', fontsize=ax_label_fontsize)
        axes[i, 1].set_ylabel(stat.capitalize(), fontsize=ax_label_fontsize)
        axes[i, 1].set_xlim(min_x, max_x)  # Set x-limits to the calculated min and max
        
        axes[i, 1].axvline(mean_value2, color='r', linestyle='-.', label=f'Mean: {mean_value2:.2f}')
        axes[i, 1].axvline(median_value2, color='g', linestyle='-.', label=f'Median: {median_value2:.2f}')
        axes[i, 1].axvline(mode_value2, color='b', linestyle='-.', label=f'Mode: {mode_value2:.2f}')
        axes[i, 1].axvline(min_value2, color='cyan', linestyle='-.', label=f'Min: {min_value2:.2f}')
        axes[i, 1].axvline(max_value2, color='violet', linestyle='-.', label=f'Max: {max_value2:.2f}')
        axes[i, 1].legend(loc='upper right', fontsize=ax_label_fontsize)
        axes[i, 1].grid()
        if kde:
            sns.kdeplot(data_list2[i], ax=axes[i, 1], color='red', linewidth=1, alpha=alpha)


    # Adjust layout to prevent overlapping
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath,  dpi=100)
        plt.close()
    else:
        plt.show()
        
    
# def plot_multiple_hist(
#     data_list1: List[np.ndarray | list],
#     subplot_title: List[str] = None,
#     bins: int|list = 30,
#     alpha: float = 0.4,
#     figsize: tuple = (),
#     suptitle: str = 'Suptitle',
#     savepath: str = '',
#     kde: bool = True,
#     ax1_label: str = 'Train Data',
#     ax_label_fontsize: int = 10,
#     x_lims=None
# ):
#     # Figure size handling
#     if not figsize:
#         figsize = (6, 4 * len(data_list1))

#     # Default subplot titles
#     if not subplot_title:
#         subplot_title = [''] * len(data_list1)
#     else:
#         assert len(data_list1) == len(subplot_title), "Titles must match number of datasets."

#     fig, axes = plt.subplots(len(data_list1), 1, figsize=figsize)
#     axes = np.atleast_1d(axes)  # Ensure iterable axes

#     plt.suptitle(suptitle, fontsize=ax_label_fontsize + 10, fontweight='bold')

#     for i, data in enumerate(data_list1):
#         # Get statistics (replace with your own function)
#         min_value1, max_value1, mean_value1, median_value1, mode_value1, _ = data_stat(data)
        
#         if x_lims:
#             min_x, max_x = x_lims[0], x_lims[1]
            
#         else:
#             # Determine the x-limits for both data_list1 and data_list2
#             min_x = min(min(data_list1[0]), min(data_list1[0]))
#             max_x = max(max(data_list1[0]), max(data_list1[0]))
            
#             # Determine the x-limits for both data_list1 and data_list1
#             mean_min_x = max(np.array(data_list1[0]).mean(), np.array(data_list1[0]).mean())
#             mean_max_x = max(np.array(data_list1[0]).mean(), np.array(data_list1[0]).mean())
            
#             min_x = min_x+mean_min_x if mean_min_x<0 else min_x-mean_min_x
#             max_x = max_x+mean_max_x if mean_max_x>0 else max_x-mean_max_x

#         # Histogram on correct axis
#         sns.histplot(data, ax=axes[i], color='green', bins=bins, stat="density",
#                      edgecolor='black', linewidth=1, alpha=alpha)
#         if kde:
#             sns.kdeplot(data, ax=axes[i], color='red', linewidth=1)

#         axes[i].set_title(f'{ax1_label}: {subplot_title[i]}', fontsize=ax_label_fontsize + 5)
#         axes[i].set_xlabel(subplot_title[i], fontsize=ax_label_fontsize)
#         axes[i].set_ylabel('Frequency', fontsize=ax_label_fontsize)

#         # Add vertical lines
#         axes[i].axvline(mean_value1, color='r', linestyle='-.', label=f'Mean: {mean_value1:.2f}')
#         axes[i].axvline(median_value1, color='g', linestyle='-.', label=f'Median: {median_value1:.2f}')
#         axes[i].axvline(mode_value1, color='b', linestyle='-.', label=f'Mode: {mode_value1:.2f}')
#         axes[i].axvline(min_value1, color='cyan', linestyle='-.', label=f'Min: {min_value1:.2f}')
#         axes[i].axvline(max_value1, color='purple', linestyle='-.', label=f'Max: {max_value1:.2f}')

#         axes[i].legend(loc='upper right', fontsize=ax_label_fontsize)
#         axes[i].grid()
#         axes[i].set_xlim(min_x, max_x)  # Set x-limits to the calculated min and max

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#     if savepath:
#         plt.savefig(savepath, dpi=100)
#         plt.close()
#     else:
#         plt.show()


def plot_multiple_hist(data_list1, n_cols=0, figsize=None, subplot_title=None,
                    suptitle="Histograms", x_label="Property", subplot_type='Dataset',
                    ax_label_fontsize=12, bins=30, alpha=0.6, kde=True,
                    x_lims=None, savepath=None, stat="density", add_count=False):
    
    """
    Generates a figure with multiple Histograms for visualizing the frequency 
    or density distribution of several datasets.

    Each dataset in `data_list1` is plotted as a histogram in a separate subplot, 
    with `stat="density"` used for the histogram to make it comparable with the 
    optional KDE overlay. Vertical lines indicate key statistical measures (mean, 
    median, mode, min, max). The function attempts to standardize the x-axis 
    limits across all subplots for comparison.

    This function requires `matplotlib.pyplot` (as `plt`), `numpy` (as `np`), 
    `seaborn` (as `sns`), and a custom `data_stat` function (which must 
    return: min, max, mean, median, mode, and one ignored value).

    Args:
        data_list1 (List[np.ndarray | list]): A list of data arrays/lists, where each element 
                                              will be plotted as a separate distribution.
        n_cols (int, optional): The number of columns for the subplot grid. 
                                If 0, it's calculated as the square root of the number of datasets. Defaults to 0.
        figsize (tuple | None, optional): Figure size (width, height). If None, it's auto-calculated 
                                          based on the number of rows and columns. Defaults to None.
        subplot_title (List[str] | None, optional): A list of titles for each subplot (e.g., feature names). 
                                                    Must match the length of `data_list1`. Defaults to None (empty titles).
        suptitle (str, optional): Main title for the entire figure. Defaults to 'Histograms'.
        x_label (str, optional): Label for the x-axis of all subplots. Defaults to 'Property'.
        subplot_type (str, optional): Prefix for the subplot title (e.g., 'Dataset: Title'). Defaults to 'Dataset'.
        ax_label_fontsize (int, optional): Base font size for axis labels and legend. Titles are larger. Defaults to 12.
        bins (int, optional): The number of bins to use for the histogram. Defaults to 30.
        alpha (float, optional): Transparency level for the histogram bars. Defaults to 0.6.
        kde (bool, optional): If True, overlays a Kernel Density Estimate (KDE) plot on the histogram. Defaults to True.
        x_lims (tuple | None, optional): Tuple (min_x, max_x) to manually set the x-axis limits for all subplots. 
                                         If None, limits are auto-calculated from the *first* dataset in `data_list1`. 
                                         NOTE: The limit calculation logic in the function is currently flawed as 
                                         it only uses `data_list1[0]` to determine the global limits. Defaults to None.
        savepath (str | None, optional): File path to save the figure. If None, the plot is displayed. Defaults to None.
        stat (str): Control the normalization of the bars
                    - stat='count': Y-axis is the number of observations (Frequency). Area not equal 1.
                    - stat='density': Y-axis is the density. Area = 1.
                    - stat='probability': Y-axis is the probability. Sum of bar heights = 1.

    Raises:
        AssertionError: If `subplot_title` is provided but its length does not match `data_list1`.
    
    Returns:
        None: Displays or saves the plot.
    """
    
    kde = False if stat == 'probability' else kde
    if not n_cols:
        n_cols = int(np.sqrt(len(data_list1)))
    # --- Figure size ---
    n_datasets = len(data_list1)
    rows = int(np.ceil(n_datasets / n_cols))
    cols = n_cols
    if figsize is None:
        figsize = (6 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    # --- Titles handling ---
    if subplot_title is None:
        subplot_title = [f'Type - {i+1}' for i in range(len(data_list1))]
    else:
        assert len(data_list1) == len(subplot_title), "Titles must match number of datasets."

    plt.suptitle(suptitle, fontsize=ax_label_fontsize + 10, fontweight='bold')

    # --- Global x-limits ---
    if x_lims:
            min_x, max_x = x_lims[0], x_lims[1]
    else:
        # Determine the x-limits for both data_list1 and data_list2
        min_x = min(min(data_list1[0]), min(data_list1[0]))
        max_x = max(max(data_list1[0]), max(data_list1[0]))
        
        # Determine the x-limits for both data_list1 and data_list1
        mean_min_x = max(np.array(data_list1[0]).mean(), np.array(data_list1[0]).mean())
        mean_max_x = max(np.array(data_list1[0]).mean(), np.array(data_list1[0]).mean())
        
        min_x = min_x+mean_min_x if mean_min_x<0 else min_x-mean_min_x
        max_x = max_x+mean_max_x if mean_max_x>0 else max_x-mean_max_x

    # --- Loop over datasets ---
    for idx, data in enumerate(data_list1):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]

        # Compute stats (replace with your function)
        _data_stat = data_stat(data)
        min_value1, max_value1, mean_value1, median_value1, mode_value1 = _data_stat['Min'], _data_stat['Max'], _data_stat['Mean'], _data_stat['Median'], _data_stat['Mode']

        sns.histplot(data, ax=ax, color='green', bins=bins, stat=stat,
                     edgecolor='black', linewidth=1, alpha=alpha)
        if kde:
            sns.kdeplot(data, ax=ax, color='red', linewidth=1)

        ax.set_title(f'{subplot_type}: {subplot_title[idx]}', fontsize=ax_label_fontsize + 3, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=ax_label_fontsize, fontweight='bold')
        ax.set_ylabel(stat.capitalize(), fontsize=ax_label_fontsize, fontweight='bold')

        # Vertical lines
        ax.axvline(mean_value1, color='r', linestyle='-.', label=f'Mean: {mean_value1:.2f}')
        ax.axvline(median_value1, color='g', linestyle='-.', label=f'Median: {median_value1:.2f}')
        ax.axvline(mode_value1, color='b', linestyle='-.', label=f'Mode: {mode_value1:.2f}')
        ax.axvline(min_value1, color='cyan', linestyle='-.', label=f'Min: {min_value1:.2f}')
        ax.axvline(max_value1, color='purple', linestyle='-.', label=f'Max: {max_value1:.2f}')

        ax.legend(loc='upper right', fontsize=ax_label_fontsize-1)
        ax.grid()
        ax.set_xlim(min_x, max_x)
        
        if add_count:
            for patch in ax.patches:
                height = patch.get_height()
                if height > 0:
                    ax.annotate(
                        f'{int(height)}',
                        (patch.get_x() + patch.get_width() / 2, height),
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        color='black',
                        xytext=(0, 3),
                        textcoords='offset points'
                    )

    # --- Remove empty axes ---
    for idx in range(n_datasets, rows * cols):
        r, c = divmod(idx, n_cols)
        fig.delaxes(axes[r, c])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    
    
def dist_hist_comparison(data_list1: List[np.ndarray | list],
                         data_list2: List[np.ndarray | list],
                         subplot_title: List[str] = None,
                         figsize: tuple = (12, 12),
                         list1_label: str = 'Data - 1',
                         list2_label: str = 'Data - 2',
                         x_label: str = 'X - axis',
                         suptitle: str = 'Suptitle',
                         bins: int = 30,
                         alpha: float = 0.4,
                         x_lims: tuple = None,
                         stat: str = 'density',
                         savepath: str = '',
                         ):
    
    """
    Generates a figure with side-by-side Histogram and Kernel Density Estimate (KDE) 
    plots for comparing the distribution of paired data sets.

    For each corresponding pair of data arrays/lists from `data_list1` and `data_list2`, 
    it creates two subplots:
    1. A **Histogram** with overlaid bars to compare the frequency counts.
    2. A **KDE Plot** with overlaid distributions to compare the density shapes.

    The x-axis limits are standardized across both the histogram and the KDE plot 
    for each pair to facilitate direct visual comparison.

    This function requires `matplotlib.pyplot` (as `plt`), `numpy` (as `np`), 
    and `seaborn` (as `sns`).

    Args:
        data_list1 (List[np.ndarray | list]): A list of data arrays/lists for the first set (e.g., 'Original' data).
        data_list2 (List[np.ndarray | list]): A list of data arrays/lists for the second set (e.g., 'Predicted' data), 
                                              must have the same length as `data_list1`.
        subplot_title (List[str]): A list of base titles for each pair of subplots (e.g., condition or feature names), 
                                   must have the same length as `data_list1`.
        figsize (tuple, optional): Figure size (width, height). Defaults to (12, 12).
        list1_label (str, optional): Legend label for the data in `data_list1`. Defaults to 'Original'.
        list2_label (str, optional): Legend label for the data in `data_list2`. Defaults to 'Predicted'.
        x_label (str, optional): The label for the x-axis in all subplots. Defaults to 'logPapp Value'.
        suptitle (str, optional): Main title for the entire figure. Defaults to 'Suptitle'.
        bins (int, optional): The number of bins to use for the histograms. Defaults to 30.
        alpha (float, optional): Transparency level for the histogram bars and KDE fill. Defaults to 0.4.
        x_lims (tuple | None, optional): Tuple (min_x, max_x) to manually set the x-axis limits for all subplots. 
                                         If None, limits are auto-calculated based on the data. Defaults to None.
        savepath (str, optional): File path to save the figure. If empty, the plot is displayed. Defaults to ''.

    Raises:
        AssertionError: If `data_list1`, `data_list2`, and `subplot_title` do not have the same length. 
                        (Note: This is a necessary pre-condition for the loops to work correctly).

    Returns:
        None: Displays or saves the plot.
    """
    kde = False if stat == 'probability' else kde
    if subplot_title is None:
        subplot_title = [f'Type - {i+1}' for i in range(len(data_list1))]
    else:
        assert len(data_list1) == len(data_list2) == len(subplot_title), "Titles must match number of datasets."
    
    # Create a 3x2 grid of subplots
    fig, axes = plt.subplots(len(data_list1), 2, figsize=figsize)
    
    # # Convert axes to 2D array format if len(data_list1) is 1
    if len(data_list1) == 1:
        axes = np.array([axes])

    for i, _ in enumerate(data_list1):
        
        if x_lims:
            min_x, max_x = x_lims[0], x_lims[1]
            
        else:
            # Determine the x-limits for both data_list1 and data_list2
            min_x = min(min(data_list1[i]), min(data_list2[i]))
            max_x = max(max(data_list1[i]), max(data_list2[i]))
            
            # Determine the x-limits for both data_list1 and data_list2
            mean_min_x = max(np.array(data_list1[i]).mean(), np.array(data_list2[i]).mean())
            mean_max_x = max(np.array(data_list1[i]).mean(), np.array(data_list2[i]).mean())
            
            min_x = min_x+mean_min_x if mean_min_x<0 else min_x-mean_min_x
            max_x = max_x+mean_max_x if mean_max_x>0 else max_x-mean_max_x
        
        # Bar plot (on the left column)
        sns.histplot(data_list1[i], ax=axes[i, 0], color='green', bins=bins, stat=stat,
                     edgecolor='black', linewidth=1, alpha=alpha, label=list1_label)
        sns.histplot(data_list2[i], ax=axes[i, 0], color='red', bins=bins, stat=stat,
                     edgecolor='black', linewidth=1, alpha=alpha, label = list2_label)
        axes[i, 0].set_title(f'{subplot_title[i]} - Bar Plot')
        axes[i, 0].set_xlabel(x_label)
        axes[i, 0].set_ylabel(stat.capitalize())
        axes[i, 0].legend(loc='upper right')
        
        axes[i, 0].set_xlim(min_x, max_x)  # Set x-range from -9 to -3
        # axes[i, 0].set_ylim(0, 300)
        
        # KDE plot (on the right column)
        sns.kdeplot(data_list1[i], ax=axes[i, 1], alpha=alpha, color='green', fill=True, label=list1_label)
        sns.kdeplot(data_list2[i], ax=axes[i, 1], alpha=alpha, color='red', fill=True, label = list2_label)
        axes[i, 1].set_title(f'{subplot_title[i]} -KDE Plot')
        axes[i, 1].set_xlabel(x_label)
        axes[i, 1].set_ylabel('Density')
        axes[i, 1].legend(loc='upper right')
        axes[i, 1].set_xlim(min_x, max_x)
        # axes[i, 1].set_ylim(0, 1)
        
    plt.suptitle(suptitle, fontsize=20, fontweight='bold')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath,  dpi=100)
        plt.close()
    else:
        plt.show()

def plot_stacked_bars(df, target_col, feature_cols='all', class_order=None,
                      figsize=(10, 6), colormap='tab10', rotation=45):
    """
    Plot stacked bar charts for categorical feature(s) showing the distribution of target classes.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    
    target_col : str
        Name of the target column (must be categorical or discrete).
    
    feature_cols : list or str, default='all'
        List of feature columns to plot. If 'all', uses all object-type columns except the target.
    
    class_order : list, optional
        List specifying the desired order of target classes in the stacked bars and legend.
        If None, uses sorted unique values from the data.
    
    figsize : tuple, default=(10, 6)
        Figure size for each plot.
    
    colormap : str or matplotlib colormap, default='tab10'
        Colormap to use for the bars (one color per target class).
    
    rotation : int, default=45
        Rotation angle for x-axis tick labels.
    
    Returns:
    --------
    None
        Displays the stacked bar plots.
    """
    
    if feature_cols == 'all':
        # Automatically select object (categorical) columns excluding the target
        categorical_columns = [col for col in df.columns if col != target_col]
    else:
        categorical_columns = [col for col in feature_cols if col != target_col]
    
    # Determine the order of classes
    if class_order is None:
        class_order = sorted(df[target_col].dropna().unique())

    for col in categorical_columns:
        # Crosstab of feature column vs. target column, with explicit class order
        ct = pd.crosstab(df[col], df[target_col])

        # Reorder the columns (i.e., target classes) as per class_order
        for cls_ in class_order:
            if cls_ not in ct.columns:
                ct[cls_] = 0  # add missing class as 0s

        ct = ct[class_order]  # re-order columns

        # Plot
        ct.plot(kind='bar', stacked=True, figsize=figsize, colormap=colormap)

        plt.title(f"Distribution of target by category in '{col}'")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend(title=target_col)
        plt.xticks(rotation=rotation)
        plt.tight_layout()
        plt.show()
  

def percentage_within_percent_error(y_true, y_pred, percent=10, transform_func=lambda x: x):
    """
    Calculate the percentage of repeated measurements within a given percentage of error.
    For example: if y is the true value and p is the percentage then:
        To find the number of datapoints within (y - yp/100, y + yp/100).

    Parameters:
    - y_true (list or numpy array): The true experimental values.
    - y_pred (list or numpy array): The predicted experimental values.
    - percent (float): The allowable percentage of error. Sould be between [0, 100]. 

    Returns:
    - float: Percentage of values within the specified error percentage.
    """
    y_true = transform_func(np.asarray(y_true, dtype=float))
    y_pred = transform_func(np.asarray(y_pred, dtype=float))
    
    # use magnitude for percentage band
    scale = np.abs(y_true)
    interval_hi = y_true + scale * (percent / 100)
    interval_lo = y_true - scale * (percent / 100)
    within = (interval_lo <= y_pred) & (y_pred <= interval_hi)

    return round(100 * np.mean(within), 2)

def percentage_within_fold_change(y_true, y_pred, fold=2, transform_func=lambda x: x):
    """
    Calculate the percentage of repeated measurements within a given fold range.

    Parameters:
    - y_true (list or numpy array): The true experimental values.
    - y_pred (list or numpy array): The predicted experimental values.
    - fold (float): The fold change range (default is 2).

    Returns:
    - float: Percentage of values within the specified fold range.
    """
    y_true = transform_func(np.array(y_true, dtype=float))
    y_pred = transform_func(np.array(y_pred, dtype=float))
    
    # Ensure there are no zero values to avoid division issues
    y_true[y_true == 0] = 1e-3
    assert np.all(y_true != 0), "True values must not contain zero."
    
    # Calculate fold change
    fold_change = y_pred / y_true 
    max_, min_ = np.round(max(fold_change), 2), np.round(min(fold_change), 2)
    within_fold = (1/fold <= fold_change) & (fold_change <= fold)
    
    # Calculate the percentage
    percentage = round(np.sum(within_fold) / len(y_true)*100, 2)
    
    return percentage, max_, min_

def geometric_mean_fold_error(
    y_true, 
    y_pred, 
    transform_func=lambda x: x,
    use_log_domain=False,
    eps=1e-12
):
    """
    Compute Geometric Mean Fold Error (GMFE).

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    transform_func : callable, optional
        Transformation applied before computation.
    use_log_domain : bool, optional
        If True, assumes inputs are already in log scale (e.g., pKa, logP).
        GMFE = 10 ** MAE.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    gmfe : float
    invalid : int
        Number of excluded samples
    """

    y_true = transform_func(np.asarray(y_true, dtype=float))
    y_pred = transform_func(np.asarray(y_pred, dtype=float))

    # Case 1: Already log-transformed values (preferred in chemistry)
    if use_log_domain:
        mae = np.mean(np.abs(y_pred - y_true))
        return 10 ** mae, 0

    # Case 2: Raw values → use absolute ratio
    valid = (np.abs(y_true) > eps) & (np.abs(y_pred) > eps)

    invalid = np.sum(~valid)
    if invalid > 0:
        warnings.warn(f"{invalid} values ignored due to near-zero entries.")

    y_true_valid = np.abs(y_true[valid])
    y_pred_valid = np.abs(y_pred[valid])

    log_fold_errors = np.abs(np.log10(y_pred_valid / y_true_valid))
    gmfe = 10 ** np.mean(log_fold_errors)

    return gmfe, invalid

def min_max(y_true, y_pred, range_list, eps = 0.5, transform_func=lambda x: x):
    
    range_list = range_list.apply(lambda x: ast.literal_eval(x)).tolist()
    
    y_true = transform_func(np.array(y_true, dtype=float))
    y_pred = transform_func(np.array(y_pred, dtype=float))
    
    num = 0
    for i in range(len(y_true)):
        x = y_pred[i]
        z = range_list[i]
        if len(z) == 1:
            min_, max_ = z[0]-eps, z[0]+eps
        else:
            min_, max_ = min(z), max(z)
        if x <= max_ and x >= min_:
            num += 1
    percent = num/len(y_pred)#*100
    
    return percent

def gmfe(y_true, y_pred, eps=1e-8, base=10, transform_func=lambda x: x):
    """
    Generalized Geometric Mean Fold Error (GMFE).
    
    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    eps : float, optional
        Small constant added to avoid division by zero (default: 1e-8).
    base : float, optional
        Logarithm base (default: 10, use np.e for natural log).
    
    Returns
    -------
    float
        Geometric Mean Fold Error.
    """
    y_true = transform_func(np.array(y_true, dtype=float))
    y_pred = transform_func(np.array(y_pred, dtype=float))

    # Ratios with absolute values to avoid undefined logs
    ratio = (np.abs(y_pred) + eps) / (np.abs(y_true) + eps)

    # Compute log in chosen base
    logs = np.log(ratio) / np.log(base)

    # GMFE formula
    gmfe_value = base ** (np.mean(np.abs(logs)))
    
    return gmfe_value

def regression_test_metrics(y_true: np.ndarray[float] | List[float],
                 y_pred: np.ndarray[float] | List[float],
                 transform_func=lambda x: x,
                 fold_transform_fn=lambda x:x,
                 use_log_domain_gmfe=False,
                 decimals=2
            ) -> OrderedDict[str, float]:
    
    '''
    Input:
        - model: Random Forest Model
        - y_true: Array/List Object
        - y_pred: Array/List Object
    
    Output: A dictionary containing MSE, RMSE, MAE and r2 Values.
    '''
    if not isinstance(y_pred, np.ndarray):
        y_pred = transform_func(np.array(y_pred))
    if not isinstance(y_true, np.ndarray):
        y_true = transform_func(np.array(y_true))

    assert y_pred.shape == y_true.shape, f"Shape mismatch: y_pred -> {y_pred.shape}, y_true -> {y_true.shape}"
    
    # Calculate metrics on Test Set
    rmse_test = root_mean_squared_error(y_true, y_pred)
    mse_test = rmse_test**2
    mae_test = mean_absolute_error(y_true, y_pred)
    r2_val_test = r2_score(y_true, y_pred)
    pcc = np.corrcoef(y_true, y_pred,)[0, 1]
    
    fold2, _, _ = percentage_within_fold_change(y_true=y_true, y_pred=y_pred, fold=2, transform_func=fold_transform_fn)
    fold3, _, _ = percentage_within_fold_change(y_true=y_true, y_pred=y_pred, fold=3, transform_func=fold_transform_fn)
    # fold2 = percentage_within_percent_error(y_true=y_true, y_pred=y_pred, percent=10)
    # fold3 = percentage_within_percent_error(y_true=y_true, y_pred=y_pred, percent=20)
    fold5, _, _ = percentage_within_fold_change(y_true=y_true, y_pred=y_pred, fold=5, transform_func=fold_transform_fn)
    _gmfe, _ = geometric_mean_fold_error(y_true=y_true, y_pred=y_pred, transform_func=fold_transform_fn, use_log_domain=use_log_domain_gmfe)
    # _gmfe = gmfe(y_true=y_true, y_pred=y_pred, transform_func=fold_transform_fn)
    
    
    results = OrderedDict({'mse':round(mse_test, decimals),
                           'r2':round(r2_val_test, decimals),
                           'rmse':round(rmse_test, decimals),
                           'mae':round(mae_test, decimals),
                           'PCC':round(pcc, decimals),
                           'GMFE':round(_gmfe, decimals),
                           'fold2':round(fold2, decimals),
                           'fold3':round(fold3, decimals),
                        #    '10%':round(fold2, decimals),
                        #    '20%':round(fold3, decimals),
                           'fold5':round(fold5, decimals),
                           })
    
    return results

def q2_f1(
    y_test_obs: Union[np.ndarray, pd.Series, list],
    y_test_pred: Union[np.ndarray, pd.Series, list],
    y_train_obs: Union[np.ndarray, pd.Series, list]
) -> float:
    """
    Computes Q^2_{F_1}, the training-mean–referenced external predictive squared correlation coefficient.

    Input:
        - y_test_obs: Observed response values of the external test set
        - y_test_pred: Model-predicted response values for the external test set
        - y_train_obs: Observed response values of the training set

    Output:
        - Q^2_{F_1} value (float), measuring true external predictivity
          relative to the training-set mean
    """

    y_test_obs = np.asarray(y_test_obs)
    y_test_pred = np.asarray(y_test_pred)
    y_train_obs = np.asarray(y_train_obs)

    y_train_mean = np.mean(y_train_obs)

    press = np.sum((y_test_obs - y_test_pred) ** 2)
    denom = np.sum((y_test_obs - y_train_mean) ** 2)+1e-8

    return 1.0 - press / denom


def q2_f2(
    y_test_obs: Union[np.ndarray, pd.Series, list],
    y_test_pred: Union[np.ndarray, pd.Series, list]
) -> float:
    """
    Computes Q^2_{F_2}, the test-mean–referenced external predictive squared correlation coefficient.

    Input:
        - y_test_obs: Observed response values of the external test set
        - y_test_pred: Model-predicted response values for the external test set

    Output:
        - Q^2_{F_2} value (float), measuring explained variance
          within the test set
    """

    y_test_obs = np.asarray(y_test_obs)
    y_test_pred = np.asarray(y_test_pred)

    y_test_mean = np.mean(y_test_obs)

    press = np.sum((y_test_obs - y_test_pred) ** 2)
    denom = np.sum((y_test_obs - y_test_mean) ** 2)+1e-8

    return 1.0 - press / denom


def q2_f3(
    y_test_obs: Union[np.ndarray, pd.Series, list],
    y_test_pred: Union[np.ndarray, pd.Series, list],
    y_train_obs: Union[np.ndarray, pd.Series, list]
) -> float:
    """
    Computes Q^2_{F_3}, the prediction-variance–referenced external predictive
    squared correlation coefficient.

    Input:
        - y_test_obs: Observed response values of the external test set
        - y_test_pred: Model-predicted response values for the external test set
        - y_train_obs: Observed response values of the training set

    Output:
        - Q^2_{F_3} value (float), detecting systematic slope or scaling bias
          in external predictions
    """

    y_test_obs = np.asarray(y_test_obs)
    y_test_pred = np.asarray(y_test_pred)
    y_train_obs = np.asarray(y_train_obs)

    y_train_mean = np.mean(y_train_obs)

    press = np.sum((y_test_obs - y_test_pred) ** 2)
    denom = np.sum((y_test_pred - y_train_mean) ** 2)+1e-8

    return 1.0 - press / denom



def test_rf(model: RandomForestRegressor | SVC,
            X_test: pd.DataFrame,
            y_test: pd.DataFrame
            ) -> OrderedDict[str, float]:
    
    '''
    Input:
        - model: Random Forest Model
        - X_test: DataFrame Object
        - y_test: Datafrane Object
    
    Output: A dictionary containing MSE, RMSE, MAE and r2 Values.
    '''
    
    y_pred = model.predict(X_test.astype(np.float16))
    y_prediced = np.array(y_pred)
    y_test_eval = np.array(y_test)

    # Calculate metrics on Test Set
    rmse_test = root_mean_squared_error(y_test_eval, y_prediced)
    mse_test = rmse_test**2
    mae_test = mean_absolute_error(y_test_eval, y_prediced)
    r2_val_test = r2_score(y_test_eval, y_prediced)
    fold2, _, _ = percentage_within_fold_change(y_true=y_test_eval, y_pred=y_prediced, fold=2)
    fold3, _, _ = percentage_within_fold_change(y_true=y_test_eval, y_pred=y_prediced, fold=3)
    
    
    results = OrderedDict({'mse':round_up(mse_test, 2),
                           'rmse':round_up(rmse_test, 2),
                           'mae':round_up(mae_test, 2),
                           'r2':round_up(r2_val_test, 2),
                           'fold2':round_up(fold2, 2),
                           'fold3':round_up(fold3, 2)
                           })
    
    return results

def subset_loader(dataloader: DataLoader,
                  batch_size: int,
                  subset_ratio: float = 0.2,
                  collate_fn=None
                  ) -> Tuple[DataLoader, DataLoader]:
    
    '''
    Input: 
        - A dataloader
        - Batch Size of the generated dataloader
        - Subset Ratio
        
    Output: A random subset of the dataloader
    '''
    
    complement_loader = None    
    original_dataset = dataloader.dataset
    subset_size = int(subset_ratio*len(original_dataset))
    perm = np.random.permutation(np.arange(len(original_dataset)))
    subset_indices = list(perm[:subset_size])
    subset_dataset = Subset(original_dataset, subset_indices)
    subset_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    if float(subset_ratio) != 1.0:
        complement_indices = list(perm[subset_size:])
        complement_subset_dataset = Subset(original_dataset, complement_indices)
        complement_loader = torch.utils.data.DataLoader(complement_subset_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return subset_loader, complement_loader


def count_linear_layers(model: nn.Module) -> Tuple[int, List[int]]:
    
    '''
    Input: A Neural Network Model
    Output: The number of Linear Layers in the Model
    '''
    
    linear_layer_count = 0
    linear_layers = []
    
    # Iterate through all modules (layers) of the model
    for layer in model.modules():
        if isinstance(layer, nn.Linear):  # Check if the layer is a Linear layer
            linear_layer_count += 1
            linear_layers.append(layer.in_features)
            last_layer = layer
    linear_layers.append(last_layer.out_features)
    
    return linear_layer_count, linear_layers

def normalize_data_labels(X, y, feature_normalised=True, label_normalised=True):
    ''' Normalizes the features and labels using StandardScaler.
        If feature_normalised or label_normalised is False, it will not normalize the respective data and will set mean to 0 and scale to 1.
        
        Returns:
            - Normalized X and y
            - Scaler objects for features and labels (useful for inverse transformation)
            - Output: X, y, scaler_feats, scaler_labels
    '''
    
    scaler_feats = StandardScaler()
    if feature_normalised:
        
        scaler_feats = scaler_feats.fit(X)
    else:
        scaler_feats.mean_ = 0
        scaler_feats.scale_ = 1
        
    scaler_labels = StandardScaler()
    if label_normalised:
        scaler_labels = scaler_labels.fit(y.reshape(-1, 1))
    else:
        scaler_labels.mean_ = 0
        scaler_labels.scale_ = 1
        
    X = scaler_feats.transform(X)
    y = scaler_labels.transform(y.reshape(-1, 1)).flatten()
    
    return X, y, scaler_feats, scaler_labels


class ClassificationResultAnalyzer:
    """
    A class to analyze and visualize classification performance metrics.
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    
    y_pred : array-like of shape (n_samples,) or (n_samples, n_classes)
        Predicted class labels or probabilities for ROC/PR curves.
    
    target_names : list of str or list of int
        Names or indices of target classes.
    """
    
    def __init__(self, y_true, y_pred, cmap = 'coolwarm', target_names=None):
        
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.cmap = cmap
        self.target_names = target_names if target_names else np.unique(self.y_true).tolist()

    def get_confusion_matrix(self, savepath=None):
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        df_cm = pd.DataFrame(cm, index=self.target_names, columns=self.target_names)
        if savepath:
            df_cm.to_csv(savepath)
        return df_cm

    def plot_confusion_matrix(self, savepath=None, name=''):
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.target_names)
        disp.plot(cmap=self.cmap)
        plt.title(f"Confusion Matrix ({name})")
        if savepath:
            plt.savefig(savepath, dpi=100)
            plt.close()
        else:
            plt.show()

    def plot_confusion_matrix_percentage(self, savepath=None, name=''):
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap=self.cmap,
                    xticklabels=self.target_names, yticklabels=self.target_names)
        plt.title(f"Confusion Matrix with Percentages {name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        if savepath:
            plt.savefig(savepath, dpi=100)
            plt.close()
        else:
            plt.show()
        return cm_percent

    def get_accuracy_score(self, savepath=None):
        
        acc = accuracy_score(self.y_true, self.y_pred)
        if savepath:
            with open(savepath, 'w') as f:
                f.write(f"Accuracy: {acc:.4f}")
        return acc

    def get_classification_report(self, savepath=None):
        
        report = classification_report(self.y_true, self.y_pred, target_names=self.target_names, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        if savepath:
            df_report.to_csv(savepath, index=False)
        return df_report

    def plot_classification_report(self, savepath=None, name=''):
        
        report = classification_report(self.y_true, self.y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose().iloc[:-1, :-1]
        plt.figure(figsize=(8, 6))
        sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Classification Report {name}")
        plt.ylabel("Class")
        plt.xlabel("Metrics")
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=100)
            plt.close()
        else:
            plt.show()

    def plot_roc_curve_multi_class(self, y_pred_proba, savepath=None, name=''):
        """
        Plot ROC curves for multi-class classification using predicted probabilities.
        """
        y_true_bin = label_binarize(self.y_true, classes=list(range(len(self.target_names))))
        fpr, tpr, roc_auc = {}, {}, {}
        n_classes = len(self.target_names)

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2,
                     label=f'{self.target_names[i]} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Multi-class {name}')
        plt.legend(loc="lower right")

        if savepath:
            plt.savefig(savepath, dpi=100)
            plt.close()
        else:
            plt.show()

        return fpr, tpr, roc_auc

    def plot_precision_recall_curve_multi_class(self, y_pred_proba, savepath=None, name=''):
        """
        Plot Precision-Recall curves for multi-class classification.
        """
        y_true_bin = label_binarize(self.y_true, classes=list(range(len(self.target_names))))
        precision, recall, pr_auc = {}, {}, {}
        n_classes = len(self.target_names)

        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
            pr_auc[i] = auc(recall[i], precision[i])

        plt.figure()
        for i in range(n_classes):
            plt.plot(recall[i], precision[i], lw=2,
                     label=f'{self.target_names[i]} (AUC = {pr_auc[i]:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - Multi-class {name}')
        plt.legend(loc="lower right")

        if savepath:
            plt.savefig(savepath, dpi=100)
            plt.close()
        else:
            plt.show()

        return precision, recall, pr_auc
    
    def plot_class_distribution(self, savepath=None, name = ""):
        """
        Plot the distribution of classes in the dataset.
        """
        df = pd.DataFrame({'True': self.y_true, 'Predicted': self.y_pred})
        class_order = [0, 1, 2]
        class_labels = ['Low', 'Medium', 'High']

        counts = pd.concat([
            df['True'].value_counts().reindex(class_order, fill_value=0).rename('Actual'),
            df['Predicted'].value_counts().reindex(class_order, fill_value=0).rename('Predicted')
        ], axis=1).fillna(0).astype(int)

        ax = counts.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'salmon'])
        plt.title(f"Class Distribution: Actual vs Predicted {name}")
        plt.ylabel("Count")
        plt.xlabel("Class")
        plt.xticks(ticks=range(len(class_labels)), labels=class_labels, rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add count labels inside the bars (centered)
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height}',
                                xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                                ha='center', va='center', fontsize=10, color='black')

        if savepath:
            plt.tight_layout()
            plt.savefig(savepath, dpi=100)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
        
        # df = pd.DataFrame({'True': self.y_true, 'Predicted': self.y_pred})
        # class_order = [0, 1, 2]
        # class_labels = ['Low', 'Medium', 'High']


        # counts = pd.concat([
        #     df['True'].value_counts().reindex(class_order, fill_value=0).rename('Actual'),
        #     df['Predicted'].value_counts().reindex(class_order, fill_value=0).rename('Predicted')
        # ], axis=1).fillna(0).astype(int)
        # print(counts)
        # counts.plot(kind='bar')
        # plt.title(f"Class Distribution: Actual vs Predicted {name}")
        # plt.ylabel("Count")
        # plt.xlabel("Class")
        # plt.xticks(ticks=range(len(class_labels)), labels=class_labels, rotation=0)
        # plt.grid(axis='y')
        # if savepath:
        #     plt.savefig(savepath, dpi=100)
        #     plt.close()
        # else:
        #     plt.show()


class TrendAnalyser:
    
    try:
        from scipy.signal import find_peaks
        from scipy.stats import kendalltau, spearmanr, pearsonr
        _HAS_SCIPY = True
    except Exception:
        _HAS_SCIPY = False
    
    def __init__(self, x, y, threshold=0.75, alpha=0.05):
        self.x, self.y = self._safe_sort_xy(x, y)
        self.threshold = threshold
        self.alpha = alpha

    # ---------- helpers ----------
    def _safe_sort_xy(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape.")
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        keep = np.r_[True, np.diff(x) != 0]
        return x[keep], y[keep]

    def _monotonicity(self):
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        deriv = dy / dx
        total = len(deriv)
        if total == 0:
            return 0.0, 0.0, "insufficient"
        inc_r = np.sum(deriv > 0) / total
        dec_r = np.sum(deriv < 0) / total
        if inc_r >= self.threshold:
            decision = f"increasing (~{inc_r:.1%})"
        elif dec_r >= self.threshold:
            decision = f"decreasing (~{dec_r:.1%})"
        else:
            decision = f"None (↑{inc_r:.1%}, ↓{dec_r:.1%})"
        return inc_r, dec_r, decision

    def _global(self):
        if len(self.y) < 2:
            return 0.0, "insufficient"
        delta = self.y[-1] - self.y[0]
        if delta > 0:
            return delta, "up"
        elif delta < 0:
            return delta, "down"
        return delta, "flat"

    def _regression(self):
        if len(self.x) < 2:
            return np.nan, "insufficient"
        slope, intercept = np.polyfit(self.x, self.y, 1)
        if slope > 0:
            return slope, "up"
        elif slope < 0:
            return slope, "down"
        return slope, "flat"

    def _peaks_troughs(self):
        y = np.asarray(self.y, dtype=float)
        if len(y) < 3:
            return np.array([]), np.array([])
        if TrendAnalyser._HAS_SCIPY:
            peaks, _ = TrendAnalyser.find_peaks(y)
            troughs, _ = TrendAnalyser.find_peaks(-y)
        else:
            dy1 = y[1:-1] - y[:-2]
            dy2 = y[2:] - y[1:-1]
            mids = np.arange(1, len(y) - 1)
            peaks = mids[(dy1 > 0) & (dy2 < 0)]
            troughs = mids[(dy1 < 0) & (dy2 > 0)]
        return peaks, troughs

    def _seq_trend(self, values):
        if len(values) < 2:
            return "None", 0.0, 0.0
        diffs = np.diff(values)
        total = len(diffs)
        up = np.sum(diffs > 0) / total
        down = np.sum(diffs < 0) / total
        no_change=np.sum(diffs == 0) / total
        if down >= self.threshold:
            return "down", up, down, total, no_change
        elif up >= self.threshold:
            return "up", up, down, total, no_change
        return "None", up, down, total, no_change

    # ---------- public API ----------
    def analyse(self):
        inc_r, dec_r, mono_decision = self._monotonicity()
        delta, global_decision = self._global()
        slope, reg_decision = self._regression()
        p_idx, t_idx = self._peaks_troughs()
        p_decision, p_up, p_down, p_pairs, p_no_change = self._seq_trend(self.y[p_idx])
        t_decision, t_up, t_down, t_pairs, t_no_change = self._seq_trend(self.y[t_idx])

        peaks_text = {"up": "Higher Highs", "down": "Lower Highs", "None": "Mixed Highs"}[p_decision]
        troughs_text = {"up": "Higher Lows", "down": "Lower Lows", "None": "Mixed Lows"}[t_decision]
        
        # Pearson
        pearson_corr, pearson_p = TrendAnalyser.pearsonr(self.x, self.y)
        # Spearman
        spearman_corr, spearman_p = TrendAnalyser.spearmanr(self.x, self.y)
        # Kendall
        kendall_corr, kendall_p = TrendAnalyser.kendalltau(self.x, self.y)

        summary = (
            f"{'-'*80}\n"
            f"Local monotonicity: {mono_decision}; \n"
            f"Global: {global_decision} (Δ={delta:.3g}); \n"
            f"Regression: {reg_decision} (slope={slope:.3g}); \n"
            f"Pearson Correlation: {f"r={pearson_corr:.3f}, p={pearson_p:.4f}, significance={pearson_p < self.alpha}"}\n"
            f"Spearman Correlation: {f"r={spearman_corr:.3f}, p={spearman_p:.4f}, significance={spearman_p < self.alpha}"}\n"
            f"Kendall Correlation: {f"r={kendall_corr:.3f}, p={kendall_p:.4f}, significance={kendall_p < self.alpha}"}\n"
            f"Peaks: {peaks_text}\n"
                + (f" (pairs={p_pairs}, ↑{p_up:.2%}, ↓{p_down:.2%}, ={p_no_change:2%})" if p_pairs else " (insufficient peaks)")
                + "; "
            f"\nTroughs: {troughs_text}\n"
                + (f" (pairs={t_pairs}, ↑{t_up:.2%}, ↓{t_down:.2%}, ={t_no_change:2%})" if t_pairs else " (insufficient troughs)")
                + "."
            f"\n{'-'*80}\n"
            )
        print(summary)

        return {
            "monotonic": {
                "increasing_ratio": inc_r,
                "decreasing_ratio": dec_r,
                "decision": mono_decision,
            },
            "global": {"decision": global_decision, "delta": delta},
            "regression": {"slope": slope, "decision": reg_decision},
            "peaks": {
                "indices": p_idx,
                "values": self.y[p_idx],
                "decision": p_decision,
                "up_ratio": p_up,
                "down_ratio": p_down,
            },
            "troughs": {
                "indices": t_idx,
                "values": self.y[t_idx],
                "decision": t_decision,
                "up_ratio": t_up,
                "down_ratio": t_down,
            },
            "pearson_corr":{
                'r':pearson_corr,
                'p':pearson_p,
                'significance':pearson_p < self.alpha
            },
            "spearman_corr":{
                'r':spearman_corr,
                'p':spearman_p,
                'significance':spearman_p < self.alpha
            },
            "kendall_corr":{
                'r':kendall_corr,
                'p':kendall_p,
                'significance':kendall_p < self.alpha
            }
        }

class RandomForestSHAPAnalyzer:
    def __init__(self, model, X, y, smiles: Optional[List[str]] = None, scaler=None):
        """
        Initialize the analyzer.

        Parameters
        ----------
        model : RandomForestClassifier or Regressor
            The trained Random Forest model.
        X : pd.DataFrame
            Test/validation dataset.
        y : pd.Series or np.ndarray
            True labels/targets.
        smiles : list[str], optional
            List of SMILES strings for molecules (if chemical data).
        scaler : fitted scaler, optional
            If features were scaled, provide the scaler (i.e. mean and std) to recover original values.
        """
        self.model = model
        self.X = X
        self.y = np.array(y)
        self.smiles = smiles
        self.scaler = scaler

        # Build explainer
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(X)
        
        # Handle classification (multi-class) vs regression
        if isinstance(self.shap_values, list):
            # For binary classification, shap_values[1] is usually most relevant
            self.shap_values = self.shap_values[1]

    # ------------------ Save & Load ------------------ #
    def save(self, path_prefix: str = "./rf_shap"):
        with open(f"{path_prefix}_explainer.pkl", "wb") as f:
            pickle.dump(self.explainer, f)
        with open(f"{path_prefix}_values.pkl", "wb") as f:
            pickle.dump(self.shap_values, f)

    def load(self, path_prefix: str = "./rf_shap"):
        with open(f"{path_prefix}_explainer.pkl", "rb") as f:
            self.explainer = pickle.load(f)
        with open(f"{path_prefix}_values.pkl", "rb") as f:
            self.shap_values = pickle.load(f)

    # ------------------ Global Analysis ------------------ #
    def summary_plot(self, features: Optional[List[str]] = None, top_n: int = 20):
        """Global SHAP summary plot."""
        if features:
            X_filtered = self.X[features]
            indices = [self.X.columns.tolist().index(f) for f in features]
            shap_values_filtered = self.shap_values[:, indices]
            shap.summary_plot(shap_values_filtered, X_filtered, feature_names=features)
        else:
            shap.summary_plot(self.shap_values, self.X, max_display=top_n)

    def feature_importance(self):
        """Return feature importance DataFrame sorted by abs SHAP."""
        importance = np.mean(self.shap_values, axis=0)
        importance_abs = np.mean(np.abs(self.shap_values), axis=0)
        df = pd.DataFrame({
            "Feature": self.X.columns,
            "SHAP Importance": importance,
            "SHAP Importance abs": importance_abs
        }).sort_values("SHAP Importance abs", ascending=False)
        return df

    # ------------------ Local Analysis ------------------ #
    def force_plot_instance(self, idx: int, features: Optional[List[str]] = None, top_n: int = 10):
        """Force plot for a single instance."""
        if self.scaler:
            X_reversed = (self.scaler.mean_ + self.X.to_numpy() * np.sqrt(self.scaler.var_))
        else:
            X_reversed = self.X.to_numpy()

        if features:
            indices = [self.X.columns.tolist().index(f) for f in features]
        else:
            # Take top_n by absolute SHAP for this sample
            indices = np.argsort(-np.abs(self.shap_values[idx]))[:top_n]

        shap.force_plot(
            base_value=float(self.explainer.expected_value),
            shap_values=self.shap_values[idx, indices],
            features=X_reversed[idx, indices],
            feature_names=np.array(self.X.columns)[indices],
            matplotlib=True,
            show=True,
            figsize=(15, 3),
            link="identity",
            text_rotation=90
        )

        # Print contextual info
        if self.smiles:
            print(f"SMILES: {self.smiles[idx]}")
        print(f"True Value: {self.y[idx]}")
        print(f"Predicted Value: {self.model.predict(self.X.iloc[idx:idx+1])[0]}")

    # ------------------ Common Feature Analysis ------------------ #
    def common_features(self, num_features: int = 200, ratio: float = 0.75, positive: bool = True):
        """
        Identify features consistently appearing in top/bottom SHAP rankings.

        Parameters
        ----------
        num_features : int
            Number of top/bottom features to consider per sample.
        ratio : float
            Minimum fraction of samples in which a feature must appear.
        positive : bool
            If True, consider top features (positive influence).
            If False, consider bottom features (negative influence).
        """
        sorted_idx = np.argsort(self.shap_values, axis=1)[:, ::-1]
        if not positive:
            top_features = np.array(self.X.columns)[sorted_idx[:, -num_features:]]
        else:
            top_features = np.array(self.X.columns)[sorted_idx[:, :num_features]]

        intersection = []
        for col in self.X.columns:
            count = sum(col in row for row in top_features)
            if count >= ratio * len(top_features):
                intersection.append(col)

        return intersection
    
    def fix_matplotlib_params(self):

        # =============================
        # Global Matplotlib Style Setup
        # =============================

        plt.rcParams.update({

            # ---- Figure ----
            'figure.figsize': (8, 6),           # default figure size
            'figure.dpi': 100,                  # resolution
            'figure.facecolor': 'white',        # background color

            # ---- Axes ----
            'axes.labelsize': 14,               # font size of x/y labels
            'axes.labelweight': 'bold',         # weight of axis labels
            'axes.titlesize': 16,               # title font size
            'axes.titleweight': 'bold',         # title weight
            'axes.edgecolor': 'black',          # border color
            'axes.linewidth': 1.2,              # border line width

            # ---- Ticks ----
            'xtick.labelsize': 12,              # x tick label size
            'ytick.labelsize': 12,              # y tick label size
            'xtick.direction': 'in',            # in, out, inout
            'ytick.direction': 'in',
            'xtick.major.size': 6,              # major tick length
            'ytick.major.size': 6,
            'xtick.minor.size': 3,              # minor tick length
            'ytick.minor.size': 3,

            # ---- Lines & Markers ----
            'lines.linewidth': 2,               # default line width
            'lines.markersize': 6,              # default marker size
            'lines.marker': None,               # default marker style

            # ---- Legend ----
            'legend.fontsize': 12,
            'legend.loc': 'best',
            'legend.frameon': True,
            'legend.edgecolor': 'black',

            # ---- Font ----
            'font.family': 'serif',             # font family (e.g., serif, sans-serif)
            'font.style': 'italic',
            'font.size': 12,                    # base font size
            'font.weight': 'normal',

            # ---- Grid ----
            'grid.color': 'gray',
            'grid.linestyle': '--',
            'grid.linewidth': 0.5,
        })
