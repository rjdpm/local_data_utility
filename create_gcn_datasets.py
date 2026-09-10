#!/usr/bin/env python3
import pandas as pd
import numpy as np
from dataclasses import asdict
import os
import warnings
warnings.filterwarnings("ignore")
from dataclasses import asdict

import pandas as pd
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_module import Tee
from Generalised_data_utils import load_json, create_folder
from graph_utils_smiles_V2 import *
from load_dataset import GCNDataset, SMILES_PRUNING, GCNDataset_Config

########################################################################################################
config = GCNDataset_Config(
    # Data reading
    path_init = '/home/rkmvu/Codes/pKa_Fu_Rb/pKa/GCN',
    file_name = 'test_df_2.csv',

    # Column names
    smiles_column_name = 'Canonicalised_SMILES',
    target_column_name = 'lgFu',
    data_split_column_name = 'Data_Split',

    # Output
    graph_dataset_path = './graph_data_for_test',
    details_filepath = 'mol_details',

    # Dataset preparation
    graph_backend_version = 'new',
    train_partition = 'Tr',
    partition_labels = None, #['Tr', 'Te', 'Val'],

    all_atoms = ['Br', 'Cl', 'P', 'I', 'F', 'H', 'S', 'N', 'O', 'C', 'B', 'Si', 'Na', 'K'],
    filters = None,
    label_normalised = True,
    force = True,

    # GCN features
    gcn_features_path = "/home/rkmvu/Codes/data_utils_local/graph_data_info/atom_props_keys_numeric.json",
    )
########################################################################################################
# Save EVERYTHING that prints
orig_stdout = sys.stdout
orig_stderr = sys.stderr
log_f = open(config.log_filepath, "w")
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)
print(f'Source: {os.path.abspath(__file__)}')
########################################################################################################
config.features_list = load_json(config.gcn_features_path)
########################################################################################################
## Load Data
datapath = f'{config.path_init}/{config.file_name}'
print(f'Loading CLint Data from: {datapath}')
all_data = pd.read_csv(datapath)

print('='*80)
print(f'Data Split Counts for Fold:')
print(all_data[config.data_split_column_name].value_counts())
print('-'*80)
########################################################################################################
def recursive_print(d, indent=0):
    for key, value in d.items():
        prefix = "    " * indent
        
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            recursive_print(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")

def sanitize_smiles(smiles):

    kwargs = {'sanitize':True,
            'canonical': True,
            'normalize_functional_groups': True,
            'sanitize_final': True,
            'remove_explicit_h': True,

            'keep_largest_fragment': True,
            'require_organic_fragment': True,
            'disconnect_metals': True,
            'strip_salts': True,
            }

    sanitize_obj = SMILES_PRUNING(smiles_list=[smiles], **kwargs)
    cleaned_smiles = sanitize_obj.sanitize_smiles_list([smiles])[0]

    return cleaned_smiles

if __name__ == "__main__":
    
    all_data[config.smiles_column_name] = all_data[config.smiles_column_name].apply(sanitize_smiles)

    gcn_dataset = GCNDataset(df_init=all_data,
                    smiles_column_name=config.smiles_column_name,
                    target_column_name=config.target_column_name,
                    partition_col=config.data_split_column_name,
                    graph_dataset_path=config.graph_dataset_path,
                    graph_backend_version =config.graph_backend_version
                    )
    datasets = gcn_dataset.prepare_datasets(train_partition = config.train_partition,
                                        partition_labels = config.partition_labels,
                                        all_atoms = config.all_atoms,
                                        filters = config.filters,
                                        symb_hyb_chirl_file = config.details_filepath,
                                        features_list = config.features_list,
                                        label_normalised = config.label_normalised,
                                        force = config.force
                                        )
    
    print('='*80)
    print('Configuration and Dataset Preparation Completed Successfully!')
    print('-'*80)
    print('Dataset Summary:')
    print('-'*80)
    train_dataset = datasets['train']["dataset"]
    test_dataset = datasets['test']["dataset"]
    val_dataset = datasets['val']["dataset"]
    scaler_train_label = gcn_dataset.StandardScaler_labels

    smiles_list_train = datasets['train']["smiles"]
    smiles_list_test = datasets['test']["smiles"]
    smiles_list_val = datasets['val']["smiles"]

    print(f'Train Dataset: {len(train_dataset)} samples')
    print(f'Test Dataset: {len(test_dataset)} samples')
    print(f'Val Dataset: {len(val_dataset)} samples')
    print('-'*80)

    print('Dataset paths:')
    print('-'*80)
    for k, v in gcn_dataset.datapaths.items():
        print(f'{k.capitalize()}: {v}')

    print('-'*80)
    print(f'Example SMILES from Train Dataset: {smiles_list_train[:3]}')
    print(f'Example SMILES from Test Dataset: {smiles_list_test[:3]}')
    print(f'Example SMILES from Val Dataset: {smiles_list_val[:3]}')          
    
    print('='*80)
    print('Config files used:')
    print('-'*80)
    config_dict = asdict(config)
    recursive_print(config_dict)
    print('-'*80)
    print('='*80)