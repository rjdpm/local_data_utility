#!/usr/bin/env python3
import pandas as pd
import numpy as np
from dataclasses import asdict
import os
import warnings
warnings.filterwarnings("ignore")
from dataclasses import asdict

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from data_utils_local.train_module import Trainer_functions, Tee
from data_utils_local.Generalised_data_utils import load_json, create_folder
from data_utils_local.graph_utils_smiles_V2 import *
from data_utils_local.load_dataset import GCNDataset
from config.gcn_config import load_gcn_config

########################################################################################################
## Load Configurations
fold = 10
config = load_gcn_config()
########################################################################################################
## Data Reading & Preprocessing
path_init           = config.base.paths.path_init
file_path           = config.base.paths.file_path
metadata_path       = config.base.paths.metadata_path
results_savepath    = config.base.paths.results_path
models_savepath     = config.base.paths.models_savepath

graph_dataset_path  = config.gcn.paths.graph_dataset_path
graph_dataset_path  = graph_dataset_path.format(FOLD=fold)
details_filepath    = config.gcn.paths.details_filepath
log_filepath        = f'{graph_dataset_path}/data_preparation_logfile.log'
create_folder(os.path.dirname(log_filepath))
########################################################################################################
# Save EVERYTHING that prints
orig_stdout = sys.stdout
orig_stderr = sys.stderr
log_f = open(log_filepath, "w")
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)
print(f'Source: {os.path.abspath(__file__)}')
print(f'Creating Fold-specific Datasets for Fold: {fold}')
########################################################################################################
## Column Names
smiles_column_name                  = config.base.columns.smiles_column_name
cannonicalized_smiles_column_name   = config.base.columns.cannonicalized_smiles_column_name
target_list_column_name             = config.base.columns.target_list_column_name
target_list_std_column_name         = config.base.columns.target_list_std_column_name
target_column_name                  = config.base.columns.target_column_name
data_split_column_name              = config.base.columns.data_split_column_name
########################################################################################################
node_feature_path = config.gcn.paths.gcn_features_path
features_list = load_json(node_feature_path)
########################################################################################################
## Load Data
datapath = f'{path_init}/{file_path}'
print(f'Loading CLint Data from: {datapath}')
all_data = pd.read_csv(datapath)

## Create Fold-specific Data Splits
temp_data_split_col = np.array(['Tr']*len(all_data), dtype=object)
te_mask = all_data[data_split_column_name] == f'CV_{fold}'
temp_data_split_col[te_mask] = 'Te'
n_ = 1 if fold == 10 else 10
val_mask = all_data[data_split_column_name] == f'CV_{n_}'
temp_data_split_col[val_mask] = 'Val'
# non_test_indices = np.where(~te_mask)[0]
# temp_data_split_col[non_test_indices[:3]] = 'Val'
all_data[data_split_column_name] = temp_data_split_col

print('='*80)
print(f'Fold-specific Data Split Counts for Fold {fold}:')
print(all_data[data_split_column_name].value_counts())
print('-'*80)
########################################################################################################
#Random Forest Parameters
rf_params=asdict(config.models.rf)
########################################################################################################
## SVM Parameters
svm_params = asdict(config.models.svm)
########################################################################################################
## XGBoost model parameters
xgb_params = asdict(config.models.xgb)
########################################################################################################
## NN Model Parameters
nn_params = asdict(config.models.nn)
nn_lr = nn_params["lr"]
nn_num_epochs = nn_params["num_epochs"]
nn_dropout = nn_params["dropout"]
nn_act_func = nn_params["act_func"]
nn_norm_type = nn_params["norm_type"]
nn_weights_initialize = nn_params["weights_initialize"]
nn_criterion = Trainer_functions.loss_function_selector(nn_params["criterion"])
device=config.base.runtime.device

num_reg_nn_layers = 5
# layers = [512, 512, 512, 512, 1]
shape0 = 512
nn_reg_layers = [int(x) for x in np.linspace(1, shape0, num_reg_nn_layers, endpoint=True)][::-1]
########################################################################################################
save_regressors=config.base.runtime.save_regressors
########################################################################################################
def recursive_print(d, indent=0):
    for key, value in d.items():
        prefix = "    " * indent
        
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            recursive_print(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")


if __name__ == "__main__":
    gcn_dataset = GCNDataset(df_init=all_data,
                    smiles_column_name=smiles_column_name,
                    target_column_name=target_column_name,
                    partition_col=data_split_column_name,
                    graph_dataset_path=graph_dataset_path,
                    graph_backend_version = config.gcn.preprocess.graph_backend_version
                    )

    datasets = gcn_dataset.prepare_datasets(train_partition = config.gcn.preprocess.train_partition,
                                        partition_labels = config.gcn.preprocess.partition_labels,
                                        all_atoms = config.gcn.preprocess.all_atoms,
                                        filters = config.gcn.preprocess.filters,
                                        symb_hyb_chirl_file = details_filepath,
                                        features_list = features_list,
                                        label_normalised = config.gcn.preprocess.label_normalised,
                                        force = config.gcn.preprocess.force
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
    print(f'Example SMILES from Train Dataset: {smiles_list_train[:3]}')
    print(f'Example SMILES from Test Dataset: {smiles_list_test[:3]}')
    print(f'Example SMILES from Val Dataset: {smiles_list_val[:3]}')          
    
    print('='*80)
    print('Config files used:')
    print('-'*80)
    
    config_dict = asdict(config)
    recursive_print(config_dict)
    print('='*80)