#!/usr/bin/env python3

import os
import ast
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool, cpu_count, get_context

import torch
from rdkit import Chem
import multiprocessing as mp

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils_local.data_utils_smiles import *
from data_utils_local.Generalised_data_utils import create_folder, savedict2json, load_json
from data_utils_local.graph_utils_smiles import *

torch.manual_seed(30)
np.random.seed(30)
# mp.set_start_method("forkserver", force=True)

__all__ = ['process_smiles_filter',
           'parallel_filter_smiles'
           ]

# === Parallel SMILES processing ===
def process_smiles_filter(smi,
                          all_atoms,
                          max_num_atoms,
                          hybridization_list,
                          chiraltypes,
                          bonds_list,
                          *args, **kwargs
                          ):
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return None
        num_atoms, atom_syms, hybridizations, chiral_tags, bond_types, _ = extract_features_from_smiles(smi)
        is_less_num_atoms = True if num_atoms<=max_num_atoms else False
        isatom_subset = True if set(atom_syms).issubset(set(all_atoms)) else False
        ishyb = True if set(hybridizations).issubset(set(hybridization_list)) else False
        ischi = True if set(chiral_tags).issubset(set(chiraltypes)) else False
        isbond = True if set(bond_types).issubset(set(bonds_list)) else False
        
        if all([isatom_subset, is_less_num_atoms, ishyb, ischi, isbond]):
            return smi
        return None
    except Exception as e:
        # Optional: log exception if needed
        print(f'Exception triggered: {e}')
        return None

def parallel_filter_smiles(smiles_list,
                           all_atoms,
                           max_num_atoms,
                           hybridization_list,
                           chiraltypes,
                           bonds_list,
                           num_workers = cpu_count()-2,
                           dataname='',
                           *args, **kwargs
                           ):
    
    func = partial(process_smiles_filter,
                   all_atoms=all_atoms,
                   max_num_atoms=max_num_atoms,
                   hybridization_list=hybridization_list,
                   chiraltypes=chiraltypes,
                   bonds_list=bonds_list)
    with Pool(num_workers) as p:
        filtered = list(tqdm(p.imap(func,
                                    smiles_list,
                                    chunksize=100
                                    ),
                             total=len(smiles_list),
                             desc=f"Filtering SMILES - {dataname}"
                             )
                        )

    return list(filter(None, filtered))


path_init = '/home/rkmvu/Dataset/selfies/zinc/'
all_data_filenames = OrderedDict({
                        'train':'properties_dtratio_0.25_train_0_5_canon_smiles_selfies_with_descriptors_train_0_5_val_0_2_test_0_3.csv',
                        'test':'properties_dtratio_0.25_test_0_3_canon_smiles_selfies_with_descriptors_train_0_5_val_0_2_test_0_3.csv',
                        'val':'properties_dtratio_0.25_val_0_2_canon_smiles_selfies_with_descriptors_train_0_5_val_0_2_test_0_3.csv'
                        })
graph_dataset_path = '/home/rkmvu/Dataset/CL_int/Graph_data'
# molecular_wt_filter = 900
# all_atoms = ['Br', 'Cl', 'P', 'I', 'F', 'H', 'S', 'N', 'O', 'C']
smile_column_name = 'SMILES'
target_column_name = [
                    'MolLogP', 'FilterItLogS', 'EState_VSA1', 'ATSC1pe', 'AATSC0c', 'ATSC1c', 'fr_COO', 'MATS1se', 'nAcid', 'NsOH',
                    'ATSC1se', 'NHOHCount', 'AATSC1c', 'nBase', 'fr_COO', 'EState_VSA5', 'AUTOCORR2D_156','Kappa1', 'Kappa2', 'Kappa3',
                    'HallKierAlpha', 'Chi0', 'Chi1', 'PEOE_VSA1', 'PEOE_VSA2', 'BalabanJ', "MolMR", "fr_aryl_methyl",
                    ]
num_workers = 8

#Creating Required Folders
create_folder(graph_dataset_path)

for dataset_name, data_filename in all_data_filenames.items():
    
    all_data = pd.read_csv(f'{path_init}/{data_filename}')
    smiles_list = all_data[smile_column_name].tolist()
    y_true = all_data[target_column_name].values
    all_dict = dict(zip(smiles_list, y_true))
    
    path = f'{graph_dataset_path}/Dataset_atoms_maxatoms_hybds_chityp_bndlst.json'
    if os.path.isfile(path):
        print(f'Loading all_atoms, max_num_atoms, hybridization_list, chiraltypes, bonds_list and neighbors_max_len from: \n                 {path}')
        kwargs = load_json(path)
    else:
        ## Calculating all the Hybridization, Atom Symbols and Chiral Types present in a molecule in a SMILES list
        print(f'No pre-defined details found in: {path}')
        print(f'Creating all_atoms, max_num_atoms, hybridization_list, chiraltypes, bonds_list and neighbors_max_len to: \n                 {path}')
        kwargs = list_smiles2symbols_hybridization_chiraltype(smiles_list)
        savedict2json(kwargs, path)
        
    max_num_atoms = kwargs['max_num_atoms']
    atom_symbols = kwargs['atom_symbols']
    hybridization_list = kwargs['hybridization_list']
    chiraltypes = kwargs['chiraltypes']
    bonds_list = kwargs['bonds_list']
    neighbors_max_len = kwargs['neighbors_max_len']
    original_list = smiles_list
    
    smiles_list = parallel_filter_smiles(smiles_list=smiles_list,
                                         all_atoms=atom_symbols,
                                         max_num_atoms=max_num_atoms,
                                         hybridization_list=hybridization_list,
                                         chiraltypes=chiraltypes,
                                         bonds_list=bonds_list,
                                         dataname = dataset_name
                                         )
    rejected = set(original_list) - set(smiles_list)
    with open(f"{graph_dataset_path}/rejected_smiles_{dataset_name}.txt", "w") as f:
        f.write("\n".join(rejected))
    all_dict = {k:all_dict[k] for k in smiles_list}
    
    print('-'*80)
    print(f'## {dataset_name.upper()} Dataset Description ##')
    print('-'*80)
    print(f'All Present Atoms in the Dataset = {atom_symbols}')
    print(f'Maximum Number of Atoms for the Dataset= {max_num_atoms}')
    print(f'All Present Hybridizations in the Dataset = {hybridization_list}')
    print(f'All Present Chiral Types in the Dataset = {chiraltypes}')
    print(f'All Present Bonds in the Dataset = {bonds_list}')
    print(f'Maximum number of neighbours for an atoms = {neighbors_max_len}')
    print(f'Number of datapoints: {len(all_dict)}')
    print(f'Number of rejected smiles: {len(rejected)}')
    print('-'*80)
    all_dataset = Multirelational_GraphDataset(smi_list=list(all_dict.keys()),
                                               labels=list(all_dict.values()),
                                               **kwargs
                                               )
    print(f'Saving Garph Data in Folder: {graph_dataset_path}')
    SmilesDataset_graph_gen(split=dataset_name,
                            dataset=all_dataset,
                            out_path=f'{graph_dataset_path}/SmilesDataset_graph',
                            mean_std_flag=(dataset_name=='train'),
                            num_workers=num_workers,
                            chunksize=100,#max(1, len(all_dataset) // (num_workers * 4))
                            )
    print('-'*80)
    print('='*80)
print('+'*80, '\n')
    
