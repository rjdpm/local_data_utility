#!/usr/bin/env python3
import pandas as pd
import time

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from data_utils_local.data_utils_smiles import mol_to_features_batch
from data_utils_local.Generalised_data_utils import attach_ids, read_data_from_excel, create_folder
from data_utils_local.load_dataset import SMILES_PRUNING

# from rdkit import Chem
# from rdkit.Chem.MolStandardize import rdMolStandardize
# from typing import Optional, Dict, Any, Tuple
# from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
import logging
logging.getLogger('mordred').setLevel(logging.CRITICAL)


def calculate_descriptors(df, smiles_column_name):
        
    desc_df = mol_to_features_batch(df[smiles_column_name].tolist(), smiles_column_name=smiles_column_name)
    df = pd.merge(df, desc_df, how='inner', on=smiles_column_name)
    
    return df

## Use for single dataset
if __name__=='__main__':
    start = time.time()
    path_init = "/home/rkmvu/Dataset/Fraction_unbound/all_datasets"
    create_folder(path_init)
    source_datapath = f"{path_init}/all_data_merged_source.xlsx"
    data_savepath = f'{path_init}/comparison_datasets/Fu_cleaned_data_091125.csv'
    smiles_column_name='Canonicalised_SMILES'
    target_col_name = 'Fu_mean'
    id_col = 'Unique_ID'
    prefix='FU091125'
    pruning = True#False
    
    df = read_data_from_excel(source_datapath, 8)
    df = attach_ids(df, prefix=prefix, id_col=id_col)
    ## Uncemment if ID is needed to add in the dataframe
    # df = attach_ids(df, prefix=prefix, id_col=id_col)
    kwargs = {'sanitize':True,
            'canonical': True,
            'normalize_functional_groups': True,
            'sanitize_final': True,
            'remove_explicit_h': True,

            'keep_largest_fragment': True,
            'require_organic_fragment': True,
            'disconnect_metals': True,
            'strip_salts': True,
            # 'prefix': '',
            }
    
    if pruning:
        smiles_pruning = SMILES_PRUNING(smiles_list=df[smiles_column_name].tolist(),
                                    labels=df[id_col],
                                    # filters=[('MolWt', '<', 1000)],
                                    smiles_column_name=smiles_column_name,
                                    **kwargs
                                    )
        df_temp = smiles_pruning.get_cleaned_smiles()[['labels', smiles_column_name]]
        df_temp = df_temp.rename(columns={'labels':id_col})
        
        df = pd.merge(df_temp, df.drop(columns=[smiles_column_name]), how='inner', on=id_col) 
    df = calculate_descriptors(df, smiles_column_name)

    # Append as a new sheet
    df.to_csv(data_savepath, index=False)
    
    avg_time =  (time.time()-start)/len(df)
    print(f'Time -> Total: {time.time()-start} sec | average: {avg_time} sec | Items: {len(df)}')
    
# ## Use for multiple dataset
# if __name__=='__main__':
#     start = time.time()
    
#     ## Inputs ##
#     #---------#
#     smiles_column_name='Canonicalised_SMILES'
#     target_col_name = 'Fu_mean'
#     id_col = 'Unique_ID'
#     pruning = True#False
#     datapath = "/home/rkmvu/Dataset/Fraction_unbound/all_datasets/all_data_merged_source_updated_04012026.xlsx"
#     df = pd.read_excel(datapath, sheet_name=None)
#     all_keys = df.keys()
#     key = 'Xulian_2025'
#     prefix_list = {'Watanabe_2018':'W18',
#                    'Ingle_2016':'I16',
#                    'Lombardo_2018':'L18',
#                    'Mulpuru_2021':'M21',
#                    'Drug3D_database':'D3D',
#                    'Kunal_2024':'K24',
#                    'Krumpholz_2024':'Kz24',
#                    'Xulian_2025':'X25',
#                    'Hiroaki_2022':'H22'}
#     prefix=f'{prefix_list[key]}040126'
    
#     ## Inputs for the computation are the dataset path and the dataframe
#     data_savepath = f"/home/rkmvu/Dataset/Fraction_unbound/all_datasets/comparison_datasets/{key}.csv"
#     df = df[key]
    
#     ## Calculating descriptors ##
#     df = attach_ids(df, prefix=prefix, id_col=id_col)
#     if pruning:
#         smiles_pruning = SMILES_PRUNING(smiles_list=df[smiles_column_name].tolist(),
#                                     labels=df[id_col],
#                                     # filters=[('MolWt', '<', 1000)],
#                                     smiles_column_name=smiles_column_name,
#                                     prefix=''
#                                     )
#         df_temp = smiles_pruning.get_cleaned_smiles()[['labels', smiles_column_name]]
#         df_temp = df_temp.rename(columns={'labels':id_col})
        
#         df = pd.merge(df_temp, df.drop(columns=[smiles_column_name]), how='inner', on=id_col) 
#     df = calculate_descriptors(df, smiles_column_name)

#     # Append as a new sheet
#     df.to_csv(data_savepath, index=False)
    
#     avg_time =  (time.time()-start)/len(df)
#     print(f'Time -> Total: {time.time()-start} sec | average: {avg_time} sec | Items: {len(df)}')