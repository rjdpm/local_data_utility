import os
import ast
import sys, time
import copy, gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, List, Dict, Tuple, Union, Set, Callable, Optional

from rdkit.Chem import Descriptors, Crippen
from mordred import Calculator, descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, AdditionalOutput

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore", module="mordred")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from basic_ops_rm import canonicalize_smiles
from calc_descs_rm import mol_to_fetures

__all__ = [  
    'df_group_duplicates',
    'df2cleandf',
    'modify_df1_wrt_df2',
    ]

def df_group_duplicates(df: pd.DataFrame, 
                        reference_col: str, 
                        target_cols: List[str], 
                        keep_old_cols: bool = True,
                        drop_duplicates:bool = True
                        ) -> pd.DataFrame:
    
    """
    Group selected columns by a reference column and collect their values into lists.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    reference_col : str
        Column used for grouping.
    target_cols : list of str
        Columns whose grouped values are collected into lists.
    keep_old_cols : bool, default=True
        Whether to retain the original target columns.
    drop_duplicates : bool, default=True
        Whether to keep only the first row for each unique reference value.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the grouped list-valued columns.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "ID": ["A", "A", "B"],
    ...     "Value": [1, 2, 3],
    ...     "Source": ["X", "Y", "Z"]
    ... })

    >>> df_group_duplicates(
    ...     df,
    ...     reference_col="ID",
    ...     target_cols=["Value", "Source"]
    ... )

      ID  Value Source Value_list Source_list
    0  A      1      X     [1, 2]      [X, Y]
    1  B      3      Z        [3]         [Z]
    """

    if keep_old_cols:
        df_grouped = df.copy()
    else:
        df_grouped = df[[reference_col, *target_cols]].copy()
        # df_grouped = df[[reference_col]].copy()
    
    for col in target_cols:
        values_by_reference = df.groupby(reference_col)[col].apply(list)
        reference_to_values = values_by_reference.to_dict()
        new_col_name = f"{col.replace(' ', '_')}_list"
        df_grouped[new_col_name] = df[reference_col].map(reference_to_values)

    df_duplicate = df_grouped[df_grouped.duplicated(subset=reference_col)].copy()
    if drop_duplicates:
        df_grouped = df_grouped.drop_duplicates(subset=reference_col, keep='first')
        df_grouped = df_grouped.reset_index(drop=True)

    df_returns = {'grouped':df_grouped, 'duplicate':df_duplicate}

    return df_returns


def df2cleandf(df: pd.DataFrame, 
               smiles_col: str, 
               target_col: str, 
               columns_to_list: Optional[List[str]] = None,
               split_data: bool = False,
               train_idx=None,
               val_idx=None,
               test_idx=None
               ) -> pd.DataFrame:
    """
    Clean, aggregate, and featurize a molecular property dataset.

    Molecules are canonicalized from the specified SMILES column and grouped
    by their canonical representation. Multiple records corresponding to the
    same molecule are aggregated into lists, and the mean and standard
    deviation of the target values are calculated. Molecular features are
    then generated for each unique molecule using ``mol_to_fetures``.

    If train, validation, and test indices are provided, the corresponding
    split labels are assigned. Otherwise, the unique molecules are randomly
    divided into training (50%), test (30%), and validation (20%) sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input molecular dataset.

    smiles_col : str
        Name of the column containing SMILES strings.

    target_col : str
        Name of the column containing the numerical target values.

    columns_to_list : list of str, optional
        Additional columns whose values should be aggregated into lists for
        duplicate molecules.

    train_idx, val_idx, test_idx : array-like, optional
        Indices used to assign molecules to the training, validation, and
        test sets. All three must be provided to use predefined splits.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing one row per unique canonical molecule,
        aggregated values, target summary statistics, molecular features,
        and dataset split labels.

    Notes
    -----
    This function requires the externally defined functions
    ``canonicalize_smiles`` and ``mol_to_fetures``.
    """

    # --------------------------------------------------------------
    # Validate input columns
    # --------------------------------------------------------------
    df = df.copy()
    columns_to_list = columns_to_list or []
    required_columns = {smiles_col, target_col, *columns_to_list}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {sorted(missing_columns)}")
    
    splits_provided = all(idx is not None for idx in (train_idx, val_idx, test_idx))
    if splits_provided and not split_data:
        warnings.warn(f'All the train, test, and validation indices for data split are provided but `split_data = False`')
        time.sleep(10)

    # --------------------------------------------------------------
    # Generate output column names dynamically
    # --------------------------------------------------------------
    canonical_col = f"canonical_{smiles_col.replace(' ', '_')}"
    smiles_list_col = f"{smiles_col.replace(' ', '_')}_list"
    target_list_col = f"{target_col.replace(' ', '_')}_list"
    target_mean_col = f"{target_col.replace(' ', '_')}_mean"
    target_std_col = f"{target_col.replace(' ', '_')}_sd"

    # --------------------------------------------------------------
    # Canonicalize molecular structures
    # --------------------------------------------------------------
    df[canonical_col] = df[smiles_col].apply(canonicalize_smiles)

    # --------------------------------------------------------------
    # Define aggregation rules and modified column names
    # --------------------------------------------------------------
    aggregation = {smiles_col: list, target_col: list}
    rename_cols = {smiles_col: smiles_list_col, target_col: target_list_col}

    for col in columns_to_list:
        if col not in aggregation:
            aggregation[col] = list
        if col not in {smiles_col, target_col}:
            rename_cols[col] = f"{col.replace(' ', '_')}_list"

    # --------------------------------------------------------------
    # Group equivalent molecules
    # --------------------------------------------------------------
    df_grouped = df.groupby(canonical_col, as_index=False).agg(aggregation)
    df_grouped = df_grouped.rename(columns=rename_cols)
    df_grouped.insert(0, smiles_col, df_grouped[canonical_col])
    print(f"Number of unique datapoints: {len(df_grouped)}")

    # --------------------------------------------------------------
    # Calculate target statistics
    # --------------------------------------------------------------
    df_grouped[target_mean_col] = df_grouped[target_list_col].apply(np.mean)
    df_grouped[target_std_col] = df_grouped[target_list_col].apply(np.std)

    # --------------------------------------------------------------
    # Calculate molecular features
    # --------------------------------------------------------------
    molecular_features = [None]*len(df_grouped)
    all_smiles = df_grouped[smiles_col].copy()

    for _idx, smi in tqdm(enumerate(all_smiles), total=len(df_grouped), desc="Calculating Features"):
        molecular_features[_idx] = mol_to_fetures(smi)

    molecular_features = pd.DataFrame(molecular_features).select_dtypes(include=[np.number])
    molecular_features = molecular_features.reset_index(drop=True)
    df_grouped = df_grouped.reset_index(drop=True)
    df_features = pd.concat([df_grouped, molecular_features], axis=1)

    # --------------------------------------------------------------
    # Assign dataset splits
    # --------------------------------------------------------------
    if split_data:
        df_features.insert(2, "Data_Split", "Other")
        if not splits_provided:
            print("Data splitting indices not provided. Splitting dataset randomly.")
            indices = np.random.permutation(len(df_features))

            train_end = len(indices) // 2
            test_end = len(indices) * 4 // 5

            train_idx = indices[:train_end]
            test_idx = indices[train_end:test_end]
            val_idx = indices[test_end:]

        df_features.loc[df_features.index.isin(train_idx), "Data_Split"] = "Tr"
        df_features.loc[df_features.index.isin(test_idx), "Data_Split"] = "Te"
        df_features.loc[df_features.index.isin(val_idx), "Data_Split"] = "Val"

    return df_features


def merge_list_columns_by_key(target_df: pd.DataFrame,
                               reference_df: pd.DataFrame,
                               key_column: str,
                               list_column: str,
                               mean_column: Optional[str] = None,
                               std_column: Optional[str] = None,
                               parse_strings_to_lists: bool = True
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

    return updated_df, row_mapping


def modify_df1_wrt_df2(df_changing: pd.DataFrame,
                       df_wrt: pd.DataFrame,
                       cannonical_smi_col_name: str = 'Cannonicalized_SMILES',
                       logpapp_list_col_name: str = 'logPapp_Values_list',
                       mean_col_name: str = 'Mean_logPapp_Values',
                       std_col_name: str = 'logPapp_Values_Std'
                       ) -> Tuple[pd.DataFrame, Dict[int, int]]:
    
    df_changing_copy = copy.deepcopy(df_changing)

    common_df_changing = df_changing_copy[df_changing_copy[cannonical_smi_col_name].isin(df_wrt[cannonical_smi_col_name])]
    df_changing_indices = dict(zip(common_df_changing[cannonical_smi_col_name], common_df_changing.index))
    
    common_df_wrt = df_wrt[df_wrt[cannonical_smi_col_name].isin(df_changing_copy[cannonical_smi_col_name])]
    df_wrt_indices = dict(zip(common_df_wrt[cannonical_smi_col_name], common_df_wrt.index))

    changing_dict_idx = {v: df_wrt_indices.get(k) for k, v in df_changing_indices.items() if k in df_wrt_indices}
    
    # Ensure columns are lists
    df_changing_copy[logpapp_list_col_name] = df_changing_copy[logpapp_list_col_name].apply(lambda x: ast.literal_eval(x))
    df_wrt[logpapp_list_col_name] = df_wrt[logpapp_list_col_name].apply(lambda x: ast.literal_eval(x))

    # Update the copied DataFrame
    for all_data_idx, rr_idx in changing_dict_idx.items():
        df_changing_copy.loc[all_data_idx, logpapp_list_col_name].extend(df_wrt.loc[rr_idx, logpapp_list_col_name])
        
    # Define functions for mean and standard deviation
    mean_func = lambda x: sum(x) / len(x) if len(x) > 0 else None
    std_func = lambda x: np.std(x) if len(x) > 0 else None

    # Apply the functions to the new columns
    df_changing_copy[std_col_name] = df_changing_copy[logpapp_list_col_name].apply(std_func)
    df_changing_copy[mean_col_name] = df_changing_copy[logpapp_list_col_name].apply(mean_func)
    
    return df_changing_copy, changing_dict_idx