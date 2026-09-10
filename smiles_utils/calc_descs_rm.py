import os
import ast
import sys
import math
import copy, gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from orderedset import OrderedSet
from typing import Any, List, Dict, Tuple, Union, Set, Callable, Optional

import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdDepictor, AllChem, rdMolDescriptors, Descriptors, Crippen
from rdkit.Chem.Draw import rdMolDraw2D
from mordred import Calculator, descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, AdditionalOutput
from rdkit.Chem.MolStandardize import rdMolStandardize

from rdkit.Chem.Descriptors3D import (Asphericity, Eccentricity, InertialShapeFactor, NPR1, NPR2, PBF, 
                                      PMI1, PMI2, PMI3, RadiusOfGyration, SpherocityIndex
                                      )
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore", module="mordred")

# --- Precompute heavy objects once ---
MORDRED_CALC = Calculator(descriptors, ignore_3D=True)
MORGAN_GENERATOR = GetMorganGenerator(radius=2, fpSize=1024)
RDKit_FUNCS = [
    (name, func)
    for name, func in Descriptors.__dict__.items()
    if callable(func)
]

__all__ = [
    'MORDRED_CALC',
    'MORGAN_GENERATOR',
    'RDKit_FUNCS',
    'calculate_properties',
    'compute_all_3d_descriptors',
    'get_descriptor_functions_from_name',
    'compute_descriptors_from_name',
    'compute_descriptors',
    'process_in_parallel',
    'listsmiles2propdf',
    'load_or_compute_descriptors',
    'mol_to_fetures_with_descriptor_function',
    'mol_to_fetures',
    'mol_to_features_single',
    'mol_to_features_batch',
]


def calculate_properties(smiles: str) -> Dict[str, Any]:
    
    mol = Chem.MolFromSmiles(smiles)
    rdkit_descriptors = {
        "Molecular Mass": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "Molar Refractivity": Descriptors.MolMR(mol),
        "Polarizability": Descriptors.MolMR(mol) / 2.5,
        "qed": Descriptors.qed(mol),
        "TPSA": Descriptors.TPSA(mol),
        "VSA_EState3": Descriptors.VSA_EState3(mol),
        "NHOHCount": Descriptors.NHOHCount(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "MolLogP": Descriptors.MolLogP(mol)
    }
    
    calc = Calculator(descriptors, ignore_3D=True)
    mordred_values = calc(mol)
    mordred_descriptors = {
        "ATSC1pe": mordred_values["ATSC1pe"],
        "ATSC1are": mordred_values["ATSC1are"],
        "AATSC1dv": mordred_values["AATSC1dv"],
        "AATSC1are": mordred_values["AATSC1are"]
    }
    
    # Combine RDKit and Mordred descriptors
    all_descriptors = {**rdkit_descriptors, **mordred_descriptors}
    
    return all_descriptors


def compute_all_3d_descriptors(smiles: str) -> Dict[str, float|int]:
    """
    Compute all 3D descriptors for a molecule.
    Parameters:
        smiles (str): Input SMILES string.
    Returns:
        dict: A dictionary of 3D descriptors.
    """
    try:
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)  # Add explicit hydrogens

        # Generate 3D conformer
        params = AllChem.ETKDG()
        params.randomSeed = 42  # For reproducibility
        success = AllChem.EmbedMolecule(mol, params)
        if success != 0:
            raise ValueError("Failed to embed molecule.")

        # Optimize geometry
        AllChem.UFFOptimizeMolecule(mol)

        # Compute 3D descriptors
        descriptors = {
            "Asphericity": Asphericity(mol),
            "Eccentricity": Eccentricity(mol),
            "InertialShapeFactor": InertialShapeFactor(mol),
            "NPR1": NPR1(mol),
            "NPR2": NPR2(mol),
            "PBF": PBF(mol),
            "PMI1": PMI1(mol),
            "PMI2": PMI2(mol),
            "PMI3": PMI3(mol),
            "RadiusOfGyration": RadiusOfGyration(mol),
            "SpherocityIndex": SpherocityIndex(mol),
        }
        
    except Exception as e:
        descriptors = {
            "Asphericity": None,
            "Eccentricity": None,
            "InertialShapeFactor": None,
            "NPR1": None,
            "NPR2": None,
            "PBF": None,
            "PMI1": None,
            "PMI2": None,
            "PMI3": None,
            "RadiusOfGyration": None,
            "SpherocityIndex": None,
        }
        
    return descriptors


    
# # --- GLOBALS for Worker Access ---
_RDKit_descriptor_funcs = None
_mordred_calc = None


# --- Descriptor Setup ---
def get_descriptor_functions_from_name(descriptor_names: List[str]) -> Tuple[List[Callable], Calculator]:
    """Creates RDKit and Mordred descriptor function sets."""
    RDKit_descriptor_funcs = OrderedDict({
        k: v for k, v in Descriptors.__dict__.items()
        if k in descriptor_names and callable(v)
    })

    mordred_calc_all = Calculator(descriptors, ignore_3D=True)
    mordred_names = list(OrderedSet(descriptor_names) - OrderedSet(RDKit_descriptor_funcs.keys()))
    mordred_selected = [d for d in mordred_calc_all.descriptors if str(d) in mordred_names]
    mordred_calc = Calculator(mordred_selected, ignore_3D=True)

    return RDKit_descriptor_funcs, mordred_calc


def compute_descriptors_from_name(smiles: str,
                        descriptor_names: List[str]
                        ) -> Dict[str, float] | None:
    """Computes descriptors for a single SMILES string."""
    try:
        RDKit_descriptor_funcs, mordred_calc = get_descriptor_functions_from_name(descriptor_names)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f'Invalid SMILES Error...')
            return None
        rdkit_vals = {name: func(mol) for name, func in RDKit_descriptor_funcs.items()}
        mordred_vals = mordred_calc(mol) if mordred_calc.descriptors else {}
        mordred_vals = {str(desc): val for desc, val in mordred_vals.items()}
        return {'SMILES': smiles, **rdkit_vals, **mordred_vals}
    except Exception as e:
        print(f'Exception Triggered: {e}')
        return None


# --- Initializer ---
def initializer(rdkit_funcs: List[Callable],
                mordred_calculator: Calculator
                ) -> None:
    
    """Sets global descriptor functions for each worker."""
    global _RDKit_descriptor_funcs, _mordred_calc
    _RDKit_descriptor_funcs = rdkit_funcs
    _mordred_calc = mordred_calculator

# --- Descriptor Calculation per SMILES ---
def compute_descriptors(smiles: str) -> Dict[str, float] | None:
    """Computes descriptors for a single SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f'Invalid SMILES Error...')
            return None

        rdkit_vals = {name: func(mol) for name, func in _RDKit_descriptor_funcs.items()}
        mordred_vals = _mordred_calc(mol) if _mordred_calc.descriptors else {}
        mordred_vals = {str(desc): val for desc, val in mordred_vals.items()}
        return {'SMILES': smiles, **rdkit_vals, **mordred_vals}
    except Exception as e:
        print(f'Exception Triggered: {e}')
        return None

# --- Parallel Processing Function ---
def process_in_parallel(smiles_list: List[str],
                        rdkit_funcs: List[Callable],
                        mordred_calc: Calculator,
                        n_jobs: int =8,
                        chunk_size:int = 1000,
                        ordered: bool = False,
                        df_savepath: str = 'temp',
                        save_per_item: int = 10000,
                        ) -> List[Dict[str, int|float|str]]:
    """Process SMILES list in parallel with descriptor calculation."""
    
    if os.path.isfile(df_savepath):
        results = pd.read_csv(df_savepath)
        print(f'Found {len(results)} records at: {df_savepath}')
        existing_smiles = set(results['SMILES'].tolist())
        smiles_list = [s for s in tqdm(smiles_list) if s not in existing_smiles]
        results = results.to_dict(orient="records")
        print(f"Number of remaining SMILES: {len(smiles_list)}")
    else:
        results = []
        
    for i in tqdm(range((len(smiles_list)//save_per_item)+1)):
        temp_smiles_list = smiles_list[save_per_item*i:save_per_item*(i+1)]
        with mp.Pool(processes=n_jobs,
                initializer=initializer,
                initargs=(rdkit_funcs, mordred_calc)
                ) as pool:
            imap_func = pool.imap if ordered else pool.imap_unordered
            temp_results = list(tqdm(imap_func(compute_descriptors,
                                        temp_smiles_list,
                                        # chunksize=chunk_size
                                        ),
                                total=len(temp_smiles_list),
                                desc="Calculating Descriptors"
                                )
                        )
        results = results + temp_results
        # if len(results)%save_per_item == 0:
        temp = [r for r in results if r is not None]
        df = pd.DataFrame(temp)
        for col in df.columns:
            if col != 'SMILES':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if df_savepath:
            df.to_csv(df_savepath, index=False)
            print(f"Descriptor data saved to: {df_savepath}")
    return [r for r in results if r is not None]


# --- High-level Wrapper ---
def listsmiles2propdf(smiles_list: List[str],
                      descriptor_names: List[str],
                      df_savepath: str = 'descriptors_output.csv',
                      ordered_parallel_processing: bool = False
                      ) -> pd.DataFrame:
    """Takes list of SMILES and outputs descriptor DataFrame (optionally saves to CSV)."""
    
    print("Preparing descriptor functions...")
    rdkit_funcs, mordred_calc = get_descriptor_functions_from_name(descriptor_names)

    n_jobs = int(mp.cpu_count()/2) - 1
    chunk_size = min(10, len(smiles_list) // (10 * n_jobs))
    print(f"Using {n_jobs} CPU cores with chunk size: {chunk_size}")
    descriptor_data = process_in_parallel(
        smiles_list=smiles_list,
        rdkit_funcs=rdkit_funcs,
        mordred_calc=mordred_calc,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        ordered=ordered_parallel_processing,
        df_savepath = f"{df_savepath}",
        save_per_item=100000,
    )

    df = pd.DataFrame(descriptor_data)
    for col in descriptor_names:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df_savepath:
        df.to_csv(df_savepath, index=False)
        print(f"Descriptor data saved to: {df_savepath}")

    return df


def load_or_compute_descriptors(df: pd.DataFrame,
                                 descriptor_names: List[str],
                                 smiles_column_name: str = 'smiles',
                                 df_savepath: str = '',
                                 ordered_parallel_processing: bool = True
                                 ) -> pd.DataFrame:
    """
    Ensures all required molecular descriptors are present in the DataFrame.
    Computes missing descriptors and merges them with the existing DataFrame.

    Args:
        df (pd.DataFrame): Existing DataFrame with SMILES and possibly some descriptors.
        descriptor_names (List[str]): List of required descriptors.
        smiles_column_name (str): Column name where SMILES strings are stored.
        df_savepath (str): Path to save newly computed descriptors (optional).
        ordered_parallel_processing (bool): Whether to compute descriptors in order and in parallel.

    Returns:
        pd.DataFrame: DataFrame with all required descriptors.
    """
    existing_cols = set(df.columns)
    required_cols = set(descriptor_names)
    missing_desc_names = list(required_cols - existing_cols)

    if missing_desc_names:
        smiles_list = df[smiles_column_name].tolist()
        new_df = listsmiles2propdf(smiles_list=smiles_list,
                                   descriptor_names=missing_desc_names,
                                   df_savepath=f'{df_savepath}_temp',
                                   ordered_parallel_processing=ordered_parallel_processing)

        # Ensure column match before merging
        if 'SMILES' in new_df.columns and smiles_column_name != 'SMILES':
            new_df = new_df.rename(columns={'SMILES': smiles_column_name})

        final_df = pd.merge(df, new_df, on=smiles_column_name, how='inner')

        
        if df_savepath:
            final_df.to_csv(df_savepath, index=False)

        return final_df

    else:
        return df


def mol_to_fetures_with_descriptor_function(smile: str,
                                            flag_name2descriptor = True
                                            ) -> Dict[Any | str, Any]:
    
    '''
    Input: A SMILES
    Output: All descriptors calculated from that smiles using different libraries.
    '''
    try:
        # Step 1: Convert SMILES to RDKit Mol object
        mol = Chem.MolFromSmiles(smile)

        # Step 2: Initialize Mordred Calculator to compute all descriptors
        calc = Calculator(descriptors, ignore_3D=True)

        # Step 3: Compute Mordred descriptors (physicochemical, MOE-type, Kappa, etc.)
        desc_values = calc(mol)

        # Collect the Mordred descriptors
        mordred_descriptors = {desc: value for desc, value in desc_values.items()}

        # Step 4: Compute RDKit-based physicochemical properties (e.g., molecular weight, LogP, TPSA)
        rdkit_properties = {}
        # for name, func in Descriptors._descList:
        for name, func in Descriptors.__dict__.items():
            if callable(func):
                try:
                    rdkit_properties[name] = func(mol)
                    '''TPSA: Calculated using the formula: TPSA = 60.0 * (NHOH + NNH) + 20.0 * NOH
                    Whereas, TopoPSA: Calculated using the formula: TopoPSA = 60.0 * (NHOH + NNH) + 20.0 * NOH + 10.0 * (NCO + NOC) + 5.0 * (NNO + NNN)
                    '''
                except:# Exception as e:
                    pass

        # Step 5: Compute Morgan fingerprints
        radius = 2  # Set radius for Morgan fingerprint
        nBits = 1024  # Set number of bits for the fingerprint
        # morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        # Initialize the MorganGenerator
        morgan_generator = GetMorganGenerator(radius=radius, fpSize=nBits)
        morgan_fp = morgan_generator.GetFingerprint(mol)

        # Convert Morgan fingerprints to a dictionary
        morgan_fp_dict = {f'MorganFP_{i}': int(bit) for i, bit in enumerate(morgan_fp)}
        
        # 3D descriptors
        descriptors_3d = compute_all_3d_descriptors(smile)

        # Step 6: Combine all descriptors into one dictionary
        all_descriptors = {**rdkit_properties, **mordred_descriptors, **morgan_fp_dict, **descriptors_3d}
        
        names2descriptors = None
        if flag_name2descriptor:
            names2descriptors = {str(k):k for k in all_descriptors.keys()}
    except:
        pass
    
    return all_descriptors, names2descriptors


def mol_to_fetures(smile: str) -> Dict[Any | str, Any]:
    
    '''
    Input: A SMILES
    Output: All descriptors calculated from that smiles using different libraries where keys are strings.
    '''
    all_descriptors, _ = mol_to_fetures_with_descriptor_function(smile, flag_name2descriptor=False)
    all_descriptors = {str(k):v for k, v in all_descriptors.items()}
    
    return all_descriptors


def mol_to_features_single(smiles: str|Chem.Mol,
                           compute_3d: bool = True
                           ) -> Dict[str, Any]:
    """Compute descriptors for a single RDKit Mol (fast path)."""
    features = {'smiles':smiles}

    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol=smiles

    # --- RDKit built-in descriptors ---
    for name, func in RDKit_FUNCS:
        try:
            features[name] = func(mol)
        except:
            pass
        
    # --- Mordred descriptors ---
    try:
        desc_values = MORDRED_CALC(mol)
        features.update({str(k): v for k, v in desc_values.items()})
    except:
        pass
    
    # --- Morgan fingerprint ---
    try:
        fp = MORGAN_GENERATOR.GetFingerprint(mol)
        features.update({f"MorganFP_{i}": int(bit) for i, bit in enumerate(fp)})
    except:
        pass

    # --- Optional 3D descriptors ---
    if compute_3d:
        try:
            features.update(compute_all_3d_descriptors(Chem.MolToSmiles(mol)))
        except:
            pass
            
    return features


def mol_to_features_batch(smiles_list: List[str],
                          smiles_column_name='smiles',
                          compute_3d: bool = True,
                          verbose: bool = True
                          ) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Optimized batch descriptor computation for a list of SMILES.
    Reuses global calculator and avoids redundant initialization.
    """
    results = []

    for i, smi in tqdm(enumerate(smiles_list), desc='Calculating descriptors:', total=len(smiles_list)):
        try:
            descs = mol_to_features_single(smi, compute_3d=compute_3d)
            results.append(descs)
        except:
            pass

        if verbose and (i + 1) % 100 == 0:
            print(f"[{i + 1}/{len(smiles_list)}] processed")
    df = pd.DataFrame(results)
    df=df.rename(columns={'smiles':smiles_column_name})
    num_df = df.select_dtypes(include=['number'])
    final_df = pd.concat([df[smiles_column_name], num_df], axis=1)
    
    return final_df