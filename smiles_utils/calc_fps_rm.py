import os
import sys
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from collections import defaultdict, OrderedDict
from orderedset import OrderedSet
from typing import Any, List, Dict, Tuple, Union, Set, Callable, Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, AdditionalOutput
from rdkit.Chem.MolStandardize import rdMolStandardize

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore", module="mordred")

__all__ = [
    'calculate_fingerprint_from_smiles',
    'calculate_multiple_fingerprints_from_smiles',
    'calculate_multiple_fingerprints_all',
    'compute_similarity_distance_metrics',
    'calc_metric_listsmiles2smiles',
    'calculate_similarities_distances',
    'calculate_similarity_distance',
    'calculate_all_fingerprints',
    'smiles_to_morgan_fps',
]


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else float('inf')

def get_fingerprint(mol, name: str):
    
    name = name.strip().lower()
    
    if isinstance(mol, str):
        mol = Chem.MolfromSmiles(mol)
        
    if name == 'rdkit':
        return Chem.RDKFingerprint(mol)
    elif name == 'pattern':
        return Chem.PatternFingerprint(mol)
    elif name == 'maccskeys':
        return rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    elif name == 'topologicaltorsion':
        return AllChem.GetTopologicalTorsionGenerator().GetFingerprint(mol)
    elif name == 'morgan':
        return AllChem.GetMorganGenerator(radius=2).GetFingerprint(mol)
    elif name == 'atompair':
        return AllChem.GetAtomPairGenerator().GetFingerprint(mol)
    else:
        raise ValueError(f"Invalid fingerprint name: {name}")

def fingerprint_to_array(fp) -> np.ndarray:
    
    arr = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    
    return arr

def calculate_fingerprint_from_mol(mol: str, fingerprint_name: str) -> np.ndarray:
    
    if isinstance(mol, str):
        mol = Chem.MolfromSmiles(mol)
    elif isinstance(mol, Chem.Mol):
        pass
    else:
        raise AttributeError(f'Invalid input ({mol}) of type {type(mol)}')
        
    if mol is None:
        raise ValueError("Invalid SMILES string")
    fp = get_fingerprint(mol, fingerprint_name)
    
    return fingerprint_to_array(fp)

def calculate_fingerprint_from_smiles(smiles: str, fingerprint_name: str) -> np.ndarray:
    
    return calculate_fingerprint_from_mol(Chem.MolFromSmiles(smiles), fingerprint_name)

def calculate_multiple_fingerprints_from_mol(mol: str, fingerprint_names: List[str]) -> dict:

    all_results = OrderedDict({})
    if mol is None:
        raise ValueError("Invalid SMILES string")
    for fp_name in fingerprint_names:
        fp = get_fingerprint(mol, fp_name)
        all_results[fp_name] = fingerprint_to_array(fp)
    
    return all_results

def calculate_multiple_fingerprints_from_smiles(smiles: str, fingerprint_names: List[str]) -> dict:

    all_results = OrderedDict({'smiles':smiles})
    mol = Chem.MolFromSmiles(smiles)
    temp = calculate_multiple_fingerprints_from_mol(mol, fingerprint_names)
    all_results = {**all_results, **temp}

    return all_results

def calculate_multiple_fingerprints_all(smiles_list:List[str],
                                        fingerprint_names: List[str]
                                        ) -> pd.DataFrame:
    
    all_results = [None]*len(smiles_list)
    for i, smiles in tqdm(enumerate(smiles_list), desc='Calculating FPs', total=len(smiles_list)):
        all_results[i] = calculate_multiple_fingerprints_from_smiles(smiles, fingerprint_names)
    
    return all_results

def get_abcd(array_1: np.ndarray, array_2: np.ndarray) -> Tuple[int, int, int, int]:
    
    a = np.sum((array_1 == 1) & (array_2 == 0))
    b = np.sum((array_1 == 0) & (array_2 == 1))
    c = np.sum((array_1 == 1) & (array_2 == 1))
    d = np.sum((array_1 == 0) & (array_2 == 0))
    
    return a, b, c, d

def compute_similarity_distance_metrics(a, b, c, d) -> Tuple[dict, dict]:
    sim = {
        "Tanimoto": safe_divide(c, a + b + c),
        "Dice": safe_divide(c, 0.5 * ((a + c) + (b + c))),
        "Cosine": safe_divide(c, np.sqrt((a + c) * (b + c))),
        "Russell_Rao": safe_divide(c, a + b + c + d),
        "Baroni_Urbani": safe_divide((np.sqrt(c * d) + c), (np.sqrt(c * d) + a + b + c)),
        "Rogers": safe_divide((c + d), (2 * a + 2 * b + c + d)),
        "Matching_Coefficient": safe_divide((c + d), (a + b + c + d)),
        "Overlap": c,
    }

    dist = {
        "Euclidean": np.sqrt(a + b),
        "Hamming": a + b,
        "Mean_Hamming": safe_divide((a + b), (a + b + c + d)),
        "Soergel": safe_divide((a + b), (a + b + c)),
        "Pattern": safe_divide((a * b), (a + b + c + d) ** 2),
        "Variance": safe_divide((a + b), (4 * (a + b + c + d))),
        "Size": safe_divide((a - b) ** 2, (a + b + c + d) ** 2),
    }
    return sim, dist

def calculate_similarities_distances(smiles1: str, smiles2: str, fingerprint_name: str = 'RDKit') -> Tuple[dict, dict]:
    
    fp1 = calculate_fingerprint_from_smiles(smiles1, fingerprint_name)
    fp2 = calculate_fingerprint_from_smiles(smiles2, fingerprint_name)
    a, b, c, d = get_abcd(fp1, fp2)
    
    return compute_similarity_distance_metrics(a, b, c, d)

def calculate_similarity_distance(smiles1: str, smiles2: str, fingerprint_name: str = 'RDKit',
                                   distance_metric: str = 'Euclidean', similarity_metric: str = 'Tanimoto') -> Tuple[float, float]:
    
    fp1 = calculate_fingerprint_from_smiles(smiles1, fingerprint_name)
    fp2 = calculate_fingerprint_from_smiles(smiles2, fingerprint_name)
    a, b, c, d = get_abcd(fp1, fp2)
    sims, dists = compute_similarity_distance_metrics(a, b, c, d)

    if similarity_metric not in sims:
        raise ValueError(f"Similarity metric not found. Choose from: {list(sims.keys())}")
    if distance_metric not in dists:
        raise ValueError(f"Distance metric not found. Choose from: {list(dists.keys())}")

    return sims[similarity_metric], dists[distance_metric]

def calc_metric_listsmiles2smiles(smiles:str, from_list:List[str], fingerprint_name: str = 'RDKit', 
                                metric: str = 'Euclidean', ascending=None):

    similarity_list = ["Tanimoto", "Dice", "Cosine", "Russell_Rao", "Baroni_Urbani", "Rogers", "Matching_Coefficient", "Overlap"]
    distnace_list = ["Euclidean", "Hamming", "Mean_Hamming", "Soergel", "Pattern", "Variance", "Size",]

    all_results = {'smiles':[], f'{metric}_metric':[]}
    for smi in from_list:
        fp1 = calculate_fingerprint_from_smiles(smi, fingerprint_name)
        fp2 = calculate_fingerprint_from_smiles(smiles, fingerprint_name)
        a, b, c, d = get_abcd(fp1, fp2)
        sims, dists = compute_similarity_distance_metrics(a, b, c, d)

        all_results['smiles'].append(smi)
        if metric in similarity_list:
            all_results[f'{metric}_metric'].append(sims[metric])
        elif metric in distnace_list:
            all_results[f'{metric}_metric'].append(dists[metric])

        out = pd.DataFrame(all_results)
        if ascending is not None:
            out = out.sort_values(by=f'{metric}_metric', ascending=ascending)

    return out


def calculate_all_fingerprints(smiles: str) -> dict:
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    names = ['RDKit', 'MACCSKeys', 'AtomPair', 'TopologicalTorsion', 'Morgan', 'Pattern']
    
    return {name: fingerprint_to_array(get_fingerprint(mol, name)) for name in names}

def smiles_to_morgan_fps(df: pd.DataFrame, smiles_col: str = "smiles", radii=(0, 1, 2), n_bits: int = 2048) -> pd.DataFrame:
    """
    Compute Morgan fingerprints for a DataFrame of SMILES strings using RDKit's MorganGenerator.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing SMILES.
        smiles_col (str): Column name that has the SMILES strings.
        radii (tuple): Radii for Morgan fingerprints.
        n_bits (int): Length of fingerprint bit vector.
    
    Returns:
        pd.DataFrame: Concatenated DataFrame of fingerprints.
    """
    fps_dfs = []

    for radius in radii:
        gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
        fps = []
        for smi in df[smiles_col]:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = gen.GetFingerprint(mol)  # returns ExplicitBitVect
                fps.append(list(fp))
            else:
                fps.append([0] * n_bits)  # fallback for invalid SMILES
        
        fps_df = pd.DataFrame(fps, columns=[f"MorganFP_{i}/r{radius}" for i in range(n_bits)])
        fps_dfs.append(fps_df)
    
    return pd.concat([df.reset_index(drop=True)] + fps_dfs, axis=1)