import re
import os
import ast
import sys
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Any, List, Dict, Tuple, Union, Set, Callable, Optional
from rdkit.Chem.Scaffolds import MurckoScaffold

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdDepictor, AllChem, rdMolDescriptors, Descriptors, Crippen
from mordred import Calculator, descriptors
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
    'canonicalize_smiles',
    'canonicalize_smiles_list',
    'smi2bonds',
    'smi2atoms',
    'get_unique_atoms_from_smiles',
    'get_all_unique_atoms',
    'get_main_organic_smiles',
    'standardize_molecules',
    'randomize_smiles',
    'augment_smiles_with_labels',
    'smiles_validity_check',
    'atom_filter_smiles',
    'generate_scaffold',
]


def canonicalize_smiles(smiles: str,
                        isomericSmiles: bool = False,#True
                        ) -> (str | None):
    
    '''
    Input: A SMILES
    Output: The Cannonicalised Form of the SMILES
    '''
    
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomericSmiles)    
        return canonical_smiles
    else:
        return None


def canonicalize_smiles_list(smiles_list: list) -> list:
    
    '''
    Input: A list of SMILES
    Output: The Cannonicalised Form of the SMILES in the list
    '''
    
    canonical_smiles_list = [None]*len(smiles_list)
    for i in range(len(smiles_list)):
        canonical_smiles = canonicalize_smiles(smiles_list[i])
        canonical_smiles_list[i] = canonical_smiles
        
    return canonical_smiles_list

def smi2bonds(smi):
    mol = Chem.MolFromSmiles(smi)
    bond_types=set()
    for atom in mol.GetAtoms():
        bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
        bond_types |= set(bonds)
    if 'DATIVE' in bond_types:
        return True
    else:
        return False
    
def smi2atoms(smi):
    
    try:
        atoms=[]
        mol = Chem.MolFromSmiles(smi)
        for atom in mol.GetAtoms():
            atoms.append(atom.GetSymbol())
        return atoms
    except:
        return []

def smiles_validity_check(smiles):
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    else:
        return True
        
def atom_filter_smiles(smiles,
                       all_atoms=set(['Br', 'Cl', 'P', 'I', 'F', 'H', 'S', 'N', 'O', 'C', 'B', 'Si', 'Na', 'K'])
                       ):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
        return (symbols.issubset(all_atoms) and mol.GetNumAtoms() >= 5)
    except Exception:
        return False

def get_unique_atoms_from_smiles(smi):
    
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return set()
        return set(atom.GetSymbol() for atom in mol.GetAtoms())
    except:
        return set()

def get_all_unique_atoms(smiles_list, num_workers=None):
    
    if num_workers is None:
        num_workers = min(cpu_count()-2, 16)  # Limit to avoid over-parallelization

    with Pool(num_workers) as pool:
        results = pool.map(get_unique_atoms_from_smiles, smiles_list)

    all_atoms = set().union(*results)
    
    return sorted(all_atoms)

def standardize_molecules(smiles):
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol) 

        # if many fragments, get the "parent" (the actual mol we are interested in) 
        clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # try to neutralize molecule
        uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
        clean_mol = uncharger.uncharge(clean_mol)

        # # try to Canonicalize tautomers
        # te = rdMolStandardize.TautomerEnumerator() 
        # clean_mol = te.Canonicalize(clean_mol)
        return Chem.MolToSmiles(clean_mol)
    except:
        return None

def get_main_organic_smiles(smiles):
    
    # Convert input SMILES to RDKit Mol object
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        # Fragment the molecule into disconnected components
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)

        # Filter out inorganic/small fragments (e.g., H+, Cl-) by number of heavy atoms
        # You may also choose by molecular weight, logP, or other criteria
        organic_frags = [frag for frag in frags if rdMolDescriptors.CalcNumHeavyAtoms(frag) > 4]

        # If multiple remain, pick the one with highest heavy atom count
        main_frag = max(organic_frags, key=rdMolDescriptors.CalcNumHeavyAtoms)

        # Convert back to SMILES
        return Chem.MolToSmiles(main_frag)
    except:
        return None


def randomize_smiles(smiles, n_aug=5):
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    
    return [Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True) for _ in range(n_aug)]


def augment_smiles_with_labels(smiles_list: List[str],
                               labels: List[int|float],
                               n_aug: int =10,
                               noise_scale: float = 0.05
                               ) -> Tuple[List[str], List[int|float]]:
    
    augmented_smiles = []
    augmented_labels = []

    for smi, label in zip(smiles_list, labels):
        randomized = randomize_smiles(smi, n_aug)
        std = max(label * noise_scale, 1e-3)  # prevent std=0
        noise = np.random.normal(loc=label, scale=std, size=len(randomized))
        
        augmented_smiles.extend(randomized)
        augmented_labels.extend(noise.tolist())

    return augmented_smiles, np.array(augmented_labels)

def generate_scaffold(smiles, include_chirality=False):
    """
    Generate Bemis-Murcko scaffold for a SMILES string.
    Returns None if molecule parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=include_chirality
    )
    return scaffold