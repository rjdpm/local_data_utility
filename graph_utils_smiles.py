from rdkit import Chem
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle, gzip
from torch_geometric.data import Data
from torch_geometric.data import Dataset as pyg_dataset
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from logP_values import *

from rdkit.Chem import AllChem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem import Descriptors
from rdkit.Chem.Descriptors3D import (
    Asphericity, Eccentricity, InertialShapeFactor, NPR1, NPR2, PBF,
    PMI1, PMI2, PMI3, RadiusOfGyration, SpherocityIndex
)

import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Generalised_data_utils import selective_range_data_sampling

__all__ = [
    'extract_features_from_smiles',
    'list_smiles2symbols_hybridization_chiraltype',
    'compute_mean_std_welford',
    'list_to_onehot',
    'list_to_onehot_oov',
    'get_element_properties',
    'get_atom_properties',
    'get_properties_smiles',
    'get_3d_descriptors',
    'smiles_filter_3d_descriptor',
    'get_atom_info_vector',
    'get_bond_matrix',
    'get_multirelational_bond_matrix',
    'GraphDataset',
    'SMILEStoPyGGraphDataset',
    'Multirelational_GraphDataset',
    'Multirelational_GraphDataset_Embeddings',
    'SmilesDataset_graph_gen',
    'limit_open_files',
    'GraphData_from_pickle',
    'GraphData_from_pickle_3d_descriptor',
    'feature_representation',
    'feature_representation_pyg',
    'gcn_pred_func',
    'filter_data',
    'separate_data',
    'weights_initilizer'
]
with open('/home/rkmvu/Dataset/Coca-2/Experimental_Data_P_app/train_test_partition_literature/All_possible_atoms_X.pkl', 'rb') as fp:
    logP_dict = pickle.load(fp)
    
def extract_features_from_smiles(smi):
    """
    Extracts features from a single SMILES string.
    Returns: 
        Tuple[List[str], List[str], List[str], List[str], int] -> (num_atoms, atom_syms, hybridizations, chiral_tags, bond_types, max_nbrs)
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        num_atoms = mol.GetNumAtoms()
        mol = Chem.AddHs(mol)

        atom_syms = set()
        hybridizations = set()
        chiral_tags = set()
        bond_types = set()
        max_nbrs = 0

        for atom in mol.GetAtoms():
            atom_syms.add(atom.GetSymbol())
            hybridizations.add(str(atom.GetHybridization()))
            chiral_tags.add(str(atom.GetChiralTag()))

            neighbors = [nbr.GetSymbol() for nbr in atom.GetNeighbors()]
            bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
            bond_types |= set(bonds)
            max_nbrs = max(max_nbrs, len(neighbors))

        return num_atoms, atom_syms, hybridizations, chiral_tags, bond_types, max_nbrs
    except:
        pass

def welford_update(x, n_total, mean, M2):
    n = x.shape[0]
    batch_mean = x.mean(dim=0)
    delta = batch_mean - mean
    mean += delta * (n / (n_total + n))
    M2 += ((x - mean)**2).sum(dim=0)
    n_total += n
    return n_total, mean, M2

def compute_mean_std_welford(dataset):
    n_total = 0
    mean = None
    M2 = None

    for data in dataset:
        x = data.x
        if mean is None:
            mean = torch.zeros(x.shape[1])
            M2 = torch.zeros(x.shape[1])
        n_total, mean, M2 = welford_update(x, n_total, mean, M2)

    variance = M2 / (n_total - 1)
    std = torch.sqrt(variance)
    std[std == 0] = 1.0
    return mean, std

def list_smiles2symbols_hybridization_chiraltype(smi_list, num_workers=int(cpu_count()/4)):
    
    """
    Parallel function to extract:
    - max num atoms
    - atom symbols
    - hybridizations
    - chiral types
    - bond types
    - max neighbors

    Args:
        smi_list (List[str]): List of SMILES strings.
        num_workers (int): Number of parallel workers.

    Returns:
        Dict -> results
    """
    all_hybridization = Chem.rdchem.HybridizationType.__dict__['names'].keys()
    print('-'*80)
    print(f'All possible hybridizations are: {all_hybridization}')
    print(f'All possible chiraltypes are: {list(Chem.rdchem.ChiralType.__dict__['names'].keys())}')
    print('-'*80)
    with Pool(num_workers) as pool:
        results = list(pool.map(extract_features_from_smiles, smi_list))

    atom_symbols = set(['Br', 'Cl', 'P', 'I', 'F', 'H', 'S', 'N', 'O', 'C'])
    hybridizations = {'UNSPECIFIED', 'S'}  # Pre-set
    chiraltypes = set()
    bond_types = set()
    max_neighbors = 0
    max_num_atoms = 0

    for natoms, syms, hybs, chirs, bonds, max_nbr in results:
        atom_symbols |= syms
        hybridizations |= hybs
        chiraltypes |= chirs
        bond_types |= bonds
        max_neighbors = max(max_neighbors, max_nbr)
        max_num_atoms = max(max_num_atoms, natoms)

    atom_symbols_sorted = sorted(atom_symbols, key=len, reverse=True)

    results = {'max_num_atoms': max_num_atoms+1,
               'atom_symbols': atom_symbols_sorted,
               'hybridization_list': list(hybridizations),
               'chiraltypes': list(chiraltypes),
               'bonds_list': list(bond_types),
               'neighbors_max_len': max_neighbors
               }
    
    return results


def list_to_onehot(element, elements_list, val=1.0) -> list:
    
    onehot = [0.]*len(elements_list)
    idx = elements_list.index(element)
    onehot[idx] = val
    
    return onehot

def list_to_onehot_oov(element, elements_list, val=1.0):
    """
    One-hot encoder with an OOV bucket.
    - Boolean vector if val is 0 or 1.
    - Float vector otherwise.
    """

    size = len(elements_list)
    use_bool = float(val) in (0.0, 1.0)
    onehot = [False] * (size + 1) if use_bool else [0.0] * (size + 1)

    try:
        idx = elements_list.index(element)
    except:
        idx = size
    onehot[idx] = bool(val) if use_bool else val

    return onehot


## Use for CLint prediction (Old)
def get_element_properties(symbol):
    
    data_elec = {
        "H": {"VdW Radius": 120, "Sanderson Electronegativity": 2.592, "Polarizability": 0.667, "Pauling Electronegativity": 2.2},
        "B": {"VdW Radius": 191, "Sanderson Electronegativity": 2.275, "Polarizability": 3.030, "Pauling Electronegativity": 2.04},
        "C": {"VdW Radius": 177, "Sanderson Electronegativity": 2.746, "Polarizability": 1.760, "Pauling Electronegativity": 2.55},
        "N": {"VdW Radius": 166, "Sanderson Electronegativity": 3.194, "Polarizability": 1.100, "Pauling Electronegativity": 3.04},
        "O": {"VdW Radius": 150, "Sanderson Electronegativity": 3.654, "Polarizability": 0.802, "Pauling Electronegativity": 3.44},
        "F": {"VdW Radius": 146, "Sanderson Electronegativity": 4.000, "Polarizability": 0.557, "Pauling Electronegativity": 3.98},
        "Al": {"VdW Radius": 225, "Sanderson Electronegativity": 1.714, "Polarizability": 6.800, "Pauling Electronegativity": 1.61},
        "Si": {"VdW Radius": 219, "Sanderson Electronegativity": 2.138, "Polarizability": 5.380, "Pauling Electronegativity": 1.9},
        "P": {"VdW Radius": 190, "Sanderson Electronegativity": 2.515, "Polarizability": 3.630, "Pauling Electronegativity": 2.19},
        "S": {"VdW Radius": 189, "Sanderson Electronegativity": 2.957, "Polarizability": 2.900, "Pauling Electronegativity": 2.58},
        "Cl": {"VdW Radius": 182, "Sanderson Electronegativity": 3.475, "Polarizability": 2.180, "Pauling Electronegativity": 3.16},
        "Fe": {"VdW Radius": 244, "Sanderson Electronegativity": 2.200, "Polarizability": 8.400, "Pauling Electronegativity": 1.83},
        "Co": {"VdW Radius": 240, "Sanderson Electronegativity": 2.560, "Polarizability": 7.500, "Pauling Electronegativity": 1.88},
        "Ni": {"VdW Radius": 240, "Sanderson Electronegativity": 1.940, "Polarizability": 6.800, "Pauling Electronegativity": 1.91},
        "Cu": {"VdW Radius": 238, "Sanderson Electronegativity": 1.980, "Polarizability": 6.100, "Pauling Electronegativity": 1.9},
        "Zn": {"VdW Radius": 239, "Sanderson Electronegativity": 2.223, "Polarizability": 7.100, "Pauling Electronegativity": 1.65},
        "Br": {"VdW Radius": 186, "Sanderson Electronegativity": 3.219, "Polarizability": 3.050, "Pauling Electronegativity": 2.96},
        "Sn": {"VdW Radius": 242, "Sanderson Electronegativity": 1.490, "Polarizability": 7.700, "Pauling Electronegativity": 1.96},
        "I": {"VdW Radius": 204, "Sanderson Electronegativity": 2.778, "Polarizability": 5.350, "Pauling Electronegativity": 2.66},
        "Na": {"VdW Radius": 250, "Sanderson Electronegativity": 0.56, "Polarizability": 24.11, "Pauling Electronegativity": 0.93},
        "Ca": {"VdW Radius": 262, "Sanderson Electronegativity": 0.950, "Polarizability": 23.70, "Pauling Electronegativity": 1.00},
        "Se": {"VdW Radius": 182, "Sanderson Electronegativity": 3.01, "Polarizability": 4.45, "Pauling Electronegativity": 2.55},
        "K": {"VdW Radius": 273, "Sanderson Electronegativity": 0.45, "Polarizability": 43.40, "Pauling Electronegativity": 0.82},
        
    }
    
    if symbol in data_elec:
        return data_elec[symbol]
    else:
        print(f"Element '{symbol}' not found in the data.")
        
## Updated few values (New)        
def get_element_properties_updated(symbol):
    
    data_elec = {
        "H": {"VdW Radius": 120, "Sanderson Electronegativity": 2.592, "Polarizability": 4.5071, "Pauling Electronegativity": 2.2},
        "B": {"VdW Radius": 191, "Sanderson Electronegativity": 2.275, "Polarizability": 20.5, "Pauling Electronegativity": 2.04},
        "C": {"VdW Radius": 177, "Sanderson Electronegativity": 2.746, "Polarizability": 11.3, "Pauling Electronegativity": 2.55},
        "N": {"VdW Radius": 166, "Sanderson Electronegativity": 3.194, "Polarizability": 7.4, "Pauling Electronegativity": 3.04},
        "O": {"VdW Radius": 150, "Sanderson Electronegativity": 3.654, "Polarizability": 5.3, "Pauling Electronegativity": 3.44},
        "F": {"VdW Radius": 146, "Sanderson Electronegativity": 4.000, "Polarizability": 3.74, "Pauling Electronegativity": 3.98},
        "Na": {"VdW Radius": 250, "Sanderson Electronegativity": 0.56, "Polarizability": 162.7, "Pauling Electronegativity": 0.93},
        "Al": {"VdW Radius": 225, "Sanderson Electronegativity": 1.714, "Polarizability": 57.8, "Pauling Electronegativity": 1.61},
        "Si": {"VdW Radius": 219, "Sanderson Electronegativity": 2.138, "Polarizability": 37.3, "Pauling Electronegativity": 1.9},
        "P": {"VdW Radius": 190, "Sanderson Electronegativity": 2.515, "Polarizability": 25, "Pauling Electronegativity": 2.19},
        "S": {"VdW Radius": 189, "Sanderson Electronegativity": 2.957, "Polarizability": 19.4, "Pauling Electronegativity": 2.58},
        "Cl": {"VdW Radius": 182, "Sanderson Electronegativity": 3.475, "Polarizability": 14.6, "Pauling Electronegativity": 3.16},
        "K": {"VdW Radius": 273, "Sanderson Electronegativity": 0.45, "Polarizability": 289.7, "Pauling Electronegativity": 0.82},
        "Ca": {"VdW Radius": 262, "Sanderson Electronegativity": 0.950, "Polarizability": 160.8, "Pauling Electronegativity": 1.00},
        "Fe": {"VdW Radius": 244, "Sanderson Electronegativity": 2.200, "Polarizability": 62.0, "Pauling Electronegativity": 1.83},
        "Co": {"VdW Radius": 240, "Sanderson Electronegativity": 2.560, "Polarizability": 55.0, "Pauling Electronegativity": 1.88},
        "Ni": {"VdW Radius": 240, "Sanderson Electronegativity": 1.940, "Polarizability": 49.0, "Pauling Electronegativity": 1.91},
        "Cu": {"VdW Radius": 238, "Sanderson Electronegativity": 1.980, "Polarizability": 46.5, "Pauling Electronegativity": 1.9},
        "Zn": {"VdW Radius": 239, "Sanderson Electronegativity": 2.223, "Polarizability": 38.67, "Pauling Electronegativity": 1.65},
        "Se": {"VdW Radius": 182, "Sanderson Electronegativity": 3.01, "Polarizability": 28.9, "Pauling Electronegativity": 2.55},
        "Br": {"VdW Radius": 186, "Sanderson Electronegativity": 3.219, "Polarizability": 21.0, "Pauling Electronegativity": 2.96},
        "Sn": {"VdW Radius": 242, "Sanderson Electronegativity": 1.490, "Polarizability": 53.0, "Pauling Electronegativity": 1.96},
        "I": {"VdW Radius": 204, "Sanderson Electronegativity": 2.778, "Polarizability": 32.9, "Pauling Electronegativity": 2.66},
    }
    
    if symbol in data_elec:
        return data_elec[symbol]
    else:
        print(f"Element '{symbol}' not found in the data.")
        
        
def get_atom_properties(atom):
    
    ## Atom Properties from RDKit
    atom_properties = [
        atom.GetAtomicNum(),               
        atom.GetDegree(), 
        # atom.GetTotalDegree(),                 
        atom.GetFormalCharge(),            
        atom.GetTotalNumHs(),  
        # atom.GetTotalValence(),            
        atom.GetExplicitValence(),         
        atom.GetImplicitValence(),         
        atom.GetNumExplicitHs(),           
        atom.GetNumImplicitHs(),           
        atom.GetMass(),
        Chem.rdchem.GetNumPiElectrons(atom)
        # atom.GetNumRadicalElectrons()      
    ]   
    atom_properties = [float(prop) for prop in atom_properties]
    
    ## Some other molecular properties including Electronegetivity
    atom_properties.extend(list(get_element_properties(atom.GetSymbol()).values()))
    
    return atom_properties


def get_properties_smiles(smiles):
    
    '''
        Given a single SMILES it gives the feature representation of that smiles
    '''
    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol=smiles
    ComputeGasteigerCharges(mol)  # Assign partial charges
    all_properties = [None]*mol.GetNumAtoms()
    if mol is None:
        print("Invalid SMILES string")
        return []

    # Iterate over atoms in the molecule and get properties
    for i, atom in enumerate(mol.GetAtoms()):
        atom_properties = get_atom_properties(atom)
        GasteigerCharge = float(atom.GetProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.
        atom_properties.append(GasteigerCharge)
        all_properties[i] = atom_properties
         
    # mol_H = Chem.AddHs(mol)
    # num_atoms = mol_H.GetNumAtoms() - mol.GetNumAtoms()
    # for i in range(num_atoms):
    #     all_properties.append([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.008, 0.0, 120, 2.592, 0.667, 2.2, 0])
         
    return all_properties


def get_3d_descriptors(smiles):
    
    """
    Computes all possible 3D descriptors for atoms and the molecule.

    Parameters:
        smiles (str): Input SMILES string.

    Returns:
        dict: A dictionary containing atomic and molecular 3D descriptors.
    """
    exclusion_flag = False
    atom_descriptors_all = []
    try:
        if isinstance(smiles, str):
            mol = Chem.MolFromSmiles(smiles)
        else:
            mol=smiles
        mol = Chem.AddHs(mol)  # Add hydrogens
        
        params = AllChem.ETKDG()
        params.randomSeed = 42  # Set a random seed
        params.maxAttempts = 100  # Increase the maximum attempts
        params.clearConfs = True  # Clear existing conformers
        params.useExpTorsionAnglePrefs = True  # Use experimental torsion-angle preferences
        params.useSmallRingTorsions = True  # Small ring torsion preferences
        embed = AllChem.EmbedMolecule(mol, params)  # 3D conformer generation
        
        if embed != 0:
            raise ValueError
        uff = AllChem.UFFOptimizeMolecule(mol)  # Optimize geometry
        
        # Compute atomic descriptors
        conformer = mol.GetConformer()
        for atom in mol.GetAtoms():
            # print(atom.GetSymbol())
            idx = atom.GetIdx()
            pos = conformer.GetAtomPosition(idx)
            atom_descriptors = [pos.x, pos.y, pos.z]
            atom_descriptors_all.append(atom_descriptors)
    except Exception as e:
        smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        # mol.RemoveAllConformers()
        exclusion_flag = True
        
    mol_H = Chem.RemoveHs(mol)
    num_atoms = mol.GetNumAtoms() - mol_H.GetNumAtoms()
        
    return atom_descriptors_all[:-num_atoms], exclusion_flag


def smiles_filter_3d_descriptor(smiles_list):
    
    exclude_list = []
    for smiles in smiles_list:
        _, flag = get_3d_descriptors(smiles)
        if flag:
            exclude_list.append(smiles)
    remain_smiles = [smi for smi in smiles_list if smi not in exclude_list]
    
    return remain_smiles, exclude_list
        

def get_atom_info_vector(smiles,
                         atom_symbols,
                         hybridization_list,
                         chiraltypes,
                         bonds_list
                         ):

    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.AddHs(mol)  # Add hydrogens
    num_atoms = mol.GetNumAtoms()
    # atom_descriptors_3d = [[]]*num_atoms
    atom_properties_mol = [[]]*num_atoms
    hybridization_atoms_list = [None]*num_atoms
    symbol_list = [None]*num_atoms
    aromaticity_list = [None]*num_atoms
    ring_list = [None]*num_atoms
    chirality_list = [None]*num_atoms
    bond_info = [None]*num_atoms
    neigbor_symbol_info = [None]*num_atoms
    neighbour_info_list = [None]*num_atoms
    logP_list = [None]*num_atoms
    
    
    if mol is None:
        print("Invalid SMILES string")
        return []
    
    # Iterate over atoms in the molecule and get properties
    for i, atom in enumerate(mol.GetAtoms()):
        aromatic = [0., 0.]
        ring = [0., 0.]
        bonds_onehot = [0.]*len(bonds_list)
        neighbor_onehot = [0.]*len(atom_symbols)
        
        ## Atom Properties from RDKit
        onehot_symbol = list_to_onehot(atom.GetSymbol(), atom_symbols)
        symbol_list[i] = onehot_symbol
        
        ## Chirality Encoding
        chirality = list_to_onehot(str(atom.GetChiralTag()), chiraltypes)
        chirality_list[i] = chirality
            
        
        ## Hybridization Encoding
        hybridization = list_to_onehot(str(atom.GetHybridization()), hybridization_list)
        hybridization_atoms_list[i] = hybridization
        
        ## Aromaticity & Ring Information Encoding
        aromatic[int(atom.GetIsAromatic())] = 1.0 
        aromaticity_list[i] = aromatic
            
        ring[int(atom.IsInRing())] = 1.0
        ring_list[i] = ring
        
        logP_list[i] = [get_logP_from_csv(logp_dict=logP_dict, mol=mol, atom_idx=i)]
        # logP_list[i] = Atom_logP_Value(mol, i)
            
        ##Neighbour Symbols Encoding
        all_neighbours = [neighbor.GetSymbol() for neighbor in atom.GetNeighbors()]
        all_bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
        bond_value_dict = dict({'SINGLE':1, 'DOUBLE':2, 'TRIPLE':3, 'AROMATIC':4})
        all_neighbours_counts = Counter(all_neighbours)
        temp_matrix = np.zeros((len(atom_symbols), len(bonds_list)))
        for j in range(len(all_neighbours)):
            temp_matrix[(atom_symbols.index(all_neighbours[j]), bonds_list.index(all_bonds[j]))] = bond_value_dict[all_bonds[j]]*float(all_neighbours_counts.get(all_neighbours[j]))
        neighbour_info_list[i] = list(temp_matrix.flatten())
        
        ## Bond Information Encoding
        all_bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
        all_bonds_counts = Counter(all_bonds)
        bond_dict = {bond:all_bonds_counts.get(bond) for bond in all_bonds}
        for bond in all_bonds:
            bonds_onehot[bonds_list.index(bond)] = float(bond_dict[bond])
        bond_info[i] = bonds_onehot
            
        ##Neighbour Symbols Encoding
        all_neighbours = [neighbor.GetSymbol() for neighbor in atom.GetNeighbors()]
        all_neighbours_counts = Counter(all_neighbours)
        for neigh in all_neighbours:
            neighbor_onehot[atom_symbols.index(neigh)] = float(all_neighbours_counts.get(neigh))#1.0
        neigbor_symbol_info[i] = neighbor_onehot
            
   
    ## Atom 2D Properties 
    atom_properties_mol = get_properties_smiles(smiles)
     
    ## Atom 3D Properties
    # atom_descriptors_3d, _ = get_3d_descriptors(smiles=smiles)
    # print(logP_list, ring_list)
    atom_feature_vector = {
        'symbol':np.array(symbol_list),
        'rdkit_2d_atom_prop':np.array(atom_properties_mol)[:, :10],
        'electronegetivity_infos':np.array(atom_properties_mol)[:, 10:-1],
        'gasteiger_charge':np.array(atom_properties_mol)[:, -1:],
        'atom_properties':np.array(atom_properties_mol),
        'hybridization':np.array(hybridization_atoms_list),
        'aromaticity':np.array(aromaticity_list),
        'ring':np.array(ring_list),
        'logP_values':np.array(logP_list),
        'chirality':np.array(chirality_list),
        'bond_info':np.array(bond_info),
        'neigbor_symbol_info':np.array(neigbor_symbol_info),
        # 'coordinates':np.array(atom_descriptors_3d)[:, 1:],
        # '3d_descriptors':np.array(atom_descriptors_3d),
        'neighbour_info':np.array(neighbour_info_list)
    }
    # atom_descs = ['AtomicNum', 'Degree', 'FormalCharge', 'TotalNumHs', 'ExplicitValence', 'ImplicitValence',
    #               'NumExplicitHs', 'NumImplicitHs', 'Mass', 'NumPiElectrons',
    #               'VdW_Radius', 'Sanderson_Electronegativity', 'Polarizability', 'Pauling_Electronegativity',
    #               'GasteigerCharge', 'LogP_Value']
    
    return atom_feature_vector#, smi_feature_size


def get_atom_info_vector_with_embeddings(smiles,
                         atom_symbols_embedding,
                         atom_symbols,
                         bond_list,
                         hybridization_embedding,
                         chiraltypes_embedding,
                         ring_embedding,
                         aromatic_embedding,
                         feature_list=['atom_properties', 'hybridization', 'aromaticity', 'ring', 'logP_values']
                         ):

    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.AddHs(mol)  # Add hydrogens
    num_atoms = mol.GetNumAtoms()
    atom_descriptors_3d = [[]]*num_atoms
    atom_properties_mol = [[]]*num_atoms
    hybridization_atoms_list = [None]*num_atoms
    symbol_list = [None]*num_atoms
    aromaticity_list = [None]*num_atoms
    ring_list = [None]*num_atoms
    chirality_list = [None]*num_atoms
    logP_list = [None]*num_atoms
    
    
    if mol is None:
        print("Invalid SMILES string")
        return []
    
    # Iterate over atoms in the molecule and get properties
    for i, atom in enumerate(mol.GetAtoms()):
        
        ## Atom Properties from RDKit
        symbol_list[i] = list(atom_symbols).index(atom.GetSymbol())
        
        ## Chirality Encoding
        chirality_list[i] = int(atom.GetChiralTag())
        
        ## Hybridization Encoding
        hybridization_atoms_list[i] = int(atom.GetHybridization())
        
        ## Aromaticity & Ring Information Encoding
        aromaticity_list[i] = int(atom.GetIsAromatic())
        ring_list[i] = int(atom.IsInRing())
        
        logP_list[i] = [get_logP_from_csv(logp_dict=logP_dict, mol=mol, atom_idx=i)]
        # logP_list[i] = Atom_logP_Value(mol, i)
        
        
    # Embedded Hybridization Form
    hybridization_atoms_list = hybridization_embedding(torch.tensor(hybridization_atoms_list))
    hybridization_atoms_list = hybridization_atoms_list.detach().numpy()
    # Embedded Atom Symbols Form
    symbol_list = atom_symbols_embedding(torch.tensor(symbol_list))
    symbol_list = symbol_list.detach().numpy()
    # Embedded Chiral Types Form
    chirality_list = chiraltypes_embedding(torch.tensor(chirality_list))
    chirality_list = chirality_list.detach().numpy()
    # Embedded Aromatic Form
    aromaticity_list = aromatic_embedding(torch.tensor(aromaticity_list))
    aromaticity_list = aromaticity_list.detach().numpy()
    # Embedded Ring Form
    ring_list = ring_embedding(torch.tensor(ring_list))
    ring_list = ring_list.detach().numpy()
   
    ## Atom 2D Properties 
    atom_properties_mol = get_properties_smiles(smiles)
     
    ## Atom 3D Properties
    atom_descriptors_3d, _ = get_3d_descriptors(smiles=smiles)
    # print(logP_list, ring_list)
    atom_feature_vector = {
        'symbol':np.array(symbol_list),
        'rdkit_2d_atom_prop':np.array(atom_properties_mol)[:, :10],
        'electronegetivity_infos':np.array(atom_properties_mol)[:, 10:],
        'atom_properties':np.array(atom_properties_mol),
        'hybridization':np.array(hybridization_atoms_list),
        'aromaticity':np.array(aromaticity_list),
        'ring':np.array(ring_list),
        'logP_values':np.array(logP_list),
        'chirality':np.array(chirality_list),
        'gasteiger_charge':np.array(atom_descriptors_3d)[:, :1],
        'coordinates':np.array(atom_descriptors_3d)[:, 1:],
        '3d_descriptors':np.array(atom_descriptors_3d),
    }
    
    return atom_feature_vector#, smi_feature_size


def get_bond_matrix(smiles, weighted_flag = True):
        
    mol = Chem.MolFromSmiles(smiles)
    
    num_atoms = mol.GetNumAtoms()
    bond_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
    bond_matrix = Chem.GetAdjacencyMatrix(mol)
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        
        if bond_type == Chem.BondType.SINGLE:
            bond_matrix[i, j] = 1
            bond_matrix[j, i] = 1
        elif bond_type == Chem.BondType.DOUBLE:
            bond_matrix[i, j] = 2 if weighted_flag else 1
            bond_matrix[j, i] = 2 if weighted_flag else 1
        elif bond_type == Chem.BondType.TRIPLE:
            bond_matrix[i, j] = 3 if weighted_flag else 1
            bond_matrix[j, i] = 3 if weighted_flag else 1
        elif bond_type == Chem.BondType.AROMATIC:
            bond_matrix[i, j] = 4 if weighted_flag else 1
            bond_matrix[j, i] = 4 if weighted_flag else 1
    
    return bond_matrix


def get_multirelational_bond_matrix(smiles, bonds_list, weighted_flag=True, include_self_loop = True):
    """
    Generates a multi-relational bond matrix for a given molecule.
    
    Args:
        smiles (str): SMILES representation of the molecule.
        bonds_list (list): List of bond types as strings, e.g., ["SINGLE", "DOUBLE"].
        weighted_flag (bool): Whether to assign numeric weights based on bond order.

    Returns:
        np.ndarray: A [num_relations, num_atoms, num_atoms] bond matrix.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    num_atoms = mol.GetNumAtoms()
    num_relations = len(bonds_list)

    if include_self_loop:
        bond_matrix = np.array([np.eye(num_atoms, num_atoms) for _ in range(num_relations)], dtype=float)
    else:
        bond_matrix = np.zeros((num_relations, num_atoms, num_atoms), dtype=float)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_type_str = str(bond.GetBondType())  # E.g., "SINGLE", "DOUBLE"
        if bond_type_str not in bonds_list:
            continue

        relation_idx = bonds_list.index(bond_type_str)
        
        # Use numeric weight (1.0, 2.0, etc.) or binary
        weight = float(bond.GetBondTypeAsDouble()) if weighted_flag else 1.0

        bond_matrix[relation_idx, i, j] = weight
        bond_matrix[relation_idx, j, i] = weight

    return torch.tensor(bond_matrix)

class SMILEStoPyGGraphDataset(pyg_dataset):
    def __init__(self,
                 smi_list,
                 labels,
                 atom_symbols,
                 hybridization_list,
                 chiraltypes,
                 bonds_list,
                 features_list=['atom_properties', 'hybridization', 'aromaticity', 'ring', 'logP_values']
                 ):
        self.smi_list = smi_list
        self.labels = labels
        self.atom_symbols = atom_symbols
        self.hybridization_list = hybridization_list
        self.chiraltypes = chiraltypes
        self.bonds_list = bonds_list
        self.features_list = features_list
        
        # Compute mean/std internally (train dataset only ideally)
        self.mean, self.std = self._compute_feature_stats()

    # ==========================================================
    # Compute dataset-wide statistics
    # ==========================================================
    def _compute_feature_stats(self):

        all_features = []

        for smi in self.smi_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            atom_features = self.atom_vector(
                mol,
                atom_symbols=self.atom_symbols,
                hybridization_list=self.hybridization_list,
                chiraltypes=self.chiraltypes,
                bonds_list=self.bonds_list,
                features_list=self.features_list
            )

            atom_features = np.concatenate([atom_features[key] for key in self.features_list], axis=1)
            all_features.append(atom_features)

        stacked = np.vstack(all_features)

        mean = torch.tensor(stacked.mean(axis=0), dtype=torch.float32)
        std = torch.tensor(stacked.std(axis=0), dtype=torch.float32)

        std[std == 0] = 1.0  # avoid division by zero
        
        return mean, std

    def __len__(self):
        return len(self.smi_list)
    
    def get_smiles(self, idx):
        return self.smi_list[idx]
    
    def bond_features(self, bond):
        """Return bond feature vector."""
        bt = bond.GetBondType()
        return torch.as_tensor([
            int(bt == Chem.rdchem.BondType.SINGLE),
            int(bt == Chem.rdchem.BondType.DOUBLE),
            int(bt == Chem.rdchem.BondType.TRIPLE),
            int(bt == Chem.rdchem.BondType.AROMATIC)
        ], dtype=torch.float)
    
    def atom_vector(self, smiles,
                    atom_symbols,
                    hybridization_list,
                    chiraltypes,
                    bonds_list,
                    features_list=['atom_properties', 'hybridization', 'aromaticity', 'ring', 'logP_values'],
                    logP_dict=None):
        if isinstance(smiles, str):
            mol = Chem.MolFromSmiles(smiles)
        else:
            mol = smiles  # Assume it's already an RDKit Mol object
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            return {}

        num_atoms = mol.GetNumAtoms()
        symbol_to_idx = {s: i for i, s in enumerate(atom_symbols)}
        bond_to_idx = {b: i for i, b in enumerate(bonds_list)}
        bond_value_dict = {'SINGLE':1, 'DOUBLE':2, 'TRIPLE':3, 'AROMATIC':4}

        # === Preallocate arrays only for requested features ===
        feature_arrays = {}
        if 'symbol' in features_list:
            feature_arrays['symbol'] = np.zeros((num_atoms, len(atom_symbols)), dtype=np.float32)
        if 'hybridization' in features_list:
            feature_arrays['hybridization'] = np.zeros((num_atoms, len(hybridization_list)), dtype=np.float32)
        if 'chirality' in features_list:
            feature_arrays['chirality'] = np.zeros((num_atoms, len(chiraltypes)), dtype=np.float32)
        if 'aromaticity' in features_list:
            feature_arrays['aromaticity'] = np.zeros((num_atoms, 2), dtype=np.float32)
        if 'ring' in features_list:
            feature_arrays['ring'] = np.zeros((num_atoms, 2), dtype=np.float32)
        if 'logP_values' in features_list:
            feature_arrays['logP_values'] = np.zeros((num_atoms, 1), dtype=np.float32)
        if 'bond_info' in features_list:
            feature_arrays['bond_info'] = np.zeros((num_atoms, len(bonds_list)), dtype=np.float32)
        if 'neigbor_symbol_info' in features_list:
            feature_arrays['neigbor_symbol_info'] = np.zeros((num_atoms, len(atom_symbols)), dtype=np.float32)
        if 'neighbour_info' in features_list:
            feature_arrays['neighbour_info'] = np.zeros((num_atoms, len(atom_symbols)*len(bonds_list)), dtype=np.float32)
        if 'atom_properties' in features_list:
            atom_properties_mol = np.array(get_properties_smiles(smiles), dtype=np.float32)
            feature_arrays['atom_properties'] = atom_properties_mol
            feature_arrays['rdkit_2d_atom_prop'] = atom_properties_mol[:, :10]
            feature_arrays['electronegetivity_infos'] = atom_properties_mol[:, 10:-1]
            feature_arrays['gasteiger_charge'] = atom_properties_mol[:, -1:]

        # === Iterate over atoms ===
        for i, atom in enumerate(mol.GetAtoms()):
            if 'symbol' in features_list:
                feature_arrays['symbol'][i] = list_to_onehot(atom.GetSymbol(), atom_symbols)
            if 'chirality' in features_list:
                feature_arrays['chirality'][i] = list_to_onehot(str(atom.GetChiralTag()), chiraltypes)
            if 'hybridization' in features_list:
                feature_arrays['hybridization'][i] = list_to_onehot(str(atom.GetHybridization()), hybridization_list)
            if 'aromaticity' in features_list:
                feature_arrays['aromaticity'][i, int(atom.GetIsAromatic())] = 1.0
            if 'ring' in features_list:
                feature_arrays['ring'][i, int(atom.IsInRing())] = 1.0
            if 'logP_values' in features_list and logP_dict is not None:
                feature_arrays['logP_values'][i] = get_logP_from_csv(logp_dict=logP_dict, mol=mol, atom_idx=i)

            neighbors = atom.GetNeighbors()
            if neighbors:
                neighbor_symbols = [n.GetSymbol() for n in neighbors]
                bonds = [str(mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondType()) for n in neighbors]
                neighbor_count = Counter(neighbor_symbols)
                bond_count = Counter(bonds)

                if 'neigbor_symbol_info' in features_list:
                    for neigh in neighbor_count:
                        feature_arrays['neigbor_symbol_info'][i, symbol_to_idx[neigh]] = float(neighbor_count[neigh])
                if 'bond_info' in features_list:
                    for b in bond_count:
                        feature_arrays['bond_info'][i, bond_to_idx[b]] = float(bond_count[b])
                if 'neighbour_info' in features_list:
                    temp_matrix = np.zeros((len(atom_symbols), len(bonds_list)), dtype=np.float32)
                    for nsym, bond_type in zip(neighbor_symbols, bonds):
                        temp_matrix[symbol_to_idx[nsym], bond_to_idx[bond_type]] = bond_value_dict[bond_type] * neighbor_count[nsym]
                    feature_arrays['neighbour_info'][i] = temp_matrix.flatten()

        return feature_arrays


    def mol_to_graph(self, smiles, add_Hs = False):
        """Convert a SMILES string into a PyG Data graph."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add hydrogens (optional)
        if add_Hs:
            mol = Chem.AddHs(mol)

        # Node features
        atom_features = self.atom_vector(mol, 
                                         atom_symbols=self.atom_symbols,
                                         hybridization_list=self.hybridization_list,
                                         chiraltypes=self.chiraltypes,
                                         bonds_list=self.bonds_list,
                                         features_list=self.features_list
                                         )
        atom_features = np.concatenate([atom_features[key] for key in self.features_list], axis=1)
        x = torch.as_tensor(atom_features, dtype=torch.float)
        x = (x - self.mean) / self.std

        # x = torch.stack([atom_vector(atom) for atom in mol.GetAtoms()])

        # Edges
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            f = self.bond_features(bond)

            edge_index.append([i, j])
            edge_index.append([j, i])  # undirected
            edge_attr.append(f)
            edge_attr.append(f)

        edge_index = torch.as_tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr) if len(edge_attr) > 0 else None

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data

    def __getitem__(self, idx):
        smiles = self.smi_list[idx]
        graph = self.mol_to_graph(smiles)
        
        if graph is None:
            return None
        if self.labels is not None:
            graph.labels = torch.as_tensor([self.labels[idx]], dtype=torch.float)
        return graph

class GraphDataset(Dataset):
    def __init__(self, smi_list, labels, max_num_atoms, atom_symbols, hybridization_list, chiraltypes, bonds_list):
        self.smi_list = smi_list
        self.labels = labels
        self.max_num_atoms = max_num_atoms
        self.atom_symbols = atom_symbols
        self.hybridization_list = hybridization_list
        self.chiraltypes = chiraltypes
        self.bonds_list = bonds_list
    

    def __len__(self):
        return len(self.smi_list)

    def __getitem__(self, idx):
        
        smi = self.smi_list[idx]
        atom_feature_vector = get_atom_info_vector(smi,
                                                 atom_symbols=self.atom_symbols,
                                                 hybridization_list=self.hybridization_list,
                                                 chiraltypes=self.chiraltypes,
                                                 bonds_list=self.bonds_list
                                                 )
        adj_matrix_temp = get_bond_matrix(smi)#rdmolops.GetAdjacencyMatrix(Chem.MolFromSmiles(smi))
        adjacency_matrix = torch.eye(self.max_num_atoms, self.max_num_atoms)
        adjacency_matrix[:adj_matrix_temp.shape[0], :adj_matrix_temp.shape[1]] += adj_matrix_temp
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
        y  = self.labels[idx]
        
        return smi, atom_feature_vector, adjacency_matrix, degree_matrix, y
    
    
class Multirelational_GraphDataset(Dataset):
    def __init__(self,
                 smi_list,
                 labels,
                 max_num_atoms = 100,
                 atom_symbols = ['Br', 'Cl', 'F', 'I', 'C', 'N', 'O', 'H', 'S', 'P'],
                 bonds_list = ['SINGLE', 'AROMATIC', 'DOUBLE', 'TRIPLE'],
                 hybridization_list=['UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP2D', 'SP3D', 'SP3D2', 'OTHER'],
                 chiraltypes=['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW'],
                 bond_weight_flag=False,
                 include_self_loop = True,
                 **kwargs
                 ):
        self.smi_list = smi_list
        self.labels = labels
        self.max_num_atoms = max_num_atoms
        self.atom_symbols = atom_symbols
        self.hybridization_list = hybridization_list
        self.chiraltypes = chiraltypes
        self.bonds_list = bonds_list
        self.num_relations = len(bonds_list)
        self.bond_weight_flag = bond_weight_flag
        self.include_self_loop = include_self_loop
    
    def __len__(self):
        return len(self.smi_list)

    def _get_one(self, idx):
        
        smi = self.smi_list[idx]
        atom_feature_vector, adjacency_tensor, degree_tensor = self.smi2feature(smi)
        y  = self.labels[idx]
        adjacency_tensor = adjacency_tensor.cpu().numpy()
        degree_tensor = degree_tensor.cpu().numpy()
        
        return smi, atom_feature_vector, adjacency_tensor, degree_tensor, y
    
    def __getitem__(self, idx):
        if isinstance(idx, (slice, list, tuple)):  
            if isinstance(idx, slice):
                indices = range(*idx.indices(len(self)))
                print(len(self))
            else:
                indices = idx
            items = [self._get_one(i) for i in indices]
            return self.collate_fn(items)
        return self._get_one(idx)
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate a list of (smi, atom_feature_dict, adjacency_matrix, degree_matrix, y)
        into numpy arrays (safe for multiprocessing).
        """
        smi_list, atom_feature_dicts, adjacency_tensors, degree_tensors, ys = zip(*batch)

        smi_list = list(smi_list)

        # --- collate atom feature dicts ---
        feature_keys = atom_feature_dicts[0].keys()
        atom_feature_vectors = {}
        for key in feature_keys:
            try:
                atom_feature_vectors[key] = np.stack([afd[key] for afd in atom_feature_dicts], axis=0).astype(np.float32)
            except ValueError:
                # if shapes differ across molecules, keep as list
                atom_feature_vectors[key] = [afd[key].astype(np.float32) for afd in atom_feature_dicts]

        # --- collate adjacency, degree, labels ---
        adjacency_tensors = np.array(adjacency_tensors, dtype=np.float32)
        degree_tensors    = np.array(degree_tensors, dtype=np.float32)
        ys                = np.array(ys, dtype=np.float32)
        return_vals = {'smiles':smi_list,
                       'atom_feature_vectors':atom_feature_vectors,
                       'adjacency_tensors':adjacency_tensors,
                       'degree_tensors':degree_tensors,
                       'labels':ys,
                       }
        
        return return_vals
    
    def smi2feature(self, smi):
        
        atom_feature_vector = get_atom_info_vector(smi,
                                                atom_symbols=self.atom_symbols,
                                                hybridization_list=self.hybridization_list,
                                                chiraltypes=self.chiraltypes,
                                                bonds_list=self.bonds_list
                                                )
        # adj_matrix_temp = get_bond_matrix(smi)#rdmolops.GetAdjacencyMatrix(Chem.MolFromSmiles(smi))
        adj_matrix_temp = get_multirelational_bond_matrix(smi,
                                                        self.bonds_list,
                                                        weighted_flag=self.bond_weight_flag,
                                                        include_self_loop = self.include_self_loop
                                                        )
        adjacency_tensor = torch.zeros((self.num_relations, self.max_num_atoms, self.max_num_atoms))
        adjacency_tensor[:adj_matrix_temp.shape[0], :adj_matrix_temp.shape[1], :adj_matrix_temp.shape[2]] = adj_matrix_temp # A = A + I
        degree_tensor = torch.stack([torch.diag(vector) for vector in adjacency_tensor.sum(axis=1)]) # D = D + I
        
        #**************************************************************************************
        ## To use D^(-1/2) A D^(-1/2) = D^(-1/2) A_1 D^(-1/2) + D^(-1/2) A_2 D^(-1/2) uncomment this part
        # identity_tensor = torch.stack([torch.eye(self.max_num_atoms, self.max_num_atoms) for _ in range(self.num_relations)])
        # degree_tensor = degree_tensor - identity_tensor
        # degree_tensor = degree_tensor.sum(axis=0)
        # temp_idt = torch.zeros_like(degree_tensor)
        # temp_idt[:adj_matrix_temp.shape[1], :adj_matrix_temp.shape[2]] += torch.eye(adj_matrix_temp.shape[1])
        # degree_tensor = degree_tensor + temp_idt
        #**************************************************************************************
        
        return atom_feature_vector, adjacency_tensor, degree_tensor
    
    @staticmethod
    def smiles2GcnDictData(
                    smi,
                    atom_symbols,
                    hybridization_list,
                    chiraltypes,
                    bonds_list,
                    max_num_atoms,
                    bond_weight_flag=False,
                    include_self_loop = True,
                    *args, **kwargs
                    ):
        num_relations = len(bonds_list)
        atom_feature_vector = get_atom_info_vector(smi,
                                                    atom_symbols=atom_symbols,
                                                    hybridization_list=hybridization_list,
                                                    chiraltypes=chiraltypes,
                                                    bonds_list=bonds_list
                                                    )

        adj_matrix_temp = get_multirelational_bond_matrix(
                                                        smi,
                                                        bonds_list,
                                                        weighted_flag=bond_weight_flag,
                                                        include_self_loop=include_self_loop
                                                        )
        adjacency_tensor = torch.zeros((num_relations, max_num_atoms, max_num_atoms))
        adjacency_tensor[ :adj_matrix_temp.shape[0], :adj_matrix_temp.shape[1], :adj_matrix_temp.shape[2]] = adj_matrix_temp
        degree_tensor = torch.stack([torch.diag(vector) for vector in adjacency_tensor.sum(axis=1)])
        
        return_vals = {'SMILES': smi,
                       'atom_feature_vector': atom_feature_vector,
                       'adjacency_matrix': adjacency_tensor.numpy(),
                       'degree_matrix': degree_tensor.numpy(),
                       }

        return return_vals
    
class Multirelational_GraphDataset_Embeddings(Dataset):
    def __init__(self, smi_list, labels, max_num_atoms,
                         atom_symbols_embedding,
                         bonds_list,
                         atom_symbols,
                         hybridization_embedding,
                         chiraltypes_embedding,
                         ring_embedding,
                         aromatic_embedding,
                         bond_weight_flag=False):
        self.smi_list = smi_list
        self.labels = labels
        self.max_num_atoms = max_num_atoms
        self.atom_symbols = atom_symbols
        self.atom_symbols_embedding = atom_symbols_embedding
        self.hybridization_embedding = hybridization_embedding
        self.chiraltypes_embedding = chiraltypes_embedding
        self.aromatic_embedding = aromatic_embedding
        self.bonds_list = bonds_list
        self.ring_embedding = ring_embedding
        self.num_relations = len(bonds_list)
        self.bond_weight_flag = bond_weight_flag
    
    def __len__(self):
        return len(self.smi_list)

    def _get_one(self, idx):
        
        smi = self.smi_list[idx]
        atom_feature_vector, adjacency_tensor, degree_tensor = self.smi2feature(smi)
        y  = self.labels[idx]
        
        ## Changing to numpy for consistency with Multiprocessing (To aviod "OSError: To many opening file")
        adjacency_tensor = adjacency_tensor.cpu().numpy()
        degree_tensor = degree_tensor.cpu().numpy()
        
        return smi, atom_feature_vector, adjacency_tensor, degree_tensor, y
    
    def __getitem__(self, idx):
        if isinstance(idx, (slice, list, tuple)):  
            if isinstance(idx, slice):
                indices = range(*idx.indices(len(self)))
                print(len(self))
            else:
                indices = idx
            items = [self._get_one(i) for i in indices]
            return self.collate_fn(items)
        return self._get_one(idx)
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate a list of (smi, atom_feature_dict, adjacency_matrix, degree_matrix, y)
        into numpy arrays (safe for multiprocessing).
        """
        smi_list, atom_feature_dicts, adjacency_tensors, degree_tensors, ys = zip(*batch)

        smi_list = list(smi_list)

        # --- collate atom feature dicts ---
        feature_keys = atom_feature_dicts[0].keys()
        atom_feature_vectors = {}
        for key in feature_keys:
            try:
                atom_feature_vectors[key] = np.stack([afd[key] for afd in atom_feature_dicts], axis=0).astype(np.float32)
            except ValueError:
                # if shapes differ across molecules, keep as list
                atom_feature_vectors[key] = [afd[key].astype(np.float32) for afd in atom_feature_dicts]

        # --- collate adjacency, degree, labels ---
        adjacency_tensors = np.array(adjacency_tensors, dtype=np.float32)
        degree_tensors    = np.array(degree_tensors, dtype=np.float32)
        ys                = np.array(ys, dtype=np.float32)
        return_vals = {'smiles':smi_list,
                            'atom_feature_vectors':atom_feature_vectors,
                            'adjacency_tensors':adjacency_tensors,
                            'degree_tensors':degree_tensors,
                            'labels':ys,
                            }
        
        return return_vals
    
    def smi2features(self, smi):
        
        atom_feature_vector = get_atom_info_vector_with_embeddings(smiles=smi,
                                                 atom_symbols_embedding=self.atom_symbols_embedding,
                                                 hybridization_embedding=self.hybridization_embedding,
                                                 chiraltypes_embedding=self.chiraltypes_embedding,
                                                 bond_list=self.bonds_list,
                                                 atom_symbols=self.atom_symbols,
                                                 ring_embedding=self.ring_embedding,
                                                 aromatic_embedding=self.aromatic_embedding
                                                 )
        # adj_matrix_temp = get_bond_matrix(smi)#rdmolops.GetAdjacencyMatrix(Chem.MolFromSmiles(smi))
        adj_matrix_temp = get_multirelational_bond_matrix(smi, self.bonds_list, weighted_flag=self.bond_weight_flag)
        adjacency_tensor = torch.stack([torch.eye(self.max_num_atoms, self.max_num_atoms) for _ in range(self.num_relations)])
        adjacency_tensor[:adj_matrix_temp.shape[0], :adj_matrix_temp.shape[1], :adj_matrix_temp.shape[2]] += adj_matrix_temp
        degree_tensor = torch.stack([torch.diag(vector) for vector in adjacency_tensor.sum(axis=1)])
        if not self.bond_weight_flag:
            mask = degree_tensor != 0.0
            degree_tensor[mask] = degree_tensor[mask] - 1
        
        return atom_feature_vector, adjacency_tensor, degree_tensor
            
def process_single_sample(data_point):
        
        smi, atom_feature_vector, adjacency_matrix, degree_matrix, y = data_point
        if atom_feature_vector is None:
            return None
        try:
            return {
                'SMILES': smi,
                'atom_feature_vector': atom_feature_vector,
                'adjacency_matrix': adjacency_matrix,
                'degree_matrix': degree_matrix,
                'label': y
            }     
        except:
            print(f'Error found at: {data_point}')
        
import resource  # for setting file limits on Unix-like systems
def limit_open_files(n=4096):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(n, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))  
         
def SmilesDataset_graph_gen(split, dataset, out_path = 'SmilesDataset_graph', mean_std_flag = False, num_workers=4, chunksize=1000, filetype='zip'):
    
    # with Pool(num_workers) as pool:
    #     print('in pool loop:')
    #     raw_data = [dataset[i] for i in tqdm(range(len(dataset)))]
    #     SmilesDataset_graph = list(tqdm(pool.imap(process_single_sample, raw_data, chunksize=chunksize),
    #                                     total=len(raw_data), desc=split))
        
    with Pool(num_workers) as pool:
        print('in pool loop:')
        SmilesDataset_graph = list(
                                tqdm(
                                    pool.imap(process_single_sample, (dataset[i] for i in range(len(dataset))), chunksize=chunksize),
                                    total=len(dataset),
                                    desc=split
                                )
                            )
    
    # Filter out None values
    # SmilesDataset_graph = [d for d in SmilesDataset_graph if d is not None]
    if filetype == 'zip':
        filename = out_path + '_' + split + '.pkl.gz'
        with gzip.open(filename, "wb") as outfile:
            pickle.dump(SmilesDataset_graph, outfile)
    if filetype == 'pkl':
        filename = out_path+'_'+split+'.pkl'
        with open(filename, "wb") as outfile:
            pickle.dump(SmilesDataset_graph, outfile)
    
    if mean_std_flag:
        _, atom_feature_vector, _, _, _ = dataset[0]
        features_keys = list(atom_feature_vector.keys())
        
        ## Subsampling for large datasets to reduce required time
        subset_size = min(50000, len(SmilesDataset_graph))
        sampled_idxs = torch.randperm(len(SmilesDataset_graph))[:subset_size]
        SmilesDataset_graph = [SmilesDataset_graph[i] for i in sampled_idxs]
        
        mean_dict, std_dict = dict({}), dict({})
        for key in features_keys:
            temp = [data['atom_feature_vector'][key] for data in SmilesDataset_graph]
            concatenated_features = np.concatenate(temp, axis=0)
            mean = concatenated_features.mean(axis=0)
            std = concatenated_features.std(axis=0)
            std[std==0] = 1.0
            mean_dict[key] = mean
            std_dict[key] = std
            
        filename = '/'.join(out_path.split('/')[:-1]+['dataset_mean.pkl'])
        with open(filename, "wb") as outfile:
            pickle.dump(mean_dict, outfile)
        filename = '/'.join(out_path.split('/')[:-1]+['dataset_std.pkl'])
        with open(filename, "wb") as outfile:
            pickle.dump(std_dict, outfile)
            
    # return None # SmilesDataset_graph, mean_dict, std_dict
    
class GraphData_from_pickle(Dataset):
    
    def __init__(self,
                 dataset_path,
                 features_list = ['symbol', 'atom_properties', 'hybridization', 'aromaticity', 'ring', 'chirality', '3d_descriptors', 'neighbour_info'],
                 max_num_atoms=65,
                 padding=True
                 ):
        self.features_list = features_list
        self.max_num_atoms = max_num_atoms
        self.padding = padding
        if dataset_path.endswith('.pkl'):
            with open(dataset_path, 'rb') as fp:
                self.dataset = pickle.load(fp)
        if dataset_path.endswith('.pkl.gz'):
            with gzip.open(dataset_path, 'rb') as fp:
                self.dataset = pickle.load(fp)
        
        filename = '/'.join(dataset_path.split('/')[:-1]+['dataset_mean.pkl'])
        with open(filename, 'rb') as fp:
            self.dataset_mean = pickle.load(fp)
        filename = '/'.join(dataset_path.split('/')[:-1]+['dataset_std.pkl'])
        with open(filename, 'rb') as fp:
            self.dataset_std = pickle.load(fp)
            
    def __len__(self):
        return len(self.dataset)

    def _get_one(self, idx):
        
        data=self.dataset[idx]
        feature_vector, adjacency_matrix, degree_matrix, y = self.smi2feature(data)
        
        return feature_vector, adjacency_matrix, degree_matrix, y
    
    def __getitem__(self, idx):
        if isinstance(idx, (slice, list, tuple)):  
            if isinstance(idx, slice):
                indices = range(*idx.indices(len(self)))
            else:
                indices = idx
            items = [self._get_one(i) for i in indices]
            return self.collate_fn(items)
        return self._get_one(idx)
    
    def collate_fn(self, batch):
        """
        Collate a list of (feature_vector, adjacency_matrix, degree_matrix, y)
        into batched tensors.
        """
        feature_vectors, adjacency_matrices, degree_matrices, ys = zip(*batch)

        # Convert to torch tensors
        feature_vectors   = [torch.as_tensor(fv, dtype=torch.float32) for fv in feature_vectors]
        adjacency_matrices = [torch.as_tensor(adj, dtype=torch.float32) for adj in adjacency_matrices]
        degree_matrices    = [torch.as_tensor(deg, dtype=torch.float32) for deg in degree_matrices]
        ys                 = [torch.as_tensor(y, dtype=torch.float32) for y in ys]

        # If feature vectors are already same shape → stack, else keep as list
        try:
            feature_vectors = torch.stack(feature_vectors, dim=0)
        except RuntimeError:
            pass  # keep list if shapes differ

        try:
            adjacency_matrices = torch.stack(adjacency_matrices, dim=0)
        except RuntimeError:
            pass

        try:
            degree_matrices = torch.stack(degree_matrices, dim=0)
        except RuntimeError:
            pass

        ys = torch.stack(ys, dim=0)

        return feature_vectors, adjacency_matrices, degree_matrices, ys
    
    def smi2feature(self, data):
        
        atom_feature_vector = data['atom_feature_vector']
        try:
            features = [atom_feature_vector[k] for k in self.features_list]
            mean = [self.dataset_mean[k] for k in self.features_list]
            std = [self.dataset_std[k] for k in self.features_list]
            
            # features = [v for k, v in atom_feature_vector.items() if k in self.features_list]
            # mean = [v for k, v in self.dataset_mean.items() if k in self.features_list]
            # std = [v for k, v in self.dataset_std.items() if k in self.features_list]
        except:
            raise AttributeError(f'Features should be in: {list(atom_feature_vector.keys())}')
        
        try:
            atom_feature_vector = np.concatenate(features, axis=1)
        except:
            raise ValueError(f'Error found in: SMILES - {data['SMILES']}, Features - {atom_feature_vector}')
        
         # Convert to tensor
        atom_feature_vector = torch.from_numpy(atom_feature_vector).float()
        
        # --- prepare mean/std tensors ---
        dataset_mean = torch.from_numpy(np.concatenate(mean)).float()
        dataset_std  = torch.from_numpy(np.concatenate(std)).float()
        
        # --- normalize and pad if needed ---
        if self.padding:
            feature_vector = torch.zeros((self.max_num_atoms, atom_feature_vector.shape[1]), dtype=torch.float32)
            feature_vector[:atom_feature_vector.shape[0], :] = (atom_feature_vector - dataset_mean) / dataset_std
        else:
            feature_vector = (atom_feature_vector - dataset_mean) / dataset_std
        
        # --- get adjacency, degree, label ---
        adjacency_matrix = torch.from_numpy(data['adjacency_matrix']).float()
        degree_matrix    = torch.from_numpy(data['degree_matrix']).float()
        y                = torch.tensor(data['label'], dtype=torch.float32)
        
        return feature_vector, adjacency_matrix, degree_matrix, y
    
    def get_smiles(self, idx):
        smi = self.dataset[idx]['SMILES']
        return smi
    
    def get_atom_info_vector(self, idx):
        
        data=self.dataset[idx]
        atom_feature_vector = data['atom_feature_vector']
        
        return atom_feature_vector
    
    @staticmethod
    def GcnDictData2GcnInput(data,
                            dataset_mean,
                            dataset_std,
                            max_num_atoms,
                            features_list=['symbol', 'atom_properties', 'hybridization', 'aromaticity', 'ring', 'chirality', 'neighbour_info'], #'3d_descriptors'
                            padding=True
                            ):
        atom_feature_vector = data['atom_feature_vector']

        try:
            features = [atom_feature_vector[k] for k in features_list]
            mean     = [dataset_mean[k] for k in features_list]
            std      = [dataset_std[k] for k in features_list]
        except KeyError:
            raise AttributeError(f'Features should be in: {list(atom_feature_vector.keys())}')

        try:
            atom_feature_vector = np.concatenate(features, axis=1)
        except Exception as e:
            raise ValueError(f"Error found in: SMILES - {data.get('SMILES', 'UNKNOWN')}, "f"Features - {atom_feature_vector}") from e

        atom_feature_vector = torch.from_numpy(atom_feature_vector).float()
        dataset_mean = torch.from_numpy(np.concatenate(mean)).float()
        dataset_std  = torch.from_numpy(np.concatenate(std)).float()

        if padding:
            feature_vector = torch.zeros((max_num_atoms, atom_feature_vector.shape[1]), dtype=torch.float32)
            feature_vector[:atom_feature_vector.shape[0], :] = ((atom_feature_vector - dataset_mean) / dataset_std)
        else:
            feature_vector = (atom_feature_vector - dataset_mean) / dataset_std

        adjacency_matrix = torch.from_numpy(data['adjacency_matrix']).float()
        degree_matrix    = torch.from_numpy(data['degree_matrix']).float()

        return feature_vector, adjacency_matrix, degree_matrix
    
class GraphData_from_pickle_3d_descriptor(Dataset):
    
    def __init__(self,
                 dataset_path,
                 features_list = ['symbol', 'atom_properties', 'hybridization', 'aromaticity', 'ring', 'chirality'],#, '3d_descriptors', 'neighbour_info'],
                 max_num_atoms=65
                 ):
        self.dataset = GraphData_from_pickle(dataset_path, features_list = features_list, max_num_atoms=max_num_atoms, padding=True)
    
    def get_smiles(self, idx):
        smi = self.dataset.get_smiles(idx)
        return smi
    
    def get_atom_info_vector(self, idx):
        
        atom_feature_vector = self.dataset.get_atom_info_vector(idx)
        return atom_feature_vector
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        feature_vector, adjacency_matrix, degree_matrix, y = self.dataset[idx]
        descriptors_3d = self.dataset.get_atom_info_vector(idx)['3d_descriptors']
        
        return feature_vector, adjacency_matrix, degree_matrix, descriptors_3d, y
    
    
def feature_representation(dataset,
                           model,
                           device='cpu',
                           input_params = ["feature_vector", "adjacency_tensor", "degree_tensor", "labels"]
                           ):
    
    '''
    dataset: A dataset with first entry of every datapoint is a SMILES
    '''
    
    # Determine the feature dimension from the model
    model.eval()
    feature_dim = model.feature_dim
    num_samples = len(dataset)

    # Initialize an array to hold all feature representations
    features = np.zeros((num_samples, feature_dim))
    labels = np.zeros((num_samples))
    smi_list = [None]*num_samples

    with torch.no_grad():  
        for i, data in tqdm(enumerate(dataset), total=len(dataset), desc=f'Feature Extraction: '):
            inp_data = {k: torch.tensor(v).to(device).unsqueeze(0) for k, v in zip(input_params, data)}
            targets = inp_data.pop('labels').squeeze()
            outputs = model.get_features(**inp_data)

            # Convert outputs to CPU for NumPy compatibility
            outputs_np = outputs.cpu().numpy()
            features[i] = outputs_np
            labels[i] = targets.cpu().numpy()
            smi_list[i] = dataset.get_smiles(i)

    return features, labels, smi_list

def feature_representation_pyg(dataloader,
                           model,
                           smi_list=None,
                           device='cpu',
                           input_params = ["x", "edge_index", "batch", "labels"]
                           ):
    
    '''
    dataset: A dataset with first entry of every datapoint is a SMILES
    '''
    
    # Determine the feature dimension from the model
    model.eval()
    feature_dim = model.feature_dim
    num_samples = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    print(f'Number of samples: {num_samples}, Batch size: {batch_size}, Number of batches: {len(dataloader)}')

    # Initialize an array to hold all feature representations
    features = np.zeros((num_samples, feature_dim))
    labels = np.zeros((num_samples))

    with torch.no_grad():  
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Feature Extraction: '):
            targets = data.pop('labels').to(device)
            try:
                inp_data = {k: data[k].to(device) for k in input_params if (k in data.keys()) and (data[k] is not None)}
            except:
                inp_data = {k: [temp_v.to(device) for temp_v in data[k] if temp_v is not None] for k in input_params if k in data.keys()}
            # print({k: v.shape for k, v in inp_data.items()})
            outputs = model.get_features(**inp_data)

            # Convert outputs to CPU for NumPy compatibility
            outputs_np = outputs.cpu().numpy()
            features[i*batch_size:(i+1)*batch_size] = outputs_np
            labels[i*batch_size:(i+1)*batch_size] = targets.cpu().numpy()
            
    return features, labels, smi_list


def gcn_pred_func(smi, gcn_model, rf_model, dataset_mean, dataset_std, features_list,
                  atom_symbols ,hybridization_list,
                  chiraltypes, bonds_list, max_num_atoms):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    atom_feature_vector = get_atom_info_vector(smi,
                                                atom_symbols=atom_symbols,
                                                hybridization_list=hybridization_list,
                                                chiraltypes=chiraltypes,
                                                bonds_list=bonds_list
                                                )
    # adj_matrix_temp = get_bond_matrix(smi)#rdmolops.GetAdjacencyMatrix(Chem.MolFromSmiles(smi))
    adj_matrix_temp = get_multirelational_bond_matrix(smi, bonds_list, weighted_flag=False)
    identity_tensor = torch.stack([torch.eye(max_num_atoms, max_num_atoms) for _ in range(len(bonds_list))])
    adjacency_tensor = identity_tensor.clone()
    adjacency_tensor[:adj_matrix_temp.shape[0], :adj_matrix_temp.shape[1], :adj_matrix_temp.shape[2]] += adj_matrix_temp
    degree_tensor = torch.stack([torch.diag(vector) for vector in adjacency_tensor.sum(axis=1)])

    #**************************************************************************************
    ## To get previous results comment this part
    ## To change to use D^(-1/2)AD^(-1/2) = D^(-1/2)A_1D^(-1/2) + D^(-1/2)A_2D^(-1/2)
    degree_tensor = degree_tensor - identity_tensor
    degree_tensor = degree_tensor.sum(axis=0)
    temp_idt = torch.zeros_like(degree_tensor)
    temp_idt[:adj_matrix_temp.shape[1], :adj_matrix_temp.shape[2]] += torch.eye(adj_matrix_temp.shape[1])
    degree_tensor = degree_tensor + temp_idt

    adjacency_tensor = adjacency_tensor.unsqueeze(0).to(device)
    degree_tensor = degree_tensor.unsqueeze(0).to(device)
    #**************************************************************************************

    features = [atom_feature_vector[k] for k in features_list]
    mean = [dataset_mean[k] for k in features_list]
    std = [dataset_std[k] for k in features_list]

    atom_feature_vector = np.concatenate(features, axis=1)
    atom_feature_vector = torch.from_numpy(atom_feature_vector)

    dataset_mean = np.concatenate(mean)
    dataset_std = np.concatenate(std)

    feature_vector = torch.zeros((max_num_atoms, atom_feature_vector.shape[-1]))
    feature_vector[:len(atom_feature_vector)] = (atom_feature_vector - dataset_mean)/dataset_std
    gcn_model, feature_vector= gcn_model.to(device), feature_vector.to(device)

    with torch.no_grad(): 
        input_vector = gcn_model.get_features(feature_vector, adjacency_tensor=adjacency_tensor, degree_tensor=degree_tensor)
        input_vector = input_vector.detach().cpu().numpy()    
    
    y_pred = rf_model.predict(input_vector).item()
    
    with torch.no_grad(): 
        y_pred = gcn_model(feature_vector, adjacency_tensor=adjacency_tensor, degree_tensor=degree_tensor).item()
    
    return y_pred


def filter_data(data, column_name, max_, min_, frac=0.5):
    
    train_df1 = data[data[column_name] <= max_]
    train_df1 = train_df1[train_df1[column_name] >= min_]
    train_df2 = data.loc[~data.index.isin(train_df1.index)]

    ticks = np.linspace(min_, max_, num=10)
    train_df3, _ = selective_range_data_sampling(train_df1, ticks=ticks, column= column_name, frac=frac)
    data = pd.concat([train_df3, train_df2])
    
    return data


def separate_data(df, data_percent=75):
    
    sorted_df = df.sort_values(by = 'Difference', ignore_index=True)
    index = int(len(sorted_df)*data_percent/100)
    selected_df = sorted_df.loc[:index]
    remaininig_df = sorted_df.loc[index:]
    
    return selected_df, remaininig_df


# network weight inilializer
class weights_initilizer():
    
    def __init__(self, network, initializer='xvr_unifrm'):
        
        if initializer=='default':
            pass
        elif initializer=='zeros':
            network.apply(self.weights_init_zeros)
        elif initializer=='ones':
            network.apply(self.weights_init_ones)
        elif initializer=='unifrm':
            network.apply(self.weights_init_uniform)
        elif initializer=='nrmal':
            network.apply(self.weights_init_normal)
        elif initializer=='xvr_unifrm':
            network.apply(self.weights_init_xavior_uniform)
        elif initializer=='xvr_nrmal':
            network.apply(self.weights_init_xavior_normal)
        else:
            raise ValueError('Define your parameters initializer: "{}"' .format(initializer))
    #------------------------------------------------------
    
    # weight initializer help function
    
    def weights_init_zeros(self, m):
        if((isinstance(m, nn.Linear)) or \
           (isinstance(m, nn.Conv1d)) or (isinstance(m, nn.ConvTranspose1d)) or \
           (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.ConvTranspose2d)) or \
           (isinstance(m, nn.Conv3d)) or (isinstance(m, nn.ConvTranspose3d)) or \
           (isinstance(m, nn.BatchNorm1d)) or \
           (isinstance(m, nn.BatchNorm2d)) or \
           (isinstance(m, nn.BatchNorm3d))):
            torch.nn.init.zeros_(m.weight)
            torch.nn.init.zeros_(m.bias)
            print('Weights "{}" initialized by "zeros" scheme' .format(m))
    #------------------------------------------------------
    
    def weights_init_ones(self, m):
        if((isinstance(m, nn.Linear)) or \
           (isinstance(m, nn.Conv1d)) or (isinstance(m, nn.ConvTranspose1d)) or \
           (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.ConvTranspose2d)) or \
           (isinstance(m, nn.Conv3d)) or (isinstance(m, nn.ConvTranspose3d)) or \
           (isinstance(m, nn.BatchNorm1d)) or \
           (isinstance(m, nn.BatchNorm2d)) or \
           (isinstance(m, nn.BatchNorm3d))):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
            print('Weights "{}" initialized by "ones" scheme' .format(m))
    #------------------------------------------------------
    
    def weights_init_uniform(self, m):
        if((isinstance(m, nn.Linear)) or \
           (isinstance(m, nn.Conv1d)) or (isinstance(m, nn.ConvTranspose1d)) or \
           (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.ConvTranspose2d)) or \
           (isinstance(m, nn.Conv3d)) or (isinstance(m, nn.ConvTranspose3d)) or \
           (isinstance(m, nn.BatchNorm1d)) or \
           (isinstance(m, nn.BatchNorm2d)) or \
           (isinstance(m, nn.BatchNorm3d))):
            torch.nn.init.uniform_(m.weight)
            torch.nn.init.uniform_(m.bias)
            print('Weights "{}" initialized by "uniform" scheme' .format(m))
    #------------------------------------------------------
    
    def weights_init_normal(self, m):
        if((isinstance(m, nn.Linear)) or \
           (isinstance(m, nn.Conv1d)) or (isinstance(m, nn.ConvTranspose1d)) or \
           (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.ConvTranspose2d)) or \
           (isinstance(m, nn.Conv3d)) or (isinstance(m, nn.ConvTranspose3d)) or \
           (isinstance(m, nn.BatchNorm1d)) or \
           (isinstance(m, nn.BatchNorm2d)) or \
           (isinstance(m, nn.BatchNorm3d))):
            torch.nn.init.normal_(m.weight)
            torch.nn.init.normal_(m.bias)
            print('Weights "{}" initialized by "normal_dist" scheme' .format(m))
    #------------------------------------------------------
    
    def weights_init_xavior_uniform(self, m):
        if((isinstance(m, nn.Linear)) or \
           (isinstance(m, nn.Conv1d)) or (isinstance(m, nn.ConvTranspose1d)) or \
           (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.ConvTranspose2d)) or \
           (isinstance(m, nn.Conv3d)) or (isinstance(m, nn.ConvTranspose3d)) or \
           (isinstance(m, nn.BatchNorm1d)) or \
           (isinstance(m, nn.BatchNorm2d)) or \
           (isinstance(m, nn.BatchNorm3d))):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias != None:
                torch.nn.init.zeros_(m.bias)
            print('Weights "{}" initialized by "xavior_uniform" scheme' .format(m))
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    torch.nn.init.orthogonal_(param.data)
                else:
                    torch.nn.init.normal_(param.data)
            print('Weights "{}" initialized by "orthogonal" scheme' .format(m))
    #------------------------------------------------------
    
    def weights_init_xavior_normal(self, m):
        if((isinstance(m, nn.Linear)) or \
           (isinstance(m, nn.Conv1d)) or (isinstance(m, nn.ConvTranspose1d)) or \
           (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.ConvTranspose2d)) or \
           (isinstance(m, nn.Conv3d)) or (isinstance(m, nn.ConvTranspose3d)) or \
        #    (isinstance(m, nn.BatchNorm1d)) or \
           (isinstance(m, nn.BatchNorm2d)) or \
           (isinstance(m, nn.BatchNorm3d))):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
            print('Weights "{}" initialized by "xavior_normal" scheme' .format(m))
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    torch.nn.init.orthogonal_(param.data)
                else:
                    torch.nn.init.normal_(param.data)
            print('Weights "{}" initialized by "orthogonal" scheme' .format(m))