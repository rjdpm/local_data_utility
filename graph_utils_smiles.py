from rdkit import Chem
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm

from logP_values import *

from rdkit.Chem import AllChem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem import Descriptors
from rdkit.Chem.Descriptors3D import (
    Asphericity, Eccentricity, InertialShapeFactor, NPR1, NPR2, PBF,
    PMI1, PMI2, PMI3, RadiusOfGyration, SpherocityIndex
)

import sys
sys.path.append('../../')
from data_utils_smiles import selective_range_data_sampling

__all__ = [
    'list_smiles2symbols_hybridization_chiraltype',
    'list_smiles2prop',
    'list_to_onehot',
    'get_element_properties',
    'get_atom_properties',
    'get_properties_smiles',
    'get_3d_descriptors',
    'smiles_filter_3d_descriptor',
    'get_atom_info_vector',
    'get_atom_info_vector_for_test_purpose',
    'get_bond_matrix',
    'get_multirelational_bond_matrix',
    'GraphDataset',
    'Multirelational_GraphDataset',
    'Multirelational_GraphDataset_Embeddings',
    'SmilesDataset_graph_gen',
    'GraphData_from_pickle',
    'GraphData_from_pickle_3d_descriptor',
    'feature_representation',
    'gcn_pred_func',
    'filter_data',
    'separate_data',
    'weights_initilizer'
]
## Molecules Representation
def list_smiles2symbols_hybridization_chiraltype(smi_list):
    
    all_hybridization = Chem.rdchem.HybridizationType.__dict__['names'].keys()
    print(f'All possible hybridizations are: {all_hybridization}')
    print(f'All possible chiraltypes are: {list(Chem.rdchem.ChiralType.__dict__['names'].keys())}')
    hybridization_val = {'UNSPECIFIED', 'S'}
    atom_symbols = set()
    chiraltypes = set()
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        for atom in mol.GetAtoms():
            hybridization_val = hybridization_val | {str(atom.GetHybridization())}
            atom_symbols = atom_symbols | {str(atom.GetSymbol())}
            chiraltypes = chiraltypes | {str(atom.GetChiralTag())}
    
    # list_hyb = [k for k, v in all_hybridization.items() if v in hybridization_val]    
    atom_symbols = sorted(list(atom_symbols), key=len, reverse=True)
    print(f'All atoms in the whole SMILES list are: {atom_symbols}')
    print(f'All hybridizations in the whole SMILES list are: {hybridization_val}')
    print(f'All chiraltypes in the whole SMILES list are: {chiraltypes}')
    
    return list(atom_symbols), list(hybridization_val), list(chiraltypes)


def list_smiles2prop(smi_list):
    
    neighbors_max_len = 0
    all_bonds = set()
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        for atom in mol.GetAtoms():
           neighbors = [neighbor.GetSymbol() for neighbor in atom.GetNeighbors()]
           bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
           all_bonds = all_bonds | set(bonds)
           if len(neighbors) > neighbors_max_len:
               neighbors_max_len = len(neighbors)
               
    return list(all_bonds), neighbors_max_len


def list_to_onehot(element, elements_list) -> list:
    
    onehot = [0.]*len(elements_list)
    idx = elements_list.index(element)
    onehot[idx] = 1.0
    
    return onehot


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
        
        
def get_atom_properties(atom):
    
    ## Atom Properties from RDKit
    atom_properties = [
        atom.GetAtomicNum(),               
        atom.GetDegree(),                  
        atom.GetFormalCharge(),            
        atom.GetTotalNumHs(),              
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
    
    mol = Chem.MolFromSmiles(smiles)
    ComputeGasteigerCharges(mol)  # Assign partial charges
    all_properties = [None]*mol.GetNumAtoms()
    if mol is None:
        print("Invalid SMILES string")
        return []

    # Iterate over atoms in the molecule and get properties
    for i, atom in enumerate(mol.GetAtoms()):
        GasteigerCharge = float(atom.GetProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.
        atom_properties = get_atom_properties(atom)
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
        mol = Chem.MolFromSmiles(smiles)
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
        
        with open('/home/rkmvu/Dataset/Coca-2/Experimental_Data_P_app/train_test_partition_literature/All_possible_atoms_X.pkl', 'rb') as fp:
            logP_dict = pickle.load(fp)
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
    
    return atom_feature_vector#, smi_feature_size


def get_atom_info_vector_with_embeddings(smiles,
                         atom_symbols_embedding,
                         atom_symbols,
                         bond_list,
                         hybridization_embedding,
                         chiraltypes_embedding,
                         ring_embedding,
                         aromatic_embedding,
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
        
        with open('/home/rkmvu/Dataset/Coca-2/Experimental_Data_P_app/train_test_partition_literature/All_possible_atoms_X.pkl', 'rb') as fp:
            logP_dict = pickle.load(fp)
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



def get_atom_info_vector_for_test_purpose(smiles,
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
    # symbol_list = [None]*num_atoms
    aromaticity_list = [None]*num_atoms
    ring_list = [None]*num_atoms
    # chirality_list = [None]*num_atoms
    # bond_info = [None]*num_atoms
    # neigbor_symbol_info = [None]*num_atoms
    # neighbour_info_list = [None]*num_atoms
    
    if mol is None:
        print("Invalid SMILES string")
        return []
    
    # Iterate over atoms in the molecule and get properties
    for i, atom in enumerate(mol.GetAtoms()):
        aromatic = [0., 0.]
        ring = [0., 0.]
        bonds_onehot = [0.]*len(bonds_list)
        neighbor_onehot = [0.]*len(atom_symbols)
        
        # ## Atom Properties from RDKit
        # onehot_symbol = list_to_onehot(atom.GetSymbol(), atom_symbols)
        # symbol_list[i] = onehot_symbol
        # # print('atom_symbols', len(atom_symbols))
        
        # ## Chirality Encoding
        # chirality = list_to_onehot(atom.GetChiralTag(), chiraltypes)
        # chirality_list[i] = chirality
        # # print('chirality', len(chirality))
            
        
        ## Hybridization Encoding
        hybridization = list_to_onehot(atom.GetHybridization(), hybridization_list)
        hybridization_atoms_list[i] = hybridization
        # print('hybridization', len(hybridization))
        
        ## Aromaticity & Ring Information Encoding
        aromatic[int(atom.GetIsAromatic())] = 1.0 
        aromaticity_list[i] = aromatic
        # print('aromatic', len(aromatic))
            
        ring[int(atom.IsInRing())] = 1.0
        ring_list[i] = ring
        # print('ring', len(ring))
        
            
        # ##Neighbour Symbols Encoding
        # all_neighbours = [neighbor.GetSymbol() for neighbor in atom.GetNeighbors()]
        # all_bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
        # bond_value_dict = dict({'SINGLE':1, 'DOUBLE':2, 'TRIPLE':3, 'AROMATIC':4})
        # all_neighbours_counts = Counter(all_neighbours)
        # temp_matrix = np.zeros((len(atom_symbols), len(bonds_list)))
        # for j in range(len(all_neighbours)):
        #     temp_matrix[(atom_symbols.index(all_neighbours[j]), bonds_list.index(all_bonds[j]))] = bond_value_dict[all_bonds[j]]*float(all_neighbours_counts.get(all_neighbours[j]))
        # neighbour_info_list[i] = list(temp_matrix.flatten())
        
        # ## Bond Information Encoding
        # all_bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
        # all_bonds_counts = Counter(all_bonds)
        # bond_dict = {bond:all_bonds_counts.get(bond) for bond in all_bonds}
        # for bond in all_bonds:
        #     bonds_onehot[bonds_list.index(bond)] = float(bond_dict[bond])
        # bond_info[i] = bonds_onehot
            
        # ##Neighbour Symbols Encoding
        # all_neighbours = [neighbor.GetSymbol() for neighbor in atom.GetNeighbors()]
        # all_neighbours_counts = Counter(all_neighbours)
        # for neigh in all_neighbours:
        #     neighbor_onehot[atom_symbols.index(neigh)] = float(all_neighbours_counts.get(neigh))#1.0
        # neigbor_symbol_info[i] = neighbor_onehot
            
   
    ## Atom 2D Properties 
    atom_properties_mol = get_properties_smiles(smiles)
     
    ## Atom 3D Properties
    # atom_descriptors_3d, _ = get_3d_descriptors(smiles=smiles)
    
    atom_feature_vector = {
        # 'symbol':np.array(symbol_list),
        'rdkit_2d_atom_prop':np.array(atom_properties_mol)[:, :10],
        'electronegetivity_infos':np.array(atom_properties_mol)[:, 10:],
        'atom_properties':np.array(atom_properties_mol),
        'hybridization':np.array(hybridization_atoms_list),
        'aromaticity':np.array(aromaticity_list),
        'ring':np.array(ring_list),
        # 'chirality':np.array(chirality_list),
        # 'bond_info':np.array(bond_info),
        # 'neigbor_symbol_info':np.array(neigbor_symbol_info),
        # 'gasteiger_charge':np.array(atom_descriptors_3d)[:, :1],
        # 'coordinates':np.array(atom_descriptors_3d)[:, 1:],
        # '3d_descriptors':np.array(atom_descriptors_3d),
        # 'neighbour_info':np.array(neighbour_info_list)
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


def get_multirelational_bond_matrix(smiles, bonds_list, weighted_flag=True):
    """
    Generates a multi-relational bond matrix for a given molecule.
    Args:
        smiles (str): SMILES representation of the molecule.
        bonds_list (list): List of bond types (as strings, e.g., "SINGLE", "DOUBLE").
        weighted_flag (bool): Whether to use bond type weights. Default is False.
    Returns:
        np.ndarray: A multi-relational bond matrix of shape [num_relations, num_atoms, num_atoms].
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Extract bond types
    names = Chem.rdchem.BondType.names
    values = Chem.rdchem.BondType.values

    # Create a dictionary mapping names to their corresponding values
    bond_type_dict = {name: value for name, value in zip(names.keys(), values)}
    num_relations = len(bonds_list)
    num_atoms = mol.GetNumAtoms()

    # Initialize bond matrix
    bond_matrix = np.zeros((num_relations, num_atoms, num_atoms), dtype=float)

    # Populate bond matrix
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = str(bond.GetBondType())

        if bond_type not in bonds_list:
            continue  # Skip bond types not in the list

        relation_idx = bonds_list.index(bond_type)
        weight = bond_type_dict[bond_type] if weighted_flag else 1
        bond_matrix[relation_idx, i, j] = weight
        bond_matrix[relation_idx, j, i] = weight

    return bond_matrix



class GraphDataset(Dataset):
    def __init__(self, smi_list, labels, max_num_atom, atom_symbols, hybridization_list, chiraltypes, bonds_list):
        self.smi_list = smi_list
        self.labels = labels
        self.max_num_atom = max_num_atom
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
        adjacency_matrix = torch.eye(self.max_num_atom, self.max_num_atom)
        adjacency_matrix[:adj_matrix_temp.shape[0], :adj_matrix_temp.shape[1]] += adj_matrix_temp
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
        y  = self.labels[idx]
        
        return smi, atom_feature_vector, adjacency_matrix, degree_matrix, y
    
    
class Multirelational_GraphDataset(Dataset):
    def __init__(self, smi_list, labels, max_num_atom, atom_symbols, hybridization_list, chiraltypes, bonds_list, bond_weight_flag=False):
        self.smi_list = smi_list
        self.labels = labels
        self.max_num_atom = max_num_atom
        self.atom_symbols = atom_symbols
        self.hybridization_list = hybridization_list
        self.chiraltypes = chiraltypes
        self.bonds_list = bonds_list
        self.num_relations = len(bonds_list)
        self.bond_weight_flag = bond_weight_flag
    
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
        # adj_matrix_temp = get_bond_matrix(smi)#rdmolops.GetAdjacencyMatrix(Chem.MolFromSmiles(smi))
        adj_matrix_temp = get_multirelational_bond_matrix(smi, self.bonds_list, weighted_flag=self.bond_weight_flag)
        identity_tensor = torch.stack([torch.eye(self.max_num_atom, self.max_num_atom) for _ in range(self.num_relations)])
        adjacency_tensor = identity_tensor.clone()
        adjacency_tensor[:adj_matrix_temp.shape[0], :adj_matrix_temp.shape[1], :adj_matrix_temp.shape[2]] += adj_matrix_temp
        degree_tensor = torch.stack([torch.diag(vector) for vector in adjacency_tensor.sum(axis=1)])
        
        #**************************************************************************************
        ## To get previous results comment this part
        ## To change to use D^(-1/2)A_D^(-1/2) = D^(-1/2)A_1D^(-1/2) + D^(-1/2)A_2D^(-1/2)
        degree_tensor = degree_tensor - identity_tensor
        degree_tensor = degree_tensor.sum(axis=0)
        temp_idt = torch.zeros_like(degree_tensor)
        temp_idt[:adj_matrix_temp.shape[1], :adj_matrix_temp.shape[2]] += torch.eye(adj_matrix_temp.shape[1])
        degree_tensor = degree_tensor + temp_idt
        #**************************************************************************************
        
        #**************************************************************************************
        # ## To get previous results uncomment this part
        # if not self.bond_weight_flag:
        #     mask = degree_tensor != 0.0
        #     degree_tensor[mask] = degree_tensor[mask] - 1
        #**************************************************************************************
        y  = self.labels[idx]
        
        return smi, atom_feature_vector, adjacency_tensor, degree_tensor, y
    
class Multirelational_GraphDataset_Embeddings(Dataset):
    def __init__(self, smi_list, labels, max_num_atom,
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
        self.max_num_atom = max_num_atom
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

    def __getitem__(self, idx):
        
        smi = self.smi_list[idx]
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
        adjacency_tensor = torch.stack([torch.eye(self.max_num_atom, self.max_num_atom) for _ in range(self.num_relations)])
        adjacency_tensor[:adj_matrix_temp.shape[0], :adj_matrix_temp.shape[1], :adj_matrix_temp.shape[2]] += adj_matrix_temp
        degree_tensor = torch.stack([torch.diag(vector) for vector in adjacency_tensor.sum(axis=1)])
        if not self.bond_weight_flag:
            mask = degree_tensor != 0.0
            degree_tensor[mask] = degree_tensor[mask] - 1
        y  = self.labels[idx]
        
        return smi, atom_feature_vector, adjacency_tensor, degree_tensor, y
            
            
def SmilesDataset_graph_gen(split, dataset, out_path = 'SmilesDataset_graph', mean_std_flag = False):
    
    SmilesDataset_graph = []
    _, atom_feature_vector, _, _, _ = dataset[0]
    features_keys = list(atom_feature_vector.keys())
    mean_dict, std_dict = dict({}), dict({})
    for (smi, atom_feature_vector, adjacency_matrix, degree_matrix, y) in tqdm(dataset, desc=split):
        if atom_feature_vector is None:
            continue
        storing_data = {
            'SMILES': smi,
            'atom_feature_vector': atom_feature_vector,
            'adjacency_matrix': adjacency_matrix,
            'degree_matrix': degree_matrix,
            'label': y
        }
        SmilesDataset_graph.append(storing_data)
        
    filename = out_path+'_'+split+'.pkl'
    with open(filename, "wb") as outfile:
        pickle.dump(SmilesDataset_graph, outfile)
    
    if mean_std_flag:
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
        with open(dataset_path, 'rb') as fp:
            self.dataset = pickle.load(fp)
        
        filename = '/'.join(dataset_path.split('/')[:-1]+['dataset_mean.pkl'])
        with open(filename, 'rb') as fp:
            self.dataset_mean = pickle.load(fp)
        filename = '/'.join(dataset_path.split('/')[:-1]+['dataset_std.pkl'])
        with open(filename, 'rb') as fp:
            self.dataset_std = pickle.load(fp)
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        data=self.dataset[idx]
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
            atom_feature_vector = torch.from_numpy(atom_feature_vector)
        except:
            raise ValueError(f'Error found in: Idx - {idx}, SMILES - {self.get_smiles(idx)}, Features - {atom_feature_vector}')
        
        dataset_mean = np.concatenate(mean)
        dataset_std = np.concatenate(std)
        if self.padding: 
            feature_vector = torch.zeros((self.max_num_atoms, atom_feature_vector.shape[-1]))
            feature_vector[:len(atom_feature_vector)] = (atom_feature_vector - dataset_mean)/dataset_std
        else:
            atom_feature_vector = (atom_feature_vector - dataset_mean)/dataset_std
            
        adjacency_matrix = data['adjacency_matrix']
        degree_matrix = data['degree_matrix']
        y  = data['label']
        
        return feature_vector, adjacency_matrix, degree_matrix, y
    
    def get_smiles(self, idx):
        smi = self.dataset[idx]['SMILES']
        return smi
    
    def get_atom_info_vector(self, idx):
        
        data=self.dataset[idx]
        atom_feature_vector = data['atom_feature_vector']
        
        return atom_feature_vector
    
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
    
    
def feature_representation(dataset, model, device):
    
    '''
    dataset: A dataset with first entry of every datapoint is a SMILES
    '''
    
    # Determine the feature dimension from the model
    feature_dim = model.feature_dim
    num_samples = len(dataset)

    # Initialize an array to hold all feature representations
    features = np.zeros((num_samples, feature_dim))
    labels = np.zeros((num_samples))
    smi_list = [None]*num_samples

    model.eval()
    with torch.no_grad():  
        for i, (*data, label) in enumerate(dataset):
            _inputs = [inp.to(device).unsqueeze(0) for inp in data] 
            outputs = model.get_features(*[x.type(torch.float32) for x in _inputs])

            # Convert outputs to CPU for NumPy compatibility
            outputs_np = outputs.cpu().numpy()
            features[i] = outputs_np
            labels[i] = label
            smi_list[i] = dataset.get_smiles(i)

    return features, labels, smi_list


def gcn_pred_func(smi, gcn_model, rf_model, dataset_mean, dataset_std, features_list,
                  atom_symbols ,hybridization_list,
                  chiraltypes, bonds_list, max_num_atom):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    atom_feature_vector = get_atom_info_vector(smi,
                                                atom_symbols=atom_symbols,
                                                hybridization_list=hybridization_list,
                                                chiraltypes=chiraltypes,
                                                bonds_list=bonds_list
                                                )
    # adj_matrix_temp = get_bond_matrix(smi)#rdmolops.GetAdjacencyMatrix(Chem.MolFromSmiles(smi))
    adj_matrix_temp = get_multirelational_bond_matrix(smi, bonds_list, weighted_flag=False)
    identity_tensor = torch.stack([torch.eye(max_num_atom, max_num_atom) for _ in range(len(bonds_list))])
    adjacency_tensor = identity_tensor.clone()
    adjacency_tensor[:adj_matrix_temp.shape[0], :adj_matrix_temp.shape[1], :adj_matrix_temp.shape[2]] += adj_matrix_temp
    degree_tensor = torch.stack([torch.diag(vector) for vector in adjacency_tensor.sum(axis=1)])


    #**************************************************************************************
    ## To get previous results comment this part
    ## To change to use D^(-1/2)A_D^(-1/2) = D^(-1/2)A_1D^(-1/2) + D^(-1/2)A_2D^(-1/2)
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

    feature_vector = torch.zeros((max_num_atom, atom_feature_vector.shape[-1]))
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
           (isinstance(m, nn.BatchNorm1d)) or \
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