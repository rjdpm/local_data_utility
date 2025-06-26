import re
import ast
import math
import copy
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from ordered_set import OrderedSet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KDTree, NearestNeighbors
from typing import Any, List, Dict, Tuple, Union, Set, Callable, Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdDepictor, AllChem, rdMolDescriptors, Descriptors, Crippen
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, AdditionalOutput
from mordred import Calculator, descriptors

import py3Dmol
from rdkit.Chem.Descriptors3D import (
    Asphericity, Eccentricity, InertialShapeFactor, NPR1, NPR2, PBF,
    PMI1, PMI2, PMI3, RadiusOfGyration, SpherocityIndex
)

import torch
import torch.nn as nn

__all__ = [  
    'canonicalize_smiles',
    'canonicalize_smiles_list',
    'randomize_smiles',
    'augment_smiles_with_labels',
    
    'draw_molecule',
    'plot_smiles_grid',
    'draw_mol_grid_with_legends',
    'plot_3d_molecule_with_labels',
    'smiles2morganbifinfo',
    'on_morganbits',
    'show_all_on_bits',
    'visualize_morgan_fps_in_molecule',
    'visualize_all_morgan_fps',
    'visualize_morgan_fp_bits',
    
    'df2cleandf',
    'modify_df1_wrt_df2',
    
    'calculate_properties',
    'compute_all_3d_descriptors',
    'create_descriptor_functions',
    'process_in_parallel',
    'compute_descriptors',
    'compute_descriptors_from_name',
    'listsmiles2propdf',
    'mol_to_fetures',
    'mol_to_fetures_with_descriptor_function',
    
    'calculate_similarities_distances',
    'calculate_similarity_distance',
    'calculate_fingerprints',
    'calculate_all_fingerprints',
    'closest_neighbour_smiles',
    
    'smi2chars',
    'listsmi2chars',
    'padding',
    'smiles_max_len',
    'get_tokens',
    'smi2onehot',
    'listsmi2onehot',
    'onehot2smiles',
    'float_onehot_encodings2smiles',
    'bool_onehot_encodings2smiles',
    
    'nearest_neighbours',
]
           
            
def canonicalize_smiles(smiles: str,
                        isomericSmiles: bool = False
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


def draw_molecule(smiles: str,
                  add_hydrogen: bool = False,
                  size: Tuple[int] = (600, 400),
                  highlight_idx: List[int] = [],
                  savepath: str = ''
                  ) -> Image.Image:
    
    mol = Chem.MolFromSmiles(smiles)
    if add_hydrogen:
        mol = Chem.AddHs(mol)
        
    img = Chem.Draw.MolToImage(mol, size=size, highlightAtoms=highlight_idx)
    if savepath:
        img.save(savepath)
    
    return img


def plot_smiles_grid(smiles_list: List[str],
                     legends: List[str]=None,
                     cols: int = 8,
                     image_size: Tuple[int] = (400, 400),
                     figsize: Tuple[int] = (24, 12),
                     legendfontsize = 12,
                     savepath: str = '',
                     suptitle = ''
                     ) -> None:
    """
    Create a grid plot of molecules from SMILES strings.

    Parameters:
        smiles_list (list of str): List of SMILES strings.
        cols (int): Number of columns in the grid.
        image_size (tuple): Size of each image (width, height).
        figsize (tuple): Overall size of the grid figure (width, height).
    """
    num_molecules = len(smiles_list)
    rows = math.ceil(num_molecules / cols)
    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    for i, ax in enumerate(axs.flat):
        if i < num_molecules:
            smiles = smiles_list[i]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                img = Draw.MolToImage(mol, size=image_size)
                ax.imshow(img)
                if legends:
                    ax.text(0.5, 0.1, legends[i], fontsize=legendfontsize, ha="center", va="center", transform=ax.transAxes) 
            else:
                ax.text(0.5, 0.5, 'Invalid SMILES', horizontalalignment='center', verticalalignment='center')
        ax.axis('off')  # Turn off axes for all subplots

    # Hide unused axes
    for j in range(num_molecules, rows * cols):
        axs.flat[j].axis('off')
    if suptitle:
        plt.suptitle(suptitle, fontsize=18, fontweight='normal')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath,  dpi=100)
        plt.close()
    else:
        plt.show()
        
        
def draw_mol_grid_with_legends(smiles_list: List[str],
                               legends: List[str],
                               highlights: List[int],
                               grid_size: Tuple[int] = (3, 3),
                               subimage_size: Tuple[int] = (400, 400),
                               savepath: str = ""
                               ) -> Image.Image:
    """
    Draw a grid of molecules with legends and specific atoms highlighted for each molecule.

    Args:
        smiles_list (list): List of SMILES strings for the molecules.
        legends (list): List of legend strings corresponding to each molecule.
        highlights (list): List of dictionaries, each mapping atom indices to colors for each molecule.
        grid_size (tuple): Number of rows and columns in the grid.
        subimage_size (tuple): Size of each molecule image in the grid.
        savepath (str): Path to save the grid image (optional).

    Returns:
        PIL.Image.Image: The generated grid image.
    """
    # Convert SMILES to RDKit molecule objects
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    # Prepare the highlight data
    highlight_atoms = [list(h.keys()) for h in highlights]
    highlight_colors = [{idx: color for idx, color in h.items()} for h in highlights]
    
    # Generate the grid image with legends
    grid_img = Draw.MolsToGridImage(
        mols,
        legends=legends,
        
        molsPerRow=grid_size[1],
        subImgSize=subimage_size,  # Larger size for each molecule image
        highlightAtomLists=highlight_atoms,
        highlightAtomColors=highlight_colors,
        returnPNG=False
    )
    
    # Save the image if a save path is provided
    if savepath:
        grid_img.save(savepath)
        print(f"Grid image saved to {savepath}")
    
    return grid_img

 
        
def plot_3d_molecule_with_labels(smiles: str) -> Image.Image:
    """
    Plot a 3D representation of a molecule with labeled atoms.
    Parameters:
        smiles (str): Input SMILES string.
    """
    # Convert SMILES to a molecule and add hydrogens
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Generate 3D conformer
    params = AllChem.ETKDG()
    params.randomSeed = 42  # For reproducibility
    AllChem.EmbedMolecule(mol, params)
    AllChem.UFFOptimizeMolecule(mol)

    # Convert to 3D structure for visualization
    mol_block = Chem.MolToMolBlock(mol)
    conformer = mol.GetConformer()

    # Use Py3Dmol to visualize the molecule
    view = py3Dmol.view(width=800, height=600)
    view.addModel(mol_block, "mol")  # Load the molecule
    view.setStyle({"stick": {}})    # Set style to stick model

    # Add labels for each atom
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conformer.GetAtomPosition(idx)
        symbol = atom.GetSymbol()
        view.addLabel(
            symbol,
            {"position": {"x": pos.x, "y": pos.y, "z": pos.z}, "backgroundColor": "black", "fontSize": 12},
        )
    
    view.zoomTo()  # Zoom to fit the molecule
    return view.show()


def color_maps()-> dict:
    """
    Task:
        Defines a fixed color mapping for different atom roles and bond types 
        in a molecular environment visualization.

    Returns:
        dict: A dictionary mapping descriptive labels to RGB color tuples.
              Keys include 'Center atom', 'Atom in a ring', 'Aromatic atom', 
              'Other atoms', and 'Bonds'.
    """
    
    # Define color palette for different atom types and bonds
    COLOR_FRAC = [
        (1, 1, 0),      # Yellow for center atom
        (1, 0.1, 1),    # Pink for ring atom
        (0.1, 1, 1),    # Cyan for aromatic atom
        (0.9, 0.9, 0.9),# Light gray for others
        (0.6, 1, 0),    # Green-yellow for bonds
    ]
    COLOR_MAP = {
        "Center atom": COLOR_FRAC[0],
        "Atom in a ring": COLOR_FRAC[1],
        "Aromatic atom": COLOR_FRAC[2],
        "Other atoms": COLOR_FRAC[3],
        "Bonds": COLOR_FRAC[4],
    }

    return COLOR_MAP

def get_atom_colors(molecule: Chem.rdchem.Mol,
                    atoms: List[int],
                    centers: Tuple[Set[int], Any] = None
                    ) -> Dict[int, Tuple[float, float, float]]:
    '''
    Task: 
        Assigning visualization colors to atoms based on chemical context.

    Args:
        molecule (Chem.rdchem.Mol): The RDKit molecule object.
        atoms (List[int]): Atom indices to assign colors to.
        centers (Set[int], optional): Atom indices to be treated as central (highlighted in yellow).

    Returns:
        Dict[int, Tuple[float, float, float]]: Mapping from atom index to RGB color.
    '''
    COLOR_MAP = color_maps()
    colors = {}
    for atom in atoms:
        if centers is not None and atom in centers:
            colors[atom] = COLOR_MAP["Center atom"]
        else:
            if molecule.GetAtomWithIdx(atom).GetIsAromatic():
                colors[atom] = COLOR_MAP["Aromatic atom"]
            elif molecule.GetAtomWithIdx(atom).IsInRing():
                colors[atom] = COLOR_MAP["Atom in a ring"]
            else:
                colors[atom] = COLOR_MAP["Other atoms"]
    return colors


def get_bond_colors(bonds: List[int]) -> Dict[int, Tuple[float, float, float]]:
    '''
    Task:
        Assigning a uniform color to highlight specific bonds in visualization.

    Args:
        bonds (List[int]): Bond indices.

    Returns:
        Dict[int, Tuple[float, float, float]]: Mapping from bond index to RGB color.
    '''
    COLOR_MAP = color_maps()
    bond_colors = {bond: COLOR_MAP["Bonds"] for bond in bonds}
    
    return bond_colors

def get_environment(molecule: Chem.rdchem.Mol,
                    center: int,
                    radius: int
                    ) -> Tuple[List[int], List[int], Dict[int, Tuple], Dict[int, Tuple]]:
    '''
    Task:
        Extracting atom and bond environment around a central atom for substructure visualization.

    Args:
        molecule (Chem.rdchem.Mol): The molecule.
        center (int): Index of the center atom.
        radius (int): Radius (in bonds) around the center atom to define the environment.

    Returns:
        Tuple[List[int], List[int], Dict[int, Tuple], Dict[int, Tuple]]:
            atoms, bonds, atom_colors, bond_colors
    '''
    if not molecule.GetNumConformers():
        rdDepictor.Compute2DCoords(molecule)
    env = Chem.FindAtomEnvironmentOfRadiusN(molecule, radius, center)
    atoms = set([center])
    bonds = set([])
    for bond in env:
        atoms.add(molecule.GetBondWithIdx(bond).GetBeginAtomIdx())
        atoms.add(molecule.GetBondWithIdx(bond).GetEndAtomIdx())
        bonds.add(bond)
    atoms = list(atoms)
    bonds = list(bonds)

    atom_colors = get_atom_colors(molecule, atoms, centers=set([center]))
    bond_colors = get_bond_colors(bonds)

    return atoms, bonds, atom_colors, bond_colors

def smiles2morganbifinfo(smiles:str,
                         radius:int = 2,
                         nBits: int = 1024
                         ) -> Tuple[Chem.rdchem.Mol, DataStructs.cDataStructs.ExplicitBitVect, dict]:
    """
    Task:
        Compute Morgan fingerprint and extract substructure information for each activated bit.

    Args:
        smiles (str): SMILES string of the molecule.
        radius (int, optional): Morgan fingerprint radius. Default is 2.
        nBits (int, optional): Length of the fingerprint vector. Default is 1024.

    Returns:
        tuple: A tuple (fp, bitInfo) where:
            - fp (ExplicitBitVect): The binary Morgan fingerprint.
            - bitInfo (dict): Dictionary mapping bit indices to lists of (atomIdx, radius) pairs.
    """
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f'Error: Invalid molecule...')
            return None, None, None
        
        output = AdditionalOutput()
        output.AllocateBitInfoMap()
        morgan_generator = GetMorganGenerator(radius=radius, fpSize=nBits)
        morgan_fp = morgan_generator.GetFingerprint(mol=mol, additionalOutput=output)
        bitInfo = output.GetBitInfoMap()
        
    except Exception as e:
        print(f'Exception triggered with error: {e}')
        return None, None, None
    
    return mol, morgan_fp, bitInfo

def on_morganbits(smiles: str,
                  radius: int = 2,
                  nBits: int = 1024) -> List[int]:
    """
    Task:
        Identify which Morgan fingerprint bits are active (ON) in the molecule.

    Args:
        smiles (str): SMILES string of the molecule.
        radius (int, optional): Morgan fingerprint radius. Default is 2.
        nBits (int, optional): Length of the fingerprint vector. Default is 1024.

    Returns:
        OrderedSet: A set of bit indices that are active (ON) in the fingerprint.
    """
    
    _, morgan_fp, _ = smiles2morganbifinfo(smiles, radius = radius, nBits = nBits)
    present_fp_indices = OrderedSet(list(morgan_fp.GetOnBits()))
    
    return present_fp_indices

def show_all_on_bits(molecule, info, legends):
    '''
    Task: 
        Generating a grid of molecule images with highlighted substructures (used for visualizing fingerprint bits).

    Args:
        molecule (Chem.rdchem.Mol): The molecule to visualize.
        info (Dict[int, Tuple[int, int]]): Mapping of bit index to (center_atom_idx, radius) tuple.
        legends (List[str]): Legend for each substructure image.

    Returns:
        PIL.Image.Image: Composite image showing all substructures.
    '''
    atoms_to_highlight = []
    bonds_to_highlight = []
    atoms_to_highlight_colors = []
    bonds_to_highlight_colors = []
    molecules_to_draw = [molecule]
    atoms_to_highlight.append([])
    bonds_to_highlight.append([])
    atoms_to_highlight_colors.append({})
    bonds_to_highlight_colors.append({})
    for key in info.keys():
        center, radius = info[key][0]
        atoms, bonds, atom_colors, bond_colors = get_environment(
            molecule, center, radius
        )
        atoms_to_highlight.append(atoms)
        bonds_to_highlight.append(bonds)
        atoms_to_highlight_colors.append(atom_colors)
        bonds_to_highlight_colors.append(bond_colors)
        molecules_to_draw.append(molecule)

    options = Draw.rdMolDraw2D.MolDrawOptions()
    options.prepareMolsForDrawing = True
    options.fillHighlights = True
    options.legendFontSize = 25
    img =  Draw.MolsToGridImage(
        molecules_to_draw,
        molsPerRow=min(3, len(molecules_to_draw)),
        subImgSize=(400, 400),
        legends=legends,
        highlightAtomLists=atoms_to_highlight,
        highlightBondLists=bonds_to_highlight,
        highlightAtomColors=atoms_to_highlight_colors,
        highlightBondColors=bonds_to_highlight_colors,
        drawOptions=options,
        maxMols = 1000
    )
    
    return img
    
def visualize_morgan_fps_in_molecule(smiles:str,
                                     bit_indices: List[int],
                                     radius: int = 2,
                                     nBits: int = 1024,
                                     legends: List[str] = [],
                                     savepath: str = ''
                                     ) -> Image.Image:
    """
    Task:
        Visualize substructures corresponding to specific Morgan fingerprint bits
        for a given molecule, along with optional saving support.

    Args:
        smiles (str): SMILES string representing the molecule.
        bit_indices (List[int]): List of bit indices to visualize.
        radius (int, optional): Radius for Morgan fingerprint. Default is 2.
        nBits (int, optional): Fingerprint vector size. Default is 1024.
        legends (List[str], optional): Legends for the substructure visualizations.
                                       First entry is for the full molecule.
        savepath (str, optional): Path to save the generated image. If empty, does not save.

    Returns:
        PIL.Image.Image: Composite image showing the molecule and its substructures
                         corresponding to the specified fingerprint bits.
    """
    
    mol, morgan_fp, bitInfo = smiles2morganbifinfo(smiles, radius = radius, nBits = nBits)
    onBits = OrderedSet(list(morgan_fp.GetOnBits()))
    bitInfo_ = {k:bitInfo[k] for k in bit_indices if k in onBits}
    if not legends:
        legends = ['Main Molecule']
        for x in bit_indices:
            legends.append(('MorganFP_'+str(x)))
        
    assert len(bit_indices) == len(legends)-1, f'Size Mismatch: Bit Indices Size = {len(bit_indices)+1} & Legends Size = {len(legends)}'
    if len(bitInfo_) < len(bit_indices):
        absent_fps = list(OrderedSet(bit_indices) - OrderedSet(bitInfo_.keys()))
        print(f'Fingerprints: {absent_fps} not present in the Molecule')
    if bitInfo_:
        img = show_all_on_bits(mol, bitInfo_, legends=legends)
        if savepath:
            img.save(savepath)
        return img
    else:
        return None
    
    
def visualize_all_morgan_fps(smiles:str,
                         radius:int = 2,
                         nBits: int = 1024,
                         molsPerRow=4
                         ) -> Image.Image:
    
    mol, morgan_fp, bitInfo = smiles2morganbifinfo(smiles, radius = radius, nBits = nBits)
    tpls = []
    legends = []
    for x in morgan_fp.GetOnBits():
        tpls.append((mol,x,bitInfo))
        legends.append(f'MorganFP_{x}')
    # Drawing options
    options = rdMolDraw2D.MolDrawOptions()
    options.legendFontSize = 25
    if tpls:
    # Generate the image
        img = Draw.DrawMorganBits(
            tpls=tpls,
            molsPerRow=molsPerRow,
            legends=legends,
            subImgSize=(350, 350),
            baseRad=0.3,
            useSVG=False,
            aromaticColor=(0.1, 1, 1),
            ringColor=(1, 0.1, 1),
            centerColor=(1, 1, 0),
            extraColor=(0.9, 0.9, 0.9),
            drawOptions=options
        )
    
    return img
    
    

def visualize_morgan_fp_bits(bit_indices: List[int],
                             smiles_list: List[str],
                             radius: int = 2,
                             nBits: int = 1024,
                             molsPerRow: int = 4
                             ) -> Image.Image:
    '''
    Task: 
        Visualize substructures in molecules that activate specific Morgan fingerprint bits.

    Args:
        bit_names (List[str]): List of bit names like 'MorganFP_140'.
        smiles_list (List[str]): List of SMILES strings (usually from a DataFrame column).
        radius (int): Morgan fingerprint radius (default = 2).
        nBits (int): Fingerprint length (default = 1024).
        molsPerRow (int): Number of molecules per row in the grid visualization.

    Returns:
        PIL.Image.Image: Grid image of substructures activating given fingerprint bits.
    '''
    
    tpls = []
    for bit in bit_indices:
        for smiles in smiles_list:
            mol, fp, bitInfo = smiles2morganbifinfo(smiles, radius = radius, nBits = nBits)
            if mol is None:
                continue
            if bit in fp.GetOnBits():
                tpls.append((mol, bit, bitInfo))
                break  # Only one example per bit

    # Drawing options
    options = rdMolDraw2D.MolDrawOptions()
    options.legendFontSize = 25
    if tpls:
    # Generate the image
        img = Draw.DrawMorganBits(
            tpls=tpls,
            molsPerRow=molsPerRow,
            legends=['MorganFP_' + str(bit) for bit in bit_indices],
            subImgSize=(350, 350),
            baseRad=0.3,
            useSVG=False,
            aromaticColor=(0.1, 1, 1),
            ringColor=(1, 0.1, 1),
            centerColor=(1, 1, 0),
            extraColor=(0.9, 0.9, 0.9),
            drawOptions=options
        )

        return img
    else:
        print(f'No SMILES found with the given fingerprint indices list')
        return None
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
########################################################################################################################


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
def create_descriptor_functions(descriptor_names: List[str]) -> Tuple[List[Callable], Calculator]:
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
        RDKit_descriptor_funcs, mordred_calc = create_descriptor_functions(descriptor_names)
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
                        ordered: bool = False
                        ) -> List[Dict[str, int|float|str]]:
    """Process SMILES list in parallel with descriptor calculation."""
    
    with mp.Pool(processes=n_jobs,
                 initializer=initializer,
                 initargs=(rdkit_funcs, mordred_calc)
                 ) as pool:
        imap_func = pool.imap if ordered else pool.imap_unordered
        results = list(tqdm(imap_func(compute_descriptors,
                                      smiles_list,
                                      chunksize=chunk_size
                                      ),
                            total=len(smiles_list),
                            desc="Calculating Descriptors"
                            )
                       )
    return [r for r in results if r is not None]


# --- High-level Wrapper ---
def listsmiles2propdf(smiles_list: List[str],
                      descriptor_names: List[str],
                      df_savepath: str = 'descriptors_output.csv',
                      ordered_parallel_processing: bool = False
                      ) -> pd.DataFrame:
    """Takes list of SMILES and outputs descriptor DataFrame (optionally saves to CSV)."""
    
    print("Preparing descriptor functions...")
    rdkit_funcs, mordred_calc = create_descriptor_functions(descriptor_names)

    n_jobs = mp.cpu_count() - 1
    chunk_size = max(1000, len(smiles_list) // (10 * n_jobs))
    print(f"Using {n_jobs} CPU cores with chunk size: {chunk_size}")

    descriptor_data = process_in_parallel(
        smiles_list=smiles_list,
        rdkit_funcs=rdkit_funcs,
        mordred_calc=mordred_calc,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        ordered=ordered_parallel_processing
    )

    df = pd.DataFrame(descriptor_data)
    for col in descriptor_names:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df_savepath:
        df.to_csv(df_savepath, index=False)
        print(f"Descriptor data saved to: {df_savepath}")

    return df
    


def mol_to_fetures_with_descriptor_function(smile: str,
                                            flag_name2descriptor = True
                                            ) -> Dict[Any | str, Any]:
    
    '''
    Input: A SMILES
    Output: All descriptors calculated from that smiles using different libraries.
    '''
    
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
    
    return all_descriptors, names2descriptors


def mol_to_fetures(smile: str) -> Dict[Any | str, Any]:
    
    '''
    Input: A SMILES
    Output: All descriptors calculated from that smiles using different libraries where keys are strings.
    '''
    all_descriptors, _ = mol_to_fetures_with_descriptor_function(smile, flag_name2descriptor=False)
    all_descriptors = {str(k):v for k, v in all_descriptors.items()}
    
    return all_descriptors


def calculate_all_fingerprints(smiles: Any) -> dict:
    
    mol = Chem.MolFromSmiles(smiles)
    fingerprint_values = {}
    
    rdkit = Chem.RDKFingerprint
    maccskeys = rdMolDescriptors.GetMACCSKeysFingerprint
    atompair = AllChem.GetAtomPairGenerator().GetFingerprint
    topological_torsion = AllChem.GetTopologicalTorsionGenerator().GetFingerprint
    morgan = AllChem.GetMorganGenerator(radius=2).GetFingerprint
    pattern = Chem.rdmolops.PatternFingerprint
    
    all_fingerprints = [rdkit, maccskeys, atompair, topological_torsion, morgan, pattern]
    fingerprint_names = ['RDKit', 'MACCSkeys', 'AtomPair', 'TopologicalTorsion', 'Morgan', 'Pattern']
    
    for i in range(len(all_fingerprints)):
        array = np.zeros((1,), dtype=int)
        fp = all_fingerprints[i](mol)
        DataStructs.ConvertToNumpyArray(fp, array)
        fingerprint_values[fingerprint_names[i]] = array

    return fingerprint_values


# Function to calculate fingerprints
def calculate_fingerprints(smiles: Any,
                           fingerprint_name: str = 'RDKit'
                           ) -> np.ndarray[Any]:
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    if fingerprint_name == 'RDKit':
        fingerprint = Chem.RDKFingerprint(mol)

    elif fingerprint_name == 'Pattern':
        fingerprint = Chem.PatternFingerprint(mol)

    elif fingerprint_name == 'MACCSKeys':
        fingerprint = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        
    elif fingerprint_name == 'TopologicalTorsion':
        fpgen = AllChem.GetTopologicalTorsionGenerator()
        fingerprint = fpgen.GetFingerprint(mol)
        
    elif fingerprint_name == 'Morgan':
        fpgen = AllChem.GetMorganGenerator(radius=2)
        fingerprint = fpgen.GetFingerprint(mol)

    elif fingerprint_name == 'AtomPair':
        fpgen = AllChem.GetAtomPairGenerator()
        fingerprint = fpgen.GetFingerprint(mol)
        
    else:
        raise ValueError("'fingerprints_name' not found in: ['RDKit', 'Pattern', 'TopologicalTorsion', 'MACCSKeys', 'Morgan', 'AtomPair']")
    
    # Convert to NumPy arrays for calculations
    array = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(fingerprint, array)

    return array


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else -1e15


# Function to calculate various similarities and distances
def calculate_similarities_distances(smiles1: str,
                                     smiles2: str,
                                     fingerprint_name: str = 'RDKit'
                                     ) -> Tuple[dict[str, float], dict[str, float]]:
    
    fp1 = calculate_fingerprints(smiles=smiles1, fingerprint_name=fingerprint_name)
    fp2 = calculate_fingerprints(smiles=smiles2, fingerprint_name=fingerprint_name)
    
    a = len(np.where((fp1 == 1) & (fp2 == 0))[0]) # present in smiles1 but not in smiles2
    b = len(np.where((fp1 == 0) & (fp2 == 1))[0]) # present in smiles2 but not in smiles1
    c = len(np.where((fp1 == 1) & (fp2 == 1))[0]) # present both in smiles1 and smiles2
    d = len(np.where((fp1 == 0) & (fp2 == 0))[0]) # absent both in smiles1 and smiles2
    
    # Similarities
    tanimoto = safe_divide(c, (a + b + c))
    dice = safe_divide(c, 0.5 * ((a + c) + (b + c)))
    cosine = safe_divide(c, np.sqrt((a + c) * (b + c)))
    russell_rao = safe_divide(c, (a + b + c + d))
    baroni_urbani = safe_divide((np.sqrt(c * d) + c), (np.sqrt(c * d) + a + b + c))
    rogers = safe_divide((c + d), (2 * a + 2 * b + c + d))
    matching = safe_divide((c + d), (a + b + c + d))

    # Distances
    euclidean = np.sqrt(a + b)  # Euclidean is always valid
    hamming = a + b  # Hamming, Manhattan, Taxi-Cab, City-Block
    mean_hamming = safe_divide((a + b), (a + b + c + d))
    soergel = safe_divide((a + b), (a + b + c))
    pattern = safe_divide((a * b), (a + b + c + d) ** 2)
    variance = safe_divide((a + b), (4 * (a + b + c + d)))
    size = safe_divide((a - b) ** 2, (a + b + c + d) ** 2)
    
    similarities = {"Tanimoto": tanimoto,
                    "Dice": dice,
                    "Cosine": cosine,
                    "Rogers": rogers,
                    "Russell_Rao": russell_rao,
                    "Baroni_Urbani": baroni_urbani,
                    "Matching_Coefficient":matching,
                    "Overlap": c
                    }
    
    distances = {"Euclidean": euclidean,
                 "Hamming": hamming,
                 "Mean_Hamming": mean_hamming,
                 "Soergel": soergel,
                 "Pattern": pattern,
                 "Variance": variance,
                 "Size": size
                 }

    return similarities, distances

# Function to calculate various similarities and distances
def calculate_similarity_distance(smiles1: str,
                                  smiles2: str,
                                  fingerprint_name: str = 'RDKit',
                                  distance_metric: str = 'Tanimoto',
                                  similarity_metric: str = 'Euclidean'
                                  ) -> Tuple[int | float, int | float]:
    
    fp1 = calculate_fingerprints(smiles=smiles1, fingerprint_name=fingerprint_name)
    fp2 = calculate_fingerprints(smiles=smiles2, fingerprint_name=fingerprint_name)
    
    a = len(np.where((fp1 == 1) & (fp2 == 0))[0]) # present in smiles1 but not in smiles2
    b = len(np.where((fp1 == 0) & (fp2 == 1))[0]) # present in smiles2 but not in smiles1
    c = len(np.where((fp1 == 1) & (fp2 == 1))[0]) # present both in smiles1 and smiles2
    d = len(np.where((fp1 == 0) & (fp2 == 0))[0]) # absent both in smiles1 and smiles2
    
    # Similarities
    if similarity_metric == 'Tanimoto':
        similarity = safe_divide(c, (a + b + c))
    elif similarity_metric == 'Dice':
        similarity = safe_divide(c, 0.5 * ((a + c) + (b + c)))
    elif similarity_metric == 'Cosine':
        similarity = safe_divide(c, np.sqrt((a + c) * (b + c)))
    elif similarity_metric == 'Russell_Rao':
        similarity = safe_divide(c, (a + b + c + d))
    elif similarity_metric == 'Baroni_Urbani':
        similarity = safe_divide((np.sqrt(c * d) + c), (np.sqrt(c * d) + a + b + c))
    elif similarity_metric == 'Rogers':
        similarity = safe_divide((c + d), (2 * a + 2 * b + c + d))
    elif similarity_metric == 'Matching_Coefficient':
        similarity = safe_divide((c + d), (a + b + c + d))
    else:
        raise ValueError("similarity metric not found in: ['Tanimoto', 'Dice', 'Cosine', 'Rogers','Russell_Rao', 'Baroni_Urbani', 'Matching_Coefficient', 'Overlap']")

    # Distances
    if distance_metric == 'Euclidean':
        distance = np.sqrt(a + b)  # Euclidean is always valid
    elif distance_metric == 'Hamming':
        distance = a + b  # Hamming, Manhattan, Taxi-Cab, City-Block
    elif distance_metric == 'Mean_Hamming':
        distance = safe_divide((a + b), (a + b + c + d))
    elif distance_metric == 'Soergel':
        distance = safe_divide((a + b), (a + b + c))
    elif distance_metric == 'Pattern':
        distance = safe_divide((a * b), (a + b + c + d) ** 2)
    elif distance_metric == 'Variance':
        distance = safe_divide((a + b), (4 * (a + b + c + d)))
    elif distance_metric == 'Size':
        distance = safe_divide((a - b) ** 2, (a + b + c + d) ** 2)
    else:
        raise ValueError("distance metric not found in: ['Euclidean', 'Hamming', 'Mean_Hamming', 'Soergel', 'Pattern', 'Variance', 'Size']")
    

    return similarity, distance


def closest_neighbour_smiles(smiles: str,
                             smiles_list: List[str],
                             similarity_metric: str = 'Tanimoto',
                             distance_metric: str = 'Soergel',
                             fingerprint_name: str = 'RDKit'
                             ) -> Tuple[str, str]:
    
    max_similarity, min_distance = -1e6, 1e6
    optimum_smiles_similarity, optimum_smiles_distance = '', ''
    
    for smi in smiles_list:
        similarities, distances = calculate_similarities_distances(smiles1=smiles, smiles2=smi, fingerprint_name=fingerprint_name)
        try:
            similarity = similarities[similarity_metric]
        except:
            raise ValueError('Similarity metric not found in: [Tanimoto, Dice, Cosine, Rogers, Russell-Rao, Baroni-Urbani, Kunczynski]')
        
        try:
            distance = distances[distance_metric]
        except:
            raise ValueError('Distance metric not found in: [Hamming, Mean Hamming, Euclidean, Soergel, Pattern, Variance, Size]')
        
        if similarity>max_similarity:
            optimum_smiles_similarity = smi
            max_similarity = similarity
        if distance<min_distance:
            optimum_smiles_distance = smi
            min_distance = distance
    
    return optimum_smiles_similarity, optimum_smiles_distance    


def df2cleandf(df: pd.DataFrame,
               smi_col_name: str = 'SMILES',
               target_col: str = 'LogPapp Value',
               target_col_rename: str = 'logPapp_Values_list',
               mean_col_name: str = 'Mean_logPapp_Values',
               std_col_name: str = 'logPapp_Values_Std',
               data_split: bool = True,
               train_idx: Union[list, tuple, np.ndarray] = [],
               val_idx: Union[list, tuple, np.ndarray] = [],
               test_idx: Union[list, tuple, np.ndarray] = []
               ) -> pd.DataFrame:
    
    '''
    Input: A dataframe containing SMILES and LogPapp Value columns
    Output: A cleaned and featured dataframe
    '''
    
    df['Cannonicalized_SMILES'] = df[smi_col_name].apply(canonicalize_smiles)
    mapping = dict(zip(df['Cannonicalized_SMILES'], df[smi_col_name]))
    df_grouped = df.groupby('Cannonicalized_SMILES')[target_col].apply(list).to_dict()
    df_grouped_list = pd.DataFrame(list(df_grouped.items()), columns=['Cannonicalized_SMILES', target_col_rename])
    
    df_grouped_list.insert(0, smi_col_name, ['Other']*len(df_grouped_list))
    df_grouped_list[smi_col_name] = df_grouped_list['Cannonicalized_SMILES'].map(mapping)
    
    mean_func = lambda x: sum(x) / len(x) if len(x) > 0 else None
    std_func = lambda x: np.std(x) if len(x) > 0 else None
    
    df_grouped_list[mean_col_name] = df_grouped_list[target_col_rename].apply(mean_func)
    df_grouped_list[std_col_name] = df_grouped_list[target_col_rename].apply(std_func)
    
    df_grouped_list_features = df_grouped_list[smi_col_name].apply(mol_to_fetures).tolist()
    df_grouped_list_features = pd.DataFrame(df_grouped_list_features)
    
    # df_grouped_list_features = df_grouped_list_features.dropna(axis=1, how='any')
    df_grouped_list_features = df_grouped_list_features.select_dtypes(include=[np.number])
    
    df_features = pd.concat([df_grouped_list, df_grouped_list_features], axis=1)
    
    if data_split:
        df_features.insert(2, 'Data_Split', ['Other']*len(df_features))
        if len(train_idx)==0 or len(test_idx)==0 or len(val_idx)==0:
            print('Data splitting indices not found. Splitting dataset randomly:')
            idx = np.random.permutation(np.arange(len(df_features)))
            train_idx, test_idx, val_idx = idx[:len(idx)//2], idx[len(idx)//2:len(idx)*4//5], idx[len(idx)*4//5:len(idx)]
            
        df_features.loc[df_features.index.isin(train_idx), 'Data_Split'] = 'Tr'
        df_features.loc[df_features.index.isin(test_idx), 'Data_Split'] = 'Te'
        df_features.loc[df_features.index.isin(val_idx), 'Data_Split'] = 'Val'
    
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


def smi2chars(smiles: str) -> OrderedSet:
    
    '''
    Input: A SMILES
    Outpur: List of unique characters of that particular SMILES
    '''
    
    # Convert SMILES to a molecule object
    mol = Chem.MolFromSmiles(smiles)
    
    # Check if the molecule object is valid
    if mol is not None:
        # Generate and return the canonical SMILES
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        
        char_list = []
               
        two_string_atoms = ['Br', 'Cl']
        
        if '[' in smiles or ']' in smiles:
            char_list += ['[', ']']
            bracket_contents = re.findall(r"\[(.*?)\]", smiles)
            char_list += bracket_contents
            smiles = re.sub(r"\[.*?\]", "", smiles)
            
        for atom in two_string_atoms:
            if atom in smiles:
                char_list.append(atom)
                smiles = smiles.replace(atom, "")          
        
        all_chars = set(char_list) | set(smiles)
        
        for char in all_chars:
            for x in ['@@', '@']:
                if x in char:
                    all_chars = all_chars - {char}
                    char = char.replace(x, '')
                    all_chars = all_chars | {x} | set(char)
    
    return OrderedSet(sorted(list(all_chars), key = len, reverse=True))


def listsmi2chars(smiles_list: List[str]) -> List[str]:
    
    '''
    Input: List of SMILES
    Outpur: List of unique characters form all the SMILES
    '''
    
    all_chars = set()
    for smi in tqdm(smiles_list):
        if Chem.MolFromSmiles(smi):
            chars = smi2chars(smi)
            all_chars = all_chars | chars
        
    return sorted(list(all_chars), key = len, reverse=True)


def padding(string: str,
            max_string_len: int = 250,
            padding: str = ' '
            ) -> str:
        
    if len(string) <= max_string_len:
        if padding:
            string = string + padding * (max_string_len - len(string))
        else:
            raise ValueError('Define your padding character: "{}"'.format(padding))
        
    return string


def smiles_max_len(smiles_list: Any,
                   char_list: Any
                   ) -> tuple[int, str]:
    
    '''
    Input: A list of SMILES
    Output: The Maximum length of the SMILES
    '''
    
    # Create a regex pattern for the vocabulary
    char_list = sorted(char_list, key=len, reverse=True)
    vocab_pattern = '|'.join(map(re.escape, char_list))
    
    max_len = 0
    for smi in tqdm(smiles_list):
        tokens = re.findall(vocab_pattern, smi)
        smi_len = len(tokens)
        
        if smi_len >= max_len:
            max_len = smi_len
            max_len_smiles = smi
            
    return max_len, max_len_smiles


def get_tokens(smi: str,
               char_list: list
               ) -> list:
    
    # Precompile the regex for faster repeated usage
    vocab_pattern = '|'.join(map(re.escape, char_list))
    vocab_regex = re.compile(vocab_pattern)
    
    # Tokenize the SMILES string
    tokens = vocab_regex.findall(smi)
    
    return tokens


def smi2onehot(smi: str,
               char_list: list,
               max_str_len: int
               ) -> tuple[np.ndarray[np.bool_], list[str]]:
    
    '''
    Input: 
        - A SMILES
        - Unique Characters List
        - Maximum length of the SMILES
        
    Output: 
        - One Hot Encoding of that SMILES
        - All the tokens in the sorted form
        - Unique tokens
        - Length of the SMILES w.r.t the tokens
    '''
    
    # Tokenize the SMILES string
    tokens = get_tokens(smi=smi, char_list=char_list)
    
    # Pre-allocate one-hot encoding array
    one_hot = np.zeros((max_str_len, len(char_list) + 1), dtype=bool)
    
    # Create a lookup table for one-hot encoding of each character
    token_to_index = {token: idx for idx, token in enumerate(char_list)}
    
    # Process tokens and fill in the one-hot matrix
    for i, token in enumerate(tokens):
        if token in token_to_index:
            one_hot[i, token_to_index[token]] = True #1
        else:
            # If token is not in vocabulary, ValueError will be raised
            raise ValueError(f'Bad SMILES Error. Token - {token} not in the Character List: {char_list}.')
        
    # Fill remaining rows with padding (last column set to 1)
    if len(tokens) < max_str_len:
        one_hot[len(tokens):, -1] = True #1
    
    # Unique tokens in the SMILES string
    # unique_tokens = sorted(set(tokens), key=len, reverse=True)
    
    # Length of SMILES with respect to tokens
    # smiles_len = len(tokens)
    
    return one_hot, tokens#, unique_tokens, smiles_len


def listsmi2onehot(smiles_list: List[str],
                   char_list: List[str],
                   max_str_len: int
                   ) -> np.ndarray[np.bool_]:
    
    '''
    Input:
        - List of SMILES
        - Unique Character List
        - Maximum length of the SMILES
        
    Output: An Array containing all the one hot encodings of the smiles
    '''
    
    onehot_encodings = np.zeros((len(smiles_list), max_str_len, len(char_list)+1), dtype=bool)
    for i in tqdm(range(len(smiles_list))):
        onehot, _ = smi2onehot(smiles_list[i], char_list, max_str_len)
        onehot_encodings[i] = onehot
        
    return onehot_encodings



def onehot2smiles(onehot_mat: np.ndarray,
                  vocabulary: List[str],
                  ) -> str:
    
    string = ''
    for i in range(len(onehot_mat)):
        char_idx_x = np.where(onehot_mat[i] == 1)
        if np.size(char_idx_x[0]) != 0:
            string += vocabulary[char_idx_x[0][0]] if char_idx_x[0][0] < len(vocabulary) else ''
        
    return string


def bool_onehot_encodings2smiles(onehot_encodings: np.ndarray[np.bool_],
                            char_list: List[str]):
    
    smiles_list = [None]*len(onehot_encodings)
    for i in range(len(onehot_encodings)):
        smiles_list[i] = onehot2smiles(onehot_encodings[i], char_list)
        
    return smiles_list

def float_onehot_encodings2smiles(onehot_encodings: np.ndarray[np.bool_],
                            char_list: List[str]):
    
    smiles_list = [None]*len(onehot_encodings)
    indices = np.argmax(onehot_encodings, axis=-1)
    shape = onehot_encodings.shape
    temp = np.zeros((shape), dtype=bool)
    for i in range(shape[0]):
        temp[i, range(shape[1]), indices[i]] = True
    
    smiles_list = bool_onehot_encodings2smiles(temp, char_list)
        
    return smiles_list
# #------------------------------------------------------

def nearest_neighbours(model: (nn.Module | RandomForestRegressor),
                       X_train: (pd.DataFrame | torch.Tensor),
                       X_rr: (pd.DataFrame | torch.Tensor),
                       y_train: (pd.DataFrame | torch.Tensor),
                       y_rr: (pd.DataFrame | torch.Tensor),
                       smiles_list_rr: Any,
                       smiles_list_train: Any,
                       n_neighbors: int = 1,
                       check_fingerprint: bool = False,
                       fingerprint_names: Any = ['RDKit', 'Pattern', 'TopologicalTorsion', 'MACCSKeys', 'Morgan', 'AtomPair'],
                       distance_metric: str = 'Euclidean',
                       similarity_metric: str = 'Tanimoto',
                       device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                       ) -> Tuple[List[str], List[str], pd.DataFrame]:
    predicted_rr = []
    n_neighbour_original = []
    n_neighbour_predicted = []
    original_rr = []
    
    rdkit_fp = []
    pattern_fp = []
    topologicaltorsion_fp = []
    maccskeys_fp = []
    morgan_fp = []
    atompair_fp = []
    fp_lists = [rdkit_fp, pattern_fp, topologicaltorsion_fp, maccskeys_fp, morgan_fp, atompair_fp]

    smi_list_rr_then_train = []
    smi_list_rr = []
    smi_list_train = []
    legends = []
    fps_result = defaultdict(list)

    for i in tqdm(range(X_rr.shape[0])):
                
        if isinstance(model, nn.Module):
            # Build the KDTree
            nn_tree = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')
            nn_tree.fit(X_train)
            
            target_point = X_rr[i]
            ind = nn_tree.kneighbors(X = target_point.reshape(1, -1), n_neighbors=n_neighbors, return_distance=False)
            nn_idx = ind.item()
            nearest_point = X_train[nn_idx]
            
            model, target_point, nearest_point = model.to(device), target_point.to(device), nearest_point.to(device)
            rr_point_pred_val = model(target_point)
            near_pts_pred_val = model(nearest_point.reshape(*target_point.shape))
            rr_point_pred_val, near_pts_pred_val = rr_point_pred_val.cpu().detach().numpy(), near_pts_pred_val.cpu().detach().numpy()
        
        elif isinstance(model, (RandomForestClassifier, RandomForestRegressor, SVC, SVR)):
            
            # Build the KDTree
            target_point = X_rr.iloc[i].values.reshape(1, -1)
            nn_tree = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(X_train)
            target_point = pd.DataFrame(target_point, columns=X_train.columns)
            
            ind = nn_tree.kneighbors(X = target_point, n_neighbors=n_neighbors, return_distance=False)
            nn_idx = ind.item()
            nearest_point = X_train.iloc[nn_idx].values.reshape(1, -1)
            nearest_point = pd.DataFrame(nearest_point, columns=X_train.columns)
            
            rr_point_pred_val = model.predict(target_point)
            near_pts_pred_val = model.predict(nearest_point)
            
        else:
            raise TypeError( 'Model type unknown. Expected a neural network or random forest model.')
            
        if isinstance(y_rr, pd.core.series.Series):
            y_rr = y_rr.to_numpy()
        if isinstance(y_train, pd.core.series.Series):
            y_train = y_train.to_numpy()  
            
        predicted_rr.append(round(rr_point_pred_val.item(), 4))
        n_neighbour_predicted.append(round(near_pts_pred_val.item(), 4))
        original_rr.append(round(y_rr[i].item(), 4))
        n_neighbour_original.append(round(y_train[nn_idx].item(), 4))
        
        smi_list_rr_then_train.append(smiles_list_rr[i])
        smi_list_rr.append(smiles_list_rr[i])
        legends.append(f'T:{original_rr[i]}-P:{predicted_rr[i]}(In)')
        
        smi_list_rr_then_train.append(smiles_list_train[nn_idx])
        smi_list_train.append(smiles_list_train[nn_idx])
        legends.append(f'T:{n_neighbour_original[i]}-P:{n_neighbour_predicted[i]}(Tr)')
        
        fps_dict = OrderedDict({})
        
        if check_fingerprint:
            for name in fingerprint_names:
                smiles_sim, _ = closest_neighbour_smiles(smiles_list_rr[i],
                                                        smiles_list_train,
                                                        fingerprint_name=name,
                                                        distance_metric=distance_metric,
                                                        similarity_metric=similarity_metric
                                                        )
                smi_list_rr_then_train.append(smiles_sim)
                
                idx = list(smiles_list_train).index(smiles_sim)
                
                if isinstance(X_train, pd.DataFrame):
                    smiles_sim_point = X_train.iloc[idx].values.reshape(1, -1)
                    smiles_sim_point = pd.DataFrame(smiles_sim_point, columns=X_train.columns)
                else:
                    smiles_sim_point = X_train[idx].reshape(1, -1)
                    # smiles_sim_point = pd.DataFrame(smiles_sim_point, columns=[f'dim-{i}' for i in range(X_train.shape[1])])
                    
                if isinstance(model, nn.Module):
                    pred_val = model(smiles_sim_point.to(device)).item()
                elif isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
                    pred_val = model.predict(smiles_sim_point).item()
                else:
                    raise TypeError( 'Model type unknown. Expected a neural network or random forest model.')
                
                true_val = y_train[idx].item()
                legends.append(f'T:{round(true_val, 4)}-P:{round(pred_val, 4)}({name})')
                
                fps_dict[name] = smiles_sim
                
            for key, value in fps_dict.items():
                fps_result[key].append(value)

    df = pd.DataFrame({'Independent SMILES': smi_list_rr,
                       'Nearest SMILES': smi_list_train,
                       'Independent Original': original_rr,
                       'Independent Predicted': predicted_rr,
                       'NN Original': n_neighbour_original,
                       'NN Predicted': n_neighbour_predicted,
                       **fps_result
                       })
        
    
    return smi_list_rr_then_train, legends, df
