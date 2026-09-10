import re
import io
import os
import ast
import sys
import math
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from orderedset import OrderedSet
from typing import Any, List, Dict, Tuple, Union, Set, Callable, Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdDepictor, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, AdditionalOutput
from rdkit.Chem.MolStandardize import rdMolStandardize

import py3Dmol
from IPython.display import SVG, display
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Generalised_data_utils import create_folder

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore", module="mordred")

__all__ = [
    'plot_molecule',
    'plot_smiles_grid',
    'plot_mol_grid_with_legends',
    'plot_3d_molecule_with_labels',
    'smiles2morganbifinfo',
    'on_morganbits',
    'show_all_on_bits',
    'visualize_morgan_fps_in_molecule',
    'visualize_all_morgan_fps',
    'visualize_morgan_fp_bits',
    'show_match',
]

def plot_molecule(
    smiles: str,

    # -----------------------------
    # Standardization options
    # -----------------------------
    sanitize: bool = True,
    remove_salt: bool = False,
    largest_fragment: bool = False,
    uncharge: bool = False,
    normalize: bool = False,
    reionize: bool = False,
    canonical_tautomer: bool = False,

    # -----------------------------
    # Molecule preparation
    # -----------------------------
    add_hydrogen: bool = False,
    remove_hydrogen: bool = False,
    kekulize: bool = False,

    # -----------------------------
    # Drawing options
    # -----------------------------
    size: Tuple[int, int] = (600, 400),
    legend: str = "",

    # Highlighting
    highlight_atoms: Optional[List[int]] = None,
    highlight_bonds: Optional[List[int]] = None,
    atom_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
    bond_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
    highlight_radius: float = 0.4,

    # Labels and indices
    add_atom_indices: bool = False,
    add_bond_indices: bool = False,
    atom_labels: Optional[Dict[int, str]] = None,

    # Styling
    bg_color: Tuple[float, float, float] = (1, 1, 1),
    transparent_background: bool = False,
    bond_line_width: float = 2.0,
    atom_font_size: int = 16,
    fixed_bond_length: Optional[float] = None,

    # Output
    image_type: str = "PIL",   # PIL or SVG
    savepath: Optional[str] = None,
    return_mol: bool = False
) -> Union[Image.Image, str, tuple]:

    """
    Draw and optionally standardize a molecule from SMILES.

    Standardization operations include:
    - Salt removal
    - Largest fragment selection
    - Uncharging
    - Normalization
    - Reionization
    - Canonical tautomer generation

    Draw a molecule from a SMILES string with customizable options.

    Parameters
    ----------
    smiles : str
        SMILES representation of molecule.

    add_hydrogen : bool
        Whether to add explicit hydrogens.

    kekulize : bool
        Whether to kekulize aromatic structures.

    size : tuple
        Image size (width, height).

    highlight_atoms : list
        Atom indices to highlight.

    highlight_bonds : list
        Bond indices to highlight.

    atom_colors : dict
        Dictionary mapping atom index -> RGB color tuple.
        Example: {0: (1, 0, 0)}

    bond_colors : dict
        Dictionary mapping bond index -> RGB color tuple.

    highlight_radius : float
        Radius for highlighted atoms.

    add_atom_indices : bool
        Display atom indices.

    add_bond_indices : bool
        Display bond indices.

    atom_labels : dict
        Custom atom labels.
        Example: {0: "N1"}

    legend : str
        Caption below molecule.

    bg_color : tuple
        Background RGB color.

    bond_line_width : float
        Width of bond lines.

    atom_font_size : int
        Font size for atom labels.

    fixed_bond_length : float
        Optional fixed bond length.

    image_type : str
        "PIL" or "SVG".

    return_mol : bool
        Return RDKit Mol object along with image.

    savepath : str
        Path to save output image.

    Returns
    -------
    PIL.Image or SVG string or tuple
    """

    # =========================================================
    # Create molecule
    # =========================================================
    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    else:
        mol = smiles

    if mol is None:
        raise ValueError("Invalid SMILES string.")

    # =========================================================
    # Standardization pipeline
    # =========================================================

    # Remove salts / counter ions
    if remove_salt:
        remover = rdMolStandardize.SaltRemover()
        mol = remover.StripMol(mol, dontRemoveEverything=True)

    # Keep largest fragment only
    if largest_fragment:
        chooser = rdMolStandardize.LargestFragmentChooser()
        mol = chooser.choose(mol)

    # Normalize functional groups
    if normalize:
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)

    # Reionize molecule
    if reionize:
        reionizer = rdMolStandardize.Reionizer()
        mol = reionizer.reionize(mol)

    # Neutralize charges
    if uncharge:
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

    # Canonical tautomer
    if canonical_tautomer:
        enumerator = rdMolStandardize.TautomerEnumerator()
        mol = enumerator.Canonicalize(mol)

    # =========================================================
    # Hydrogen handling
    # =========================================================
    if add_hydrogen:
        mol = Chem.AddHs(mol)

    if remove_hydrogen:
        mol = Chem.RemoveHs(mol)

    # =========================================================
    # Kekulization
    # =========================================================
    if kekulize:
        try:
            Chem.Kekulize(mol)
        except:
            pass

    # =========================================================
    # Drawing setup
    # =========================================================
    if image_type.upper() == "SVG":
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])

    options = drawer.drawOptions()

    options.addAtomIndices = add_atom_indices
    options.addBondIndices = add_bond_indices
    options.bondLineWidth = bond_line_width
    options.baseFontSize = atom_font_size / 20
    if transparent_background:
        options.setBackgroundColour((1, 1, 1, 0))   # transparent RGBA
    else:
        options.setBackgroundColour(bg_color)

    if fixed_bond_length is not None:
        options.fixedBondLength = fixed_bond_length

    # Custom labels
    if atom_labels is not None:
        for idx, label in atom_labels.items():
            options.atomLabels[idx] = label

    # Defaults
    if highlight_atoms is None:
        highlight_atoms = []

    if highlight_bonds is None:
        highlight_bonds = []

    if atom_colors is None:
        atom_colors = {}

    if bond_colors is None:
        bond_colors = {}

    # =========================================================
    # Draw molecule
    # =========================================================
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        legend=legend,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
        highlightAtomRadii={
            idx: highlight_radius for idx in highlight_atoms
        }
    )

    drawer.FinishDrawing()

    # =========================================================
    # Output handling
    # =========================================================
    if image_type.upper() == "SVG":

        img = drawer.GetDrawingText()
        if savepath:
            with open(savepath, "w", encoding="utf-8") as f:
                f.write(img)

    else:
        img_data = drawer.GetDrawingText()
        img = Image.open(BytesIO(img_data)).convert("RGBA")

        if savepath:
            img.save(savepath)

    if return_mol:
        return img, mol

    return img


def mol_to_image_with_font(mol, size=(400, 400), atom_font_size=18, 
                           bond_line_width=4, dots_per_angstrom=None, 
                           hlt_atm_indcs = None, addAtomIndices=False):
    """
    Return a PIL.Image of `mol` drawn with RDKit's MolDraw2DCairo while setting atom label font size.
    atom_font_size should be an integer (point size).
    """
    try:
        w, h = size
        drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
        draw_options = drawer.drawOptions()

        # Important settings
        drawer.SetFontSize(int(atom_font_size))
        draw_options.addAtomIndices = addAtomIndices
        draw_options.highlightRadius = 0.4
        # draw_options.atomLabelFontSize = int(atom_font_size)   # integer point size
        draw_options.bondLineWidth = bond_line_width           # thicker bonds if desired
        draw_options.minFontSize = atom_font_size
        draw_options.maxFontSize = atom_font_size+5
        draw_options.scaleBondWidth = True
        draw_options.annotationFontScale = 1.0

        if dots_per_angstrom is not None:
            try:
                draw_options.dotsPerAngstrom = float(dots_per_angstrom)
            except Exception:
                pass

        ## Highlight indices
        if isinstance(hlt_atm_indcs, int):
            hlt_atm_indcs = [hlt_atm_indcs]
        highlight_atom_indices = list(hlt_atm_indcs) if hlt_atm_indcs is not None else []
        
        # Draw and return PIL image
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atom_indices)
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        # GetDrawingText may return str or bytes depending on RDKit build; normalize to bytes
        if isinstance(png, str):
            png = png.encode("utf-8")
        return Image.open(io.BytesIO(png))
    except Exception as exc:
        # fall back to the simpler MolToImage if Cairo isn't available or something else fails
        print("Warning: MolDraw2DCairo unavailable/failed (falling back). Error:", exc)
        return Draw.MolToImage(mol, size=size)


def fig_to_image(fig, dpi=100):
    import io
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)

    img = Image.open(buf)

    # --- SAFETY FIX ---
    # Ensure all metadata values are strings
    if hasattr(img, "info") and isinstance(img.info, dict):
        img.info = {k: str(v) for k, v in img.info.items()}

    return img


def plot_smiles_grid(smiles_list: List[str],
                     legends: List[str]=None,
                     titles: List[str]=None,
                     cols: int = 5,
                     image_size: Tuple[int,int] = (400, 400),
                     figsize: Tuple[int,int] = None,
                     savepath: str = '',
                     suptitle: str = '',
                     caption: str = '',
                     legendfontsize: int = 0,
                     row_lines: bool = True,
                     atom_font_size: int = 18,
                     dots_per_angstrom: float = None,
                     bond_line_width: float = 4,
                     separate_first_column: bool = False,
                     highlight_atom_indices = None,
                     addAtomIndices=False,
                     show=True,
                     ) -> None:
    """
    Create a grid plot of molecules from SMILES strings.

    Parameters:
        smiles_list (list of str): List of SMILES strings.
        legends (list of str, optional): Captions below each molecule.
        titles (list of str, optional): Titles above each molecule.
        cols (int): Number of columns in the grid.
        image_size (tuple): Size of each image (width, height).
        figsize (tuple): Overall size of the grid figure (width, height).
        suptitle (str): Title for the entire figure.
        caption (str): Caption for the entire figure.
    """
    cols = min(cols, len(smiles_list))
    if figsize is None:
        n_row = (len(smiles_list)//cols)+1
        figsize=(3.3*cols, 4 * n_row)
    if legendfontsize == 0:
        legendfontsize = figsize[0]-int(figsize[0]/cols)-1
        
    titlefontsize = legendfontsize + 2
    num_molecules = len(smiles_list)
    rows = math.ceil(num_molecules / cols)
    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    # Normalize axs to a flat list
    axs = np.atleast_1d(axs).reshape(-1)

    if highlight_atom_indices is not None:
        if isinstance(highlight_atom_indices, list):
            assert len(highlight_atom_indices) == len(smiles_list), (
                    f'length mismatch between highlight_atom_indices ({len(highlight_atom_indices)}) and smiles_list ({len(smiles_list)})'
                    )
        if isinstance(highlight_atom_indices, int):
            highlight_atom_indices = [[highlight_atom_indices]]*len(smiles_list)

    for i, ax in enumerate(axs):
        if i < num_molecules:
            smiles = smiles_list[i]
            if highlight_atom_indices is not None:
                hlt_atm_indcs = highlight_atom_indices[i]
            else:
                hlt_atm_indcs = None

            if isinstance(smiles, str):
                mol = Chem.MolFromSmiles(smiles)
            else:
                mol = smiles
            if mol:
                img = mol_to_image_with_font(mol,
                                             size=image_size,
                                             atom_font_size=atom_font_size,
                                             dots_per_angstrom=dots_per_angstrom,
                                             bond_line_width=bond_line_width,
                                             hlt_atm_indcs=hlt_atm_indcs,
                                             addAtomIndices=addAtomIndices
                                             )
                ax.imshow(img)
                # Titles and legends
                if titles and i < len(titles) and titles[i]:
                    ax.set_title(titles[i], fontsize=titlefontsize, fontweight="bold")
                if legends and i < len(legends) and legends[i]:
                    ax.text(0.5, -0.12, legends[i], fontsize=legendfontsize,
                            ha="center", va="center", transform=ax.transAxes, fontweight="bold")
            else:
                ax.text(0.5, 0.5, 'Invalid SMILES',
                        ha='center', va='center', fontsize=legendfontsize)
        ax.axis('off')

    # Hide unused axes
    for j in range(num_molecules, rows * cols):
        axs[j].axis('off')

    if suptitle:
        plt.suptitle(suptitle, fontsize=titlefontsize+2*int(figsize[0]/cols), fontweight='bold')

    if caption:
        plt.figtext(0.5, -0.02, caption, wrap=True, ha="center", fontsize=legendfontsize)

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tight_layout()
    # Compute global left/right bounds from all axes
    left = min(ax.get_position().x0 for ax in axs)
    right = max(ax.get_position().x1 for ax in axs)

    if row_lines and rows > 1:
        for r in range(rows - 1):
            # Axes in the current row
            row_axes = axs[r * cols:(r + 1) * cols]

            # Bottom of the row (minimum y0 among axes in that row)
            y_bottom = min(ax.get_position().y0 for ax in row_axes)

            # Small adaptive offset based on row height
            row_height = max(ax.get_position().height for ax in row_axes)
            y = y_bottom - 0.25 * row_height

            fig.add_artist(plt.Line2D([left, right], [y, y], transform=fig.transFigure, color="black", lw=1, alpha=0.5))
            
    if cols > 1 and separate_first_column:
        # right edge of the first column
        x = axs[0].get_position().x1

        # vertical span: from bottom row to top row
        y_bottom = axs[(rows - 1) * cols].get_position().y0
        y_top = axs[0].get_position().y1

        fig.add_artist(plt.Line2D([x, x], [y_bottom, y_top], transform=fig.transFigure, color="black", lw=1, alpha=0.5))

    if savepath:
        parent_folder = os.path.dirname(os.path.abspath(savepath))
        create_folder(parent_folder)
        print(savepath)
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f'Figure saved in: {savepath}')
        plt.close()
    elif not show:
        return fig_to_image(fig, dpi=300)
    else:
        plt.show()
                

def plot_mol_grid_with_legends(smiles_list: List[str],
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
        (1, 0.5, 0),      # Yellow for center atom
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
            centerColor=(1, 0, 0),
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
            centerColor=(1, 0, 0),
            extraColor=(0.9, 0.9, 0.9),
            drawOptions=options
        )

        return img
    else:
        print(f'No SMILES found with the given fingerprint indices list')
        return None


def show_match(smiles, atoms_indices_hlite, special_atom = None, mol_size=(300, 200), mol_name=None, return_type="show"):
    """
    Display a molecule with selected atoms and bonds highlighted.

    Parameters
    ----------
    smiles : str or rdkit.Chem.rdchem.Mol
        Molecule represented as a SMILES string or an RDKit Mol object.

    atoms_indices_hlite : iterable of int
        Atom indices to highlight. Bonds connecting highlighted atoms are also
        highlighted.

    special_atom : int or iterable of int, optional
        Atom index (or indices) to highlight in red instead of the default green.
        Useful for emphasizing one or more atoms within the highlighted substructure.

    mol_size : tuple of int, default=(300, 200)
        Width and height of the output image in pixels.

    mol_name : str, optional
        Legend displayed below the molecule.

    return_type : {"show", "svg", "img"}, default="show"
        Specifies the output format:
        - "show": display the SVG in the notebook.
        - "svg": return the SVG string.
        - "img": return an IPython.display.SVG object.

    Returns
    -------
    None, str, or IPython.display.SVG
        Output depends on `return_type`.
    """

    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = smiles

    n_atoms = mol.GetNumAtoms()

    atoms_indices_hlite = sorted(set(int(idx) for idx in atoms_indices_hlite if 0 <= idx < n_atoms))
    match_atom_set = set(atoms_indices_hlite)

    atom_colors = {idx: (0.5, 0.8, 0.5) for idx in atoms_indices_hlite}

    if special_atom is not None:
        if isinstance(special_atom, int):
            special_atom = [special_atom]
        for idx in special_atom:
            if 0 <= idx < n_atoms:
                atom_colors[idx] = (1.0, 0.3, 0.3)  # red

    match_bonds = []
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in match_atom_set and end_idx in match_atom_set:
            match_bonds.append(bond.GetIdx())

    drawer = rdMolDraw2D.MolDraw2DSVG(mol_size[0], mol_size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(drawer,
                                       mol,
                                       highlightAtoms=atoms_indices_hlite,
                                       highlightBonds=match_bonds,
                                       highlightAtomColors=atom_colors,
                                       legend=mol_name or "")

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("svg:", "")

    if return_type == "show":
        display(SVG(svg))
    elif return_type == "svg":
        return svg
    elif return_type == "img":
        return SVG(svg)
    else:
        raise ValueError("return_type must be one of {'show', 'svg', 'img'}")
