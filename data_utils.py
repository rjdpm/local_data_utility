##############################################################################################
#% FUNCTION: 
#% WRITER: SOUMITRA SAMANTA            DATE: 01-12-2020
#% For bug and others mail me at soumitramath39@gmail.com
#%--------------------------------------------------------------------------------------------
#% INPUT:
#% OUTPUT:
#%--------------------------------------------------------------------------------------------
#% EXAMPLE:
#%
##############################################################################################
import sys
import re
import pandas as pd
import yaml
import numpy as np
import pickle
import os
import sys
from tqdm import tqdm
import math
from functools import partial
import scipy.sparse as sp
from collections import OrderedDict

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
matplotlib.use('Agg')

from bokeh.plotting import ColumnDataSource, figure, output_file, show

from sklearn.metrics import pairwise_distances
import torch
from torch.utils.data import Dataset, DataLoader

# if torch.cuda.is_available():# if GPU available then load RAPIDS ML library
#     from cuml import PCA, TSNE, UMAP
# else:
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from mordred import Calculator, descriptors
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')# to disable RDkit smiles parsing standard output errors
from rdkit.Chem import Descriptors
from rdkit.Chem import RDConfig
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.AtomPairs import Pairs
sys.path.append('/home/rkmvu/Codes/data_utils_local/')
from hyperspherical_vae.distributions.von_mises_fisher import *

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer
#################################################################################################


def mol_to_fetures(smile):
    
    '''
    Given a smile this function will generate all possible descriptor values (approx 2600) of that smile.
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
    for name, func in Descriptors.__dict__.items():
        try:
            rdkit_properties[name] = func(mol)
        except:
            pass

    # Step 5: Compute Morgan fingerprints
    radius = 2  # Set radius for Morgan fingerprint
    nBits = 1024  # Set number of bits for the fingerprint
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

    # Convert Morgan fingerprints to a dictionary
    morgan_fp_dict = {f'MorganFP_{i}': int(bit) for i, bit in enumerate(morgan_fp)}

    # Step 6: Combine all descriptors into one dictionary
    all_descriptors = {**rdkit_properties, **mordred_descriptors, **morgan_fp_dict}
    
    return all_descriptors


def classify_acid_base(smile):
    
    '''
    Given a smile it will classify its acidic or basic nature. But it is not accurate.
    '''
    
    # Example classification based on a simple heuristic
    # This is a placeholder; actual classification may require more sophisticated analysis
    mol = Chem.MolFromSmiles(smile)
    hbd = Descriptors.NumHDonors(mol)
    
    hba = Descriptors.NumHAcceptors(mol)
    
    if hbd > 0 and hba > 0:
        return 'Amphoteric'
    elif hbd > 0:
        return 'Acidic'
    elif hba > 0:
        return 'Basic'
    else:
        return 'Neutral'

class MolFingerprint():
    
    def __init__(self, name='RDK'):
        
        self.name= name
        if name=='RDK':
            self.fingerprint = Chem.RDKFingerprint
        elif name=='AtomPair':
            self.fingerprint = Chem.rdMolDescriptors.GetAtomPairFingerprint
        elif name=='TopologicalTorsion':
            self.fingerprint = Chem.rdMolDescriptors.GetTopologicalTorsionFingerprint
        elif name=='MACCSKeys':
            self.fingerprint = Chem.rdMolDescriptors.GetMACCSKeysFingerprint    
        elif name=='Morgan':
            self.fingerprint = Chem.rdMolDescriptors.GetMorganFingerprint 
        elif name=='Pattern':
            self.fingerprint = Chem.rdmolops.PatternFingerprint
        elif name=='2DPharmacophore':# Has some issues
            self.fingerprint = Generate.Gen2DFingerprint
        elif name=='ErG':# Has some issues
            self.fingerprint = Chem.rdReducedGraphs.GetErGFingerprint    
        elif name=='ReducedGraph':# Has some issues
            self.fingerprint = Chem.rdReducedGraphs.GenerateErGFingerprintForReducedGraph   
        else:
            raise ValueError('Define your fingerprint "{}"' .format(name))
        

class tensor_data_loader(Dataset):
    def __init__(self, data):
        self.data = data.type(torch.float32)# data type conversion
        
    def __getitem__(self, index):
        x = self.data[index]
        
        return x, index
    
    def __len__(self):
        return len(self.data)
    
    
class PARSE_SMILES():
    """SMILES string parser and tokenization"""
    
    def __init__(self, smiles):
        self.smiles = smiles
    #------------------------------------------------------
    
    @staticmethod
    def load_smiles_from_csv(smiles_data_filename, smiles_column_ids='smiles', max_string_len=250, smiles_check_flag=0, cannonical_check=0, verbose=0):
        """
        Read smiles from a csv file
        
        INPUT:
        smiles_data_filename- smiles input file name (csv)
        smiles_column_ids- column name of smiles data
        
        OUTPUT:
        smiles- required smiles
        """
        
        print('Loading smiles form: "{}"' .format(smiles_data_filename))
        df = pd.read_csv(smiles_data_filename)
        df[smiles_column_ids] = df[smiles_column_ids].str.strip()
        df = df[df[smiles_column_ids].str.len() <= max_string_len]
        smiles = df[smiles_column_ids].tolist()

        # check smiles validity
        if smiles_check_flag:
            smiles, _ = PARSE_SMILES.check_smiles_validity(smiles, verbose=verbose)
        # convert into cannonical smiles
        if cannonical_check:
            smiles = PARSE_SMILES._canonicalise_batch(smiles, verbose=verbose)
            
        return smiles
    #------------------------------------------------------ 
    
    @staticmethod
    def load_smiles_char(char_filename):
        """
        Read smiles character from a json file
        
        INPUT:
        char_filename- smiles characters input file name (json)
        
        OUTPUT:
        smiles_char- smiles characters
        smiles_char_map- smiles characters mapping
        """

        smiles_char = yaml.safe_load(open(char_filename))
        smiles_char_map = {}
        smiles_char_map['domain'] = dict((c, i) for i, c in enumerate(smiles_char))
        smiles_char_map['co-domain'] = dict((i, c) for i, c in enumerate(smiles_char))
        
        return smiles_char, smiles_char_map
    #------------------------------------------------------
    
    @staticmethod
    def load_smiles_max_length(smiles_length_filename):
        """
        Read smiles string max-length from a 'text' file
        
        INPUT:
        smiles_length_filename- smiles string length input file name (text)
        
        OUTPUT:
        max_smiles_length- smiles string max-length
        """

        with open(smiles_length_filename, 'r') as f:
            max_smiles_length = int(f.readline())
        
        return max_smiles_length
    #------------------------------------------------------
    
    @staticmethod
    def check_smiles_validity(smiles, verbose=0):
        """
        Check smiles is valid or not!
        
        INPUT: 
        smiles- input smiles as a "list"
        flag_progress- display progress bar during checking(0- no, >0 display) (default 0)
        
        OUTPUT:
        smiles- valid smiles (only)
        check_ids- smiles validity flag (True or False) of the input smiles
        """
        
        if verbose:
            print('<===checking smiles validity===>')
            iterator = enumerate(tqdm(smiles))
        else:
            iterator = enumerate(smiles)
            
        check_ids = [True]*len(smiles)
        for i, smile in iterator:
            try:
                Chem.Kekulize(Chem.MolFromSmiles(smile))
            except:
                check_ids[i] = False 

#         if verbose:
#             print('<===checking smiles validity===>')
#             check_ids = [Chem.MolFromSmiles(smile) is not None for smile in tqdm(smiles)]
#         else:
#             check_ids = [Chem.MolFromSmiles(smile) is not None for smile in smiles]

        smiles = [smiles[i] for i, flag in enumerate(check_ids) if flag]

        return smiles, check_ids
    #------------------------------------------------------
    
    @staticmethod
    def parse_special(smiles):
        """Parse SMILES """
    
        # Canonicalise:
        smiles = PARSE_SMILES._canonicalise(smiles)
    
        # Get tokens:
        tokens = PARSE_SMILES._get_special_tokens(smiles)
    
        return smiles, tokens
    #------------------------------------------------------
    
    @staticmethod
    def parse_special_batch(smiles, verbose=0):
        """Parse SMILES in a batch"""
        
        dict_char = {}
        if verbose:
            iterator = enumerate(tqdm(smiles))
        else:
            iterator = enumerate(smiles)
        for i, sm in iterator:
            
            # Canonicalise:
            smiles[i] = PARSE_SMILES._canonicalise(smiles[i])
        
            # Get tokens:
            tokens = PARSE_SMILES._get_special_tokens(smiles[i])
            # print("token-----")
            # print(tokens)
            
            for c in tokens:
                if c in dict_char.keys():
                    dict_char[c] += 1
                else:
                    dict_char[c] = 1
    
        return smiles, dict_char
    #------------------------------------------------------
    
    @staticmethod
    def _get_special_tokens(smiles):
        """Special tokenize Cl->Cl and Br->Br and rest are one character"""
        
        num_char = len(smiles)
        
        count = 0
        tokens = []
        while(count<num_char):
            char = smiles[count]
            
            # for "Br"
            if char=='B':
                if((count+1 < num_char) and (smiles[count+1]=='r')):
                    tokens.append(smiles[count:count+2])
                    count += 2
                else:
                    tokens.append(char)
                    count += 1
            # for "Cl"
            elif char=='C':
                if((count+1 < num_char) and (smiles[count+1]=='l')):
                    tokens.append(smiles[count:count+2])
                    count += 2
                else:
                    tokens.append(char)
                    count += 1
            else:
                tokens.append(char)
                count += 1
                
        return tokens
    #------------------------------------------------------
             
    @staticmethod
    def parse(smiles):
        """Parse SMILES """
    
        # Canonicalise:
        smiles = PARSE_SMILES._canonicalise(smiles)
    
        # Get tokens:
        tokens = []
        PARSE_SMILES._get_tokens(smiles, tokens)
    
        return smiles, tokens
    #------------------------------------------------------
    
    @staticmethod
    def parse_batch(smiles, verbose=0):
        """Parse SMILES """
        
        dict_char = {}
        if verbose:
            iterator = enumerate(tqdm(smiles))
        else:
            iterator = enumerate(smiles)
        for i, sm in iterator:
            
            # Canonicalise:
            smiles[i] = PARSE_SMILES._canonicalise(smiles[i])
        
            # Get tokens:
            tokens = []
            PARSE_SMILES._get_tokens(smiles[i], tokens)
            
            for c in tokens:
                if c in dict_char.keys():
                    dict_char[c] += 1
                else:
                    dict_char[c] = 1
    
        return smiles, dict_char
    #------------------------------------------------------
    
    @staticmethod
    def _canonicalise(smiles):
        """Canonicalise"""
        
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)
    #------------------------------------------------------
    
    @staticmethod
    def _canonicalise_batch(smiles, verbose=0):
        """Canonicalise in a batch"""
        
        if verbose:
            print('<===canonicalise smiles===>')
            iterator = enumerate(tqdm(smiles))
        else:
            iterator = enumerate(smiles)
        
        return [Chem.MolToSmiles(Chem.MolFromSmiles(smile), isomericSmiles=False) for i, smile in iterator]
    #------------------------------------------------------
    
    @staticmethod
    def _get_tokens(substr, tokens):
        """ SMILES tokenize"""
    
        # Halogens (Br and Cl) or square bracketed terms:
        match = re.match(r'^(Br|Cl|\[[^]]*\])', substr)
    
        if match:
            group = match.group(1)
            tokens.append(group)
            PARSE_SMILES._get_tokens(substr[len(group):], tokens)
            return
    
        # Number(s):
        match = re.match(r'^(%?\d+)', substr)
    
        if match:
            group = match.group(1)
            tokens[-1] += group
            PARSE_SMILES._get_tokens(substr[len(group):], tokens)
            return
    
        if substr:
            tokens.append(substr[0])
            PARSE_SMILES._get_tokens(substr[1:], tokens)
    #------------------------------------------------------
    
    @staticmethod
    def check_smiles_validity_special_vocabulary(smiles, smiles_char, max_string_len, smiles_check_flag=0, cannonical_check=0, verbose=0):
    
        check_ids = [True]*len(smiles)
        # check valid smiles
        if smiles_check_flag:
            _, check_ids = PARSE_SMILES.check_smiles_validity(smiles, verbose=verbose)
    
        # check smiles unique characters
        if verbose:
            print("Checking smiles based on required vocabulary")
            iterator = enumerate(tqdm(smiles))
        else:
            iterator = enumerate(smiles)
        for i, smile in iterator:
            tokens = PARSE_SMILES._get_special_tokens(smile)
            for char in tokens:
                if char not in smiles_char:
                    check_ids[i] = False
    
        # check string length
        for i, sm in enumerate(smiles):
            if len(sm) > max_string_len:
                check_ids[i] = False
    
        # take only valid smiles
        smiles = [smiles[i] for i, val in enumerate(check_ids) if val]
        
        # convert into cannonical smiles
        # print("test------>")
        # print(smiles)
        if cannonical_check:
            smiles = PARSE_SMILES._canonicalise_batch(smiles)
        
        return smiles, check_ids
    #------------------------------------------------------
    
    @staticmethod
    def check_smiles_validity_neil_special_vocabulary(smiles, smiles_char, max_string_len, smiles_check_flag=0, cannonical_check=0, verbose=0):
    
        check_ids = [True]*len(smiles)
        # check valid smiles
        if smiles_check_flag:
            _, check_ids = PARSE_SMILES.check_smiles_validity(smiles, verbose=verbose)
    
        # check smiles unique characters
        if verbose:
            print("Checking smiles based on required vocabulary")
            iterator = enumerate(tqdm(smiles))
        else:
            iterator = enumerate(smiles)
        for i, smile in iterator:
            # Get tokens:
            tokens = []
            PARSE_SMILES._get_tokens(smile, tokens)
#             tokens = PARSE_SMILES._get_special_tokens(smile)
            for char in tokens:
                if char not in smiles_char:
                    check_ids[i] = False
    
        # check string length
        for i, sm in enumerate(smiles):
            if len(sm) > max_string_len:
                check_ids[i] = False
    
        # take only valid smiles
        smiles = [smiles[i] for i, val in enumerate(check_ids) if val]
        
        # convert into cannonical smiles
        # print("test------>")
        # print(smiles)
        if cannonical_check:
            smiles = PARSE_SMILES._canonicalise_batch(smiles)
        
        return smiles, check_ids
    #------------------------------------------------------
    
    # fixed lenght string conversion
    @staticmethod
    def padding_smiles(string, max_string_len=250, padding='right'):
        
        if len(string) <= max_string_len:
            if padding == 'right':
                string = string + " " * (max_string_len - len(string))
            elif padding == 'left':
                string = " " * (max_string_len - len(string)) + string
            elif padding == 'none':
                string = string
            else:
                raise ValueError('Define your padding scheme: "{}"' .format(padding))
            
            return string
    #------------------------------------------------------
    
    # smiles to onehot vector conversion
    @staticmethod
    def smiles2onehot_vector(smiles, char_indices, max_string_len=250, padding='right', tokenize='', verbose=0):
        
        # padding smiles (not consider string > max_string_len)
        smiles = [PARSE_SMILES.padding_smiles(i, max_string_len, padding)
                  for i in smiles if PARSE_SMILES.padding_smiles(i, max_string_len, padding)]
    
        X = np.zeros((len(smiles), max_string_len, len(char_indices.keys())), dtype=np.bool_)
                      
        if verbose:
            print("<===smiles to onehot===>")
            iterator = enumerate(tqdm(smiles))
        else:
            iterator = enumerate(smiles)
        for i, smile in iterator:
            if tokenize=='spacial':
                tokens = PARSE_SMILES._get_special_tokens(smile)
            elif tokenize=='neil_token':
                tokens = []
                PARSE_SMILES._get_tokens(smile, tokens)
            else:
                tokens = list(smile)
            for t, char in enumerate(tokens):
    #             print(t, char)
                try:
                    X[i, t, char_indices[char]] = 1
                except KeyError as e:
                    print("ERROR: Check chars file. bad SMILES:", smile)
                    raise e
        return X
    #------------------------------------------------------
    
    # onehot vector to smiles string conversion
#     @staticmethod
#     def onehot2smiles(X, indices_chars, remove_space=True):
        
#         smiles = [None]*X.shape[0]
#     #     print('<=== onehot to smiles ===>')
#         for i in range(X.shape[0]):
#             temp_str = ""
#             for j in range(X.shape[1]):
#                 index = np.argmax(X[i,j,:])
#                 temp_str += indices_chars[index]
#             if remove_space:
#                 smiles[i] = temp_str.replace(" ", "")
#             else:
#                 smiles[i] = temp_str.strip()
                           
#         return smiles
    @staticmethod
    def onehot2smiles(X, indices_chars, remove_space=True):

        smiles = [None]*X.shape[0]
    #     print('<=== onehot to smiles ===>')
        for i in range(X.shape[0]):
            smiles[i] = ""
            index = np.argmax(X[i,:,:], axis=1)
            for j in index:
                smiles[i] = ''.join([smiles[i], indices_chars[j]])
            if remove_space:
                smiles[i] = smiles[i].replace(" ", "")
            else:
                smiles[i] = smiles[i].strip()

        return smiles
    #------------------------------------------------------
    
    # draw smiles as a grid in an image
    @staticmethod
    def draw_molecules(smiles, plot_nrows_module=5, plot_subimg_size=(200, 200), legends=[]):

        num_molecules = len(smiles)
        mol = [None]*num_molecules
        for i in range(num_molecules):
            mol[i] = Chem.MolFromSmiles(smiles[i])
        # drow molecules
        if len(legends):
            img = Draw.MolsToGridImage(mol, molsPerRow=plot_nrows_module, subImgSize=plot_subimg_size, legends=legends, returnPNG = False)
        else:
            img = Draw.MolsToGridImage(mol, molsPerRow=plot_nrows_module, subImgSize=plot_subimg_size, returnPNG = False)

        return img
    #------------------------------------------------------
    
    @staticmethod
    def mol_prop_logp(smiles):
        """
        logP- Partition Coefficient: log10 (P); P = Partition Coefficient = [organic]/[aqueous]
        Where [ ] indicates the concentration of solute in the organic and aqueous partition
        """
        
        return [Descriptors.MolLogP(Chem.MolFromSmiles(sm)) for sm in smiles]
    #------------------------------------------------------
    
    @staticmethod
    def mol_prop_qed(smiles):
        """
        Quantitative estimation of drug-likeness (qed)
        """
        
        return [Descriptors.qed(Chem.MolFromSmiles(sm)) for sm in smiles]
    #------------------------------------------------------
    
    @staticmethod
    def mol_prop_sas(smiles):
        """
        Synthetic assessibility score (sas)
        """
        
        return [sascorer.calculateScore(Chem.MolFromSmiles(sm)) for sm in smiles]
    #------------------------------------------------------
    
    @staticmethod
    def mol_prop_mass(smiles):
        """
        Molecular mass (mass)
        """
        
        return [Descriptors.ExactMolWt(Chem.MolFromSmiles(sm)) for sm in smiles]
    #------------------------------------------------------
    
    @staticmethod
    def mol_formula(smiles):
        """
        Molecular formula
        """
        
        formula = [CalcMolFormula(Chem.MolFromSmiles(sm)) for sm in smiles]
        
        return formula
    #------------------------------------------------------
#################################################################################################  

class PARSE_MASSSPEC():
    """SMILES string parser and tokenization"""
    
    def __init__(self, smiles):
        self.smiles = smiles
    #------------------------------------------------------
    
    @staticmethod
    def to_numpy(array_str, sep=','):
        '''Convert array_str to numpy.'''
        
        return np.fromstring(array_str[1:-1], sep=sep)
    #------------------------------------------------------
    
    @staticmethod
    def massspec2binaryimages(MS, min_mass=30, max_mass=1000, num_div_unitmass=100, verbose=0):
    
        num_samples = len(MS)

        num_unitmass = max_mass - min_mass
        ms_binaryimages = np.zeros((num_samples, num_unitmass, num_div_unitmass), dtype=np.bool_)
        
        if verbose:
            print('<===mass-spec to one-hot===>')
            iterator = enumerate(tqdm(MS))
        else:
            iterator = enumerate(MS)
            
        for i, mass in iterator:

            mass = np.asarray(mass)
            mass = mass[(mass>min_mass) & (mass<max_mass)]
            mass = mass - min_mass

            row_id = np.floor(mass).astype(int)
            col_id = np.floor((mass - row_id)*num_div_unitmass).astype(int)
            ms_binaryimages[i, row_id, col_id] = True

        return ms_binaryimages
    #------------------------------------------------------
    
    # get mass-spec data as a iterator
    @staticmethod
    def get_massspec_data_pandas(filename, 
                                 smiles_col='smiles', 
                                 massspec_col='METFRAG_MZ', 
                                 massspec_sep=',', 
                                 num_samples=100000, 
                                 batch_size=128, 
                                 min_mass=30, 
                                 max_mass=1000, 
                                 num_div_unitmass=100, 
                                 verbose=0):
        
        # print('Reading smiles and mass-spec from: "{}"' .format(filename))
        for chunk in pd.read_csv(filename, nrows=num_samples, chunksize=batch_size):
            smiles = chunk[smiles_col].tolist()
            mass_spec = PARSE_MASSSPEC.massspec2binaryimages(chunk[massspec_col].apply(PARSE_MASSSPEC.to_numpy, sep=massspec_sep),
                                                             min_mass=min_mass, 
                                                             max_mass=max_mass, 
                                                             num_div_unitmass=num_div_unitmass, 
                                                             verbose=verbose)

            yield smiles, mass_spec
    #------------------------------------------------------
    
    # get mass-spec data as a iterator
    @staticmethod
    def load_massspec_from_csv(filename, 
                                 smiles_col='smiles', 
                                 massspec_col='METFRAG_MZ', 
                                 massspec_sep=',', 
                                 num_samples=100000, 
                                 batch_size=128, 
                                 min_mass=30, 
                                 max_mass=1000, 
                                 num_div_unitmass=100, 
                                 verbose=0):
        
        print('Reading smiles and mass-spec from: "{}"' .format(filename))
        
        if num_samples==batch_size:#just to avoid the memory swap delay
            for i, (smls, X) in enumerate(PARSE_MASSSPEC.get_massspec_data_pandas(filename, 
                                              smiles_col, 
                                              massspec_col, 
                                              massspec_sep, 
                                              num_samples, 
                                              batch_size, 
                                              min_mass, 
                                              max_mass, 
                                              num_div_unitmass, 
                                              verbose)):# iteration over each minibatch

                start_idx = (i*batch_size)%num_samples
                smiles = smls
                mass_spec = X
        else:
            num_unitmass = max_mass - min_mass
            smiles = [None]*num_samples
            mass_spec = np.zeros((num_samples, num_unitmass, num_div_unitmass), dtype=np.bool_)
            for i, (smls, X) in enumerate(PARSE_MASSSPEC.get_massspec_data_pandas(filename, 
                                              smiles_col, 
                                              massspec_col, 
                                              massspec_sep, 
                                              num_samples, 
                                              batch_size, 
                                              min_mass, 
                                              max_mass, 
                                              num_div_unitmass, 
                                              verbose)):# iteration over each minibatch

                start_idx = (i*batch_size)%num_samples
                end_idx = start_idx + len(smls)
                smiles[start_idx:end_idx] = smls
                mass_spec[start_idx:end_idx, :] = X
                
            smiles= smiles[:end_idx]
            mass_spec = mass_spec[:end_idx, :]
        print('----------------------------------------------------------------')
        
        return smiles, mass_spec        
    #------------------------------------------------------
    
    
#################################################################################################


# sampling around mean by adding some variation along each dimension
def random_neighbour_samples(data_mu, data_sigma, num_samples=100, scale=1e-2):

    data_sigma = data_sigma.repeat(num_samples, 1) #torch.exp(0.5*data_sigma).repeat(num_samples, 1)
    data_mu = data_mu.repeat(num_samples, 1)
    print(data_mu.shape, data_sigma.shape)
    vmf = VonMisesFisher(data_mu, data_sigma)
    print(vmf)
    z = vmf.rsample()
    
    # eps = scale * (1*(torch.rand_like(data_sigma) - 0.5))
#     z = data_mu.repeat(num_samples, 1) + eps*data_sigma
    
    return z    #, eps
#------------------------------------------------------

def DrawMassSpecMatchedMolecules(all_results_filename, \
                                 save_result_path=[], \
                                 sigma_scale=1, \
                                 fingerprint_name=['RDK', 'AtomPair', 'TopologicalTorsion', 'MACCSKeys', 'Morgan', 'Pattern'], \
                                 top_stats=1, \
                                 flag_mol_plot=1, \
                                 n_bins=20, \
                                 xylabel_fontsize=20, \
                                 verbose=1 \
                                ):
    
    if len(save_result_path)==0:
        save_result_path = ''.join([all_results_filename[:-len(all_results_filename.split('/')[-1])], 'images/'])
    save_result_path = create_folder(save_result_path)
    
    all_results = load_dict_pickle(all_results_filename)
    smiles_test = all_results['smiles']
    num_test_samples = len(smiles_test)
    
    stats_mol_weight = np.zeros(num_test_samples)
    stats_edist_latent = np.zeros(num_test_samples)
    stats_tsdist = np.zeros(num_test_samples)
    
    dict_result_stats = OrderedDict({'qm':[], 'qm_mf':[], 'qm_mw': [], \
                                     'sm_top1':[], 'sm_top1_mf':[], 'sm_top1_mw':[], \
                                     'sm_top1_ts':[], 'sm_top1_ts_finrprint': [], 'sm_top1_ts_ed':[], 'sm_top1_ed':[]})
    
    for fp in fingerprint_name:
        dict_result_stats[fp] = []
    

    if verbose:
        iterator = tqdm(range(num_test_samples))
    else:
        iterator = range(num_test_samples)
    for ids in iterator:

        temp_Y_test = all_results['smiles_latent_space_rep'][ids]
        neighbour_smiles = all_results['neighbour_smiles'][ids]
        neighbour_smiles_latent_space_rep = all_results['neighbour_smiles_latent_space_rep'][ids] 


        chem_formula_qm = CalcMolFormula(Chem.MolFromSmiles(smiles_test[ids]))
        mol_weight_qm = ExactMolWt(Chem.MolFromSmiles(smiles_test[ids]))
        
        dict_result_stats['qm'].append(smiles_test[ids])
        dict_result_stats['qm_mf'].append(chem_formula_qm)
        dict_result_stats['qm_mw'].append(mol_weight_qm)

        if len(neighbour_smiles):

            chem_formula_sm = [CalcMolFormula(Chem.MolFromSmiles(sm)) \
                            for sm in neighbour_smiles \
                           ]

            mol_weight_sm = [ExactMolWt(Chem.MolFromSmiles(i)) \
                          for i in neighbour_smiles \
                         ]
            # calculate the wight difference
            mol_weight_diff = np.abs(np.asarray(mol_weight_sm) - \
                                     mol_weight_qm \
                                    )
            # sort based on the wight difference
            idx_sort_mw = np.argsort(mol_weight_diff)

            # calculate euclidean dist in latent space
            edist_latent = pairwise_distances(temp_Y_test, \
                                              neighbour_smiles_latent_space_rep, metric='euclidean' \
                                             )[0,:]
            edist_latent = 1./(1+edist_latent)
            idx_sort_edl = np.argsort(-edist_latent)

            # calculate tanimoto similarity
            temp_tsdist = similarity_smiles_tanimoto([smiles_test[ids]], \
                                                neighbour_smiles, \
                                                fingerprint_name=fingerprint_name)
            top1_fingerprint_ids = temp_tsdist.argmax(0)[0,:]
            tsdist = temp_tsdist.max(0)[0,:]
            idx_sort_tms = np.argsort(-tsdist)
            
            stats_mol_weight[ids] = np.asarray([mol_weight_diff[i] for i in idx_sort_mw[:top_stats]]).mean()
            stats_edist_latent[ids] = np.asarray([edist_latent[i] for i in idx_sort_edl[:top_stats]]).mean()
            stats_tsdist[ids] = np.asarray([tsdist[i] for i in idx_sort_tms[:top_stats]]).mean()
            
            # record top-1 stats
            dict_result_stats['sm_top1'].append(neighbour_smiles[idx_sort_tms[0]])
            dict_result_stats['sm_top1_mf'].append(chem_formula_sm[idx_sort_tms[0]])
            dict_result_stats['sm_top1_mw'].append(mol_weight_sm[idx_sort_tms[0]])
            dict_result_stats['sm_top1_ts'].append(tsdist[idx_sort_tms[0]])
            dict_result_stats['sm_top1_ts_finrprint'].append(fingerprint_name[top1_fingerprint_ids[idx_sort_tms[0]]])
            dict_result_stats['sm_top1_ts_ed'].append(edist_latent[idx_sort_tms[0]])
            dict_result_stats['sm_top1_ed'].append(edist_latent[idx_sort_edl[0]])
            for _i, fp in enumerate(fingerprint_name):
                dict_result_stats[fp].append(temp_tsdist[_i, 0, idx_sort_tms[0]])
            

            # plot matched molecules
            if flag_mol_plot:
                ## plot based on molecular weights
                smiles_mw = [smiles_test[ids]] + \
                [neighbour_smiles[i] for i in idx_sort_mw]
                legends_mw = ['q_m:'+chem_formula_qm + \
                              '-'+str(round(mol_weight_qm, 5)) + \
                             '-'+str(0.0)] + \
                [chem_formula_sm[i] + \
                 '-'+str(round(mol_weight_sm[i], 5)) + \
                 '-'+str(round(tsdist[i], 3)) \
                 for i in idx_sort_mw \
                ]
                img = PARSE_SMILES.draw_molecules(smiles_mw, legends=legends_mw)
                img.save(''.join([save_result_path, \
                                  'test_org_scale_', str(sigma_scale), \
                                  '_', str(ids), 
                                  '_mw', '.png']))

                # plot based on euclidean distance similarity
                smiles_edl = [smiles_test[ids]] + \
                [neighbour_smiles[i] for i in idx_sort_edl]
                legends_edl = ['q_m:'+chem_formula_qm + \
                              '-'+str(0.0) + \
                              '-'+str(round(mol_weight_qm, 4))] + \
                [chem_formula_sm[i] + \
                 '-'+str(round(edist_latent[i], 4)) + \
                 '-'+str(round(mol_weight_sm[i], 4)) \
                 for i in idx_sort_edl \
                ]
                img = PARSE_SMILES.draw_molecules(smiles_edl, legends=legends_edl)
                img.save(''.join([save_result_path, \
                                  'test_org_scale_', str(sigma_scale), \
                                  '_', str(ids), 
                                  '_edl', '.png']))

                # plot based on tanimoto similarity
                smiles_tms = [smiles_test[ids]] + \
                [neighbour_smiles[i] for i in idx_sort_tms]
                legends_tms = ['q_m:'+chem_formula_qm + \
                              '-'+str(0.0) + \
                              '-'+str(round(mol_weight_qm, 4))] + \
                [chem_formula_sm[i] + \
                 '-'+str(round(tsdist[i], 3)) + \
                 '-'+str(round(mol_weight_sm[i], 4)) \
                 for i in idx_sort_tms \
                ]
                img = PARSE_SMILES.draw_molecules(smiles_tms, legends=legends_tms)
                img.save(''.join([save_result_path, \
                                  'test_org_scale_', str(sigma_scale), \
                                  '_', str(ids), 
                                  '_tms', '.png']))

        else:
            print('No match found!')
            # record top-1 stats
            dict_result_stats['sm_top1'].append('not_found')
            dict_result_stats['sm_top1_mf'].append('not_found')
            dict_result_stats['sm_top1_mw'].append('not_found')
            dict_result_stats['sm_top1_ts'].append('not_found')
            dict_result_stats['sm_top1_ts_finrprint'].append('not_found')
            dict_result_stats['sm_top1_ts_ed'].append('not_found')
            dict_result_stats['sm_top1_ed'].append('not_found')
            for _i, fp in enumerate(fingerprint_name):
                dict_result_stats[fp].append('not_found')
            
        # save top-1 stats as a csv file
        all_result_save_filename = ''.join([save_result_path, \
                                            'all_result_stats_top1_nsamples_', str(num_test_samples), \
                                            '.xlsx'])
        pd.DataFrame.from_dict(dict_result_stats).to_excel(all_result_save_filename, index=False)
            
    # plot based on molecular weights
    xticks_min = stats_mol_weight.min()
    xticks_max = stats_mol_weight.max()
    xticks_step = (xticks_max - xticks_min)/10.0

    count, bins, _ = plt.hist(stats_mol_weight, bins=n_bins)

    plt.grid(linestyle='--')
    plt.xlabel( 'Molecular weights', fontsize=xylabel_fontsize)
    plt.xticks(np.arange(xticks_min, xticks_max+np.finfo(float).eps, xticks_step))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=count.sum()))

    plt.savefig(''.join([save_result_path, 'mol_weight_', str(len(stats_mol_weight)), '.png']), bbox_inches='tight')
    plt.show(block=False)
    plt.close()


    # plot based on Tanimoto similarity
    xticks_min = 0.0#stats_tsdist.min()
    xticks_max = 1.0#stats_tsdist.max()
    xticks_step = (xticks_max - xticks_min)/10.0

    count, bins, _ = plt.hist(stats_tsdist, bins=n_bins)

    plt.grid(linestyle='--')
    plt.xlabel( 'Tanimoto similarity', fontsize=xylabel_fontsize)
    plt.xticks(np.arange(xticks_min, xticks_max+np.finfo(float).eps, xticks_step))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=sum(count)))

    plt.savefig(''.join([save_result_path, 'tanimoto_sim_', str(len(stats_tsdist)), '.png']), bbox_inches='tight')
    plt.show(block=False)
    plt.close()



    # plot based on euclidean distance similarity
    xticks_min = 0.0#stats_edist_latent.min()
    xticks_max = 1.0#stats_edist_latent.max()
    xticks_step = (xticks_max - xticks_min)/10.0

    count, bins, _ = plt.hist(stats_edist_latent, bins=n_bins)

    plt.grid(linestyle='--')
    plt.xlabel( 'Euclidean similarity', fontsize=xylabel_fontsize)
    plt.xticks(np.arange(xticks_min, xticks_max+np.finfo(float).eps, xticks_step))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=sum(count)))

    plt.savefig(''.join([save_result_path, 'euclidean_sim_', str(len(stats_edist_latent)), '.png']), bbox_inches='tight')
    plt.show(block=False)
    plt.close()


    # plot based on euclidean distance similarity
    plt.scatter(stats_tsdist, stats_mol_weight, \
                s=80, alpha=0.3)

    plt.grid(linestyle='--')
    plt.xlabel('TS', fontsize=xylabel_fontsize)
    plt.ylabel('MW', fontsize=xylabel_fontsize)
    plt.title('TS vs MW', fontsize=xylabel_fontsize)

    plt.savefig(''.join([save_result_path, 'TS_vs_MW_', str(len(stats_edist_latent)), '.png']), bbox_inches='tight')
    plt.show(block=False)
    plt.close()
    
    return dict_result_stats, stats_tsdist, stats_mol_weight, stats_edist_latent
#------------------------------------------------------

# # plot the embedded data
def plot_data_embedding(data_embedding, embedding_method, figure_size=(10, 10), xticks_step=5, yticks_step=5, xylabel_fontsize=25, xyticks_fontsize=20, image_save_filename='embeding_plot'):
    
    # for plot axis ticks
    xticks_min = np.min(data_embedding[:, 0]) - (np.min(data_embedding[:, 0])%10.)
    xticks_max = np.max(data_embedding[:, 0]) + (10-abs(np.max(data_embedding[:, 0])%10.)) + 1

    yticks_min = np.min(data_embedding[:, 1]) - (np.min(data_embedding[:, 1])%10.)
    yticks_max = np.max(data_embedding[:, 1]) + (10-abs(np.max(data_embedding[:, 1])%10.)) + 1

    plt.rcParams["figure.figsize"] = figure_size

    # plot test data
    plt.scatter(data_embedding[:, 0], data_embedding[:, 1], s=3, alpha=0.3)

    #     plt.legend(loc='upper right')
    plt.xticks(np.arange(xticks_min, xticks_max, step=xticks_step), fontsize=xyticks_fontsize)
    plt.yticks(np.arange(yticks_min, yticks_max, step=yticks_step), fontsize=xyticks_fontsize)
    plt.grid(linestyle='--')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=xyticks_fontsize)
    plt.xlabel(''.join(['1st ', embedding_method, ' component']), fontsize=xylabel_fontsize)
    plt.ylabel(''.join(['2nd ', embedding_method, ' component']), fontsize=xylabel_fontsize)

    # save the plot
    plt.savefig('' .join([image_save_filename,'.png']), bbox_inches='tight')#save the results in eps form

    plt.show(block=False)
    plt.close()

# data embedding
def data_2D_embedding(data, embedding_method, figure_size=(10, 10),  xticks_step=5, yticks_step=5, xylabel_fontsize=25, xyticks_fontsize=20, image_save_filename='embeding_plot'):
    
    # PCA embedding
    if embedding_method == 'PCA':
        reducer = PCA(n_components=2)
        reducer.fit(data)
        data_embedding = reducer.transform(data)

    # t-SNE embedding
    elif embedding_method == 't-SNE':
        reducer = TSNE(n_components=2, init='random', random_state=0)
        data_embedding = reducer.fit_transform(data)

    # UMAP embedding
    elif embedding_method == 'UMAP':
        reducer = UMAP(random_state=np.random.seed(42))
        reducer.fit(data)
        data_embedding = reducer.transform(data)

    else:
        raise ValueError('define your embedding method "{}"' .format(embedding_method))
    
    plot_data_embedding(data_embedding, embedding_method, figure_size=figure_size, xticks_step=xticks_step, yticks_step=yticks_step, xylabel_fontsize=xylabel_fontsize, xyticks_fontsize=xyticks_fontsize, image_save_filename=image_save_filename)
    
    return data_embedding, reducer


# dataset partion indices
def data_partition_train_val_test_smiles(num_data_points, train_ratio=0.5, val_ratio=0.2, test_ratio=0.3):
    
    num_train_data_points = int(train_ratio*num_data_points)
    num_val_data_points = int(val_ratio*num_data_points)
    
    idx = np.random.permutation(num_data_points)
    idx_train = idx[:num_train_data_points]
    idx_val = idx[num_train_data_points:num_train_data_points+num_val_data_points]
    idx_test = idx[num_train_data_points+num_val_data_points:]

    return idx_train, idx_val, idx_test

# calculate tanimoto similariry
def similarity_smiles_tanimoto(smiles_set1, smiles_set2, fingerprint_name=['RDK'], r_morgan=3, verbose=0):
    
    similarity_mat = np.zeros((len(fingerprint_name), len(smiles_set1), len(smiles_set2)))
    for fp, name in enumerate(fingerprint_name):
        mol = MolFingerprint(name)
        if verbose:
            print('Doing TS for finger print: {}' .format(name))
            iterator = enumerate(tqdm(smiles_set1))
        else:
            iterator = enumerate(smiles_set1)
            
        for i, s1 in iterator:
            for j, s2 in enumerate(smiles_set2):

                if name=='Morgan':
                    dist = DataStructs.TanimotoSimilarity(mol.fingerprint(Chem.MolFromSmiles(s1), r_morgan), mol.fingerprint(Chem.MolFromSmiles(s2), r_morgan))
                elif name=='2DPharmacophore':
                    dist = DataStructs.TanimotoSimilarity(mol.fingerprint(Chem.MolFromSmiles(s1), Gobbi_Pharm2D.factory), mol.fingerprint(Chem.MolFromSmiles(s2), Gobbi_Pharm2D.factory))
                else:
                    dist = DataStructs.TanimotoSimilarity(mol.fingerprint(Chem.MolFromSmiles(s1)), mol.fingerprint(Chem.MolFromSmiles(s2)))

                similarity_mat[fp, i, j] = dist

    return similarity_mat#.squeeze()


def create_folder(folder_name):
    if len(folder_name):
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
        
    return folder_name

def save_dict_csv_pandas(dict_name, save_filename='temp_save_filename.csv'):
    
    # create folder (if not) 
    _ = create_folder(save_filename[:-len(save_filename.split('/')[-1])])
    pd.DataFrame.from_dict(dict_name).to_csv(save_filename, index=False)

def save_dict_pickle(dict_name, save_filename='temp_save_filename.pkl', protocol=4):
    
    # create folder (if not) 
    _ = create_folder(save_filename[:-len(save_filename.split('/')[-1])])
    with open(save_filename,'wb') as f:
        pickle.dump(dict_name, f, protocol=protocol)
        
def load_dict_pickle(input_filename):
    with open(input_filename,'rb') as f:
        dict_name = pickle.load(f)
    
    return dict_name

def line_count_csv_file(filename, chunksize=1000):
    
    print('Counting number of data point in: "{}"' .format(filename))
    count = 0
    for chunk in tqdm(pd.read_csv(filename, usecols=[0], chunksize=chunksize)):
        count += len(chunk)
            
    return count

def find_continous_nonzeros_pos_1Darray(row_sum):
    
    non_zeros_ids = np.where(row_sum)[0]
    non_zeros_ids_diff = np.abs(non_zeros_ids[1:] - non_zeros_ids[:-1] - 1)
    exact_non_zeros_ids = np.where(non_zeros_ids_diff)[0]

    non_zeros_pos_ids = np.zeros((len(exact_non_zeros_ids)+1, 2))
    non_zeros_pos_ids[0, 0], non_zeros_pos_ids[0, 1] = non_zeros_ids[0], non_zeros_ids[exact_non_zeros_ids[0]]

    for i in range(len(exact_non_zeros_ids)-1):
        non_zeros_pos_ids[i+1, 0], non_zeros_pos_ids[i+1, 1] = non_zeros_ids[exact_non_zeros_ids[i]+1], non_zeros_ids[exact_non_zeros_ids[i+1]]
    non_zeros_pos_ids[-1, 0], non_zeros_pos_ids[-1, 1] = non_zeros_ids[exact_non_zeros_ids[-1]+1], non_zeros_ids[-1]
    non_zeros_pos_ids = non_zeros_pos_ids.astype('int')
    
    return non_zeros_pos_ids

def remove_uniform_image_border(image_array, border_pixl_value=255.0, gap_thresh=5):
    
    """
    INPUT: image_array- numpy 3D array
           border_pixl_value- uniform color values 
    
    """
    
    Temp_val = (border_pixl_value - image_array.astype('float32')).sum(axis=2)
    
    row_sum = Temp_val.sum(axis=1)
    non_zeros_row_pos_ids = find_continous_nonzeros_pos_1Darray(row_sum)
    
    for i in range(non_zeros_row_pos_ids.shape[0]):
        if(i%2):
            row_sum[non_zeros_row_pos_ids[i,0]-gap_thresh:non_zeros_row_pos_ids[i,1]+3*gap_thresh] = 1
        else:
            row_sum[non_zeros_row_pos_ids[i,0]-gap_thresh:non_zeros_row_pos_ids[i,1]+gap_thresh] = 1
    
    col_sum  = Temp_val.sum(axis=0)
    non_zeros_col_pos_ids = find_continous_nonzeros_pos_1Darray(col_sum)
    for i in range(non_zeros_col_pos_ids.shape[0]):
        col_sum[non_zeros_col_pos_ids[i,0]-gap_thresh:non_zeros_col_pos_ids[i,1]+gap_thresh] = 1
    row_sum = row_sum > 0
    col_sum = col_sum > 0    
    image_array = image_array[row_sum, :]
    image_array = image_array[:, col_sum, :].astype('uint8')
    
    return image_array

def plot_embedded_smiles_string(x, y, smiles, embedding_method, fig_size=(600, 600), circle_size=3, title='Embedded smiles string plot', image_save_filename='embedded_smiles_string.html', colors=['dodgerblue'], legend=[], extra_data={}):
    
    if not isinstance(circle_size, list):
        circle_size = [circle_size]*len(x)
    if len(colors)==1:
        colors = colors*len(x)
    if len(legend)==0:
        legend = circle_size
    
    output_file(image_save_filename)# save output file
    
    source = ColumnDataSource(data=dict(
        x=x,
        y=y,
        smiles=smiles,
        circle_size=circle_size,
        colors=colors,
        legend=legend,
    ))

    tooltip = [
        ("(x,y)", "($x, $y)"),
        ("smiles", "@smiles"),
    ]
    
    # add extra disply in hover
    if len(extra_data.keys()):
        for key in extra_data.keys():
            tooltip.append((key, "@"+key))
            source.data[key] = extra_data[key]

    # call bokeh figure api
    fig = figure(plot_width=fig_size[0], plot_height=fig_size[1], tooltips=tooltip, title=title)
    # plot data
    fig.circle('x', 'y', size='circle_size', source=source, color='colors', legend_group='legend')

    fig.title.align = 'center'
    fig.xaxis.axis_label = ''.join([embedding_method, ' 1st comp'])
    fig.yaxis.axis_label = ''.join([embedding_method, ' 2nd comp'])

    show(fig)
    
    
def plot_embedded_smiles_images(x, y, smiles, embedding_method, flag_smiles_images=0, smiles_images_path='smiles_images/', fig_size=(700, 700), circle_size=3, title='Embedded smiles images plot', plot_subimg_size=(300, 300), image_format='png', image_save_filename=[], colors=['dodgerblue'], legend=[]):
    
    if not isinstance(circle_size, list):
        circle_size = [circle_size]*len(x)
    if len(colors)==1:
        colors = colors*len(x)
    if len(legend)==0:
        legend = circle_size
        
    ## smiles to images and saved in a folder
    if flag_smiles_images == 0:

        print('converting smiles to images and save as ".{}" in "{}"' .format(image_format, smiles_images_path))
        smiles_images_path = create_folder(smiles_images_path)
        decimal_file_number = len(str(len(smiles)))+2
        Temp_file_number = '%'+str(decimal_file_number)+'.'+str(decimal_file_number)+'d'
        for i, sm in enumerate(tqdm(smiles)):
            image = draw_molecules([sm], plot_nrows_module=1, plot_subimg_size=plot_subimg_size)
            image = remove_uniform_image_border(np.asarray(image))
            image = Image.fromarray(image)
            
            image_filename = ''.join([smiles_images_path,'images_', Temp_file_number%(i+1), '.', image_format])
            image.save(image_filename)
    
    # iamge files for hover
    images_filename = [None]*len(smiles)
    decimal_file_number = len(str(len(smiles)))+2
    Temp_file_number = '%'+str(decimal_file_number)+'.'+str(decimal_file_number)+'d'
    for i, sm in enumerate(tqdm(smiles)):
        images_filename[i] = ''.join([smiles_images_path.split('/')[-2],'/images_', Temp_file_number%(i+1), '.', image_format])
    
    # output image (html) file
    if len(image_save_filename)==0:
        image_save_filename = ''.join([''.join([smiles_images_path[:-(len(smiles_images_path.split('/')[-2])+1)]]), embedding_method, '_embedded_smiles_images.html'])
            
    output_file(image_save_filename)# save output file

    source = ColumnDataSource(data=dict(
        x=x,
        y=y,
        images=images_filename,
        circle_size=circle_size,
        colors=colors,
        legend=legend,
    ))

    tooltip = """
            <div>
                <div>
                    <img
                    src="@images" height="100" alt="@images"
                    style="float: left; margin: 0px 5px 5px 0px; image-rendering: pixelated;"
                    border="2"
                    ></img>
                     <span style="font-size: 10px; color: #696;">($x, $y)</span>
                </div>

            </div>
                  """
    # call bokeh figure api
    fig = figure(plot_width=fig_size[0], plot_height=fig_size[1], tooltips=tooltip, title=title)
    # plot data
    fig.circle('x', 'y', size='circle_size', source=source, color='colors', legend_group='legend')

    fig.title.align = 'center'
    fig.xaxis.axis_label = ''.join([embedding_method, ' 1st comp'])
    fig.yaxis.axis_label = ''.join([embedding_method, ' 2nd comp'])

    show(fig)

def execution_time(func, *args):
    
    tic = time.time()
    func_value = func(*args)#Calculate function value
    toc = time.time()
    
    return (toc - tic), func_value

# Evalutes 'qed', 'sas', 'logP', 'Molecular mass' of the given smiles :-
def qed_sas_logp_mass(smiles):

    qed = PARSE_SMILES.mol_prop_qed(smiles)
    sas = PARSE_SMILES.mol_prop_sas(smiles)
    logP = PARSE_SMILES.mol_prop_logp(smiles)
    mol_mass = PARSE_SMILES.mol_prop_mass(smiles)

    return torch.tensor([qed, sas, logP, mol_mass])
    
    
   
def main(smiles):
    '''main method.'''
    smiles_cano = []
    tokens  = []
    for sm in smiles:

        T_smiles_cano, T_tokens = PARSE_SMILES.parse(sm)
        smiles_cano.append(T_smiles_cano)
        tokens.extend(T_tokens)
        print(T_smiles_cano, T_tokens)
        # print()
        
    return smiles_cano, tokens
        
if __name__ == '__main__':
    
    smiles = ['C1=CC=CC=C1', 'C1:C:C:C:C:C1', 'c1ccccc1',
                    'O=Cc1ccc(O)c(OC)c1'
                    'CN=C=O',
                    'CN1CCC[C@H]1c2cccnc2',
                    'CCc(c1)ccc2[n+]1ccc3c2[nH]c4c3cccc4',
                    'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1',
                    'CCc(c%99)ccc2[n+]%99ccc3c2[nH]c4c3cccc4',
                    'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1',
                    r'CCC[C@@H](O)CC\C=C\C=C\C#CC#C\C=C\CO', 
                    'O=C1N(CCc2ccccc2)C[C@@H]2C[C@@H](c3ccc(OC(F)F)cc3)[NH+]3CCC[C@@]123', 'CCCCOCCOc1c(C)cc(C)cc1Br']
    
    
    
    smiles_cano, tokens = main(smiles)
    
    smiles_cano_batch, tokens_batch = PARSE_SMILES.parse_batch(smiles+smiles+smiles, 1)
    
    tokens = sorted(list(set(tokens)))
    tokens_batch = sorted(tokens_batch)
    
    if len(tokens) != len(tokens_batch):
        print("token are not same!")
    else:
        flag_equal = 1
        for i, _ in enumerate(tokens):
            if tokens[i] != tokens_batch[i]:
                print("token are not same! {}/{}" .format(tokens[i], tokens_batch[i]))
                flag_equal = 0
        if flag_equal:
            print("same token")
     
    tokens = [None]*len(smiles)
    for i, sm in enumerate(smiles):
        tokens[i] = PARSE_SMILES._get_special_tokens(sm)
        
        
    smiles_special, special_tokens = PARSE_SMILES.parse_special_batch(smiles, 1)
    smiles_val, val_ids = PARSE_SMILES.check_smiles_validity_special_vocabulary(smiles, list(special_tokens.keys()), 2000, 1, 1, 1)
                
    
    