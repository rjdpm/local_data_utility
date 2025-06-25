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

import re
import pandas as pd
import yaml
import numpy as np
import pickle
from tqdm import tqdm
import random

import matplotlib
matplotlib.use('Agg')
from rdkit import Chem
from rdkit.Chem import Draw
from input_output import *
#################################################################################################

__all__ = [
    'PARSE_SMILES',
]


class PARSE_SMILES():
    """SMILES string parser and tokenization"""
    
    def __init__(self, smiles):
        self.smiles = smiles
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
        
        return [Chem.MolToSmiles(Chem.MolFromSmiles(smile), isomericSmiles=False) for i, smile in tqdm(iterator)]
    
    
    @staticmethod
    def load_smiles_from_parquet(smiles_data_filename, smiles_column_ids='smiles', max_string_len=150, smiles_check_flag=0, cannonical_check=0, n_data = 'all', verbose=0):
        
        """
        Read smiles from a parquet file
        
        INPUT:
        smiles_data_filename- smiles input file name (parquet)
        smiles_column_ids- column name of smiles data
        
        OUTPUT:
        smiles- required smiles
        """
        
        print('Loading smiles form: "{}"' .format(smiles_data_filename))
        df = read_parquet_file_to_dict_pandas(smiles_data_filename)
        df = pd.DataFrame.from_dict(df)
        df[smiles_column_ids] = df[smiles_column_ids].str.strip()
        df = df[df[smiles_column_ids].str.len() <= max_string_len]
        smiles = df[smiles_column_ids].tolist()
        
        if n_data == 'all':
            smiles = smiles
        elif isinstance(n_data, int):
            smiles = random.sample(smiles, n_data)
        else:
            raise ValueError(f"n_data should be a integer value or 'all' but got {type(n_data)} instead")

        # check smiles validity
        if smiles_check_flag:
            smiles, _ = PARSE_SMILES.check_smiles_validity(smiles, verbose=verbose)
        # convert into cannonical smiles
        if cannonical_check:
            smiles = PARSE_SMILES._canonicalise_batch(smiles, verbose=verbose)
        print('Loading Complete.')
        print('='*80)
            
        return smiles
    #------------------------------------------------------ 
    
    
    
    @staticmethod
    def load_smiles_from_csv(smiles_data_filename, smiles_column_ids='smiles', max_string_len=250, smiles_check_flag=0, cannonical_check=0, n_data = 'all', verbose=0):
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
        
        if n_data == 'all':
            smiles = smiles
        elif isinstance(n_data, int):
            smiles = random.sample(smiles, n_data)
        else:
            raise ValueError(f"n_data should be a integer value or 'all' but got {type(n_data)} instead")

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
             
    
    # fixed lenght string conversion
    @staticmethod
    def padding_smiles(string, max_string_len=150, padding='right'):
        
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
    def smiles2onehot_vector(smiles, char_indices, max_string_len= 150, padding='right', tokenize='', verbose=0):
        
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
        for i, smile in tqdm(iterator):
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
    
    # draw smiles as a grid in an image
    @staticmethod
    def draw_molecules(smiles, plot_nrows_module=10, plot_subimg_size=(200, 200), legends=[]):

        num_molecules = len(smiles)
        mol = [None]*num_molecules
        for i in range(num_molecules):
            mol[i] = Chem.MolFromSmiles(smiles[i])
        # drow molecules
        if len(legends):
            img = Draw.MolsToGridImage(mol, molsPerRow=plot_nrows_module, subImgSize=plot_subimg_size, legends=legends)
        else:
            img = Draw.MolsToGridImage(mol, molsPerRow=plot_nrows_module, subImgSize=plot_subimg_size)

        return img
    #------------------------------------------------------