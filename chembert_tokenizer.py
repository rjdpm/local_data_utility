from collections import OrderedDict
from orderedset import OrderedSet
from typing import List, Dict, Tuple, Callable, Any
import re, numpy as np
from rdkit import Chem
from tqdm import tqdm
import json

__all__ = ["ChemBertTokenizer"]

class ChemBertTokenizer:
    
    def __init__(self, tokens_list: List[str], max_str_len: int = None):
        self.char_list = tokens_list
        self.max_str_len = max_str_len
        
    def tokenize(self, smiles: str) -> list:
        return get_tokens(smiles, self.char_list)
    
    def encode(self, smiles: str) -> np.ndarray[np.bool_]:
        onehot =  smi2onehot(smiles, self.char_list, self.max_str_len)[0]
        return np.where(onehot)[1]
    
    def decode(self, onehot_mat: np.ndarray) -> str:
        return onehot2smiles(onehot_mat, self.char_list)

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
    
    all_chars = OrderedSet()
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