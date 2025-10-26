import os
import ast, inspect
import random
import operator
from mol2vec.features import mol2alt_sentence, sentences2vec
from gensim.models import word2vec
import pandas as pd
import numpy as np
from collections import OrderedDict
import pickle
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import json
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from datasets import Dataset
from sklearn.utils import shuffle

import torch
import torch.nn as nn
from rdkit import Chem
from transformers import AutoTokenizer

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils_local.data_utils_smiles import *
from data_utils_local.Generalised_data_utils import *
from data_utils_local.train_module import *
from data_utils_local.models_module import *
from data_utils_local.graph_utils_smiles import *

from datetime import datetime
import pytz
ist = pytz.timezone('Asia/Kolkata')
datetime_now = datetime.now(ist).strftime('[%H:%M:%S]/%d/%m/%Y')

__all__ = ['Mol2VecDataset',
           'GCNDataset',
           'DescriptorsDataset',
           'ChemBertDataset'
           ]
    
    
def partition_data(df_init,
                  partition_col,
                  partition_labels = ['Tr', 'Val', 'Te']):
        
    train_data = df_init[df_init[partition_col] == partition_labels[0]]
    val_data = df_init[df_init[partition_col] == partition_labels[1]]
    test_data = df_init[df_init[partition_col] == partition_labels[2]]
    
    return train_data, val_data, test_data

def ensure_2d(y: np.ndarray) -> np.ndarray:
    """Ensure y is 2D: (n,) -> (n,1), (n,k) -> (n,k)."""
    y = np.asarray(y)
    return y.reshape(-1, 1) if y.ndim == 1 else y

class Mol2VecDataset:
    
    def __init__(self, df_init, 
                partition_col,
                smiles_column_name,
                target_column_name,
                mol2vec_dataset_path
                ):
        self.df = df_init.copy()
        self.smiles_column_name =smiles_column_name
        self.target_column_name =target_column_name
        self.mol2vec_dataset_path =mol2vec_dataset_path
        self.partition_col =partition_col
        self.train_data, self.val_data, self.test_data = None, None, None
        self.StandardScaler_labels = None
        
    def split_data(self, partition_labels=['Tr', 'Val', 'Te']):
        
        self.train_data, self.val_data, self.test_data = partition_data(self.df,
                                                                       partition_col=self.partition_col,
                                                                       partition_labels=partition_labels
                                                                       )
        
    def load_smiles_labels(self, label_normalised=True):
        
        self.y_all = self.df[self.target_column_name].values
        self.y_train = self.train_data[self.target_column_name].values
        self.y_test = self.test_data[self.target_column_name].values
        self.y_val = self.val_data[self.target_column_name].values
        
        self.StandardScaler_labels = StandardScaler()
        if label_normalised:
            self.StandardScaler_labels.fit(self.y_train.reshape(-1, 1))
        else:
            self.StandardScaler_labels.mean_ = 0.0
            self.StandardScaler_labels.scale_ = 1.0

        os.makedirs(self.mol2vec_dataset_path, exist_ok=True)
        with open(f'{self.mol2vec_dataset_path}/StandardScaler_labels.pkl', 'wb') as fp:
            pickle.dump(self.StandardScaler_labels, fp)

        self.smiles_list_all = self.df[self.smiles_column_name].values.tolist()
        self.smiles_list_train = self.train_data[self.smiles_column_name].values.tolist()
        self.smiles_list_test = self.test_data[self.smiles_column_name].values.tolist()
        self.smiles_list_val = self.val_data[self.smiles_column_name].values.tolist()

    def create_mol_sentence(self, smiles_list):
        
        mol_sentences = [mol2alt_sentence(mol, 1)
                 for smiles in smiles_list
                 if (mol := Chem.MolFromSmiles(smiles)) is not None]
        # mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        # mol_list = [mol for mol in mol_list if mol is not None]
        # mol_sentences = [mol2alt_sentence(mol, 1) for mol in mol_list]
        
        return mol_sentences
    
    def create_sentences(self):

        self.mol_sentences_train = self.create_mol_sentence(self.smiles_list_train)
        self.mol_sentences_test = self.create_mol_sentence(self.smiles_list_test)
        self.mol_sentences_val = self.create_mol_sentence(self.smiles_list_val)
        
    def load_mol2vec_model(self,
                           mol_sentences_train = None,
                           train_mol2vec = False,
                           vector_size = 1024,
                           window = 10,
                           min_count = 1,
                           workers = 4
                           ):
        if train_mol2vec:
            ## Training Model
            trained_model = word2vec.Word2Vec(vector_size=vector_size,
                                            window=window,
                                            min_count=min_count,
                                            workers=workers
                                            )
            trained_model.build_vocab(mol_sentences_train)
            trained_model.train(mol_sentences_train, total_examples=trained_model.corpus_count, epochs=10)
            mol2vec_model = trained_model
        else:
            # Load Pretrained Model
            mol2vec_model = word2vec.Word2Vec.load('/home/rkmvu/Codes/CL_int/Mol2Vec_representation/model_300dim.pkl')
        
        return mol2vec_model
    
    def mol2vec_repr(self, mol2vec_model, mol_sentences, y_true, train_mol2vec=False):
        
        if train_mol2vec:
            X = sentences2vec(mol_sentences, mol2vec_model)
        else:
            X = sentences2vec(mol_sentences, mol2vec_model, unseen='UNK')
            
        dims_latent = X.shape[1]
        X = pd.DataFrame(X, columns=[f'dim-{i}' for i in range(dims_latent)])
        X['labels'] = y_true
        
        return X
    
    def mol2vec_data(self):
        
        '''Output: X_train, X_test, X_val'''
        
        mol2vec_model = self.load_mol2vec_model()
        X_train = self.mol2vec_repr(mol2vec_model=mol2vec_model,
                                    mol_sentences=self.mol_sentences_train,
                                    y_true=self.y_train
                                    )
        X_test = self.mol2vec_repr(mol2vec_model=mol2vec_model,
                                    mol_sentences=self.mol_sentences_test,
                                    y_true=self.y_test
                                    )
        X_val = self.mol2vec_repr(mol2vec_model=mol2vec_model,
                                    mol_sentences=self.mol_sentences_val,
                                    y_true=self.y_val
                                    )

        
        return X_train, X_val, X_test
    
    def prepare_datasets(self,
                     partition_labels = ['Tr', 'Val', 'Te']
                     ):
        
        self.split_data(partition_labels = partition_labels)
        self.load_smiles_labels()
        self.create_sentences()
        train_data, val_data, test_data = self.mol2vec_data()
        X_train, y_train = train_data.drop(columns=['labels']), train_data['labels']
        X_test, y_test = test_data.drop(columns=['labels']), test_data['labels']
        X_val, y_val = val_data.drop(columns=['labels']), val_data['labels']
        
        train_dataset = list(zip(X_train.values, y_train.values))
        test_dataset = list(zip(X_test.values, y_test.values))
        val_dataset = list(zip(X_val.values, y_val.values))
        
        results = {'train':{'smiles':self.smiles_list_train, 'dataset':train_dataset},
                   'val':{'smiles':self.smiles_list_val, 'dataset':val_dataset},
                   'test':{'smiles':self.smiles_list_test, 'dataset':test_dataset},
                    "scaler": {"labels": self.StandardScaler_labels}
                   }
        
        return results

class GCNDataset:
    """
    A dataset preparation utility for graph-based neural networks (e.g., GCNs).

    This class provides an end-to-end pipeline for:
    - Filtering molecules based on atom types and size.
    - Applying column constraints.
    - Splitting into train/validation/test sets.
    - Normalizing labels.
    - Saving/loading datasets from preprocessed files.

    Attributes
    ----------
    df : pd.DataFrame
        Working dataframe containing SMILES strings, labels, and partition information.
    train_data, val_data, test_data : pd.DataFrame or None
        Dataframes corresponding to train, validation, and test partitions.
    StandardScaler_labels : StandardScaler or None
        Scaler used for label normalization.
    """

    def __init__(self, df_init,
                smiles_column_name="smiles",
                target_column_name="target",
                partition_col="split",
                max_num_atoms=100,
                graph_dataset_path='./datasets/graph_dataset'
                ):
        """
        Initialize the GCNDataset object.

        Parameters
        ----------
        df_init : pd.DataFrame
            Input dataframe containing SMILES, labels, and partition info.
        """
        
        self.df = df_init.copy()
        self.smiles_column_name =smiles_column_name
        self.target_column_name =target_column_name
        self.partition_col =partition_col
        self.max_num_atoms =max_num_atoms
        self.graph_dataset_path =graph_dataset_path
        self.train_data, self.val_data, self.test_data = None, None, None
        self.StandardScaler_labels = None
        
    def __repr__(self):
        return auto_repr(self)
        

    def atom_filter(self,
                    all_atoms={'Br', 'Cl', 'P', 'I', 'F', 'H', 'S', 'N', 'O', 'C', 'B', 'Si', 'Na', 'K'}):
        """
        Filter molecules based on allowed atom types and maximum atom count.

        Parameters
        ----------
        smiles_column_name : str, default="smiles"
            Column name containing SMILES strings.
        max_num_atoms : int or float, default=100
            Maximum number of atoms allowed in a molecule.
        all_atoms : set, optional
            Allowed atom symbols.

        Notes
        -----
        Updates the internal dataframe `self.df` by removing invalid molecules.
        """
        print(f'Filter molecules based on: {all_atoms}')
        mask = []
        for smiles in self.df[self.smiles_column_name]:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    mask.append(False)
                    continue
                symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
                mask.append(symbols.issubset(all_atoms) and (5<= mol.GetNumAtoms() <= self.max_num_atoms))
            except Exception:
                mask.append(False)
        self.df = self.df[np.array(mask)]
        print(f'Done')
        print('-'*80)

    def apply_column_constraints(self, filters: list[tuple]):
        """
        Apply filtering rules to dataframe columns.

        Parameters
        ----------
        filters : list of tuple
            Each tuple must be of form (col, operator, value).
            Supported operators: ==, !=, <, <=, >, >=, "between".

        Returns
        -------
        pd.DataFrame
            Filtered dataframe.
        """
        ops = {
            "==": operator.eq, "!=": operator.ne,
            "<": operator.lt, "<=": operator.le,
            ">": operator.gt, ">=": operator.ge,
            "between": lambda col, bounds: col.between(bounds[0], bounds[1]),
        }
        print(f'Applying constraints: {ops}')
        mask = np.ones(len(self.df), dtype=bool)
        for col, rel, ref in filters:
            if rel not in ops:
                raise ValueError(f"Unsupported operator: {rel}")
            mask &= ops[rel](self.df[col], ref)
        self.df = self.df[mask]
        print(f'Done')
        print('-'*80)
        return self.df

    def split_data(self, partition_labels=('Tr', 'Val', 'Te')):
        """
        Split dataframe into train/validation/test partitions.

        Parameters
        ----------
        partition_col : str
            Column name defining the partition labels.
        partition_labels : tuple of str, default=("Tr", "Val", "Te")
            Partition labels for train, validation, and test splits.

        Notes
        -----
        Populates `self.train_data`, `self.val_data`, and `self.test_data`.
        """
        print(f'Splitting Dataset into Train, Validation and Test')
        self.train_data, self.val_data, self.test_data = partition_data(self.df,
                                                                       partition_col=self.partition_col,
                                                                       partition_labels=partition_labels
                                                                       )
        print(f'Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}')
        print(f'Done')
        print('-'*80)
        
    def mol_details(self, symb_hyb_chirl_file=''):
        
        print(f'Creating Mol Deatins file for Hybridization, Chirality, Symbols:')
        full_path = symb_hyb_chirl_file#f'{self.graph_dataset_path}/{symb_hyb_chirl_file}.json'

        if os.path.isfile(full_path):
            self.kwargs = load_json(full_path)
        else:
            self.kwargs = list_smiles2symbols_hybridization_chiraltype(self.train_data[self.smiles_column_name].tolist())
            dict2json(self.kwargs, filepath=full_path) 
        self.kwargs['max_num_atoms'] = self.max_num_atoms
        print(f'Done')
        print('-'*80)

    def load_smiles_labels(self,
                           label_normalised=True):
        """
        Extract SMILES strings and labels; normalize labels if required.

        Parameters
        ----------
        smiles_column_name : str
            Column containing SMILES strings.
        target_column_name : str
            Column containing target labels.
        label_normalised : bool, default=True
            Whether to apply standardization to labels.
        graph_dataset_path : str, default="./datasets/graph_dataset"
            Path to save label scaler.

        Notes
        -----
        Populates attributes:
        - `self.y_train`, `self.y_val`, `self.y_test`
        - `self.smiles_list_train`, `self.smiles_list_val`, `self.smiles_list_test`
        Saves label scaler as pickle.
        """
        print(f'Separating SMILES and Labels from the full dataset')
        self.y_all   = np.stack(self.df[self.target_column_name].values)
        self.y_train = np.stack(self.train_data[self.target_column_name].values)
        self.y_val   = np.stack(self.val_data[self.target_column_name].values)
        self.y_test  = np.stack(self.test_data[self.target_column_name].values)

        self.StandardScaler_labels = StandardScaler()
        if label_normalised:
            self.StandardScaler_labels.fit(ensure_2d(self.y_train))
        else:
            n_features = ensure_2d(self.y_train).shape[1]
            self.StandardScaler_labels.mean_ = np.zeros(n_features)
            self.StandardScaler_labels.scale_ = np.ones(n_features)

        os.makedirs(self.graph_dataset_path, exist_ok=True)
        with open(f'{self.graph_dataset_path}/StandardScaler_labels.pkl', 'wb') as fp:
            pickle.dump(self.StandardScaler_labels, fp)

        # transform and preserve dimensionality
        self.y_train = self.StandardScaler_labels.transform(ensure_2d(self.y_train))
        self.y_val   = self.StandardScaler_labels.transform(ensure_2d(self.y_val))
        self.y_test  = self.StandardScaler_labels.transform(ensure_2d(self.y_test))

        # if single-label, flatten back to (n,)
        if self.y_train.shape[1] == 1:
            self.y_train = self.y_train.ravel()
            self.y_val   = self.y_val.ravel()
            self.y_test  = self.y_test.ravel()

        self.smiles_list_all   = self.df[self.smiles_column_name].tolist()
        self.smiles_list_train = self.train_data[self.smiles_column_name].tolist()
        self.smiles_list_val   = self.val_data[self.smiles_column_name].tolist()
        self.smiles_list_test  = self.test_data[self.smiles_column_name].tolist()
        print(f'Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}')
        print(f'Done')
        print('-'*80)
        
    def create_dataset(self, symb_hyb_chirl_file='untitled'):
        
        print(f'Creating dataset and saving to .pkl.gz file')
        self.mol_details(symb_hyb_chirl_file=symb_hyb_chirl_file)
        print(f'Saving Graph Data in Folder: {self.graph_dataset_path}')
        self.train_dataset = Multirelational_GraphDataset(smi_list=self.smiles_list_train, labels=self.y_train, **self.kwargs)
        print(f'Dataset Created')
        SmilesDataset_graph_gen('train', self.train_dataset, out_path=f'{self.graph_dataset_path}/SmilesDataset_graph', mean_std_flag=True)
        
        self.test_dataset  = Multirelational_GraphDataset(smi_list=self.smiles_list_test, labels=self.y_test, **self.kwargs)
        SmilesDataset_graph_gen('test', self.test_dataset, out_path=f'{self.graph_dataset_path}/SmilesDataset_graph')
        
        self.val_dataset   = Multirelational_GraphDataset(smi_list=self.smiles_list_val, labels=self.y_val, **self.kwargs)
        SmilesDataset_graph_gen('val', self.val_dataset, out_path=f'{self.graph_dataset_path}/SmilesDataset_graph')
        print(f'Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}')
        print(f'Done')
        print('-'*80)
        
    def load_dataset_from_pkl(self,
                              features_list=['atom_properties', 'logP_values', 'gasteiger_charge']):
        """
        Load preprocessed datasets and label scaler from pickle files.

        Parameters
        ----------
        graph_dataset_path : str
            Path containing dataset pickle files and scaler.
        features_list : list of str, optional
            Graph features to include.
        max_num_atoms : int or float, default=np.inf
            Maximum number of atoms per molecule.

        Returns
        -------
        tuple
            (train_dataset, val_dataset, test_dataset, StandardScaler_labels)
        """
        print(f'Loading dataset from: {self.graph_dataset_path}')
        kwargs = {'features_list': features_list, 'max_num_atoms': self.max_num_atoms}
        self.train_dataset = GraphData_from_pickle(f'{self.graph_dataset_path}/SmilesDataset_graph_train.pkl.gz', **kwargs)
        self.val_dataset   = GraphData_from_pickle(f'{self.graph_dataset_path}/SmilesDataset_graph_val.pkl.gz', **kwargs)
        self.test_dataset  = GraphData_from_pickle(f'{self.graph_dataset_path}/SmilesDataset_graph_test.pkl.gz', **kwargs)

        with open(f'{self.graph_dataset_path}/StandardScaler_labels.pkl', 'rb') as fp:
            self.StandardScaler_labels = pickle.load(fp)

        print('Dataset loading complete.\n' + '-'*80)
        print(f'Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}')
        
        return self.train_dataset, self.val_dataset, self.test_dataset, self.StandardScaler_labels

    def prepare_datasets(
            self,
            all_atoms={'Br', 'Cl', 'P', 'I', 'F', 'H', 'S', 'N', 'O', 'C', 'B', 'Si', 'Na', 'K'},
            partition_labels=("Tr", "Val", "Te"),
            label_normalised=True,
            filters = [],
            symb_hyb_chirl_file="",
            features_list=['atom_properties', 'logP_values', 'gasteiger_charge']
        ):
        """
        Prepare datasets end-to-end, either by loading from precomputed pickle
        or by regenerating them from scratch.

        Parameters
        ----------
        partition_labels : tuple of str, default=("Tr", "Val", "Te")
            Partition labels.
        max_num_atoms : int, default=100
            Maximum number of atoms per molecule.
        all_atoms : set, optional
            Allowed atom symbols.
        label_normalised : bool, default=True
            Whether to normalize labels.
        symb_hyb_chirl_file : str, default=""
            File for hybridization/chirality info.
        features_list : list of str, optional
            Graph features to include.

        Returns
        -------
        dict
            {
            'train': {'smiles': [...], 'dataset': train_dataset},
            'val':   {'smiles': [...], 'dataset': val_dataset},
            'test':  {'smiles': [...], 'dataset': test_dataset},
            'scaler': {'labels': StandardScaler_labels}
            }
        """
        print(f'Preparing Dataset:')
        print('-'*80)
        def build_results(train_dataset, val_dataset, test_dataset, scaler):
            return {
                'train': {'smiles': [train_dataset.get_smiles(i) for i in range(len(train_dataset))],
                        'dataset': train_dataset},
                'val':   {'smiles': [val_dataset.get_smiles(i) for i in range(len(val_dataset))],
                        'dataset': val_dataset},
                'test':  {'smiles': [test_dataset.get_smiles(i) for i in range(len(test_dataset))],
                        'dataset': test_dataset},
                'scaler': {'labels': scaler}
            }

        try:
            # self.kwargs = load_json(symb_hyb_chirl_file)
            # self.kwargs['max_num_atoms'] = self.max_num_atoms
            datasets = self.load_dataset_from_pkl(features_list=features_list)
            self.train_dataset, self.val_dataset, self.test_dataset, self.StandardScaler_labels = datasets

        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            # Regenerate dataset from scratch
            self.atom_filter(all_atoms=all_atoms)
            if filters:
                self.apply_column_constraints(filters=filters)
            self.split_data(partition_labels=partition_labels)
            self.load_smiles_labels(label_normalised=label_normalised)
            self.create_dataset(symb_hyb_chirl_file=symb_hyb_chirl_file)

            datasets = self.load_dataset_from_pkl(features_list=features_list)
            self.train_dataset, self.val_dataset, self.test_dataset, self.StandardScaler_labels = datasets
        print('Dataset Preparation complete')
        print('-'*80)

        return build_results(self.train_dataset, self.val_dataset,
                         self.test_dataset, self.StandardScaler_labels)


class DescriptorsDataset:
    
    def __init__(self, df_init,
                 smiles_column_name,
                 target_column_name,
                 partition_col,
                 descriptor_dataset_path = './descriptor_dataset_path/'
                 ):
        os.makedirs(descriptor_dataset_path, exist_ok=True)
        self.df = df_init.copy()
        self.smiles_column_name = smiles_column_name
        self.target_column_name = target_column_name
        self.partition_col = partition_col
        self.descriptor_dataset_path = descriptor_dataset_path
        self.train_data, self.val_data, self.test_data = None, None, None
        self.StandardScaler_labels = None
        self.scaler_train_feats = None
        self.optimal_features = None

    def split_data(self, partition_labels=['Tr', 'Val', 'Te']):
        self.train_data, self.val_data, self.test_data = partition_data(self.df,
                                                                       partition_col=self.partition_col,
                                                                       partition_labels=partition_labels
                                                                       )
        
    def load_smiles_labels(self):
        self.y_all = self.df[self.target_column_name].values
        self.y_train = self.train_data[self.target_column_name].values
        self.y_test = self.test_data[self.target_column_name].values
        self.y_val = self.val_data[self.target_column_name].values

        self.smiles_list_all = self.df[self.smiles_column_name].values.tolist()
        self.smiles_list_train = self.train_data[self.smiles_column_name].values.tolist()
        self.smiles_list_test = self.test_data[self.smiles_column_name].values.tolist()
        self.smiles_list_val = self.val_data[self.smiles_column_name].values.tolist()
    
    # ---------------------------
    # FEATURE PREPROCESSING
    # ---------------------------
    def _select_by_mutual_info(self, X_train, y_train, threshold):
        mi_scores = mutual_info_regression(X_train, y_train)
        return X_train.columns[mi_scores > threshold]

    def _remove_high_corr(self, X, threshold):
        return remove_highly_correlated_columns(X, threshold=threshold)

    def _remove_low_corr(self, X_train, y_train, target_col, threshold):
        
        y_train_series = pd.Series(y_train, name=target_col)
        corr_matrix = pd.concat([X_train, y_train_series], axis=1).corr()
        corr_target = corr_matrix[target_col].abs().sort_values(ascending=False)
        keep_features = list(corr_target[(corr_target > threshold) & (corr_target < 1.0)].keys())
        
        return keep_features

    def _apply_hierarchical_fs(self, X_train, y_train, X_val, y_val):
        
        feature_names =  hierarchical_feature_selection(X_train=X_train, y_train=y_train,
                                                        X_val=X_val, y_val=y_val,
                                                        threshold=0.001, n_estimators=50,
                                                        random_state=42, col_drop_threshold=0.99,
                                                        criterion='squared_error', max_features='sqrt'
                                                        )
        return feature_names

    def preprocess_features(self, X_train, X_val,
                            y_train, y_val,
                            optimal_feature_path, drop_columns=[], min_feature_variance = 1e-10,
                            preprocessing=True, feature_selection=True, fingerprints=False,
                            mi_threshold=0.05, corr_threshold=0.95, non_corr_threshold=0.05):
        """
        Orchestrates feature preprocessing in modular steps.
        """
        random.seed(10)
        np.random.seed(20)

        X_train = X_train.drop(columns=drop_columns).select_dtypes(include='number')
        mask = X_train.std(numeric_only=True) >= min_feature_variance
        selected_cols = mask[mask].index.tolist()  # keep only True columns
        X_train = X_train[selected_cols]
        optimal_features = X_train.columns
        if preprocessing:
            if os.path.isfile(optimal_feature_path):
                with open(optimal_feature_path, 'r') as f:
                    optimal_features = json.load(f)
            else:
                # Mutual info selection
                selected_features = self._select_by_mutual_info(X_train=X_train, y_train=y_train, threshold=mi_threshold)
                X_train = X_train[selected_features]

                # Remove high correlation
                X_train = self._remove_high_corr(X_train, corr_threshold)

                # Remove low correlation
                optimal_features = self._remove_low_corr(X_train=X_train, y_train=y_train,
                                                         target_col=self.target_column_name, threshold=non_corr_threshold
                                                         )

                # Optional feature selection
                if feature_selection:
                    optimal_features = self._apply_hierarchical_fs(X_train, y_train, X_val, y_val)

                save_list2json(optimal_features, filepath=optimal_feature_path)
            if not fingerprints:
                optimal_features = [s for s in optimal_features if 'MorganFP_' not in s]

        self.optimal_features = optimal_features
        
        return optimal_features

    # ---------------------------
    # NORMALIZATION
    # ---------------------------
    def _fit_scaler(self, X, enabled=True):
        scaler = StandardScaler()
        if enabled:
            scaler.fit(X)
        else:
            scaler.mean_ = 0
            scaler.scale_ = 1
        return scaler

    def _transform_with_scaler(self, scaler, *datasets):
        return [scaler.transform(ds) for ds in datasets]

    def normalize_data(self, X_train, X_val, X_test, y_train, y_val, y_test,
                       feature_normalised=True, label_normalised=True):
        """
        Normalize features and labels separately using StandardScaler.
        """
        # Feature scaling
        self.StandardScaler_features = self._fit_scaler(X_train, feature_normalised)
        X_train, X_val, X_test = self._transform_with_scaler(self.StandardScaler_features, X_train, X_val, X_test)

        # Label scaling
        self.StandardScaler_labels = self._fit_scaler(y_train.reshape(-1, 1), label_normalised)
        y_train, y_val, y_test = [np.ravel(self.StandardScaler_labels.transform(arr.reshape(-1, 1)))
                                        for arr in (y_train, y_val, y_test)
                                    ]

        with open(f'{self.descriptor_dataset_path}/StandardScaler_labels.pkl', 'wb') as fp:
            pickle.dump(self.StandardScaler_labels, fp)
        with open(f'{self.descriptor_dataset_path}/StandardScaler_features.pkl', 'wb') as fp:
            pickle.dump(self.StandardScaler_features, fp)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_datasets(self, optimal_feature_path,
                         drop_columns=[],
                         preprocessing=True,
                         feature_selection=True,
                         fingerprints=False,
                         feature_normalised=True,
                         label_normalised=True,
                         mi_threshold=0.05,
                         corr_threshold=0.95,
                         non_corr_threshold=0.05,
                         partition_labels=['Tr', 'Val', 'Te']
                         ):
        
        self.split_data(partition_labels = partition_labels)
        self.load_smiles_labels()
        self.preprocess_features(X_train=self.train_data, X_val=self.val_data,
                                y_train=self.y_train, y_val=self.y_val,
                                drop_columns=drop_columns,
                                optimal_feature_path=optimal_feature_path,
                                preprocessing=preprocessing,
                                feature_selection=feature_selection,
                                fingerprints=fingerprints,
                                mi_threshold=mi_threshold,
                                corr_threshold=corr_threshold,
                                non_corr_threshold=non_corr_threshold
                                )
        self.X_train, self.X_val, self.X_test = (self.train_data[self.optimal_features],
                                                        self.val_data[self.optimal_features],
                                                        self.test_data[self.optimal_features]
                                                        )
        X_train, X_val, X_test, y_train, y_val, y_test= self.normalize_data(self.X_train, self.X_val, self.X_test,
                                                                            self.y_train, self.y_val, self.y_test,
                                                                            feature_normalised=feature_normalised,
                                                                            label_normalised=label_normalised
                                                                            )
        train_dataset = list(zip(X_train, y_train))
        test_dataset = list(zip(X_test, y_test))
        val_dataset = list(zip(X_val, y_val))
        
        results = {'train':{'smiles':self.smiles_list_train, 'dataset':train_dataset},
                   'val':{'smiles':self.smiles_list_val, 'dataset':val_dataset},
                   'test':{'smiles':self.smiles_list_test, 'dataset':test_dataset},
                    "scaler": {
                            "features": self.StandardScaler_features,
                            "labels": self.StandardScaler_labels
                            }
                   }
        return results
    
class ChemBertDataset:
    
    def __init__(self, df_init,
                 smiles_column_name='smiles',
                 target_column_name = 'labels',
                 partition_col = 'Data_Split',
                 model_name = "DeepChem/ChemBERTa-77M-MLM"):
        self.df = df_init.copy()
        self.smiles_column_name = smiles_column_name
        self.target_column_name = target_column_name
        self.partition_col = partition_col
        self.TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        self.train_data, self.val_data, self.test_data = None, None, None
        self.StandardScaler_labels = None
        
    def split_data(self, partition_labels=['Tr', 'Val', 'Te']):
        
        self.train_data, self.val_data, self.test_data = partition_data(self.df,
                                                                       partition_col=self.partition_col,
                                                                       partition_labels=partition_labels
                                                                       )
        
    def load_smiles_labels(self, label_normalised=True, chembert_dataset_path='./'):

        self.y_all = self.df[self.target_column_name].values
        self.y_train = self.train_data[self.target_column_name].values
        self.y_test = self.test_data[self.target_column_name].values
        self.y_val = self.val_data[self.target_column_name].values

        # Single scaler
        self.StandardScaler_labels = StandardScaler()
        if label_normalised:
            self.StandardScaler_labels.fit(self.y_train.reshape(-1, 1))
        else:
            self.StandardScaler_labels.mean_ = 0.0
            self.StandardScaler_labels.scale_ = 1.0

        # Save scaler
        os.makedirs(chembert_dataset_path, exist_ok=True)
        with open(f'{chembert_dataset_path}/StandardScaler_labels.pkl', 'wb') as fp:
            pickle.dump(self.StandardScaler_labels, fp)

        # Apply transformation
        self.y_all = np.ravel(self.StandardScaler_labels.transform(self.y_all.reshape(-1, 1)))
        self.y_train = np.ravel(self.StandardScaler_labels.transform(self.y_train.reshape(-1, 1)))
        self.y_val = np.ravel(self.StandardScaler_labels.transform(self.y_val.reshape(-1, 1)))
        self.y_test = np.ravel(self.StandardScaler_labels.transform(self.y_test.reshape(-1, 1)))

        self.smiles_list_all = self.df[self.smiles_column_name].tolist()
        self.smiles_list_train = self.train_data[self.smiles_column_name].tolist()
        self.smiles_list_test = self.test_data[self.smiles_column_name].tolist()
        self.smiles_list_val = self.val_data[self.smiles_column_name].tolist()   
    
    def find_max_length(self,
                        max_len_path = './max_length_smiles.pkl'
                        ):
        
        if os.path.isfile(max_len_path):
            with open(max_len_path, 'rb') as fp:
                max_length = pickle.load(fp)
        else:
            all_tokens = self.TOKENIZER(self.smiles_list_train)
            input_ids = all_tokens['input_ids']
            max_length = len(max(input_ids, key=len))
            with open(max_len_path, 'wb') as fp:
                pickle.dump(max_length, fp)
        print('-'*80)
        print(f'Max Length SMILES: {max_length}')
        print('+'*80)
        self.tokenize = lambda batch: self.TOKENIZER(batch[self.smiles_column_name], padding="max_length", truncation=True, max_length=max_length)
        
    def create_dataset(self, data_augmentation=False):
        
        all_dataset = Dataset.from_pandas(pd.DataFrame({self.smiles_column_name: self.smiles_list_all, "labels":self.y_all}))
        train_dataset = Dataset.from_pandas(pd.DataFrame({self.smiles_column_name: self.smiles_list_train, "labels":self.y_train}))
        test_dataset = Dataset.from_pandas(pd.DataFrame({self.smiles_column_name: self.smiles_list_test, "labels":self.y_test}))
        val_dataset = Dataset.from_pandas(pd.DataFrame({self.smiles_column_name: self.smiles_list_val, "labels":self.y_val}))

        if data_augmentation:
            augmented_smiles, augmented_labels = augment_smiles_with_labels(smiles_list=self.smiles_list_train, labels=self.y_train)
            aug_smiles, y_train = shuffle(augmented_smiles, augmented_labels, random_state=42)
            train_dataset = Dataset.from_pandas(pd.DataFrame({self.smiles_column_name: aug_smiles, "labels":y_train}))

        all_dataset = all_dataset.map(self.tokenize, batched=True)
        train_dataset = train_dataset.map(self.tokenize, batched=True)
        val_dataset = val_dataset.map(self.tokenize, batched=True)
        test_dataset = test_dataset.map(self.tokenize, batched=True)
        
        results = {
                    "train": {"smiles": self.smiles_list_train, "dataset": train_dataset},
                    "val": {"smiles": self.smiles_list_val, "dataset": val_dataset},
                    "test": {"smiles": self.smiles_list_test, "dataset": test_dataset},
                    "scaler": {"labels": self.StandardScaler_labels}
                }

        return results
    
    def prepare_datasets(self, partition_labels=['Tr', 'Val', 'Te'],
                        max_len_path = './max_length_smiles.pkl'
                        ):
        
        self.split_data(partition_labels = partition_labels)
        self.load_smiles_labels()
        self.find_max_length(max_len_path=max_len_path)
        datasets_final = self.create_dataset()
        
        return datasets_final
    
        