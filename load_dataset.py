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
from dataclasses import dataclass, asdict, field
import torch
from torch.utils.data import Dataset as Torch_Dataset

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from typing import Optional, Dict, Any, Tuple
from rdkit.Chem.SaltRemover import SaltRemover
from transformers import AutoTokenizer

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils_local.data_utils_smiles import *
from data_utils_local.Generalised_data_utils import *
from data_utils_local.train_module import *
from data_utils_local.models_module import *
# from data_utils_local.graph_utils_smiles import *
from data_utils_local.graph_utils_smiles import list_smiles2symbols_hybridization_chiraltype
from data_utils_local.graph_utils_smiles import (
            Multirelational_GraphDataset as Multirelational_GraphDataset_old,
            SmilesDataset_graph_gen as SmilesDataset_graph_gen_old,
            GraphData_from_pickle as GraphData_from_pickle_old,
        )
from data_utils_local.graph_utils_smiles_V2 import (
    Multirelational_GraphDataset as Multirelational_GraphDataset_new,
    SmilesDataset_graph_gen as SmilesDataset_graph_gen_new,
    GraphData_from_pickle as GraphData_from_pickle_new,
)
from Codes.new_smiles_data_train_test_vae_svae_16_09_24.original_codes.network.SmilesImage2Smiles_Network_vae import *
from Codes.new_smiles_data_train_test_vae_svae_16_09_24.original_codes.data_utils import *
# from data_utils_local.graph_utils_smiles_V2 import *

from datetime import datetime
import pytz
ist = pytz.timezone('Asia/Kolkata')
datetime_now = datetime.now(ist).strftime('[%H:%M:%S]/%d/%m/%Y')

__all__ = ['ensure_2d',
           'Mol2VecDataset',
           'GCNDataset',
           'DescriptorsDataset',
           'ChemBertDataset',
           'VAEDataset',
           'SMILES2GRAPH_Data',
           'SMILES_PRUNING'
           ]

PARTITION_NAMES = {'Tr':'train', 'Te':'test', 'Val':'val'}

def split_data(df, partition_col, partition_labels=None):
        """
        Build partitions from the dataframe.
        If partition_labels is None, use all unique labels found.
        """
        partitions = OrderedDict({})
        if partition_labels is None:
            partition_labels = df[partition_col].unique().tolist()

        for p in partition_labels:
            partitions[p] = df[df[partition_col] == p].reset_index(drop=True)
            
        return partitions

def ensure_2d(y: np.ndarray) -> np.ndarray:
    """Ensure y is 2D: (n,) -> (n,1), (n,k) -> (n,k)."""
    y = np.asarray(y)
    return y.reshape(-1, 1) if y.ndim == 1 else y

def atom_filter(df, smiles_column_name='Cannonicalised_SMILES',
                max_num_atoms = 100,
                all_atoms={'Br','Cl','P','I','F','H','S','N','O','C','B','Si','Na','K'}):
    
    mask = []
    for smi in df[smiles_column_name]:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                mask.append(False); continue
            symbols = {a.GetSymbol() for a in mol.GetAtoms()}
            mask.append(symbols.issubset(all_atoms) and (5 <= mol.GetNumAtoms() <= max_num_atoms))
        except:
            mask.append(False)
    df = df[np.array(mask)]
    
    return df
        

def apply_column_constraints(df, filters):
    
    ops = {
        "==": operator.eq, "!=": operator.ne,
        "<": operator.lt, "<=": operator.le,
        ">": operator.gt, ">=": operator.ge,
        "between": lambda col, b: col.between(b[0], b[1])
    }
    mask = np.ones(len(df), dtype=bool)
    for col, rel, ref in filters:
        mask &= ops[rel](df[col], ref)
    df = df[mask]
    
    return df


class Mol2VecDataset:

    def __init__(self,
                 df_init,
                 partition_col,
                 smiles_column_name,
                 target_column_name,
                 mol2vec_dataset_path,
                 mol2vec_model_path = "/home/rkmvu/Codes/CL_int/Mol2Vec_representation/model_300dim.pkl"
                 ):

        self.df = df_init.copy()
        self.partition_col = partition_col
        self.smiles_column_name = smiles_column_name
        self.target_column_name = target_column_name
        self.mol2vec_dataset_path = mol2vec_dataset_path
        self.mol2vec_model_path = mol2vec_model_path

        # Dictionaries indexed by partition label
        self.partitions = {}          # label -> dataframe
        self.smiles = {}              # label -> list of smiles
        self.y = {}                   # label -> numpy array
        self.mol_sentences = {}       # label -> mol2vec sentences

        self.StandardScaler_features = None
        self.StandardScaler_labels = None

    # --------------------------------------------------------
    # Partition handling
    # --------------------------------------------------------

    def get_partitions(self, partition_labels=['Tr', 'Val', 'Te']):
        
        print(f'Partitioning data..')
        self.partitions = split_data(self.df,
                                    partition_col=self.partition_col,
                                    partition_labels=partition_labels
                                    )

    # --------------------------------------------------------
    # SMILES and labels
    # --------------------------------------------------------

    def load_smiles_labels(self):
        
        print(f'Loading labels..')
        for p, dfp in self.partitions.items():
            self.smiles[p] = dfp[self.smiles_column_name].values.tolist()
            self.y[p] = dfp[self.target_column_name].values

    # --------------------------------------------------------
    # Mol2Vec sentence creation
    # --------------------------------------------------------

    def create_mol_sentence(self, smiles_list):
        
        print(f'Mol2Vec sentence creation..')
        all_sentences =  [mol2alt_sentence(mol, 1) for s in smiles_list if (mol := Chem.MolFromSmiles(s)) is not None]
        
        return all_sentences

    def create_sentences(self):
        
        print(f'Mol2Vec sentence creation for all dataset..')
        for p in self.smiles:
            self.mol_sentences[p] = self.create_mol_sentence(self.smiles[p])

    # --------------------------------------------------------
    # Mol2Vec model
    # --------------------------------------------------------

    def load_mol2vec_model(self, train_partition="Tr",
                           train_mol2vec=False,
                           vector_size=1024,
                           window=10,
                           min_count=1,
                           workers=4):

        if train_mol2vec:
            print(f'Training Mol2Vec model..')
            sentences = self.mol_sentences[train_partition]
            model = word2vec.Word2Vec(
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=workers
            )
            model.build_vocab(sentences)
            model.train(sentences, total_examples=model.corpus_count, epochs=10)
        else:
            print(f'Loading Mol2Vec model..')
            model = word2vec.Word2Vec.load(self.mol2vec_model_path)

        return model

    # --------------------------------------------------------
    # Mol2Vec representations
    # --------------------------------------------------------

    def mol2vec_repr(self, model):
        """
        Returns dict: partition -> DataFrame(dim-*, labels)
        """
        print(f'Calculating representations..')
        X = {}

        for p, sentences in self.mol_sentences.items():
            vecs = sentences2vec(sentences, model, unseen="UNK")
            df = pd.DataFrame(vecs, columns=[f"dim-{i}" for i in range(vecs.shape[1])])
            df["labels"] = self.y[p]
            X[p] = df

        return X

    # --------------------------------------------------------
    # Normalisation utilities
    # --------------------------------------------------------

    def _fit_scaler(self, X, enabled=True):
        
        scaler = StandardScaler()
        if enabled:
            print(f'Training StandardScaler..')
            scaler.fit(X)
        else:
            print(f'No training of StandardScaler. Setting mean = 0, variance = 1..')
            scaler.mean_ = np.zeros(X.shape[1])
            scaler.scale_ = np.ones(X.shape[1])
        return scaler

    # --------------------------------------------------------
    # Normalisation (fit on reference split only)
    # --------------------------------------------------------

    def normalize_data(self, X_dict, train_partition="Tr",
                       feature_normalised=True,
                       label_normalised=True):

        X_train = X_dict[train_partition].drop(columns="labels").values
        y_train = X_dict[train_partition]["labels"].values.reshape(-1, 1)

        print('-'*60)
        print(f'Normalizing features..')
        self.StandardScaler_features = self._fit_scaler(X_train, feature_normalised)
        print('-'*60)
        print(f'Normalizing labels..')
        self.StandardScaler_labels = self._fit_scaler(y_train, label_normalised)
        print('-'*60)

        results = {}

        for p, df in X_dict.items():
            X = df.drop(columns="labels").values
            y = df["labels"].values.reshape(-1, 1)

            X = self.StandardScaler_features.transform(X)
            y = np.ravel(self.StandardScaler_labels.transform(y))

            results[p] = list(zip(X, y))

        os.makedirs(self.mol2vec_dataset_path, exist_ok=True)
        path = f'{self.mol2vec_dataset_path}/StandardScaler_labels.pkl'
        with open(path, 'wb') as fp:
            print(f'Saving StandardScaler for labels in: {path}')
            pickle.dump(self.StandardScaler_labels, fp)
            
        path = f'{self.mol2vec_dataset_path}/StandardScaler_features.pkl'
        with open(path, 'wb') as fp:
            print(f'Saving StandardScaler for features in: {path}')
            pickle.dump(self.StandardScaler_features, fp)

        return results

    # --------------------------------------------------------
    # Full pipeline
    # --------------------------------------------------------

    def prepare_datasets(self,
                         train_partition="Tr",
                         partition_labels=None,
                         feature_normalised=True,
                         label_normalised=True,
                         train_mol2vec=False):

        print(f'## Preparing full dataset ##')
        print('-'*80)
        self.get_partitions(partition_labels)
        self.load_smiles_labels()
        self.create_sentences()

        mol2vec_model = self.load_mol2vec_model(train_partition=train_partition,
                                                train_mol2vec=train_mol2vec
                                                )

        X_dict = self.mol2vec_repr(mol2vec_model)

        datasets = self.normalize_data(X_dict,
                                     train_partition=train_partition,
                                     feature_normalised=feature_normalised,
                                     label_normalised=label_normalised
                                     )

        results = {PARTITION_NAMES.get(p, p.lower()): {"smiles": self.smiles[p], "dataset": datasets[p]} for p in datasets}
        results["scaler"] = {"features": self.StandardScaler_features,
                             "labels": self.StandardScaler_labels
                             }
        print('-'*80)
        print(f'## Done ##')
        print('-'*80)
        print('='*80)
        
        return results


class DescriptorsDataset:

    def __init__(self, df_init, smiles_column_name, target_column_name,
                 partition_col, descriptor_dataset_path="./descriptor_dataset_path/"):

        os.makedirs(descriptor_dataset_path, exist_ok=True)

        self.df = df_init.copy()
        self.smiles_column_name = smiles_column_name
        self.target_column_name = target_column_name
        self.partition_col = partition_col
        self.descriptor_dataset_path = descriptor_dataset_path

        self.partitions = {}        # label -> dataframe
        self.smiles = {}            # label -> smiles list
        self.y = {}                 # label -> labels

        self.StandardScaler_features = None
        self.StandardScaler_labels = None
        self.optimal_features = None

    # ----------------------------------------------------
    # Partition handling
    # ----------------------------------------------------

    def get_partitions(self, partition_labels=['Tr', 'Val', 'Te']):
        
        print(f'Partitioning data:')
        self.partitions = split_data(self.df,
                                    partition_col=self.partition_col,
                                    partition_labels=partition_labels
                                    )
        # print(f'Number of train, test and val data: {len(self.partitions['Tr'])}, {len(self.partitions['Te'])}, {len(self.partitions['Val'])}')

    # ----------------------------------------------------
    # Load SMILES and labels
    # ----------------------------------------------------

    def load_smiles_labels(self):
        
        print(f'Loading labels:')
        for p, dfp in self.partitions.items():
            self.smiles[p] = dfp[self.smiles_column_name].values.tolist()
            self.y[p] = dfp[self.target_column_name].values

    # ----------------------------------------------------
    # Feature selection utilities
    # ----------------------------------------------------

    def _select_by_mutual_info(self, X_train, y_train, threshold):
        
        print(f'Selecting features w.r.t Mutual Information:')
        mi_scores = mutual_info_regression(X_train, y_train)
        
        return X_train.columns[mi_scores > threshold]

    def _remove_high_corr(self, X, threshold):
        
        print(f'Removing highly correlated columns:')
        
        return remove_highly_correlated_columns(X, threshold=threshold)

    def _remove_low_corr(self, X_train, y_train, target_col, threshold):
        
        print(f'Removing non-correlated columns:')
        y_series = pd.Series(y_train, name=target_col)
        corr = pd.concat([X_train, y_series], axis=1).corr()
        corr_target = corr[target_col].abs().sort_values(ascending=False)
        keep = list(corr_target[(corr_target > threshold) & (corr_target < 1.0)].keys())
        
        return keep

    def _apply_hierarchical_fs(self, X_train, y_train, X_val, y_val):
        
        print(f'Applying hierarchical feature selection:')
        optimal_feature =  hierarchical_feature_selection(X_train=X_train, y_train=y_train,
                                                        X_val=X_val, y_val=y_val,
                                                        threshold=0.001, n_estimators=50,
                                                        random_state=42,
                                                        criterion="squared_error", max_features="sqrt"
                                                        )
        
        return optimal_feature

    # ----------------------------------------------------
    # Feature preprocessing (fit on reference split only)
    # ----------------------------------------------------

    def preprocess_features(self, train_partition, val_partition,
                            optimal_feature_path, drop_columns=[],
                            min_feature_variance=1e-10,
                            preprocessing=True, feature_selection=True,
                            fingerprints=True, mi_threshold=0.05,
                            corr_threshold=0.95, non_corr_threshold=0.05,
                            force=False):

        random.seed(10)
        np.random.seed(20)

        print(f'Preprocessing features:')
        drop_columns = drop_columns+[self.target_column_name]
        X_train = self.partitions[train_partition].drop(columns=drop_columns).select_dtypes(include="number")
        y_train = self.y[train_partition]

        X_val = self.partitions[val_partition].drop(columns=drop_columns).select_dtypes(include="number")
        y_val = self.y[val_partition]

        mask = X_train.std(numeric_only=True) >= min_feature_variance
        X_train = X_train[mask[mask].index]
        optimal_features = X_train.columns.tolist()

        if preprocessing:
            if os.path.isfile(optimal_feature_path) and not force:
                print(f'Loading optimal features from: {optimal_feature_path}')
                with open(optimal_feature_path, "r") as f:
                    optimal_features = json.load(f)
                print(f'Loaded.')
            else:
                print(f'\033[1;95mNo optimal features found in: {optimal_feature_path}\033[0m')
                print(f'Applying preprocessing')
                selected = self._select_by_mutual_info(X_train, y_train, mi_threshold)
                X_train = X_train[selected]

                X_train = self._remove_high_corr(X_train, corr_threshold)

                optimal_features = self._remove_low_corr(
                                                        X_train=X_train, y_train=y_train,
                                                        target_col=self.target_column_name,
                                                        threshold=non_corr_threshold
                                                    )

                if feature_selection:
                    optimal_features = self._apply_hierarchical_fs(
                        X_train[optimal_features], y_train,
                        X_val[optimal_features], y_val
                    )

                save_list2json(optimal_features, optimal_feature_path)

            if not fingerprints:
                optimal_features = [f for f in optimal_features if "MorganFP_" not in f]

        print(f'Number of features:{len(optimal_features)}')
        self.optimal_features = optimal_features
        
        return optimal_features

    # ----------------------------------------------------
    # Scaling
    # ----------------------------------------------------

    def _fit_scaler(self, X, enabled=True):
        
        print(f'Training StandardScaler:')
        scaler = StandardScaler()
        if enabled:
            scaler.fit(X)
        else:
            scaler.mean_ = np.zeros(X.shape[1])
            scaler.scale_ = np.ones(X.shape[1])
            
        return scaler

    # ----------------------------------------------------
    # Normalize all partitions using training split
    # ----------------------------------------------------

    def normalize_data(self, train_partition='Tr', feature_normalised=True, label_normalised=True):

        print(f'Normalizing data:')
        X_train = self.partitions[train_partition][self.optimal_features].values
        y_train = self.y[train_partition].reshape(-1, 1)

        self.StandardScaler_features = self._fit_scaler(X_train, feature_normalised)
        self.StandardScaler_labels = self._fit_scaler(y_train, label_normalised)

        results = {}

        for p, df in self.partitions.items():
            X = df[self.optimal_features].values
            y = self.y[p].reshape(-1, 1)

            X = self.StandardScaler_features.transform(X)
            y = np.ravel(self.StandardScaler_labels.transform(y))

            results[p] = list(zip(X, y))

        with open(f'{self.descriptor_dataset_path}/StandardScaler_labels.pkl', 'wb') as fp:
            pickle.dump(self.StandardScaler_labels, fp)
        with open(f'{self.descriptor_dataset_path}/StandardScaler_features.pkl', 'wb') as fp:
            pickle.dump(self.StandardScaler_features, fp)

        return results

    # ----------------------------------------------------
    # Full pipeline
    # ----------------------------------------------------

    def prepare_datasets(self, optimal_feature_path,
                         train_partition="Tr",
                         val_partition="Val",
                         partition_labels=None,
                         drop_columns=[],
                         preprocessing=True,
                         feature_selection=True,
                         fingerprints=False,
                         feature_normalised=True,
                         label_normalised=True,
                         force=False,
                         mi_threshold=0.05,
                         corr_threshold=0.95,
                         non_corr_threshold=0.05):

        self.get_partitions(partition_labels)
        self.load_smiles_labels()

        self.preprocess_features(
                                train_partition=train_partition,
                                val_partition=val_partition,
                                optimal_feature_path=optimal_feature_path,
                                drop_columns=drop_columns,
                                preprocessing=preprocessing,
                                feature_selection=feature_selection,
                                fingerprints=fingerprints,
                                mi_threshold=mi_threshold,
                                corr_threshold=corr_threshold,
                                non_corr_threshold=non_corr_threshold,
                                force=force
                            )

        datasets = self.normalize_data(
                                    train_partition=train_partition,
                                    feature_normalised=feature_normalised,
                                    label_normalised=label_normalised
                                )

        results = {PARTITION_NAMES.get(p, p.lower()): {"smiles": self.smiles[p], "dataset": datasets[p]} for p in datasets}

        results["scaler"] = {"features": self.StandardScaler_features,
                             "labels": self.StandardScaler_labels
                             }

        return results
    
    
class ChemBertDataset:

    def __init__(self, df_init,
                 smiles_column_name="smiles",
                 target_column_name="labels",
                 partition_col="Data_Split",
                 model_name="DeepChem/ChemBERTa-77M-MLM",
                 metadata_path="./"):

        self.df = df_init.copy()
        self.smiles_column_name = smiles_column_name
        self.target_column_name = target_column_name
        self.partition_col = partition_col
        self.metadata_path = metadata_path

        self.TOKENIZER = AutoTokenizer.from_pretrained(model_name)

        self.partitions = {}        # label -> dataframe
        self.smiles = {}            # label -> smiles
        self.y = {}                 # label -> labels

        self.StandardScaler_labels = None
        self.max_length = None

    # ---------------------------------------------------
    # Partition handling
    # ---------------------------------------------------

    def get_partitions(self, partition_labels=['Tr', 'Val', 'Te']):
        
        print(f'Partitioning data..')
        self.partitions = split_data(self.df,
                                    partition_col=self.partition_col,
                                    partition_labels=partition_labels
                                    )

    # ---------------------------------------------------
    # Load smiles & labels
    # ---------------------------------------------------

    def load_smiles_labels(self, train_partition="Tr", label_normalised=True):

        print(f'Loading SMILES and Labels..')
        for p, dfp in self.partitions.items():
            self.smiles[p] = dfp[self.smiles_column_name].tolist()
            self.y[p] = dfp[self.target_column_name].values

        # Fit label scaler only on reference partition
        self.StandardScaler_labels = StandardScaler()
        if label_normalised:
            self.StandardScaler_labels.fit(self.y[train_partition].reshape(-1, 1))
        else:
            self.StandardScaler_labels.mean_ = 0.0
            self.StandardScaler_labels.scale_ = 1.0

        for p in self.y:
            self.y[p] = np.ravel(self.StandardScaler_labels.transform(self.y[p].reshape(-1, 1)))

        os.makedirs(self.metadata_path, exist_ok=True)
        with open(f"{self.metadata_path}/StandardScaler_labels.pkl", "wb") as fp:
            pickle.dump(self.StandardScaler_labels, fp)

    # ---------------------------------------------------
    # Find max token length (reference split only)
    # ---------------------------------------------------

    def find_max_length(self, train_partition="Tr", max_len_path="./max_length_smiles.pkl"):

        if os.path.isfile(max_len_path):
            print(f'Loading max smiles length from: {max_len_path}')
            with open(max_len_path, "rb") as fp:
                self.max_length = pickle.load(fp)
        else:
            print(f'Finding max smiles length..')
            tokens = self.TOKENIZER(self.smiles[train_partition])
            input_ids = tokens["input_ids"]
            self.max_length = max(len(x) for x in input_ids)

            with open(max_len_path, "wb") as fp:
                pickle.dump(self.max_length, fp)
            print(f'Saved max smiles length to: {max_len_path}')

        self.tokenize = lambda batch: self.TOKENIZER(batch[self.smiles_column_name],
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.max_length
                                                    )

    # ---------------------------------------------------
    # Create HuggingFace datasets for all partitions
    # ---------------------------------------------------

    def create_datasets(self, data_augmentation=False, train_partition="Tr"):

        print(f'Creating HuggingFace datasets for all partitions:')
        hf_datasets = {}
        for p in self.partitions:
            smiles = self.smiles[p]
            labels = self.y[p]

            if data_augmentation and p == train_partition:
                smiles, labels = augment_smiles_with_labels(smiles_list=smiles, labels=labels)
                smiles, labels = shuffle(smiles, labels, random_state=42)

            ds = Dataset.from_pandas(pd.DataFrame({self.smiles_column_name: smiles, "labels": labels}))
            ds = ds.map(self.tokenize, batched=True)
            ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

            hf_datasets[PARTITION_NAMES.get(p, p.lower())] = {"smiles": smiles, "dataset": ds}

        hf_datasets["scaler"] = {"labels": self.StandardScaler_labels}
        
        return hf_datasets

    # ---------------------------------------------------
    # Full pipeline
    # ---------------------------------------------------

    def prepare_datasets(self,
                         train_partition="Tr",
                         partition_labels=None,
                         label_normalised=True,
                         max_len_path="./max_length_smiles.pkl",
                         data_augmentation=False):

        print('-'*80)
        print(f'## Preparing the full dataset ##')
        print('-'*80)
        self.get_partitions(partition_labels)
        self.load_smiles_labels(train_partition=train_partition, label_normalised=label_normalised)
        self.find_max_length(train_partition=train_partition, max_len_path=max_len_path)

        datasets = self.create_datasets(data_augmentation=data_augmentation,
                                        train_partition=train_partition
                                        )
        print('-'*80)
        print('## Done ##')
        print('-'*80)

        return datasets

@dataclass
class GCNDataset_Config:

    # Data reading
    path_init: str = ''
    file_name: str = ''

    # Column names
    smiles_column_name: str = ''
    target_column_name: str = ''
    data_split_column_name: str = ''

    # Output
    graph_dataset_path: str = ''
    details_filepath: str = ''

    # Dataset preparation
    graph_backend_version: str = 'new'
    train_partition: str = 'Tr'
    partition_labels: list = field(default_factory=lambda: ['Tr', 'Te', 'Val'])

    all_atoms: list = field(default_factory=lambda: ['Br', 'Cl', 'P', 'I', 'F', 'H', 'S', 
                                                     'N', 'O', 'C', 'B', 'Si', 'Na', 'K'])
    filters: object = None
    label_normalised: bool = True
    force: bool = False

    # GCN features
    gcn_features_path: str = ''

    @property
    def log_filepath(self):
        return f'{self.graph_dataset_path}/data_preparation_logfile.log'

    def __post_init__(self):
        os.makedirs(self.graph_dataset_path, exist_ok=True)
        if not self.details_filepath.endswith('.json'):
            self.details_filepath = f'{self.details_filepath}.json'

class GCNDataset:
    
    def __init__(self, df_init,
                 smiles_column_name="smiles",
                 target_column_name="target",
                 partition_col="split",
                 max_num_atoms=None,
                 graph_dataset_path="./datasets/graph_dataset",
                 graph_backend_version = 'old',
                 *uncsry_vars, **uncsry_kvars
                 ):

        self.df = df_init.copy()
        self.smiles_column_name = smiles_column_name
        self.target_column_name = target_column_name
        self.partition_col = partition_col
        self.max_num_atoms = max_num_atoms
        self.graph_dataset_path = graph_dataset_path
        self.graph_backend_version = graph_backend_version

        self.datapaths = {}       # label -> dataframe
        self.partitions = {}       # label -> dataframe
        self.smiles = {}           # label -> smiles
        self.y = {}                # label -> labels

        self.StandardScaler_labels = None
        self.kwargs = None
        os.makedirs(self.graph_dataset_path, exist_ok=True)
        
    def load_graph_utils(self):
        
        if self.graph_backend_version == "old":
            print(f'Generating data in old style:')
            self.Multirelational_GraphDataset = Multirelational_GraphDataset_old
            self.SmilesDataset_graph_gen = SmilesDataset_graph_gen_old
            self.GraphData_from_pickle = GraphData_from_pickle_old
            
        elif self.graph_backend_version == "new":
            print(f'Generating data in new style:')
            self.Multirelational_GraphDataset = Multirelational_GraphDataset_new
            self.SmilesDataset_graph_gen = SmilesDataset_graph_gen_new
            self.GraphData_from_pickle = GraphData_from_pickle_new
        else:
            raise ValueError(f"Unknown graph backend version: {self.graph_backend_version}")

    # ----------------------------------------------------
    # Partition handling
    # ----------------------------------------------------

    def get_partitions(self, partition_labels=['Tr', 'Val', 'Te'], *uncsry_vars, **uncsry_kvars):
        
        print(f'Partitioning dataset..')
        self.partitions = split_data(self.df,
                                    partition_col=self.partition_col,
                                    partition_labels=partition_labels
                                    )

    # ----------------------------------------------------
    # Atom & column filtering
    # ----------------------------------------------------

    def atom_filter(self, all_atoms={'Br','Cl','P','I','F','H','S','N','O','C','B','Si','Na','K'}, *uncsry_vars, **uncsry_kvars):
        
        
        print('-'*80)
        print(f'Applying atom filters with atoms: {all_atoms}')
        init_len = len(self.df.copy())
        mask = []
        for smi in self.df[self.smiles_column_name]:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    mask.append(False); continue
                symbols = {a.GetSymbol() for a in mol.GetAtoms()}
                mask.append(symbols.issubset(all_atoms) and (5 <= mol.GetNumAtoms() <= self.max_num_atoms))
            except:
                mask.append(False)
        self.df = self.df[np.array(mask)]
        print(f'Number of eliminated datapoints after atom filtering: {init_len - len(self.df.copy())}')
        print('Done')
        print('-'*80)

    def apply_column_constraints(self, filters, *uncsry_vars, **uncsry_kvars):
        
        if filters:
            print('-'*80)
            print(f'Applying contraints: {filters}')
            init_len = len(self.df.copy())
            ops = {
                "==": operator.eq, "!=": operator.ne,
                "<": operator.lt, "<=": operator.le,
                ">": operator.gt, ">=": operator.ge,
                "between": lambda col, b: col.between(b[0], b[1])
            }
            mask = np.ones(len(self.df), dtype=bool)
            for col, rel, ref in filters:
                mask &= ops[rel](self.df[col], ref)
            self.df = self.df[mask]
            print(f'Number of eliminated datapoints:{len(self.df.copy())-init_len}')
            print('Done')
            print('-'*80)

    # ----------------------------------------------------
    # SMILES and label handling
    # ----------------------------------------------------
    
    def load_smiles_labels(self, train_partition="Tr", label_normalised=True, *uncsry_vars, **uncsry_kvars):

        for p, dfp in self.partitions.items():
            __partion_name = PARTITION_NAMES.get(p, p.lower())
            self.smiles[__partion_name] = dfp[self.smiles_column_name].tolist()
            self.y[__partion_name] = np.stack(dfp[self.target_column_name].values)
        
        self.scaler_path = f"{self.graph_dataset_path}/StandardScaler_labels.pkl"
        if not os.path.isfile(self.scaler_path):

            # Fit label scaler only on reference partition
            self.StandardScaler_labels = StandardScaler()
            y_train = ensure_2d(self.y[PARTITION_NAMES.get(train_partition, train_partition)])

            if label_normalised:
                self.StandardScaler_labels.fit(y_train)
            else:
                self.StandardScaler_labels.mean_ = np.array([0.0])
                self.StandardScaler_labels.scale_ = np.array([1.0])
        
            with open(self.scaler_path, "wb") as fp:
                pickle.dump(self.StandardScaler_labels, fp)
                
        with open(self.scaler_path, "rb") as fp:
            self.StandardScaler_labels = pickle.load(fp)

        for p in self.y:
            __partion_name = PARTITION_NAMES.get(p, p.lower())
            Y = self.StandardScaler_labels.transform(ensure_2d(self.y[__partion_name]))
            self.y[__partion_name] = Y.ravel() if Y.shape[1] == 1 else Y

    # ----------------------------------------------------
    # Atom metadata (fit on training split only)
    # ----------------------------------------------------

    def mol_details(self, symb_hyb_chirl_file:str, *uncsry_vars, **uncsry_kvars):

        if os.path.isfile(symb_hyb_chirl_file):
            print('-'*80)
            print(f'Loading atom metadata from: {symb_hyb_chirl_file}')
            self.kwargs = load_json(symb_hyb_chirl_file)
            print(f'Done')
            print('-'*80)
        else:
            print('-'*80)
            print(f'No atom metadata found in: {symb_hyb_chirl_file}')
            print('Creating metadata:')
            self.kwargs = list_smiles2symbols_hybridization_chiraltype(self.df[self.smiles_column_name].tolist())
            dict2json(self.kwargs, filepath=symb_hyb_chirl_file)
            print('Saved and Loaded.')
            print('-'*80)
        if self.max_num_atoms is None:
            self.max_num_atoms = self.kwargs["max_num_atoms"]

    # ----------------------------------------------------
    # Load or Create Dataset
    # ----------------------------------------------------

    def build_or_load_datasets(self,
                           features_list,
                           train_partition='Tr',
                           partition_labels=['Tr', 'Val', 'Te'],
                           force=False,
                           *uncsry_vars, **uncsry_kvars
                           ):

        self.load_graph_utils()
        out_path = f"{self.graph_dataset_path}/SmilesDataset_graph"

        # Atom metadata must be consistent across partitions
        # self.mol_details(train_partition, symb_hyb_chirl_file)

        datasets = {}
        load_kwargs = {"features_list": features_list, "max_num_atoms": self.max_num_atoms}

        if partition_labels is None:
            partition_labels = list(self.partitions.keys())

        partition_labels.remove(train_partition)
        partition_labels = [train_partition] + partition_labels

        for p in partition_labels:
            __partition_name = PARTITION_NAMES.get(p, p.lower())
            pkl_file = f"{self.graph_dataset_path}/SmilesDataset_graph_{__partition_name}.pkl.gz"
            self.datapaths[__partition_name] = pkl_file
            needs_build = force or not os.path.isfile(pkl_file)

            if not needs_build:
                try:
                    print('-'*80)
                    print(f'Loading {__partition_name.capitalize()} Graph Data from: {pkl_file}')
                    datasets[__partition_name] = self.GraphData_from_pickle(pkl_file, **load_kwargs)
                    print('Loaded')
                    print('-'*80)
                    continue
                except Exception:
                    # corrupted or incompatible file
                    needs_build = True

            if needs_build:
                print('-'*80)
                print(f'Graph Data not found in: {pkl_file}')
                print(f"Generating new graph data for {__partition_name.capitalize()}:")

                ds = self.Multirelational_GraphDataset(smi_list=self.smiles[__partition_name], labels=self.y[__partition_name], **self.kwargs)
                self.SmilesDataset_graph_gen(__partition_name, ds, out_path=out_path, mean_std_flag=(p == train_partition))
                print('Done')

                # Load freshly generated dataset
                print(f'Loading {__partition_name.capitalize()} Graph Data from: {pkl_file}')
                datasets[__partition_name] = self.GraphData_from_pickle(pkl_file, **load_kwargs)
                print('Loaded')
                print('-'*80)

        # Load label scaler (must exist if any dataset was built)
        scaler_path = f"{self.graph_dataset_path}/StandardScaler_labels.pkl"
        if not os.path.isfile(scaler_path):
            raise RuntimeError("Label scaler missing — dataset generation incomplete.")

        with open(scaler_path, "rb") as fp:
            self.StandardScaler_labels = pickle.load(fp)
            self.datapaths['StandardScaler_labels_path'] = scaler_path

        return datasets


    # ----------------------------------------------------
    # Full pipeline
    # ----------------------------------------------------

    def prepare_datasets(self,
                         train_partition="Tr",
                         partition_labels=None,
                         all_atoms={'Br','Cl','P','I','F','H','S','N','O','C','B','Si','Na','K'},
                         filters=[],
                         symb_hyb_chirl_file="",
                         features_list=['atom_properties','gasteiger_charge','logP_values'],
                         label_normalised=True,
                         force=False,
                         *uncsry_vars, **uncsry_kvars
                         ):
        if train_partition not in self.df[self.partition_col].unique():
            raise ValueError(f'Train partition - {train_partition} not in data split column - {self.partition_col}')
        
            ## To set the maximum number of datapoint containing split as train_partition, uncomment it
            # __counts = self.df[self.partition_col].value_counts().to_dict()
            # sorted_splits = sorted(__counts, key=__counts.get, reverse=True)
            # train_partition = sorted_splits[0]
            # print(f'Setting train partition column to: {train_partition}')

        self.mol_details(symb_hyb_chirl_file=symb_hyb_chirl_file)
        
        # filtering
        if all_atoms:
            self.atom_filter(all_atoms=all_atoms)
        if filters:
            self.apply_column_constraints(filters=filters)

        # partition
        self.get_partitions(partition_labels=partition_labels)

        # labels
        self.load_smiles_labels(train_partition=train_partition,
                                label_normalised=label_normalised)

        # build or load
        datasets = self.build_or_load_datasets(features_list=features_list, 
                                               train_partition=train_partition, 
                                               partition_labels=partition_labels,
                                               force=force)
        results = {PARTITION_NAMES.get(p, p.lower()): {"smiles": [datasets[p].get_smiles(i) for i in range(len(datasets[p]))],
                       "dataset": datasets[p]
                       } for p in datasets
                   }

        results["scaler"] = {"labels": self.StandardScaler_labels}
        
        return results


class VAEDataset:

    def __init__(self,
                 df_init,
                 smiles_column_name="SMILES",
                 target_column_name="Mean_logPapp_Values",
                 partition_col="Data_Split",
                 smiles_char_filepath=None,
                 smiles_max_length_filepath=None,
                 vae_model=None,
                 cache_path="./vae_cache"):

        self.df = df_init.copy()
        self.smiles_column_name = smiles_column_name
        self.target_column_name = target_column_name
        self.partition_col = partition_col

        self.smiles_char_filepath = smiles_char_filepath
        self.smiles_max_length_filepath = smiles_max_length_filepath
        self.vae_model = vae_model
        self.cache_path = cache_path

        self.partitions = {}
        self.smiles = {}
        self.y = {}

        self.StandardScaler_labels = None
        os.makedirs(cache_path, exist_ok=True)

    # --------------------------------------------------
    # Partition handling
    # --------------------------------------------------
    def get_partitions(self, partition_labels=['Tr', 'Val', 'Te', 'MD']):
            
        print(f'Partitioning data..')
        self.partitions = split_data(self.df,
                                    partition_col=self.partition_col,
                                    partition_labels=partition_labels
                                    )
        
    # --------------------------------------------------
    # Load SMILES & labels
    # --------------------------------------------------

    def load_smiles_labels(self, train_partition="Tr", label_normalised=True):

        for p, dfp in self.partitions.items():
            self.smiles[p] = dfp[self.smiles_column_name].tolist()
            self.y[PARTITION_NAMES.get(p, p.lower())] = dfp[self.target_column_name].values

        self.StandardScaler_labels = StandardScaler()
        if label_normalised:
            self.StandardScaler_labels.fit(self.y[PARTITION_NAMES.get(train_partition, train_partition.lower())].reshape(-1, 1))
        else:
            self.StandardScaler_labels.mean_ = 0.0
            self.StandardScaler_labels.scale_ = 1.0

        for p in self.y:
            self.y[PARTITION_NAMES.get(p, p.lower())] = np.ravel(self.StandardScaler_labels.transform(self.y[PARTITION_NAMES.get(p, p.lower())].reshape(-1, 1)))

        with open(f"{self.cache_path}/StandardScaler_labels.pkl", "wb") as fp:
            pickle.dump(self.StandardScaler_labels, fp)
            
    # --------------------------------------------------
    # Loading VAE pretrained model
    # --------------------------------------------------
    
    def load_pretrained_vae(self,
                              model_path_init='/home/rkmvu/Codes/new_smiles_data_train_test_vae_svae_16_09_24/',
                              dims_latent=300,
                              num_gru=4,
                              learning_rate=1e-4,
                              ):  
              
        path_init = '/home/rkmvu/Dataset/selfies/zinc/'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('='*70)
        print('Model will use: {}' .format(device))
        print('='*70)

        # dataset parameters
        # num_total_samples = 3963360#3970176
        num_train_samples = 12941195#1000#10500000#100000#10#

        # dataset_name_prefix = 'zinc6m_all4'#'zinc6m_all4_mona'
        smiles_char_filename = 'tokens_smiles_train_0_5_val_0_2_test_0_3_63.json'
        smiles_max_length_filename = 'max_len_smiles_train_0_5_val_0_2_test_0_3_241.txt'
        dataset_name = 'canon_smiles_selfies_with_descriptors_train_0_5_val_0_2_test_0_3.parquet'
        padding = 'right'

        smiles_char_filepath = ''.join([path_init, '/', smiles_char_filename])
        smiles_max_length_filepath = ''.join([path_init, '/', smiles_max_length_filename])
        
        #+++++++++++++++++++++++++++++++++++++++++++++++++++
        smiles = PARSE_SMILES([])
        # load smiles char vocabulary
        smiles.smiles_char, smiles.smiles_char_map = smiles.load_smiles_char(smiles_char_filepath)
        # load smiles max-length
        smiles.max_smiles_length = smiles.load_smiles_max_length(smiles_max_length_filepath)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #+++++++++++++++++++++++++++++++++++++++++++++++++++
        ## VAE_GRU parameters
        learning_rate = learning_rate#1e-4#1e-3#
        dims_latent = dims_latent#300#100#50#25#
        num_gru = num_gru#4#5#3#
        
        dims_input_data = (smiles.max_smiles_length, len(smiles.smiles_char_map['domain'].keys())+1) 
        dims_output_data = (smiles.max_smiles_length, len(smiles.smiles_char_map['domain'].keys())+1)
        num_kernels = [9, 9, 10]#[[11, 13, 15]#11, 13, 15]#[15, 17, 19]#[11, 13, 15]#[9, 9, 10]
        size_kernels = [9, 9, 11]#[[13, 11, 9]#9, 9, 11]#[15, 13, 11]#[9, 9, 11]#[9, 9, 11]
        num_fc_layer_encoder = 0#2#
        dropout_prob = 0.0
        act_func = 'relu'
        scale_latent_space = 1e-2
        num_fc_layer_decoder = 0
        gru_hidden_size = 500
        layer_type = '1dcnn_gru'
        weight_init = 'xvr_unifrm'
        loss_type = 'bce_kld'
        solver_type = 'adam'
        num_epoch = 500
        batch_size = 128
        save_result_ateach_epoch = 50
        model_path = ''.join(['cache/smi2smi_vae/', path_init.split('/')[-2], '_', dataset_name, '_', str(num_train_samples), '/']).replace('.','_')
        #+++++++++++++++++++++++++++++++++++++++++++++++++++

        self.smiles_vae_model = SmIm2Sm_Network(dims_input_data, dims_output_data, 
                                        smiles_char_map=smiles.smiles_char_map, 
                                        max_string_len=smiles.max_smiles_length, 
                                        padding=padding, 
                                        num_kernels=num_kernels, 
                                        size_kernels=size_kernels, 
                                        num_fc_layer_encoder=num_fc_layer_encoder, 
                                        dropout_prob=dropout_prob, 
                                        dims_latent=dims_latent, 
                                        act_func=act_func, 
                                        scale_latent_space=scale_latent_space, 
                                        num_fc_layer_decoder=num_fc_layer_decoder, 
                                        gru_hidden_size=gru_hidden_size, 
                                        num_gru=num_gru, 
                                        layer_type=layer_type, 
                                        device=device, 
                                        weight_init=weight_init, 
                                        loss_type=loss_type, 
                                        solver_type=solver_type, 
                                        num_epoch=num_epoch, 
                                        batch_size=batch_size, 
                                        learning_rate=learning_rate, 
                                        save_result_ateach_epoch=save_result_ateach_epoch, 
                                        result_savepath=model_path)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++

        temp_network_path = '' .join([model_path_init, self.smiles_vae_model.result_savepath, self.smiles_vae_model.best_network_save_filename,'.pth'])
        self.smiles_vae_model.load_test_network(temp_network_path)
        self.smiles_vae_model.network.eval()
        
        return self.smiles_vae_model

    # --------------------------------------------------
    # Full pipeline
    # --------------------------------------------------

    def prepare_datasets(self,
                         train_partition="Tr",
                         partition_labels=None,
                         label_normalised=True,
                         feature_normalised=False,
                         model_path_init: str = '/home/rkmvu/Codes/new_smiles_data_train_test_vae_svae_16_09_24/',
                         dims_latent: int = 300,
                         num_gru: int = 4,
                         learning_rate: float = 0.0001
                         ):

        self.get_partitions(partition_labels)
        self.load_smiles_labels(train_partition=train_partition,
                                label_normalised=label_normalised)
        self.load_pretrained_vae(model_path_init = model_path_init,
                                 dims_latent = dims_latent,
                                 num_gru = num_gru,
                                 learning_rate = learning_rate
                                 )

        results = {}

        self.StandardScaler_features = StandardScaler()
        smiles_train = self.smiles[train_partition]
        X_train, _, _ = self.smiles_vae_model.smiles2latentspace_representation(smiles=smiles_train)
        if feature_normalised:
            print('-'*80)
            print(f'Training StandardScaler..')
            self.StandardScaler_features.fit(X_train)
        else:
            print(f'No training of StandardScaler. Setting mean = 0, variance = 1:')
            d = X_train.shape[1]
            self.StandardScaler_features.mean_  = np.zeros(d)
            self.StandardScaler_features.scale_ = np.ones(d)

        path = f'{self.cache_path}/StandardScaler_features.pkl'
        with open(path, 'wb') as fp:
            print(f'Saving StandardScaler for features in: {path}')
            pickle.dump(self.StandardScaler_features, fp)
        print('-'*80)
            
        for p in self.partitions:
            
            smiles = self.smiles[p]
            X, _, _ = self.smiles_vae_model.smiles2latentspace_representation(smiles=smiles)
            y = self.y[PARTITION_NAMES.get(p, p.lower())]
            
            if feature_normalised:
                print(f'Applying StandardScaler on {PARTITION_NAMES.get(p, p.lower())} data:')
                X = self.StandardScaler_features.transform(X)
                
            results[PARTITION_NAMES.get(p, p.lower())] = {
                "smiles": self.smiles[p],
                "dataset": list(zip(X, y))
            }
         

        results["scaler"] = {"labels": self.StandardScaler_labels,
                             "features": self.StandardScaler_features
                             }
        
        return results


class SMILES2GRAPH_Data(Torch_Dataset):
    
    def __init__(self,
                 smiles_list,
                 mol_details,
                 labels=None,
                 eval_mode = False,
                 label_mean = 0.0,
                 label_std = 1.0,
                 dataset_mean = 0.0,
                 dataset_std = 1.0,
                 label_mean_std_norm = False,
                 graph_backend_version = 'old',
                 features_list=['atom_properties', 'aromaticity', 'ring', 'chirality']
                 ):
        
        self.smiles_list = smiles_list
        self.mol_details = mol_details
        self.labels = labels
        self.eval_mode = eval_mode
        self.label_mean = label_mean
        self.label_std = label_std
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.features_list = features_list
        self.label_mean_std_norm = label_mean_std_norm
        self.graph_backend_version = graph_backend_version
        
        self.load_graph_utils()
        self.labels_normalize()
        self.preprocessing()
        
    def __len__(self):
        return len(self.smiles_list)
        
    def _get_one_new(self, idx):
        
        data_point = self.data[idx]
        smi, atom_feature_vector, bond_matrix, y = data_point
        dp =  {'SMILES': smi,
               'atom_feature_vector': atom_feature_vector,
               'bond_matrix': bond_matrix,
               'label': y
               }     
        temp = self.GraphData_from_pickle.GcnDictData2GcnInput(dp,
                                                    self.dataset_mean,
                                                    self.dataset_std,
                                                    max_num_atoms=self.mol_details['max_num_atoms'],
                                                    features_list=self.features_list)
        return temp
    
    def _get_one_old(self, idx):
        
        data_point = self.data[idx]
        smi, atom_feature_vector, adjacency_tensor, degree_tensor, y = data_point
        dp = {'SMILES': smi,
            'atom_feature_vector': atom_feature_vector,
            'adjacency_matrix': adjacency_tensor,
            'degree_matrix': degree_tensor,
            'label': y
            }
        temp = self.GraphData_from_pickle.GcnDictData2GcnInput(dp,
                                                    self.dataset_mean,
                                                    self.dataset_std,
                                                    max_num_atoms=self.mol_details['max_num_atoms'],
                                                    features_list=self.features_list)
        return temp
        
    def _get_one(self, idx):
        if self.graph_backend_version.lower() == 'new':
            return self._get_one_new(idx)
        else:
            return self._get_one_old(idx)
        
    def __getitem__(self, idx):
        
        if isinstance(idx, (slice, list, tuple)):  
            if isinstance(idx, slice):
                indices = range(*idx.indices(len(self)))
            else:
                indices = idx
            items = [self._get_one(i) for i in indices]
            return self.collate_fn(items)
        return self._get_one(idx) 
    
    def collate_fn(self, items):
    
        n_out = len(items[0])
        all_lists = [[] for _ in range(n_out)]
        
        for i in range(n_out):
            for it in items:
                all_lists[i].append(it[i])
        
        # Only stack if shapes are consistent
        all_lists = [torch.from_numpy(np.stack(x)) for x in all_lists]
        
        return tuple(all_lists)
        
    def load_graph_utils(self):
        
        if self.graph_backend_version == "old":
            print(f'Generating data in old style:')
            self.Multirelational_GraphDataset = Multirelational_GraphDataset_old
            self.SmilesDataset_graph_gen = SmilesDataset_graph_gen_old
            self.GraphData_from_pickle = GraphData_from_pickle_old
            
        elif self.graph_backend_version == "new":
            print(f'Generating data in new style:')
            self.Multirelational_GraphDataset = Multirelational_GraphDataset_new
            self.SmilesDataset_graph_gen = SmilesDataset_graph_gen_new
            self.GraphData_from_pickle = GraphData_from_pickle_new
        else:
            raise ValueError(f"Unknown graph backend version: {self.graph_backend_version}")
        
    def labels_normalize(self):
        
        if self.labels is not None:
            self.StandardScaler_labels = StandardScaler()
            labels = ensure_2d(self.labels)

            if self.label_mean_std_norm:
                self.StandardScaler_labels.fit(labels)
            else:
                self.StandardScaler_labels.mean_ = np.array([0.0])
                self.StandardScaler_labels.scale_ = np.array([1.0])
        
            self.labels = self.StandardScaler_labels.transform(labels)
            self.labels = self.labels.ravel() if self.labels.shape[1] == 1 else self.labels
        else:
            self.labels = np.zeros((len(self.smiles_list)))
            
    def preprocessing(self):
        
        self.data = self.Multirelational_GraphDataset(self.smiles_list, labels=self.labels, **self.mol_details)
        
        return self.data
       
    
class SMILES_PRUNING:
    
    def __init__(self,
                smiles_list,
                labels='eval',
                smiles_column_name='smiles',
                filters = None,
                all_atoms={'Br', 'Cl', 'P', 'I', 'F', 'H', 'S', 'N', 'O', 'C', 'B', 'Si', 'Na', 'K'},
                prefix='',

                sanitize=False,                             ## This should be 'True' for generic use.
                canonical: bool = False,                    ## This should be 'True' for generic use.
                normalize_functional_groups: bool = False,  ## This should be 'True' for generic use.
                sanitize_final: bool = False,               ## This should be 'True' for generic use.
                remove_explicit_h: bool = False,            ## This should be 'True' for generic use.

                keep_largest_fragment: bool = False,        ## This should be 'True' for generic use.
                require_organic_fragment: bool = False,     ## This should be 'True' for generic use.
                disconnect_metals: bool = False,            ## This should be 'True' for generic use.
                strip_salts: bool = False,                  ## This should be 'True' for generic use.
                salt_remover_def: Optional[str] = None,
                
                reionize: bool = False,
                uncharge: bool = False,
                canonicalize_tautomer: bool = False,
                kekulize: bool = False,
                clear_stereo: bool = False,

                return_mol: bool = False,
                log_flag: bool = False
                 ):
        '''
            Supported columns for filter: ['MolWt', 'MolLogP', 'MolMR', 'qed', 'TPSA', 'NumHDonors']
        '''
        self.labels = labels
        self.all_atoms = all_atoms
        self.prefix = prefix
        self.filters = filters
        self.smiles_column_name = smiles_column_name
        self.strip_salts = strip_salts
        self.salt_remover_def = salt_remover_def
        self.keep_largest_fragment = keep_largest_fragment
        self.keep_largest_fragment = keep_largest_fragment
        self.require_organic_fragment = require_organic_fragment
        self.disconnect_metals = disconnect_metals
        self.normalize_functional_groups = normalize_functional_groups
        self.reionize = reionize
        self.uncharge = uncharge
        self.canonicalize_tautomer = canonicalize_tautomer
        self.remove_explicit_h = remove_explicit_h
        self.sanitize_final = sanitize_final
        self.kekulize = kekulize
        self.clear_stereo = clear_stereo
        self.canonical = canonical
        self.return_mol = return_mol
        self.log_flag = log_flag
        self.smiles_list = [Chem.MolFromSmiles(s, sanitize=sanitize) for s in smiles_list]
        self.log = []
        
        if isinstance(labels, str) and labels == 'eval':
            self.labels = [None]*len(self.smiles_list)
        else:
            assert len(self.smiles_list) == len(self.labels), f'ERROR: smiles (n_items - {len(self.smiles_list)}) and labels (n_items - {len(self.labels)}) should be of same lengths.'
            
        self.df = pd.DataFrame({self.smiles_column_name:self.smiles_list, 'labels':self.labels})
        self.df = self.df[~self.df[self.smiles_column_name].isna()]
        if self.prefix:
            self.df = attach_ids(self.df, self.prefix)
        self.df['MolWt'] = self.df[self.smiles_column_name].apply(Chem.Descriptors.MolWt)
        self.df['MolLogP'] = self.df[self.smiles_column_name].apply(Chem.Descriptors.MolLogP)
        self.df['qed'] = self.df[self.smiles_column_name].apply(Chem.Descriptors.qed)
        self.df['TPSA'] = self.df[self.smiles_column_name].apply(Chem.Descriptors.TPSA)
        self.df['NumHDonors'] = self.df[self.smiles_column_name].apply(Chem.Descriptors.NumHDonors)
        
    def _safe_mol_from_smiles(self, smiles: str, sanitize: bool = True) -> Tuple[Optional[Chem.Mol], Optional[str]]:
        """
        Try to parse a SMILES safely. Returns (mol, error_message).
        """
        if not smiles or not isinstance(smiles, str):
            return None, "empty_or_nonstring_input"

        try:
            # allow deferred sanitization if needed later
            m = Chem.MolFromSmiles(smiles, sanitize=sanitize)
            if m is None:
                return None, "rdkit_failed_to_parse"
            return m, None
        except Exception as e:
            return None, f"parse_exception:{str(e)}"


    def _select_largest_fragment(self, mol: Chem.Mol, require_organic: bool = True) -> Chem.Mol:
        """
        Select the largest fragment by heavy atom count. If require_organic is True,
        prefer fragments containing C (a simple organic heuristic).
        """
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if not frags:
            return mol

        # filter organic fragments if requested
        if require_organic:
            organic_frags = [f for f in frags if any(a.GetSymbol() == 'C' for a in f.GetAtoms())]
            if organic_frags:
                frags = organic_frags

        # choose fragment with maximum heavy atoms
        return max(frags, key=lambda m: m.GetNumHeavyAtoms())


    def sanitize_smiles_master(self,
                            smiles: str|Chem.Mol,
                            *,
                            strip_salts: bool = True,
                            salt_remover_def: Optional[str] = None,
                            keep_largest_fragment: bool = True,
                            require_organic_fragment: bool = True,
                            disconnect_metals: bool = True,
                            normalize_functional_groups: bool = True,
                            reionize: bool = True,
                            uncharge: bool = True,
                            canonicalize_tautomer: bool = True,
                            remove_explicit_h: bool = True,
                            sanitize_final: bool = True,
                            kekulize: bool = False,
                            clear_stereo: bool = False,
                            canonical: bool = True,
                            return_mol: bool = False,
                            log_flag: bool = False
                            ) -> Dict[str, Any]:
        """
        Comprehensive step-by-step SMILES sanitization pipeline.

        Returns a dictionary with:
        - 'input_smiles' : original input
        - 'status' : 'success' or 'failed'
        - 'final_smiles' : resulting canonical SMILES (or None on failure)
        - 'mol' : RDKit Mol if return_mol True and success
        - 'steps' : list of step logs (each is dict with 'step', 'smiles', 'note')
        - 'error' : error message if any

        Parameters are self-explanatory; tweak as required.
        """

        
        original = smiles
        if isinstance(smiles, str):
            mol, err = self._safe_mol_from_smiles(smiles, sanitize=True)
        else:
            mol=smiles
        if mol is None:
            if log_flag:
                self.log.append({"step": "parse", "smiles": None, "note": err})
            return {
                "input_smiles": original,
                "status": "failed",
                "final_smiles": None,
                "steps": self.log,
                "error": err,
            }
        if log_flag:
            self.log.append({"step": "parse", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "parsed"})

        try:
            # Step 1: Strip salts/solvents (SaltRemover)
            if strip_salts:
                if salt_remover_def:
                    remover = SaltRemover(defnData=salt_remover_def)
                else:
                    # default definition removes common salts (RDKit built-in)
                    remover = SaltRemover()
                mol_stripped = remover.StripMol(mol, dontRemoveEverything=True)
                if log_flag:
                    self.log.append({
                    "step": "strip_salts",
                    "smiles": Chem.MolToSmiles(mol_stripped, canonical=False) if mol_stripped else None,
                    "note": "salt_removal_applied"
                })
                mol = mol_stripped

            # Step 2: Keep largest fragment (prefer organic)
            if keep_largest_fragment:
                mol_before = Chem.MolToSmiles(mol, canonical=False) if mol else None
                mol = self._select_largest_fragment(mol, require_organic=require_organic_fragment)
                if log_flag:
                    self.log.append({
                    "step": "largest_fragment",
                    "smiles": Chem.MolToSmiles(mol, canonical=False) if mol else None,
                    "note": f"selected_largest_fragment (organic_required={require_organic_fragment}) from {mol_before}"
                })

            # Step 3: Disconnect metals (break coordinate bonds)
            if disconnect_metals:
                md = rdMolStandardize.MetalDisconnector()
                mol = md.Disconnect(mol)
                if log_flag:
                    self.log.append({
                    "step": "disconnect_metals",
                    "smiles": Chem.MolToSmiles(mol, canonical=False),
                    "note": "metal_disconnected"
                })

            # Step 4: Normalize functional groups (Normalizer)
            if normalize_functional_groups:
                normalizer = rdMolStandardize.Normalizer()
                mol = normalizer.normalize(mol)
                if log_flag:
                    self.log.append({"step": "normalize", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "normalizer_applied"})

            # Step 5: Reionize (preferred protonation)
            if reionize:
                reionizer = rdMolStandardize.Reionizer()
                mol = reionizer.reionize(mol)
                if log_flag:
                    self.log.append({"step": "reionize", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "reionizer_applied"})

            # Step 6: Uncharge (neutralize where sensible)
            if uncharge:
                uncharger = rdMolStandardize.Uncharger()
                mol = uncharger.uncharge(mol)
                if log_flag:
                    self.log.append({"step": "uncharge", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "uncharger_applied"})

            # Step 7: Tautomer canonicalization
            if canonicalize_tautomer:
                taut_enum = rdMolStandardize.TautomerEnumerator()
                mol = taut_enum.Canonicalize(mol)
                if log_flag:
                    self.log.append({"step": "tautomer", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "tautomer_canonicalized"})

            # Step 8: Clear stereo if requested
            if clear_stereo:
                Chem.RemoveStereochemistry(mol)
                if log_flag:
                    self.log.append({"step": "clear_stereo", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "stereo_removed"})

            # Step 9: Remove explicit Hs if requested
            if remove_explicit_h:
                mol = Chem.RemoveHs(mol)
                if log_flag:
                    self.log.append({"step": "remove_explicit_h", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "explicit_H_removed"})

            # Step 10: Final sanitize and optional kekulize
            if sanitize_final:
                Chem.SanitizeMol(mol)
                if log_flag:
                    self.log.append({"step": "final_sanitize", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "sanitized"})
            if kekulize:
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                    if log_flag:
                        self.log.append({"step": "kekulize", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "kekulized"})
                except Exception:
                    # some molecules cannot be kekulized; not fatal
                    if log_flag:
                        self.log.append({"step": "kekulize", "smiles": Chem.MolToSmiles(mol, canonical=False), "note": "kekulize_failed_or_not_applicable"})

            # Final canonicalization
            final_smiles = Chem.MolToSmiles(mol, canonical=canonical)
            if log_flag:
                self.log.append({"step": "final_canonical", "smiles": final_smiles, "note": "done"})

            out = {
                "input_smiles": original,
                "status": "success",
                "final_smiles": final_smiles,
                "steps": self.log,
                "error": None,
            }
            if return_mol:
                out["mol"] = mol
            return out

        except Exception as e:
            # Fail gracefully and include logs for debugging
            if log_flag:
                self.log.append({"step": "exception", "smiles": None, "note": str(e)})
            return {
                "input_smiles": original,
                "status": "failed",
                "final_smiles": None,
                "steps": self.log,
                "error": str(e),
            }
            
    def sanitize_smiles_list(self, smiles_list):
        
        cleaned_smiles=[None]*len(smiles_list)
        for i, s in enumerate(smiles_list):
            cleaned_smiles[i] = self.sanitize_smiles_master(smiles=s,
                                                            strip_salts = self.strip_salts,
                                                            salt_remover_def = self.salt_remover_def,
                                                            keep_largest_fragment = self.keep_largest_fragment,
                                                            require_organic_fragment = self.require_organic_fragment,
                                                            disconnect_metals = self.disconnect_metals,
                                                            normalize_functional_groups = self.normalize_functional_groups,
                                                            reionize = self.reionize,
                                                            uncharge = self.uncharge,
                                                            canonicalize_tautomer = self.canonicalize_tautomer,
                                                            remove_explicit_h = self.remove_explicit_h,
                                                            sanitize_final = self.sanitize_final,
                                                            kekulize = self.kekulize,
                                                            clear_stereo = self.clear_stereo,
                                                            canonical = self.canonical,
                                                            return_mol = self.return_mol,
                                                            log_flag = self.log_flag
                                                            )
        # Extract the final SMILES only (None if failed)
        cleaned_smiles = [r["final_smiles"] for r in cleaned_smiles]
        
        return cleaned_smiles

    def get_cleaned_smiles(self):
        
        self.df[self.smiles_column_name] = self.sanitize_smiles_list(self.df[self.smiles_column_name])
        # self.df[self.smiles_column_name] = self.df[self.smiles_column_name].apply(Chem.MolToSmiles)
        # if self.filters:
        #     self.df = apply_column_constraints(self.df, self.filters)
        
        return self.df
