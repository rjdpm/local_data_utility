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

        results = {PARTITION_NAMES.get(p, p): {"smiles": self.smiles[p], "dataset": datasets[p]} for p in datasets}
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
                                                        random_state=42, col_drop_threshold=0.99,
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
                            fingerprints=False, mi_threshold=0.05,
                            corr_threshold=0.95, non_corr_threshold=0.05,
                            force=False):

        random.seed(10)
        np.random.seed(20)

        print(f'Preprocessing features:')
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

    def normalize_data(self, train_partition, feature_normalised=True, label_normalised=True):

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

        results = {PARTITION_NAMES.get(p, p): {"smiles": self.smiles[p], "dataset": datasets[p]} for p in datasets}

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

            hf_datasets[PARTITION_NAMES.get(p, p)] = {"smiles": smiles, "dataset": ds}

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


class GCNDataset:

    def __init__(self, df_init,
                 smiles_column_name="smiles",
                 target_column_name="target",
                 partition_col="split",
                 max_num_atoms=100,
                 graph_dataset_path="./datasets/graph_dataset"):

        self.df = df_init.copy()
        self.smiles_column_name = smiles_column_name
        self.target_column_name = target_column_name
        self.partition_col = partition_col
        self.max_num_atoms = max_num_atoms
        self.graph_dataset_path = graph_dataset_path

        self.partitions = {}       # label -> dataframe
        self.smiles = {}           # label -> smiles
        self.y = {}                # label -> labels

        self.StandardScaler_labels = None
        self.kwargs = None

    # ----------------------------------------------------
    # Partition handling
    # ----------------------------------------------------

    def get_partitions(self, partition_labels=['Tr', 'Val', 'Te']):
        
        self.partitions = split_data(self.df,
                                    partition_col=self.partition_col,
                                    partition_labels=partition_labels
                                    )

    # ----------------------------------------------------
    # Atom & column filtering
    # ----------------------------------------------------

    def atom_filter(self, all_atoms={'Br','Cl','P','I','F','H','S','N','O','C','B','Si','Na','K'}):
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

    def apply_column_constraints(self, filters):
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

    # ----------------------------------------------------
    # SMILES and label handling
    # ----------------------------------------------------

    def load_smiles_labels(self, train_partition="Tr", label_normalised=True):

        for p, dfp in self.partitions.items():
            self.smiles[p] = dfp[self.smiles_column_name].tolist()
            self.y[p] = np.stack(dfp[self.target_column_name].values)

        # Fit label scaler only on reference partition
        self.StandardScaler_labels = StandardScaler()
        y_train = ensure_2d(self.y[train_partition])

        if label_normalised:
            self.StandardScaler_labels.fit(y_train)
        else:
            self.StandardScaler_labels.mean_ = np.zeros(y_train.shape[1])
            self.StandardScaler_labels.scale_ = np.ones(y_train.shape[1])

        os.makedirs(self.graph_dataset_path, exist_ok=True)
        with open(f"{self.graph_dataset_path}/StandardScaler_labels.pkl", "wb") as fp:
            pickle.dump(self.StandardScaler_labels, fp)

        for p in self.y:
            Y = self.StandardScaler_labels.transform(ensure_2d(self.y[p]))
            self.y[p] = Y.ravel() if Y.shape[1] == 1 else Y

    # ----------------------------------------------------
    # Atom metadata (fit on training split only)
    # ----------------------------------------------------

    def mol_details(self, train_partition, symb_hyb_chirl_file):

        if os.path.isfile(symb_hyb_chirl_file):
            self.kwargs = load_json(symb_hyb_chirl_file)
        else:
            self.kwargs = list_smiles2symbols_hybridization_chiraltype(self.smiles[train_partition])
            dict2json(self.kwargs, filepath=symb_hyb_chirl_file)

        self.kwargs["max_num_atoms"] = self.max_num_atoms

    # ----------------------------------------------------
    # Dataset creation
    # ----------------------------------------------------

    def create_datasets(self, train_partition, symb_hyb_chirl_file):

        self.mol_details(train_partition, symb_hyb_chirl_file)

        os.makedirs(self.graph_dataset_path, exist_ok=True)
        out_path = f"{self.graph_dataset_path}/SmilesDataset_graph"

        for p in self.partitions:
            ds = Multirelational_GraphDataset(smi_list=self.smiles[p], labels=self.y[p], **self.kwargs)
            SmilesDataset_graph_gen(p.lower(), ds, out_path=out_path, mean_std_flag=(p == train_partition))

    # ----------------------------------------------------
    # Load from pickle
    # ----------------------------------------------------

    def load_dataset_from_pkl(self, features_list):
        
        kwargs = {"features_list": features_list, "max_num_atoms": self.max_num_atoms}
        datasets = {}

        for p in self.partitions:
            datasets[p] = GraphData_from_pickle(f"{self.graph_dataset_path}/SmilesDataset_graph_{p.lower()}.pkl.gz", **kwargs)

        with open(f"{self.graph_dataset_path}/StandardScaler_labels.pkl", "rb") as fp:
            self.StandardScaler_labels = pickle.load(fp)
        
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
                         features_list=['atom_properties','logP_values','gasteiger_charge'],
                         label_normalised=True):

        # filtering
        self.atom_filter(all_atoms)
        if filters:
            self.apply_column_constraints(filters)

        # partition
        self.get_partitions(partition_labels)

        # labels
        self.load_smiles_labels(train_partition=train_partition,
                                label_normalised=label_normalised)

        # build or load
        try:
            datasets = self.load_dataset_from_pkl(features_list)
        except:
            self.create_datasets(train_partition, symb_hyb_chirl_file)
            datasets = self.load_dataset_from_pkl(features_list)

        results = {PARTITION_NAMES.get(p, p): {"smiles": [datasets[p].get_smiles(i) for i in range(len(datasets[p]))],
                       "dataset": datasets[p]
                       } for p in datasets
                   }

        results["scaler"] = {"labels": self.StandardScaler_labels}
        
        return results

