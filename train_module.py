#!/usr/bin/env python3

from typing import Union, Tuple, Dict, List, Any
import os, sys, pickle, copy, re, warnings
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from collections import OrderedDict
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import torch, platform
import tarfile
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau

import torchvision
import torchvision.transforms as tt
from torchvision import datasets, transforms
from torchvision.models import resnet18, mobilenet_v2
from torchvision.models import mobilenet_v3_large
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Generalised_data_utils import regression_test_metrics, datetime_now, create_folder, subset_loader, get_func_input_names, auto_repr

__all__ = ['NN_Trainer',
           'weights_initializer',
           'EarlyStopping',
           'Trainer_functions',
           'TransferWeights'
           ]

class EarlyStopping:
    def __init__(self, stop_patience: int = 10, min_delta: float = 1e-5, type_: str = 'loss'):
        """
        Early stopping mechanism for model training.

        Args:
            stop_patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change to qualify as improvement.
            type_ (str): Metric type to monitor ('loss' or 'accuracy').

        Raises:
            ValueError: If `type_` is not 'loss' or 'accuracy'.
        """
        if type_ not in ['loss', 'accuracy']:
            raise ValueError("type_ must be either 'loss' or 'accuracy'")

        self.patience = stop_patience
        self.min_delta = min_delta
        self.type_ = type_
        self.counter = 0
        self.early_stop = False
        self.best_score = np.inf if self.type_ == 'loss' else -np.inf

    def __call__(self, current_score: float):
        """
        Update the early stopping status based on the current validation metric.

        Args:
            current_score (float): Current validation loss or accuracy.

        Returns:
            None: Updates internal counter and early_stop flag.
        """
        if self.type_ == 'loss':
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

        elif self.type_ == 'accuracy':
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    
                
class Trainer_functions:
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_scheduler(scheduler_name: str,
                    optimizer: torch.optim.Optimizer,
                    **kwargs
                    ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Returns a learning rate scheduler based on the provided name and parameters.
        
        Parameters:
            scheduler_name (str): Name of the scheduler, e.g., 'StepLR', 'MultiStepLR', 'ExponentialLR', 'ReduceLROnPlateau'.
            optimizer (Optimizer): The optimizer for which to schedule the learning rate.
            **kwargs: Additional keyword arguments for the specific scheduler.
            
        Returns:
            scheduler: The instantiated learning rate scheduler.
        """
        
        scheduler_name = scheduler_name.lower()  # make case-insensitive

        if scheduler_name == 'steplr':
            scheduler = StepLR(optimizer, **kwargs)
        elif scheduler_name == 'multisteplr':
            scheduler = MultiStepLR(optimizer, **kwargs)
        elif scheduler_name == 'exponentiallr':
            scheduler = ExponentialLR(optimizer, **kwargs)
        elif scheduler_name == 'reducelronplateau':
            scheduler = ReduceLROnPlateau(optimizer, **kwargs)
        elif scheduler_name == 'cycliclr':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **kwargs)
        elif scheduler_name == 'onecyclelr':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
        else:
            raise ValueError(f"Scheduler Name: '{scheduler_name}' is not in: ['steplr', 'multisteplr', 'exponentiallr', 'reducelronplateau', 'cycliclr', 'onecyclelr'].")

        return scheduler

    @staticmethod
    def activation_func(name='relu', param={'alpha':1.0, 'negative_slope':1e-2}):
        """
        Returns an activation function module based on the provided name.

        Args:
            name (str): Name of the activation function. Supported: 
                        'sigmoid', 'tanh', 'relu', 'selu', 'elu', 'leakyrelu', 'celu'.
            param (dict): Parameters for activation functions that require alpha or negative_slope.
                        Example: {'alpha': 1.0, 'negative_slope': 1e-2}

        Returns:
            nn.Module: Corresponding PyTorch activation function module.
        
        Raises:
            ValueError: If the activation name is not supported.
        """
            
        if name=='sigmoid':
            return nn.Sigmoid()
        elif name=='tanh':
            return nn.Tanh()
        elif name=='relu':
            return nn.ReLU(inplace=True)
        elif name=='selu':
            return nn.SELU(inplace=True)
        elif name=='elu':
            return nn.ELU(param['alpha'], inplace=True)
        elif name=='leakyrelu':
            return nn.LeakyReLU(param['negative_slope'], inplace=True)
        elif name=='celu':
            return nn.CELU(param['alpha'], inplace=True)
        else:
            raise ValueError('Define your activation function: "{}"' .format(name))


    @staticmethod
    def get_optimizer(model, solver_type='adam', learning_rate=1e-4, **kwargs):
        """
        Returns a PyTorch optimizer based on the specified type.

        Args:
            model (nn.Module): The neural network whose parameters are to be optimized.
            solver_type (str): Type of optimizer. Supported:
                'adam', 'sgd', 'rprop', 'rmsprop', 'radam', 'adamw', 'adagrad', 'adamax', 'nadam'
            learning_rate (float): Learning rate.
            kwargs (dict): Additional optimizer-specific parameters (e.g., weight_decay).

        Returns:
            torch.optim.Optimizer: Instantiated optimizer.

        Raises:
            ValueError: If an unsupported optimizer is specified.
        """
        solver_type = solver_type.lower()

        optimizers = {
            'adam':     optim.Adam,
            'sgd':      optim.SGD,
            'rprop':    optim.Rprop,
            'rmsprop':  optim.RMSprop,
            'radam':    getattr(optim, 'RAdam', None),   # Some versions may not support RAdam
            'adamw':    optim.AdamW,
            'adagrad':  optim.Adagrad,
            'adamax':   optim.Adamax,
            'nadam':    getattr(optim, 'NAdam', None),   # PyTorch ≥ 1.9
        }

        optimizer_class = optimizers.get(solver_type)

        if optimizer_class is None:
            raise ValueError(f'Unsupported optimizer: "{solver_type}". Please define or check availability.')

        return optimizer_class(model.parameters(), lr=learning_rate, **kwargs)

    #-------------------------------------------------------------------------------------------------------------------------
            

    @staticmethod
    def loss_function_selector(name='mse', params=None):
        
        """
        Returns a PyTorch loss function object based on the specified name.
        
        Args:
            name (str): Name of the loss function. Supported:
                'mse', 'l1', 'crossentropy', 'bce', 'bcewithlogits', 'smoothl1', 'huber',
                'nll', 'hinge', 'cosine', 'kl', 'marginranking', 'multilabelmargin'
            params (dict, optional): Parameters to pass into the loss function constructor.
        
        Returns:
            nn.Module: Instantiated loss function object.
        
        Raises:
            ValueError: If the loss name is unrecognized.
        """
        
        if params is None:
            params = {}

        name = name.lower()

        if name == 'mse':
            return nn.MSELoss(**params)
        elif name == 'l1':
            return nn.L1Loss(**params)
        elif name == 'crossentropy':
            return nn.CrossEntropyLoss(**params)
        elif name == 'bce':
            return nn.BCELoss(**params)
        elif name == 'bcewithlogits':
            return nn.BCEWithLogitsLoss(**params)
        elif name in ('smoothl1', 'huber'):  # Huber alias
            return nn.SmoothL1Loss(**params)
        elif name == 'nll':
            return nn.NLLLoss(**params)
        elif name == 'hinge':
            return nn.HingeEmbeddingLoss(**params)
        elif name == 'cosine':
            return nn.CosineEmbeddingLoss(**params)
        elif name == 'kl':
            return nn.KLDivLoss(**params)
        elif name == 'marginranking':
            return nn.MarginRankingLoss(**params)
        elif name == 'multilabelmargin':
            return nn.MultiLabelMarginLoss(**params)
        else:
            raise ValueError(f'Unknown loss function: "{name}". Please define it or add support.')


    # # Define the test function
    @staticmethod
    def nn_input2output(model: nn.Module,
                        test_loader: DataLoader[Dict[str, torch.Tensor]],
                        criterion: nn.Module,
                        targets_col: str = 'labels',
                        description: str = 'Map',
                        device: str=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                        ) -> Tuple[np.ndarray[float], np.ndarray[float], float]:
        
        """
        Evaluates the model on the test set and returns predictions, true values, and average loss.

        Args:
            model (nn.Module): Trained neural network model.
            test_loader (DataLoader): DataLoader containing test data batches.
            criterion (nn.Module): Loss function used for evaluation.
            targets_col (str): Key in each batch corresponding to target values.
            device (str): Device to perform computation on ('cuda' or 'cpu').

        Returns:
            Tuple[np.ndarray, np.ndarray, float]: 
                - y_pred: Model predictions (NumPy array).
                - y_true: Ground-truth values (NumPy array).
                - epoch_test_loss: Average test loss over all samples.
        """
        
        model = model.to(device)
        model.eval()  # Set model to evaluation mode
        y_pred = []
        y_true = []
        epoch_test_loss = 0
        model_params = get_func_input_names(model.forward)

        with torch.no_grad():  # No need to compute gradients during evaluation
            for batch in tqdm(test_loader, desc=description):
                targets = batch.pop(targets_col).to(device)
                batch = {k: batch[k].to(device) for k in model_params}
                # batch = {k: v.to(device) for k, v in batch.items()}
                # targets = batch.pop(targets_col)
                outputs = model(**batch)
                loss = criterion(input=outputs.reshape(targets.shape).type(torch.float32), target=targets.type(torch.float32)).type(torch.float32)
                    
                # Collect all predictions and targets
                y_pred.append(outputs.reshape(targets.shape).cpu())
                y_true.append(targets.cpu())
                
                # Accumulate loss
                epoch_test_loss += loss.item()
            epoch_test_loss = epoch_test_loss/len(test_loader.dataset)
        
        # Concatenate all predictions and targets across batches
        y_pred = torch.cat(y_pred).numpy()
        y_true = torch.cat(y_true).numpy()
        
        return y_pred, y_true, epoch_test_loss

    # # Define the test function
    @staticmethod
    def test_model(model: nn.Module,
                test_loader: DataLoader[Dict[str, torch.Tensor]],
                criterion: nn.Module,
                targets_col: str = 'labels',
                description: str = 'Test',
                device: str=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                ) -> Tuple[Dict[str, float], float]:
        
        """
        Evaluates a trained model on the test data and computes regression metrics.

        Args:
            model (nn.Module): Trained neural network model.
            test_loader (DataLoader): DataLoader containing test data batches.
            criterion (nn.Module): Loss function used for evaluation.
            targets_col (str): Key in each batch corresponding to target values.
            device (str): Device to perform computation on ('cuda' or 'cpu').

        Returns:
            Tuple[Dict[str, float], float]:
                - results: Dictionary containing regression metrics (e.g., R2, MAE, RMSE).
                - epoch_test_loss: Average test loss.
        """
        
        y_pred, y_true, epoch_test_loss = Trainer_functions.nn_input2output(model=model,
                                                                test_loader=test_loader,
                                                                criterion=criterion,
                                                                targets_col=targets_col,
                                                                description = description,
                                                                device=device
                                                                )
        results = regression_test_metrics(y_true=y_true, y_pred=y_pred)
            
        return results, epoch_test_loss


    # # Define the test function
    @staticmethod
    def train_test_model(model: nn.Module,
                test_loader: DataLoader[Dict[str, torch.Tensor]],
                criterion: nn.Module,
                targets_col: str = 'labels',
                description: str = 'Train-Test',
                device: str=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                ) -> Tuple[float, float]:
        
        """
        Computes test R² score and average loss for a trained model.

        Args:
            model (nn.Module): Trained neural network model.
            test_loader (DataLoader): DataLoader containing test data batches.
            criterion (nn.Module): Loss function used for evaluation.
            targets_col (str): Key in each batch corresponding to target values.
            device (str): Device to perform computation on ('cuda' or 'cpu').

        Returns:
            Tuple[float, float]:
                - r2: Coefficient of determination (R² score) on the test set.
                - epoch_test_loss: Average loss over the test set.
        """
        y_pred, y_true, epoch_test_loss = Trainer_functions.nn_input2output(model=model,
                                                                test_loader=test_loader,
                                                                criterion=criterion,
                                                                targets_col=targets_col,
                                                                description=description,
                                                                device=device
                                                                )

        # Calculate metrics
        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        
        return r2, epoch_test_loss


    @staticmethod
    def train_1epoch(model: nn.Module,
                    train_loader: DataLoader[Dict[str, torch.Tensor]],
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    param_reg: str = '',
                    lambda_reg: float = 0.1,
                    alpha: float = 0.5,
                    targets_col: str = 'labels',
                    description: str = 'Training',
                    device: str=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    ) -> float:
            
        """
        Trains the model for one epoch using optional regularization and returns average training loss.

        Args:
            model (nn.Module): Model to be trained.
            train_loader (DataLoader): DataLoader containing training data batches.
            criterion (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
            param_reg (str): Type of regularization ('l1', 'l2', 'elastic_net', or '').
            lambda_reg (float): Regularization strength.
            alpha (float): Elastic net mixing parameter (0 = L2 only, 1 = L1 only).
            targets_col (str): Key in batch dict for ground truth labels.
            device (str): Device to perform computation on ('cuda' or 'cpu').

        Returns:
            float: Average training loss over the entire epoch.
        """
        
        model.train()# switch to train model
        model = model.to(device)
        epoch_train_loss = 0
        model_params = get_func_input_names(model.forward)
        # with torch.autograd.set_detect_anomaly(True):
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=description):
            # print(f'Train Batch: {i}\n')
            # batch = {k: v.to(device) for k, v in batch.items() if k in model_params}
            targets = batch.pop(targets_col).to(device)
            batch = {k: batch[k].to(device) for k in model_params}
            optimizer.zero_grad()
            outputs = model(**batch)
                
            loss = criterion(input=outputs.reshape(targets.shape).type(torch.float32), target=targets.type(torch.float32)).type(torch.float32)
            if param_reg in ['l1', 'elastic_net']:
                l1_temp = 0
                for param in model.parameters():
                    l1_temp = l1_temp + torch.sum(torch.abs(param))
                l1_reg = lambda_reg * l1_temp  # L1 term
                if param_reg == 'l1':
                    loss = loss + l1_reg
                
            if param_reg in ['l2', 'elastic_net']:
                l2_temp = 0
                for param in model.parameters():
                    l2_temp = l2_temp + torch.sum(param**2)
                l2_reg = lambda_reg * l2_temp    # L2 term
                if param_reg == 'l2':
                    loss = loss + l2_reg
                
            if param_reg == 'elastic_net':
                elastic_net_reg = alpha * l1_reg + (1 - alpha) * l2_reg
                loss = loss + elastic_net_reg
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            epoch_train_loss += loss.item()
        epoch_train_loss = epoch_train_loss/len(train_loader.dataset)
        
        return epoch_train_loss


    @staticmethod
    def train_model(model: nn.Module,
                    train_loader: DataLoader[Dict[str, torch.Tensor]],
                    val_loader: DataLoader[Dict[str, torch.Tensor]],
                    test_loader: DataLoader[Dict[str, torch.Tensor]],
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    targets_col: str = 'labels',
                    scheduler_name: str = '',
                    schedulers_kwargs: dict = {},
                    early_stop: bool = False,
                    early_stop_kwargs: dict = {},
                    num_epochs: int = 100,
                    batch_size: int = 128,
                    subset_ratio: int = 1,
                    xlabel: str = 'Epoch',
                    ylabel_acc: str = 'r2_value',
                    ylabel_loss: str = 'Loss_value',
                    title: str = 'Loss and Accuracy Plot',
                    results_savepath: str = '',
                    save_image_filename: str = 'Untitled',
                    loss_save_filename: str = 'loss_acc_file',
                    best_model_savename: str = 'best_model',
                    description_savename: str = 'description.txt',
                    tqdm_write: bool = False,
                    standardscaler: Any = None,
                    save_model_wrt: str = 'loss', # 'loss' or 'accuracy'
                    more_description = '',
                    save_model_per_epoch: int = 50,
                    DATETIME: str = datetime_now(path=False)[0],
                    date: str = datetime_now(path=False)[1],
                    time: str = datetime_now(path=False)[2],
                    save_model: bool = True,
                    device: Any = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    collate_fn = None
                    ) -> tuple[List[float], nn.Module]:
        
        """
        Train a PyTorch neural network model over multiple epochs with optional validation, testing,
        learning rate scheduling, and early stopping. Logs metrics, saves plots, and stores the best model.

        ...

        Args:
            model (nn.Module): The neural network model to be trained.
            train_loader (DataLoader): Dataloader containing training data.
            val_loader (DataLoader): Dataloader containing validation data.
            test_loader (DataLoader): Dataloader containing test data.
            criterion (nn.Module): Loss function to optimize.
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            targets_col (str, optional): Key for the target values in batch dictionary. Defaults to 'labels'.
            scheduler_name (str, optional): Name of the learning rate scheduler (if any). Defaults to ''.
            schedulers_kwargs (dict, optional): Keyword arguments for the scheduler. Defaults to {}.
            early_stop (bool, optional): Whether to apply early stopping. Defaults to False.
            early_stop_kwargs (dict, optional): Arguments for early stopping strategy. Defaults to {}.
            num_epochs (int, optional): Maximum number of epochs to train. Defaults to 100.
            batch_size (int, optional): Batch size for dataloaders. Defaults to 128.
            subset_ratio (int, optional): Ratio of data subset for metrics during training. Defaults to 1.
            xlabel (str, optional): Label for x-axis in plots. Defaults to 'Epoch'.
            ylabel_acc (str, optional): Label for y-axis of accuracy plot. Defaults to 'r2_value'.
            ylabel_loss (str, optional): Label for y-axis of loss plot. Defaults to 'Loss_value'.
            title (str, optional): Title of the loss/accuracy plot. Defaults to 'Loss and Accuracy Plot'.
            results_savepath (str, optional): Path to save results, logs, and models. Defaults to '.'.
            save_image_filename (str, optional): Filename to save performance plot. Defaults to 'Untitled'.
            loss_save_filename (str, optional): CSV filename to save loss/accuracy logs. Defaults to 'loss_acc_file'.
            best_model_savename (str, optional): Filename for the best model checkpoint. Defaults to 'best_model'.
            description_savename (str, optional): Filename to save model configuration description. Defaults to 'description.txt'.
            DATETIME (str, optional): Timestamp string used for saving uniquely. Defaults to current time.
            tqdm_write (bool, optional): Whether to log epoch details via tqdm. Defaults to False.
            standardscaler (Any, optional): Scaler to be stored with model if applicable. Defaults to None.
            save_model_wrt (str, optional): Criteria to save best model: 'loss' or 'accuracy'. Defaults to 'loss'.
            more_description (str, optional): Additional model description for documentation. Defaults to ''.
            device (torch.device, optional): Device to use for computation (CPU/GPU). Defaults to CUDA if available.

        Returns:
            tuple:
                - List[float]: A list containing training loss values for each epoch.
                - nn.Module: The best-performing model according to validation metric.

        Raises:
            NameError: If `save_model_wrt` is not 'loss' or 'accuracy'.
        """
        
        if not results_savepath:
            results_savepath = f"./cache/{date}/{time}"
        models_savepath = create_folder(f"{results_savepath}/models")
        best_model_savepath = f"{models_savepath}/{best_model_savename}"
        
        num_train = len(train_loader.dataset)
        num_test = len(test_loader.dataset)
        num_val = len(val_loader.dataset)
        
        out =  (
                    f"{'='*60}\n"
                    f"## NN_Trainer Configuration ##\n"
                    f"{'-'*60}\n"
                    f"{'    Date & Time':35}: {DATETIME}\n"
                    f"{'    torch_version':35}: {torch.__version__}\n"
                    f"{'    platform':35}: {platform.platform()}\n"
                    f"{'    OS':35}: {platform.system()} {platform.release()}\n"
                    f"{'    Python Version':35}: {platform.python_implementation()} {platform.python_version()}\n"
                    f"{'    Python Build':35}: {platform.python_build()}\n"
                    f"{'    Python Compiler':35}: {platform.python_compiler()}\n"
                    f"{'    Python Architecture':35}: {platform.architecture()}\n"
                    f"{'    Hostname':35}: {platform.node()}\n"
                    f"{'-'*60}\n"
                    f"{'    Model':35}: {model.__class__.__name__}\n"
                    f"{'    Device':35}: {device}\n"
                    f"{'    Epochs':35}: {num_epochs}\n"
                    f"{'    Batch Size':35}: {batch_size}\n"
                    f"{'    Optimizer':35}: {optimizer.__class__.__name__}\n"
                    f"{'    Loss Function':35}: {criterion.__class__.__name__}\n"
                    f"{'    Scheduler':35}: {scheduler_name if scheduler_name else 'None'}\n"
                    f"{'    Scheduler Arguments':35}: {schedulers_kwargs if scheduler_name else 'N/A'}\n"
                    f"{'    Early Stopping':35}: {'Enabled' if early_stop else 'Disabled'}\n"
                    f"{'    Early Stop Arguments':35}: {early_stop_kwargs if early_stop else 'N/A'}\n"
                    f"{'    Results Save Path':35}: {results_savepath}\n"
                    f"{'    Save Model Based On':35}: {save_model_wrt}\n"
                    f"{'    Best Model File Path':35}: {best_model_savepath}\n"
                    f"{'    Output Plot Filename':35}: {save_image_filename}.png\n"
                    f"\n"
                    f"{'-'*60}\n"
                    f"{'## Dataset Sizes ##'}\n"
                    f"{'-'*60}\n"
                    f"{'    Train Set':35}: {num_train} samples\n"
                    f"{'    Validation Set':35}: {num_val} samples\n"
                    f"{'    Test Set':35}: {num_test} samples\n\n"
                    f"{'    Train-Test Subset':35}: {int(num_train * subset_ratio)} samples\n"
                    f"{'    Val-Test Subset':35}: {int(num_val * subset_ratio)} samples\n"
                    f"{'    Test-Test Subset':35}: {int(num_test * subset_ratio)} samples\n"
                    f"{'-'*60}\n"
                    f"{'## Representation ##'}\n"
                    f"{'-'*60}\n"
                    f"{repr(model)}\n"
                    f"{'-'*60}\n"

                )
        out = out + more_description +'\n' + '='*80 
        description_savepath = f'{results_savepath}/{description_savename}'
        with open(description_savepath, 'w') as fp:
            fp.write(out)
        print(f"[Info] Model configuration saved to: {description_savepath}")
        
        model.to(device)
        epoch_train_losses = [None]*num_epochs
        epoch_val_losses = [None]*num_epochs
        epoch_test_losses = [None]*num_epochs
        
        train_acc_list = [None]*num_epochs
        test_acc_list = [None]*num_epochs
        val_acc_list = [None]*num_epochs
        
        lr_list = [None]*num_epochs
        time_list = [None]*num_epochs
        
        model_acc = -float('inf')
        model_loss = float('inf')
        
        # if subset_ratio < 1:
        train_loader_subset, _ = subset_loader(train_loader, batch_size=batch_size, subset_ratio=subset_ratio, collate_fn=collate_fn)
        test_loader_subset, _ = subset_loader(test_loader, batch_size=batch_size, subset_ratio=subset_ratio, collate_fn=collate_fn)
        val_loader_subset, _ = subset_loader(val_loader, batch_size=batch_size, subset_ratio=subset_ratio, collate_fn=collate_fn)

        print('+'*70)
        print(f'Number of Test-Train Samples: {len(train_loader_subset.dataset)}')
        print(f'Number of Test-Val Samples: {len(val_loader_subset.dataset)}')
        print(f'Number of Test-Test Samples: {len(test_loader_subset.dataset)}')
        print('+'*70)
        
        # LR Schduler
        scheduler = Trainer_functions.get_scheduler(scheduler_name=scheduler_name,
                                optimizer=optimizer,
                                **schedulers_kwargs
                                ) if scheduler_name and schedulers_kwargs else None # is instance of torch.optim.lr_scheduler.LRScheduler
        
        # Initializing Early Stopping
        early_stop_kwargs={**early_stop_kwargs, 'type_':save_model_wrt}
        early_stopping = EarlyStopping(**early_stop_kwargs) if early_stop and early_stop_kwargs else None
        progress_bar = tqdm(range(num_epochs), desc="Training", unit="Eps")

        for epoch in progress_bar:
            # tqdm.write(f"Epoch - {epoch + 1}/{num_epochs}:")
            model.train()
            epoch_train_loss = Trainer_functions.train_1epoch(model,
                                            train_loader,
                                            criterion,
                                            optimizer,
                                            device=device,
                                            targets_col=targets_col,
                                            description = 'Training',
                                            )
            train_acc, epoch_train_loss = Trainer_functions.train_test_model(model=model, test_loader=train_loader_subset,
                                                        criterion=criterion, device=device, targets_col=targets_col, description='Test-on-Train')
            test_acc, epoch_test_loss = Trainer_functions.train_test_model(model=model, test_loader=test_loader_subset,
                                                        criterion=criterion, device=device, targets_col=targets_col, description='Test-on-Test')
            val_acc, epoch_val_loss = Trainer_functions.train_test_model(model=model, test_loader=val_loader_subset,
                                                    criterion=criterion, device=device, targets_col=targets_col, description='Test-on-Val')

            # train_acc, test_acc, val_acc = train_acc, test_acc, val_acc
            train_acc_list[epoch] = round(train_acc, 2)
            test_acc_list[epoch] = round(test_acc, 2)
            val_acc_list[epoch] = round(val_acc, 2)
            lr_list[epoch] = optimizer.param_groups[0]['lr']
            time_list[epoch] = datetime_now(path=False)[0]
            
            epoch_train_losses[epoch] = round(epoch_train_loss, 6)
            epoch_test_losses[epoch] = round(epoch_test_loss, 6)
            epoch_val_losses[epoch] = round(epoch_val_loss, 6)
            
            if save_model_wrt == 'loss':
                condition_flag = epoch_val_loss < model_loss
            elif save_model_wrt == 'accuracy':
                condition_flag = val_acc > model_acc
            else:
                raise NameError('Condition name not known')

            if condition_flag:
                if save_model:
                    # Save the best model
                    print(f"Saving best model at epoch {epoch + 1} with validation loss: {epoch_val_loss:.4f} and accuracy: {val_acc:.4f}")
                    Trainer_functions.save_network(model=model, 
                                optimizer=optimizer,
                                epoch=epoch,
                                device = device,
                                standardscaler=standardscaler,
                                
                                train_acc=train_acc_list[:epoch+1],
                                test_acc=test_acc_list[:epoch+1],
                                val_acc=val_acc_list[:epoch+1],
                                
                                train_loss=epoch_train_losses[:epoch+1],
                                test_loss=epoch_test_losses[:epoch+1],
                                val_loss=epoch_val_losses[:epoch+1],
                                
                                network_save_filename=best_model_savepath
                                )
                best_train_loss, best_test_loss, best_val_loss = epoch_train_loss, epoch_test_loss, epoch_val_loss
                best_train_acc, best_test_acc, best_val_acc = train_acc, test_acc, val_acc
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                model_loss = epoch_val_loss
                model_acc = val_acc
                # print('-'*80, '\n')

            loss_dict = OrderedDict({'epoch': list(range(1, epoch+2)), 
                                    'Time': time_list[:epoch+1],
                                    'lr': lr_list[:epoch+1],
                                    'train_loss':epoch_train_losses[:epoch+1],
                                    'test_loss':epoch_test_losses[:epoch+1],
                                    'val_loss':epoch_val_losses[:epoch+1],
                                    
                                    'train_acc': train_acc_list[:epoch+1],
                                    'test_acc': test_acc_list[:epoch+1],
                                    'val_acc': val_acc_list[:epoch+1],
                                    })
            Trainer_functions.save_metrics_csv(data_dict=loss_dict, save_csv_filename=f"{results_savepath}/{loss_save_filename}.csv")
            
            if (epoch % save_model_per_epoch == 0):
                
                model_save_name = f"{models_savepath}/model_at_epoch_{epoch}"
                if save_model:
                    print(f"Saving model at epoch {epoch + 1} to: {model_save_name}")
                    Trainer_functions.save_network(model=model, 
                                                optimizer=optimizer,
                                                epoch=epoch,
                                                device = device,
                                                standardscaler=standardscaler,
                                                
                                                train_acc=train_acc_list[:epoch+1],
                                                test_acc=test_acc_list[:epoch+1],
                                                val_acc=val_acc_list[:epoch+1],
                                                
                                                train_loss=epoch_train_losses[:epoch+1],
                                                test_loss=epoch_test_losses[:epoch+1],
                                                val_loss=epoch_val_losses[:epoch+1],
                                                
                                                network_save_filename=model_save_name)
            
            # Update tqdm progress bar with accs
            if tqdm_write:
                tqdm.write(f"At {time_list[epoch]} |Epoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Test Loss: {epoch_test_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f} | Test Acc: {test_acc:.2f}")
            
            # Set postfix in tqdm bar for live update
            progress_bar.set_postfix({"Train Loss": f"{epoch_train_loss:.4f}", "Val Loss": f"{epoch_val_loss:.4f}", "Test Loss": f"{epoch_test_loss:.4f}",
                                    "Train Acc": f"{train_acc:.4f}", "Val Acc": f"{val_acc:.4f}", "Test Acc": f"{test_acc:.4f}"})
                
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_val_loss)
                else:
                    scheduler.step()
            
            if early_stop:  
                early_stop_val = val_acc if save_model_wrt=='accuracy' else  epoch_val_loss
                early_stopping(early_stop_val)
                if early_stopping.early_stop:
                    print(f"Early stop triggered at epoch: {epoch + 1}")
                    print('='*80)
                    break
        print(f'Best Model Details:\n  Epoch: {best_epoch+1}\n    Train Loss: {best_train_loss}, Test Loss: {best_test_loss}, Val Loss: {best_val_loss}\n   \
            Train Acc: {best_train_acc}, Test Acc: {best_test_acc}, Val Acc: {best_val_acc}')
                
        Trainer_functions.plot_loss_save_images(train_acc=train_acc_list, 
                                                test_acc=test_acc_list,
                                                val_acc=val_acc_list,
                                                train_loss=epoch_train_losses, 
                                                test_loss=epoch_test_losses,
                                                val_loss=epoch_val_losses,
                                                xlabel = xlabel,
                                                ylabel_acc = ylabel_acc,
                                                ylabel_loss = ylabel_loss,
                                                title = title,
                                                save_image_filename = f"{results_savepath}/{save_image_filename}"
                                               )
                

        print('Training completed.')
        
        return epoch_train_losses, best_model



    @staticmethod
    def plot_loss_save_images(train_acc: List[float],
                            test_acc: List[float],
                            val_acc: List[float],
                            train_loss: List[float],
                            test_loss: List[float],
                            val_loss: List[float],
                            xlabel: str = 'Epoch',
                            ylabel_loss: str = 'Loss Value',
                            ylabel_acc: str = 'r2_value',
                            title: str = 'Loss and Accuracy Plot',
                            save_image_filename: str = '',
                            marker: str = '', 
                            markersize: int = 5, 
                            linewidth: int = 2, 
                            linestyle: str = '-', 
                            title_fontsize: int = 25, 
                            xyticks_fontsize: int = 10 
                            ) -> None:
        """
        Plots training, validation, and test accuracy and loss curves over epochs.

        Args:
            train_acc (List[float]): Accuracy values for training set per epoch.
            test_acc (List[float]): Accuracy values for test set per epoch.
            val_acc (List[float]): Accuracy values for validation set per epoch.
            train_loss (List[float]): Loss values for training set per epoch.
            test_loss (List[float]): Loss values for test set per epoch.
            val_loss (List[float]): Loss values for validation set per epoch.
            xlabel (str): Label for x-axis. Defaults to 'Epoch'.
            ylabel_loss (str): Label for y-axis on loss plot. Defaults to 'Loss Value'.
            ylabel_acc (str): Label for y-axis on accuracy plot. Defaults to 'r2_value'.
            title (str): Title for the entire figure.
            save_image_filename (str): Path to save plot (without extension).
            marker (str): Marker symbol for line plot.
            markersize (int): Size of the marker.
            linewidth (int): Width of the lines in the plot.
            linestyle (str): Style of the lines.
            title_fontsize (int): Font size for titles.
            xyticks_fontsize (int): Font size for tick labels.

        Returns:
            None. Saves plot as PNG file and closes the figure.
        """
        
        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))  # 1 row, 2 columns
        fig.suptitle(title, fontsize=title_fontsize)  # Overall title for the subplots
        
        # Plot for accuracy
        axs[0].plot(
            range(len(train_acc)), train_acc, 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='train' 
        )
        axs[0].plot(
            range(len(test_acc)), test_acc, 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='test' 
        )
        axs[0].plot(
            range(len(val_acc)), val_acc, 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='val' 
        )
        axs[0].set_xlabel(xlabel, fontsize=title_fontsize-5)
        axs[0].set_ylabel(ylabel_acc, fontsize=title_fontsize-5)
        axs[0].grid(linestyle='--')
        axs[0].legend(loc='upper right', fontsize=xyticks_fontsize)
        axs[0].set_title('Epoch vs Accuracy', fontsize=title_fontsize-5)
        axs[0].tick_params(axis='both', labelsize=xyticks_fontsize)  # Set x and y ticks
        
        # Plot for loss
        axs[1].plot(
            range(len(train_loss)), train_loss, 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='train',
            color='green' 
        )
        axs[1].plot(
            range(len(test_loss)), test_loss, 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='test',
            color = 'red' 
        )
        axs[1].plot(
            range(len(val_loss)), val_loss, 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='val',
            color = 'blue' 
        )
        axs[1].set_xlabel(xlabel, fontsize=title_fontsize-5)
        axs[1].set_ylabel(ylabel_loss, fontsize=title_fontsize-5)
        axs[1].grid(linestyle='--')
        axs[1].legend(loc='upper right', fontsize=xyticks_fontsize)
        axs[1].set_title('Epoch vs Loss', fontsize=title_fontsize-5)
        axs[1].tick_params(axis='both', labelsize=xyticks_fontsize)  # Set x and y ticks
        
        # Adjust layout
        plt.tight_layout()  # Leave space for the suptitle
        
        # Save the plot if a filename is provided
        if save_image_filename:
            plt.savefig(f"{save_image_filename}.png", bbox_inches='tight')  # Save as PNG
        
        # plt.show()  # Display the plot
        plt.close()  # Close the figure
        
        
        
    # trained network save function
    @staticmethod
    def save_network(model: nn.Module = None,
                    optimizer: torch.optim.Optimizer = None,
                    epoch : int = None,
                    learning_rate : float = None,
                    train_acc: List[float] = [],
                    test_acc: List[float] = [],
                    val_acc: List[float] = [],
                    train_loss: List[float] = [],
                    test_loss: List[float] = [],
                    val_loss: List[float] = [],
                    standardscaler: Any = None,
                    device: Any = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    network_save_filename: str = 'network_best'
                    ) -> None:

        """
        Saves the model checkpoint, optimizer state, training metrics, and source file content.

        Args:
            model (nn.Module): Trained model to be saved.
            optimizer (torch.optim.Optimizer): Optimizer used during training.
            epoch (int): Epoch at which the model is saved.
            learning_rate (float): Learning rate used for training.
            train_acc (List[float]): Training accuracy values.
            test_acc (List[float]): Test accuracy values.
            val_acc (List[float]): Validation accuracy values.
            train_loss (List[float]): Training loss values.
            test_loss (List[float]): Test loss values.
            val_loss (List[float]): Validation loss values.
            standardscaler (Any): Optional scaler object used on input data.
            network_save_filename (str): Name for the saved checkpoint file (without extension).

        Returns:
            None. Saves a `.pt` checkpoint file to disk.
        """
        network_save_filename = ''.join([network_save_filename, '.pt'])
        print(f'Saving network in: "{network_save_filename}')
        
        source_file = os.path.abspath(__file__)
        with open(source_file, 'rb') as fp:
            file_ = fp.read()
            utils_pyfile = pickle.dumps(file_)
            
        torch.save({'model_class': model.__class__.__name__,
                    'torch_version': torch.__version__,
                    'python_version': platform.python_version(),
                    'device': str(device),
                    'hostname': platform.node(),
                    'epoch': epoch+1,
                    'datetime':datetime_now(),
                    'utils_pyfile': utils_pyfile,
                    'network_pyfile': getattr(model, 'model_script', None),
                    'learning_rate' : learning_rate,
                    'standardscaler':standardscaler,
                    
                    'train_loss': train_loss,
                    'val_loss': val_loss, 
                    'test_loss': test_loss, 
                    
                    'train_acc': train_acc,
                    'val_acc': val_acc, 
                    'test_acc': test_acc, 
                    
                    'state_dict': copy.deepcopy(model.state_dict()),
                    'optimizer': copy.deepcopy(optimizer.state_dict()),
                    'representation': str(repr(model).replace('\n', ''))
                    }, network_save_filename)
    #------------------------------------------------------

    #save (csv) different losses
    @staticmethod
    def save_metrics_csv(data_dict: dict,
                        save_csv_filename: str = 'temp_metrics.csv'):
        """
        Saves metrics from training/validation/testing to a CSV file.

        Args:
            data_dict (dict): Dictionary where keys are metric names (e.g., 'train_loss') 
                            and values are lists or tuples of metric values.
            save_csv_filename (str): Name of the CSV file to save the metrics.

        Returns:
            pd.DataFrame: DataFrame containing the saved metrics.
        """
        if not data_dict:
            raise ValueError("No data provided to save.")

        # Start with 'epoch' column

        for key, value in data_dict.items():
            # Truncate each list to match current epoch
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"Value for '{key}' must be list or tuple, not {type(value).__name__}")

        df = pd.DataFrame(data_dict)
        df.to_csv(save_csv_filename, index=False)
        
        return df

        
    # pre-trained network load function for test
    @staticmethod
    def load_test_network(network: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        temp_network_path: str = '',
                        strict: bool = True,
                        device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                        ):
        """
        Loads a previously saved model checkpoint for evaluation or resumption.

        Args:
            network (nn.Module): Model architecture to load weights into.
            optimizer (torch.optim.Optimizer): Optimizer to load state into.
            temp_network_path (str): Path to the `.pt` checkpoint file.
            device (str): Device to map the checkpoint onto (e.g., 'cuda' or 'cpu').

        Returns:
            None. Loads model/optimizer states and prints training summary.
            Prints error if checkpoint is missing or corrupted.
        """
        
        if os.path.isfile(temp_network_path):
            print('-'*80)
            print('Loading pre-trained network checkpoint from: "{}"'.format(temp_network_path))
            checkpoint = torch.load(temp_network_path, map_location=device, weights_only=False)
            #--------------------------------------------------
            epoch_best_network = checkpoint.get('epoch', 'UNKNOWN')
            train_loss = checkpoint.get('train_loss', 'NA')
            val_loss = checkpoint.get('val_loss', 'NA')
            test_loss = checkpoint.get('test_loss', 'NA')
            train_acc = checkpoint.get('train_acc', 'NA')
            val_acc = checkpoint.get('val_acc', 'NA')
            test_acc = checkpoint.get('test_acc', 'NA')
            network_pyfile = checkpoint.get('network_pyfile', 'NA')
            model_class = checkpoint.get('model_class', 'NA')
            standardscaler = checkpoint.get('standardscaler', False)
            model_repr = checkpoint.get('representation', False)
            #--------------------------------------------------
            #+++++++++++++++++++++++++++++++++++++++++++++++++++
                
            network.load_state_dict(checkpoint['state_dict'], strict=strict)
            
            print('Loaded pre-trained network checkpoint from "{}"\nepoch: {} train loss: {} val loss: {} test loss: {} train acc: {} val acc: {} test acc: {} ' \
                .format(temp_network_path, epoch_best_network, train_loss[-1], val_loss[-1], test_loss[-1], train_acc[-1], val_acc[-1], test_acc[-1])
                    )
            print('='*80)
            
            return model_repr, standardscaler, model_class, network_pyfile
        else:
            print('-'*80)
            print(f'No pre-trained network checkpoint found at "{temp_network_path}"')
            print('='*80)
            # warnings.warn(f'No pre-trained network checkpoint found at "{temp_network_path}"')
        
        


class weights_initializer:
    """
    Apply custom weight initialization to a PyTorch network.
    """
    def __init__(self, network, initializer='xvr_unifrm'):
        """
        Initialize the weights of a neural network based on the chosen scheme.

        Args:
            network (torch.nn.Module): The neural network whose weights will be initialized.
            initializer (str): The weight initialization strategy. 
                               Options: 'default', 'zeros', 'ones', 'unifrm', 'nrmal', 
                                        'xvr_unifrm', 'xvr_nrmal'.

        Raises:
            ValueError: If an unsupported initializer is provided.
        """
        if initializer == 'default':
            pass  # Do nothing
        elif initializer == 'zeros':
            network.apply(self.weights_init_zeros)
        elif initializer == 'ones':
            network.apply(self.weights_init_ones)
        elif initializer == 'unifrm':
            network.apply(self.weights_init_uniform)
        elif initializer == 'nrmal':
            network.apply(self.weights_init_normal)
        elif initializer == 'xvr_unifrm':
            network.apply(self.weights_init_xavier_uniform)
        elif initializer == 'xvr_nrmal':
            network.apply(self.weights_init_xavier_normal)
        else:
            raise ValueError(f'Unknown initializer: "{initializer}"')

    # -------------------------------------- #

    def weights_init_zeros(self, m):
        """
        Apply zero initialization to weights and biases of the specified module.

        Args:
            m (torch.nn.Module): A layer or module whose weights are to be initialized.

        Output:
            Modifies weights and biases in-place to zeros.
        """
        if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() >= 2:
            nn.init.zeros_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

    def weights_init_ones(self, m):
        """
        Apply one initialization to weights and zero to biases of the specified module.

        Args:
            m (torch.nn.Module): A layer or module whose weights are to be initialized.

        Output:
            Modifies weights to ones and biases to zeros in-place.
        """
        if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() >= 2:
            nn.init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

    def weights_init_uniform(self, m):
        """
        Apply uniform distribution-based initialization to weights and biases.

        Args:
            m (torch.nn.Module): A layer or module whose weights are to be initialized.

        Output:
            Modifies weights and biases in-place using a uniform distribution.
        """
        if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() >= 2:
            nn.init.uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.uniform_(m.bias)

    def weights_init_normal(self, m):
        """
        Apply normal distribution-based initialization to weights and biases.

        Args:
            m (torch.nn.Module): A layer or module whose weights are to be initialized.

        Output:
            Modifies weights and biases in-place using a normal distribution.
        """
        if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() >= 2:
            nn.init.normal_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.normal_(m.bias)

    def weights_init_xavier_uniform(self, m):
        """
        Apply Xavier uniform initialization to weights and zero to biases.

        For GRU layers, uses orthogonal for weights and normal for biases.

        Args:
            m (torch.nn.Module): A layer or module whose weights are to be initialized.

        Output:
            Modifies weights and biases in-place using Xavier or orthogonal schemes.
        """
        if isinstance(m, nn.GRU):
            for param in m.parameters():
                if param.dim() >= 2:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.normal_(param.data)
        elif isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)) and m.weight.dim() >= 2:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def weights_init_xavier_normal(self, m):
        """
        Apply Xavier normal initialization to weights and zero to biases.

        For GRU layers, uses orthogonal for weights and normal for biases.

        Args:
            m (torch.nn.Module): A layer or module whose weights are to be initialized.

        Output:
            Modifies weights and biases in-place using Xavier or orthogonal schemes.
        """
        if isinstance(m, nn.GRU):
            for param in m.parameters():
                if param.dim() >= 2:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.normal_(param.data)
        elif isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)) and m.weight.dim() >= 2:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

class NN_Trainer():
    
    def __init__(self,
                network: nn.Module,
                train_loader: DataLoader[Dict[str, torch.Tensor]],
                val_loader: DataLoader[Dict[str, torch.Tensor]],
                test_loader: DataLoader[Dict[str, torch.Tensor]],
                
                optimizer_name: str='adam',
                optimizer_kwargs: Dict[str, Any]={'betas':(0.9, 0.999), 'eps':1e-08, 'weight_decay':0, 'maximize':False},
                learning_rate: float = 1e-4,
                loss_name: str = 'mse',
                loss_kwargs: Dict[str, Any] = {'reduction':'sum'},
                targets_col: str = 'labels',
                scheduler_name: str = '',
                schedulers_kwargs: dict = {},
                early_stop: bool = False,
                early_stop_kwargs: dict = {},
                
                num_epochs: int = 100,
                batch_size: int = 128,
                subset_ratio: int = 1,
                save_model_per_epoch: int = 50,
                save_model_wrt: str = 'loss', # 'loss' or 'accuracy'
                
                results_savepath: str = '.',
                description_savename: str = 'description.txt',
                model_save_name: str = 'Untitled_model',
                save_image_filename: str = 'loss_acc_plot',
                loss_save_filename: str = 'loss_acc_file',
                more_description: str = '',
                
                tqdm_write: bool = False,
                standardscaler: Any = None,
                DATETIME: str = datetime_now(path=False)[0],
                date: str = datetime_now(path=False)[1],
                time: str = datetime_now(path=False)[2],
                
                xlabel: str = 'Epoch',
                ylabel_acc: str = 'r2_value',
                ylabel_loss: str = 'Loss_value',
                title: str = 'Loss and Accuracy Plot',
                device: Any = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                ):
        """
        Initializes the NN_Trainer class with configuration for training, testing, and evaluating a PyTorch neural network.

        Args:
            network (nn.Module): The neural network model to be trained.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            test_loader (DataLoader): DataLoader for the testing dataset.
            optimizer_name (str): Name of the optimizer to use (e.g., 'adam', 'sgd').
            optimizer_kwargs (Dict[str, Any]): Arguments for the optimizer.
            learning_rate (float): Learning rate for training.
            loss_name (str): Name of the loss function to use.
            loss_kwargs (Dict[str, Any]): Parameters for the loss function.
            targets_col (str): The key name used for labels in each batch.
            scheduler_name (str): Name of the learning rate scheduler (if any).
            schedulers_kwargs (dict): Arguments for the scheduler.
            early_stop (bool): Whether to enable early stopping.
            early_stop_kwargs (dict): Parameters for early stopping.
            num_epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
            subset_ratio (int): Ratio of subset of data to consider.
            xlabel (str): X-axis label for plots.
            ylabel_acc (str): Y-axis label for accuracy plot.
            ylabel_loss (str): Y-axis label for loss plot.
            title (str): Title for plots.
            description_savename (str): Filename to save model configuration description.
            save_image_filename (str): Filename to save loss-accuracy plots.
            loss_save_filename (str): Filename to save loss values.
            save_model_wrt (str): Metric to determine model saving ('loss' or 'accuracy').
            device (torch.device): Device to use ('cuda' or 'cpu').
            save_model_per_epoch (int): Save model every N epochs.
            result_savepath (str): Directory path to save results.
            model_save_name (str): Additional name tag for saved model.
            DATETIME (str): Timestamp string for experiment tracking.
            tqdm_write (bool): Whether to use tqdm for progress logging.
            standardscaler (Optional): StandardScaler object for post-processing.
            more_description (str): Additional notes to include in the config description.
        """

        self.network = network
        
        self.criterion = Trainer_functions.loss_function_selector(name=loss_name, params=loss_kwargs)
        self.optimizer = Trainer_functions.get_optimizer(model=self.network, solver_type=optimizer_name, learning_rate=learning_rate, **optimizer_kwargs)
        self.loss_kwargs = loss_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_name = scheduler_name
        self.schedulers_kwargs = schedulers_kwargs
        self.early_stop = early_stop
        self.early_stop_kwargs = early_stop_kwargs
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_model_per_epoch = save_model_per_epoch
        self.model_save_name = model_save_name
        self.results_savepath = results_savepath
        self.best_model_save_name = ''.join(['best_network',
                                             '_lr_', str(self.learning_rate).replace('.', '_').replace('-', '_'),
                                             '_batsz_', str(self.batch_size),
                                             '_optim_', str(self.optimizer_name),
                                             self.model_save_name,
                                             '.pth']
                                            )
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.targets_col = targets_col
        self.subset_ratio = subset_ratio
        
        self.xlabel = xlabel
        self.ylabel_acc = ylabel_acc
        self.ylabel_loss = ylabel_loss
        self.title = title 
        self.description_savename = description_savename
        self.save_image_filename = save_image_filename
        self.loss_save_filename = loss_save_filename
        self.save_model_wrt = save_model_wrt
        self.device = device
        self.num_train = len(self.train_loader.dataset)
        self.num_test = len(self.test_loader.dataset)
        self.num_val = len(self.val_loader.dataset)
        self.tqdm_write = tqdm_write,
        self.standardscaler = standardscaler,
        self.more_description = more_description,
        self.create_description()
        
        self.results_savepath = f"{self.results_savepath}/cache/{date}/{time}"
        self.models_savepath = create_folder(f"{self.results_savepath}/models")
        self.datetime = DATETIME
        
    def __str__(self):
        """
        Returns:
            str: Formatted string displaying the training configuration and dataset statistics.
        """

        out =  (
                f"NN_Trainer Configuration\n"
                f"{'-'*50}\n"
                f"{'    Date':35} : {self.datetime}\n"
                f"{'    torch_version':35}: {torch.__version__}\n"
                f"{'    platform':35}: {platform.platform()}\n"
                f"{'    OS':35}: {platform.system()} {platform.release()}\n"
                f"{'    Python Version':35}: {platform.python_implementation()} {platform.python_version()}\n"
                f"{'    Python Build':35}: {platform.python_build()}\n"
                f"{'    Python Compiler':35}: {platform.python_compiler()}\n"
                f"{'    Python Architecture':35}: {platform.architecture()}\n"
                f"{'    Hostname':35}: {platform.node()}\n"
                f"{'-'*60}\n"
                f"{'    Model':35} : {self.network.__class__.__name__}\n"
                f"{'    Device':35} : {self.device}\n"
                f"{'    Epochs':35} : {self.num_epochs}\n"
                f"{'    Batch Size':35} : {self.batch_size}\n"
                f"{'    Optimizer':35} : {self.optimizer.__class__.__name__}\n"
                f"{'    Optimizer Arguments':35} : {self.optimizer_kwargs}\n"
                f"{'    Learning Rate':35} : {self.learning_rate}\n"
                f"{'    Loss Function':35} : {self.criterion.__class__.__name__}\n"
                f"{'    Loss Arguments':35} : {self.loss_kwargs}\n"
                f"{'    Scheduler':35} : {self.scheduler_name if self.scheduler_name else 'None'}\n"
                f"{'    Scheduler Arguments':35} : {self.schedulers_kwargs if self.scheduler_name else 'N/A'}\n"
                f"{'    Early Stopping':35} : {'Enabled' if self.early_stop else 'Disabled'}\n"
                f"{'    Early Stop Arguments':35} : {self.early_stop_kwargs if self.early_stop else 'N/A'}\n"
                f"{'    Results Save Path':35} : {self.result_savepath}\n"
                f"{'    Models Save Path':35} : {self.models_savepath}\n"
                f"{'    Save Model Based On':35} : {self.save_model_wrt}\n"
                f"{'    Best Model File Name':35} : {self.best_model_savename}\n"
                f"{'    Output Plot Filename':35} : {self.save_image_filename}.png\n"
                f"{'    Loss Save Filename':35} : {self.loss_save_filename}\n"
                f"{'    Model Save Frequency (epochs)':35} : {self.save_model_per_epoch}\n"
                f"{'    Subset Ratio':35} : {self.subset_ratio}\n"
                f"{'    Standard Scaler':35} : {self.standardscaler if self.standardscaler else 'None'}\n"
                f"{'    Description File Name':35} : {self.description_savename}\n"
                f"{'    Target Column':35} : {self.targets_col}\n"
                f"\n"
                f"{'Dataset Sizes':35}\n"
                f"{'-'*50}\n"
                f"{'    Train Set':35} : {self.num_train} samples\n"
                f"{'    Validation Set':35} : {self.num_val} samples\n"
                f"{'    Test Set':35} : {self.num_test} samples\n\n"
                f"{'    Train-Test Subset':35} : {int(self.num_train * self.subset_ratio)} samples\n"
                f"{'    Val-Test Subset':35} : {int(self.num_val * self.subset_ratio)} samples\n"
                f"{'    Test-Test Subset':35} : {int(self.num_test * self.subset_ratio)} samples\n"
                f"{'-'*60}\n"
                f"{'## Class Representation ##'}\n"
                f"{'-'*60}\n"
                f"{repr(self)}\n"
                f"{'-'*60}\n"
                f"{'## Network Representation ##'}\n"
                f"{'-'*60}\n"
                f"{repr(self.network)}\n"
                f"{'-'*60}\n"
            )
        
        return out + '\n' + '-'*80 + '\n' + '='*80 + '\n' + self.more_description
    
    def __repr__(self):
        """
        Developer-focused representation: full instantiation parameters for debugging. 
        Should ideally be valid Python code that recreates the object "(eval(repr(obj)))".
        """
        return auto_repr(self)
    
    def create_description(self):
        """
        Saves a text file describing the current model configuration, training parameters, and dataset sizes.
        """

        description_savepath = f'{self.result_savepath}/{self.description_savename}'
        with open(description_savepath, 'w') as fp:
            fp.write(str(self))
        print(f"[Info] Model configuration saved to: {description_savepath}")
        
    def nn_input2output(self, dataloader: DataLoader[Dict[str, torch.Tensor]]) -> Tuple[np.ndarray[float], np.ndarray[float], List[float]]:
        
        """
        Runs the model in evaluation mode over a dataloader and collects predicted outputs, true targets, and average loss.

        Args:
            dataloader (DataLoader): Dataloader to evaluate (train/val/test).

        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                - Predictions as NumPy array.
                - Ground truth labels as NumPy array.
                - Average loss over the dataset.
        """

        self.network.to(self.device)
        self.network.eval()  # Set model to evaluation mode
        y_pred = []
        y_true = []
        epoch_test_loss = 0

        with torch.no_grad():  # No need to compute gradients during evaluation
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                targets = batch.pop(self.targets_col)
                outputs = self.network(**batch)
                loss = self.criterion(input=outputs.reshape(targets.shape).type(torch.float32),
                                      target=targets.type(torch.float32)
                                      ).type(torch.float32)
                    
                # Collect all predictions and targets
                y_pred.append(outputs.reshape(-1).cpu())
                y_true.append(targets.cpu())
                
                # Accumulate loss
                epoch_test_loss += loss.item()
            epoch_test_loss = epoch_test_loss/len(dataloader)
        
        # Concatenate all predictions and targets across batches
        y_pred = torch.cat(y_pred).numpy()
        y_true = torch.cat(y_true).numpy()
    
        return y_pred, y_true, epoch_test_loss
    
    
        # # Define the test function
    def test_model(self, dataloader: DataLoader[Dict[str, torch.Tensor]]) -> Tuple[float, float]:
        
        """
        Evaluates the model on the given dataset using regression metrics.

        Args:
            dataloader (DataLoader): Dataloader to evaluate.

        Returns:
            Tuple[Dict[str, float], float]:
                - Dictionary of regression metrics (e.g., MAE, MSE, R²).
                - Average loss across the dataset.
        """


        y_pred, y_true, epoch_test_loss = self.nn_input2output(dataloader=dataloader)
        results = regression_test_metrics(y_true=y_true, y_pred=y_pred)

        return results, epoch_test_loss


    # # Define the test function
    def train_test_model(self, dataloader: DataLoader[Dict[str, torch.Tensor]]) -> Tuple[float, float]:
        
        """
        Evaluates the model on the given dataset and returns R² score and average loss.

        Args:
            dataloader (DataLoader): Dataloader to evaluate.

        Returns:
            Tuple[float, float]:
                - R² score.
                - Average loss.
        """


        y_pred, y_true, epoch_test_loss = self.nn_input2output(dataloader=dataloader)
        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        
        return r2, epoch_test_loss
    

    def train_1epoch(self,
                    param_reg: str = '',
                    lambda_reg: float = 0.1,
                    alpha: float = 0.5,
                    ) -> float:
            
        """
        Trains the model for one epoch on the training dataset.

        Args:
            param_reg (str): Regularization method to apply ('l1', 'l2', 'elastic_net', or '').
            lambda_reg (float): Regularization strength.
            alpha (float): Mixing ratio for elastic net (0: only L2, 1: only L1).

        Returns:
            float: Average training loss for the epoch.
        """

        self.network.train()# switch to train model
        self.network.to(self.device)
        epoch_train_loss = 0
        # with torch.autograd.set_detect_anomaly(True):
        for batch in tqdm(self.train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            targets = batch.pop(self.targets_col)
            outputs = self.network(**batch)
                
            loss = self.criterion(input=outputs.reshape(targets.shape).type(torch.float32),
                                  target=targets.type(torch.float32)
                                  ).type(torch.float32)
            if param_reg in ['l1', 'elastic_net']:
                l1_temp = 0
                for param in self.network.parameters():
                    l1_temp = l1_temp + torch.sum(torch.abs(param))
                l1_reg = lambda_reg * l1_temp  # L1 term
                if param_reg == 'l1':
                    loss = loss + l1_reg
                
            if param_reg in ['l2', 'elastic_net']:
                l2_temp = 0
                for param in self.network.parameters():
                    l2_temp = l2_temp + torch.sum(param**2)
                l2_reg = lambda_reg * l2_temp    # L2 term
                if param_reg == 'l2':
                    loss = loss + l2_reg
                
            if param_reg == 'elastic_net':
                elastic_net_reg = alpha * l1_reg + (1 - alpha) * l2_reg
                loss = loss + elastic_net_reg
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Accumulate loss
            epoch_train_loss += loss.item()
        epoch_train_loss = epoch_train_loss/len(self.train_loader)
        
        return epoch_train_loss
    
    def plot_loss_save_images(self,
                              save_image_filename: str = '',
                              marker: str = '',
                              markersize: int = 5,
                              linewidth: int = 2,
                              linestyle: str = '-',
                              title_fontsize: int = 25,
                              xyticks_fontsize: int = 10 
                              ) -> None:
        """
        Plots and optionally saves training, validation, and test loss and accuracy curves over epochs.

        Args:
            save_image_filename (str): Filename to save the plot image (without extension).
            marker (str): Marker style for the plot lines.
            markersize (int): Size of the markers in the plot.
            linewidth (int): Width of the plot lines.
            linestyle (str): Line style for the plots.
            title_fontsize (int): Font size for the title.
            xyticks_fontsize (int): Font size for the tick labels.
        """

        
        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))  # 1 row, 2 columns
        fig.suptitle(self.title, fontsize=title_fontsize)  # Overall title for the subplots
        
        # Plot for accuracy
        axs[0].plot(
            range(len(self.train_acc_list[:self.epoch])), self.train_acc_list[:self.epoch], 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='train' 
        )
        axs[0].plot(
            range(len(self.test_acc_list[:self.epoch])), self.test_acc_list[:self.epoch], 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='test' 
        )
        axs[0].plot(
            range(len(self.val_acc_list[:self.epoch])), self.val_acc_list[:self.epoch], 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='val' 
        )
        axs[0].set_xlabel(self.xlabel, fontsize=title_fontsize-5)
        axs[0].set_ylabel(self.ylabel_acc, fontsize=title_fontsize-5)
        axs[0].grid(linestyle='--')
        axs[0].legend(loc='upper right', fontsize=xyticks_fontsize)
        axs[0].set_title('Epoch vs Accuracy', fontsize=title_fontsize-5)
        axs[0].tick_params(axis='both', labelsize=xyticks_fontsize)  # Set x and y ticks
        
        # Plot for loss
        axs[1].plot(
            range(len(self.epoch_train_losses[:self.epoch])), self.epoch_train_losses[:self.epoch], 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='train',
            color='green' 
        )
        axs[1].plot(
            range(len(self.epoch_test_losses[:self.epoch])), self.epoch_test_losses[:self.epoch], 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='test',
            color = 'red' 
        )
        axs[1].plot(
            range(len(self.epoch_val_losses[:self.epoch])), self.epoch_val_losses[:self.epoch], 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='val',
            color = 'blue' 
        )
        axs[1].set_xlabel(self.xlabel, fontsize=title_fontsize-5)
        axs[1].set_ylabel(self.ylabel_loss, fontsize=title_fontsize-5)
        axs[1].grid(linestyle='--')
        axs[1].legend(loc='upper right', fontsize=xyticks_fontsize)
        axs[1].set_title('Epoch vs Loss', fontsize=title_fontsize-5)
        axs[1].tick_params(axis='both', labelsize=xyticks_fontsize)  # Set x and y ticks
        
        # Adjust layout
        plt.tight_layout()  # Leave space for the suptitle
        
        # Save the plot if a filename is provided
        if save_image_filename:
            plt.savefig(f"{self.result_savepath}/{save_image_filename}.png", bbox_inches='tight')  # Save as PNG
            plt.close()  # Close the figure
        else:
            plt.show()  # Display the plot
            
        
    # trained network save function
    def save_network(self, network_save_filename: str = 'network_best') -> None:

        """
        Save the current state of the trained network and relevant training metadata.

        This method saves the model's state dict, optimizer state, training configurations,
        learning rates, losses, and accuracies up to the current epoch into a `.pt` file
        at the specified location. It also serializes the source `.py` file for reproducibility.

        Args:
            network_save_filename (str): The filename (without extension) to save the model under.
                The full path will be `{result_savepath}/{network_save_filename}.pt`.
        
        Returns:
            None
        """
        network_save_filename = f"{self.models_savepath}/{network_save_filename}.pt"
        print(f'Saving network in: "{network_save_filename}"')

        source_file = os.path.abspath(__file__)
        with open(source_file, 'rb') as fp:
            file_ = fp.read()
            self.utils_pyfile = pickle.dumps(file_)
            
        torch.save({'model_class_name': self.network.__class__.__name__,
                    'optimizer_name': self.optimizer.__class__.__name__,
                    'loss_func_name': self.criterion.__class__.__name__,
                    'torch_version': torch.__version__,
                    'python_version': platform.python_version(),
                    'device': str(self.device),
                    'hostname': platform.node(),
                    'epoch': self.epoch,
                    'best_epoch': self.best_epoch,
                    'utils_pyfile': self.utils_pyfile,
                    'network_pyfile': getattr(self.network, 'model_script', None),
                    'standardscaler' : getattr(self, 'standardscaler', None),
                    'learning_rate' : self.learning_rate,
                    'network_save_filename': self.best_model_save_name,
                    'plot_output': f'{self.save_image_filename}.png',
                    'save_model_wrt': self.save_model_wrt,
                    'scheduler_name': self.scheduler_name,
                    'schedulers_kwargs': self.schedulers_kwargs,
                    'loss_kwargs': self.loss_kwargs,
                    'optimizer_kwargs': self.optimizer_kwargs,
                    'early_stop': self.early_stop,
                    'early_stop_kwargs': self.early_stop_kwargs,
                    'num_epochs': self.num_epochs,
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate,
                    'Num_Train': self.num_train, 
                    'Num_Validation': self.num_val,
                    'Num_Test': self.num_test,
                    'Train-Test': self.num_train*self.subset_ratio,
                    'Val-Test': self.num_val*self.subset_ratio,
                    'Test-Test': self.num_test*self.subset_ratio,
                    
                    'train_loss': self.epoch_train_losses[:self.epoch],
                    'val_loss': self.epoch_val_losses[:self.epoch], 
                    'test_loss': self.epoch_test_losses[:self.epoch], 
                    
                    'train_acc': self.train_acc_list[:self.epoch],
                    'val_acc': self.val_acc_list[:self.epoch], 
                    'test_acc': self.test_acc_list[:self.epoch], 
                    
                    'state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'representation': str(repr(self.network).replace('\n', ''))
                    }, network_save_filename)
    #------------------------------------------------------

    #save (csv) different losses
    def save_loss_csv(self, save_csv_filename='temp'):
        """
        Save the training, validation, and test losses and accuracies as a CSV file.

        This function captures the history of loss and accuracy metrics recorded over training epochs 
        and stores them in tabular CSV format, which is suitable for further analysis or visualization.

        Args:
            save_csv_filename (str): Name of the CSV file (without extension) to save metrics to.
                File is saved at `{result_savepath}/{save_csv_filename}.csv`.

        Returns:
            None
        """
        loss_csv_filepath = f"{self.result_savepath}/{save_csv_filename}.csv"
        dict_loss = OrderedDict({'epoch': list(range(1, self.epoch)),
                                 'Time': self.time_list[:self.epoch],
                                 'lr': self.lr_list[:self.epoch],
                                 
                                 'train_loss': self.epoch_train_losses[:self.epoch],
                                 'val_loss': self.epoch_val_losses[:self.epoch], 
                                 'test_loss': self.epoch_test_losses[:self.epoch], 
                                
                                 'train_acc': self.train_acc_list[:self.epoch],
                                 'val_acc': self.val_acc_list[:self.epoch], 
                                 'test_acc': self.test_acc_list[:self.epoch], 
                                })
        # print(dict_loss)
        

        df_loss = pd.DataFrame.from_dict(dict_loss)
        df_loss.to_csv(loss_csv_filepath, index=False)
        #----------------------------------------------------


    def train_model(self) -> tuple[List[float], nn.Module]:
        
        """
        Train the neural network model using the specified training, validation, and test loaders.

        This method handles the entire training loop including loss/accuracy tracking, model saving,
        early stopping, scheduler stepping, and intermediate results saving. Best model is selected
        based on validation loss or accuracy, and saved accordingly.

        Returns:
            tuple:
                - List[float]: List of training losses for each epoch.
                - nn.Module: The best-performing model (based on validation performance).
        """
        self.network.to(self.device)
        self.epoch_train_losses = [None]*self.num_epochs
        self.epoch_val_losses = [None]*self.num_epochs
        self.epoch_test_losses = [None]*self.num_epochs
        
        self.train_acc_list = [None]*self.num_epochs
        self.test_acc_list = [None]*self.num_epochs
        self.val_acc_list = [None]*self.num_epochs
        
        self.lr_list = [None]*self.num_epochs
        self.time_list = [None]*self.num_epochs
        
        model_acc = -float('inf')
        model_loss = float('inf')
        
        train_loader_subset, _ = subset_loader(self.train_loader, batch_size=self.batch_size, subset_ratio=self.subset_ratio)
        test_loader_subset, _ = subset_loader(self.test_loader, batch_size=self.batch_size, subset_ratio=self.subset_ratio)
        val_loader_subset, _ = subset_loader(self.val_loader, batch_size=self.batch_size, subset_ratio=self.subset_ratio)
        
        print('+'*70)
        print(f'Number of Test-Train Samples: {len(train_loader_subset.dataset)}')
        print(f'Number of Test-Val Samples: {len(val_loader_subset.dataset)}')
        print(f'Number of Test-Test Samples: {len(test_loader_subset.dataset)}')
        print('+'*70)
        
        # LR Schduler
        self.scheduler = Trainer_functions.get_scheduler(scheduler_name=self.scheduler_name,
                                  optimizer=self.optimizer,
                                  **self.schedulers_kwargs
                                ) if self.scheduler_name and self.schedulers_kwargs else None # is instance of torch.optim.lr_scheduler.LRScheduler
        
        # Initializing Early Stopping
        early_stopping = EarlyStopping(**self.early_stop_kwargs) if self.early_stop and self.early_stop_kwargs else None
        progress_bar = tqdm(range(self.num_epochs), desc="Training", unit="Eps")

        for epoch in progress_bar:
            self.epoch = epoch+1
            self.network.train()
            epoch_train_loss = self.train_1epoch()
            train_acc, epoch_train_loss = self.train_test_model(dataloader=train_loader_subset)
            test_acc, epoch_test_loss = self.train_test_model(dataloader=test_loader_subset)
            val_acc, epoch_val_loss = self.train_test_model(dataloader=val_loader_subset)
            
            self.train_acc_list[epoch] = round(train_acc, 2)
            self.test_acc_list[epoch] = round(test_acc, 2)
            self.val_acc_list[epoch] = round(val_acc, 2)
            self.lr_list[epoch] = self.optimizer.param_groups[0]['lr']
            self.time_list[epoch] = datetime_now(path=False)[0]
            
            self.epoch_train_losses[epoch] = round(epoch_train_loss, 6)
            self.epoch_test_losses[epoch] = round(epoch_test_loss, 6)
            self.epoch_val_losses[epoch] = round(epoch_val_loss, 6)
            
            if self.save_model_wrt == 'loss':
                condition_flag = (epoch_val_loss < model_loss)
            elif self.save_model_wrt == 'accuracy':
                condition_flag = (val_acc > model_acc)
            else:
                raise NameError('Condition name not known')

            if condition_flag:
                self.best_train_loss, self.best_test_loss, self.best_val_loss = epoch_train_loss, epoch_test_loss, epoch_val_loss
                self.best_train_acc, self.best_test_acc, self.best_val_acc = train_acc, test_acc, val_acc
                self.best_epoch = self.epoch
                self.best_model = copy.deepcopy(self.network)
                self.save_network(network_save_filename=self.best_model_save_name)
                model_loss = epoch_val_loss
                model_acc = val_acc
                
            if (epoch % self.save_model_per_epoch == 0):
                model_save_name = f'{self.model_save_name}_model_at_epoch_{self.epoch}'
                self.save_network(network_save_filename=model_save_name)
                
            # Set postfix in tqdm bar for live update
            progress_bar.set_postfix({"Train Loss": f"{epoch_train_loss:.4f}", "Val Loss": f"{epoch_val_loss:.4f}", "Test Loss": f"{epoch_test_loss:.4f}",
                                    "Train Acc": f"{train_acc:.4f}", "Val Acc": f"{val_acc:.4f}", "Test Acc": f"{test_acc:.4f}"})
            
            # Update tqdm progress bar with accs
            if self.tqdm_write:
                tqdm.write(f"At {self.time_list[epoch]}  Epoch {self.epoch}/{self.num_epochs} - "
                        f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Test Loss: {epoch_test_loss:.4f}| "
                        f"Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f} | Test Acc: {test_acc:.2f}")
                
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(epoch_val_loss)
                else:
                    self.scheduler.step()
            
            if self.early_stop:    
                early_stopping(epoch_val_loss)
                if early_stopping.early_stop:
                    print(f"Early stop triggered at epoch: {epoch + 1}")
                    print('='*80)
                    break
            
            ## Save epoch losses
            self.save_loss_csv(save_csv_filename=self.loss_save_filename)
        print(f'Best Model Details:\n  Epoch: {self.best_epoch}\n    Train Loss: {self.best_train_loss}, Test Loss: {self.best_test_loss}, Val Loss: {self.best_val_loss}\n   \
            Train Acc: {self.best_train_acc}, Test Acc: {self.best_test_acc}, Val Acc: {self.best_val_acc}')
                
        ## Plot epoch Loss-Accuracy Plot
        self.plot_loss_save_images(save_image_filename = self.save_image_filename)
        print('Training completed.')
        
        return self.epoch_train_losses, self.best_model
    
        
    # pre-trained network load function for test
    def load_test_network(self, temp_network_path: str = ''):
        
        """
        Load a previously saved network for evaluation or inference.

        This method restores the model's state dict, optimizer state, and all training metadata
        such as learning rate, loss, accuracy history, and scaler from a `.pt` checkpoint file.

        Args:
            temp_network_path (str): Full path to the checkpoint file. If not provided,
                defaults to the best saved model at `{result_savepath}/{best_model_save_name}.pt`.

        Returns:
            None
        """
    
        if not temp_network_path:
            temp_network_path = f'{self.result_savepath}/{self.best_model_save_name}.pt'
        
        if os.path.isfile(temp_network_path):
            print('Loading pre-trained network checkpoint from: "{}"'.format(temp_network_path))
            checkpoint = torch.load(temp_network_path, map_location=self.device)
            #--------------------------------------------------
            
            self.model_class_name = checkpoint.get('model_class_name', 'UNKNOWN')
            self.optimizer_name = checkpoint.get('optimizer_name', 'UNKNOWN')
            self.loss_func_name = checkpoint.get('loss_func_name', 'UNKNOWN')
            self.standardscaler = checkpoint.get('standardscaler', 'UNKNOWN')
            self.learning_rate = checkpoint.get('learning_rate', 'UNKNOWN')
            self.save_model_wrt = checkpoint.get('save_model_wrt', 'UNKNOWN')
            self.scheduler_name = checkpoint.get('scheduler_name', 'UNKNOWN')
            self.schedulers_kwargs = checkpoint.get('schedulers_kwargs', 'UNKNOWN')
            self.epoch = checkpoint.get('epoch', 'UNKNOWN')
            self.epoch_best_network = checkpoint.get('best_epoch', 'UNKNOWN')
            self.loss_kwargs = checkpoint.get('loss_kwargs', 'UNKNOWN')
            self.optimizer_kwargs = checkpoint.get('optimizer_kwargs', 'UNKNOWN')
            self.train_loss = checkpoint.get('train_loss', [])
            self.val_loss = checkpoint.get('val_loss', [])
            self.test_loss = checkpoint.get('test_loss', [])
            self.train_acc = checkpoint.get('train_acc', [])
            self.val_acc = checkpoint.get('val_acc', [])
            self.test_acc = checkpoint.get('test_acc', [])
            self.network_pyfile = checkpoint.get('network_pyfile' 'NA')
            self.utils_pyfile = checkpoint.get('utils_pyfile' 'NA')
            self.network_object = checkpoint.get('representation' 'NA')
            #+++++++++++++++++++++++++++++++++++++++++++++++++++
                
            self.network.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print('Loaded pre-trained network checkpoint from "{}"\nepoch: {} train loss: {} val loss: {} test loss: {} train acc: {} val acc: {} test acc: {} ' \
                .format(temp_network_path, self.epoch_best_network, self.train_loss[-1], self.val_loss[-1],
                        self.test_loss[-1], self.train_acc[-1], self.val_acc[-1], self.test_acc[-1])
                    )

        else:
            print('No pre-trained network checkpoint found at "{}"' \
                    .format(temp_network_path) \
                    )
        print('-'*80)
            

class TransferWeights:
    def __init__(self, source: Union[str, nn.Module, dict], map_location: str = "cpu"):
        """
        Initialize loader with a path, state_dict, or model.
        
        Args:
            source (str | nn.Module | dict): Path to .pt/.pth file, 
                                            a loaded nn.Module, 
                                            or a state_dict.
            map_location (str): Device mapping for loading.
        """
        if isinstance(source, str):  # path to file
            self.checkpoint = torch.load(source, map_location=map_location, weights_only=False)['state_dict']
        elif isinstance(source, nn.Module):  # full model
            self.checkpoint = source.state_dict()
        elif isinstance(source, dict):  # state_dict
            self.checkpoint = source
        else:
            raise ValueError("Source must be path, state_dict, or nn.Module.")

    def _load_state_dict(self, model: nn.Module, state_dict: dict, strict: bool=False) -> nn.Module:
        
        """Safely load a state_dict into the model."""
        missing_keys, unexpected_keys = "", ""
        try:
            model.load_state_dict(state_dict, strict=strict)
        except:
            # Fallback: partial load
            model_dict = model.state_dict()
            model_dict.update({k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape})
            missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=strict)
            
            missings = {'missing_keys':missing_keys, 'unexpected_keys':unexpected_keys}
            
        return missings

    def transfer_into(self, model: nn.Module, strict: bool = False) -> nn.Module:
        """
        Load the checkpoint into the provided model.
        
        Args:
            model (nn.Module): Target model.
            strict (bool): Enforce exact layer match.
        
        Returns:
            nn.Module: Model with weights loaded.
        """
        return self._load_state_dict(model, self.checkpoint, strict)