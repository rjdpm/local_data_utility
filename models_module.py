#!/usr/bin/env python3
import inspect
import numpy as np
import math, os, pickle, sys
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict, Set

import torch
from torch import nn
from torchinfo import summary
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large
from torchvision.models import resnet18
from transformers import ViTModel, ViTFeatureExtractor
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers import AutoModel, AutoConfig

from torch_geometric.nn import GINConv as PyG_GINConv, global_add_pool
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import LayerNorm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Generalised_data_utils import auto_repr, collect_class_definitions, check_tensor

__all__ = [
    'network_pyfile',
    'activation_func',
    'make_mlp',
    'normalize_adjacency',
    'Identity',
    'Flatten',
    'View_',
    'ConvStandard',
    'Conv',
    'AllCNN_',
    'LeNet32_',
    'ResidualBlock_',
    'ResNet9_',
    'MobileNet',
    'ResNet18',
    'TimeDistributed',
    'GLU',
    'GatedResidualNetwork',
    'PositionalEncoder',
    'VariableSelectionNetwork',
    'TFT',
    'LSTMnetwork',
    'SMILESLinearNet',
    'FusionNet',
    'GCNLayer',
    'GCN',
    'GCN_Connected',
    'MRGCN',
    'GINConv',
    'GIN',
    'GIN_FU',
    'GIN_SINKHORN',
    'GIN_PyG',
    'ChemBERTa',
    'ChemBERTaRegressorroberta',
    'ChemBERTaRegressor',
    'ChemBERTaRegressorwithMultiheadAttention',
    'ChemBERTaRegressorwithAttention',
    'ChemBERTaRegressorwithLSTM',
    'ChemBERTaRegressorRNN',
    'AdditiveAttention',
    'DotProductAttention',
    'UnifiedAttention',
    'MultiHeadAttention',
    'GeneralizedAdditiveAttention',
    'GeneralizedDotProductAttention',
    'GeneralizedUnifiedAttention',
    'GeneralizedMultiHeadAttention',
]

source_file = os.path.abspath(__file__)
with open(source_file, 'rb') as fp:
    file_ = fp.read()
    network_pyfile = pickle.dumps(file_)
    
def activation_func(name='relu', alpha=1.0, negative_slope=1e-2):
    """
    Return a PyTorch activation module based on the provided name.

    Args:
        name (str): Name of the activation function.
        alpha (float): Parameter for ELU/CELU.
        negative_slope (float): Parameter for LeakyReLU.

    Returns:
        nn.Module: Corresponding activation function.
    """
    name = name.lower()
    if name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'selu':
        return nn.SELU()
    elif name == 'elu':
        return nn.ELU(alpha)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(negative_slope)
    elif name == 'celu':
        return nn.CELU(alpha)
    elif name == 'gelu':
        return nn.GELU()
    elif name is None:
        return nn.Identity()
    else:
        raise ValueError(f'Unsupported activation function: "{name}"')
    


def make_mlp(list_dims, dropout=0.0, act_func='relu', norm_type='layer', alpha=1.0, negative_slope=1e-2):
    
    """
    Constructs a multi-layer perceptron (MLP) using a sequence of linear layers, each optionally followed by 
    normalization, activation, and dropout layers.

    Args:
        list_dims (List[int]): A list specifying the input, hidden, and output dimensions. 
                               For example, [128, 256, 64, 1] creates 3 layers.
        dropout (float): Dropout rate to apply after each activation (except the final layer). Default is 0.0.
        act_func (Union[str, List[str], None]): Activation function(s) to use. 
            - If a single string (e.g., 'relu', 'elu', 'leakyrelu'), it is used for all hidden layers. 
              If list_dims=[128,256,64,1], then act_func list must have length 2 (for the two hidden layers [256,64]).
            - If a list of strings, it must have `len(list_dims) - 2` entries (for hidden layers only).
            - If None, no activation is applied.
        norm_type (str): Type of normalization to apply after each linear layer:
            - 'batch' for BatchNorm1d
            - 'layer' for LayerNorm
            - 'NA' for No Normalization
            - None for no normalization.
        alpha (float): The α parameter used for ELU/CELU activations. Default is 1.0.
        negative_slope (float): The negative slope parameter used for LeakyReLU. Default is 1e-2.

    Returns:
        nn.Sequential: A fully constructed MLP as a `torch.nn.Sequential` block.
    """
    layers = nn.ModuleList([])

    # Standardize act_func to a list of activation layers
    num_layers = len(list_dims) - 1
    if act_func is None or not act_func:
        act_funcs = [nn.Identity()] * (num_layers - 1)
    elif isinstance(act_func, str):
        act_funcs = [activation_func(act_func, alpha=alpha, negative_slope=negative_slope)] * (num_layers - 1)
    elif isinstance(act_func, list):
        assert (len(act_func) == num_layers - 1), f"Length of act_func list {len(act_func)} must match the number of hidden layers {(len(list_dims) - 2)}."
        act_funcs = [activation_func(name, alpha=alpha, negative_slope=negative_slope) for name in act_func]
    else:
        raise TypeError("act_func must be a string or a list of strings.")

    if num_layers > 0:
        for i in range(num_layers):
            in_dim = list_dims[i]
            out_dim = list_dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))

            if i < num_layers - 1:
                # Normalization
                if norm_type == 'batch':
                    layers.append(nn.BatchNorm1d(out_dim))
                elif norm_type == 'layer':
                    layers.append(nn.LayerNorm(out_dim))
                elif (norm_type == 'NA') or (norm_type is None):
                    layers.append(nn.Identity())
                elif norm_type is not None:
                    raise ValueError(f"Unsupported normalization type: {norm_type}")
                
                # Activation and Dropout
                layers.append(act_funcs[i])
                layers.append(nn.Dropout(dropout))
    else:
        layers.append(nn.Identity())

    return nn.Sequential(*layers)


    
def normalize_adjacency(
    adjacency,
    degree,
    laplacian=False
):
    """
    Normalize adjacency (or Laplacian) matrices for both single and multi-relational cases.

    Args:
        adjacency (Tensor): Shape [B, N, N] or [B, R, N, N].
        degree (Tensor):    Same shape as adjacency.
        laplacian (bool):   Whether to compute normalized Laplacian instead of adjacency.

    Returns:
        Tensor: Normalized adjacency or Laplacian matrix of same shape as input.
    """
    if adjacency.dim() == 4:  # [B, R, N, N] multi-relational case
        B, R, N, _ = adjacency.shape
        normalized = torch.zeros_like(adjacency)
        
        for rel in range(R):
            adj = adjacency[:, rel]       # [B, N, N]
            deg = degree[:, rel]   if degree.dim() == 4  else degree    # [B, N, N]

            deg_inv_sqrt = deg.clone()
            mask = deg_inv_sqrt != 0
            deg_inv_sqrt[mask] = deg_inv_sqrt[mask].pow(-0.5)

            if laplacian:
                lap = deg - adj
                norm = torch.bmm(torch.bmm(deg_inv_sqrt, lap), deg_inv_sqrt)
            else:
                norm = torch.bmm(torch.bmm(deg_inv_sqrt, adj), deg_inv_sqrt)

            normalized[:, rel] = norm

        return normalized

    elif adjacency.dim() == 3:  # [B, N, N] single-relation case
        deg_inv_sqrt = degree.clone()
        mask = deg_inv_sqrt != 0
        deg_inv_sqrt[mask] = deg_inv_sqrt[mask].pow(-0.5)

        if laplacian:
            lap = degree - adjacency
            normalized = torch.bmm(torch.bmm(deg_inv_sqrt, lap), deg_inv_sqrt)
        else:
            normalized = torch.bmm(torch.bmm(deg_inv_sqrt, adjacency), deg_inv_sqrt)

        return normalized

    else:
        raise ValueError("Adjacency tensor must be 3D or 4D")

    
class Identity(nn.Module):
    """
    Identity layer used to pass input without modification.

    Task:
        Acts as a placeholder in models when conditional operations like dropout 
        or batch norm are toggled off.

    Input:
        Tensor of any shape.

    Output:
        Same tensor, unchanged.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class Flatten(nn.Module):
    """
    Flattens all dimensions of a tensor except the batch dimension.

    Task:
        Prepares convolutional outputs for linear layers.

    Input:
        Tensor of shape (N, C, H, W)

    Output:
        Tensor of shape (N, C*H*W)
    """
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)
    
class View_(nn.Module):
    """
    Reshapes tensor to a specified shape using `.view`.

    Task:
        Useful in pipelines like nn.Sequential where dynamic reshaping is needed.

    Args:
        size (tuple): Target shape (can include -1 for inferred dimensions)

    Input:
        Tensor of any shape.

    Output:
        Tensor reshaped to specified dimensions.
    """
    def __init__(self, size):
        super(View_, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    

class ConvStandard(nn.Conv2d): 
    """
    2D convolutional layer with custom Gaussian weight initialization.

    Task:
        Provides more control over initialization for experimentation or 
        stable training in small-data regimes.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Convolutional kernel size.
        stride (int): Stride of convolution.
        padding (int): Zero-padding size.
        w_sig (float): Std deviation factor for weight init.

    Input:
        Tensor of shape (N, C_in, H, W)

    Output:
        Tensor of shape (N, C_out, H_out, W_out)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0, w_sig =\
                 np.sqrt(1.0)):
        super(ConvStandard, self).__init__(in_channels, out_channels,kernel_size)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.w_sig = w_sig
        self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=self.w_sig/(self.in_channels*np.prod(self.kernel_size)))
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=0)
            
    def forward(self, input):
        return F.conv2d(input,self.weight,self.bias,self.stride,self.padding)
            
class Conv(nn.Sequential):
    """
    Convolutional block with optional batch normalization and activation.

    Task:
        Modular convolution layer with support for transposed convolutions,
        batch normalization, and configurable activation.

    Args:
        in_channels, out_channels, kernel_size, stride, padding (int or tuple)
        activation_fn (nn.Module): e.g., nn.ReLU
        batch_norm (bool): Whether to apply BatchNorm2d
        transpose (bool): Whether to use ConvTranspose2d

    Input:
        Tensor of shape (N, C_in, H, W)

    Output:
        Tensor of shape determined by layer parameters.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
#             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#                                 )]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)

class AllCNN_(nn.Module):
    """
    All Convolutional Network for classification (variant of All-CNN).

    Task:
        Performs image classification using stacked convolutional layers without
        max pooling, using stride for downsampling.

    Args:
        filters_percentage (float): Scaling for channel width.
        n_channels (int): Input channels (3 for RGB).
        num_classes (int): Output class count.
        dropout (bool): Whether to apply dropout.
        batch_norm (bool): Whether to use batch norm.

    Input:
        Tensor of shape (N, C, H, W)

    Output:
        Tensor of shape (N, num_classes)
    """
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True, padding = 0):
        super(AllCNN_, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        
        self.conv1 = Conv(n_channels, n_filter1, kernel_size=3, padding = padding, batch_norm=batch_norm)
        self.conv2 = Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm)
        self.conv3 = Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm)
        
        self.dropout1 = self.features = nn.Sequential(nn.Dropout(inplace=True) if dropout else Identity())
        
        self.conv4 = Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv5 = Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv6 = Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm)
        
        self.dropout2 = self.features = nn.Sequential(nn.Dropout(inplace=True) if dropout else Identity())
        
        self.conv7 = Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv8 = Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm)
        if n_channels == 3:
            self.pool = nn.AvgPool2d(8)
        elif n_channels == 1:
            self.pool = nn.AvgPool2d(7)
        self.flatten = Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)
        actv1 = out
        
        out = self.conv2(out)
        actv2 = out
        
        out = self.conv3(out)
        actv3 = out
        
        out = self.dropout1(out)
        
        out = self.conv4(out)
        actv4 = out
        
        out = self.conv5(out)
        actv5 = out
        
        out = self.conv6(out)
        actv6 = out
        
        out = self.dropout2(out)
        
        out = self.conv7(out)
        actv7 = out
        
        out = self.conv8(out)
        actv8 = out
        
        out = self.pool(out)
        
        out = self.flatten(out)
        
        out = self.classifier(out)
        
        return out#, actv1, actv2, actv3, actv4, actv5, actv6, actv7, actv8 



class LeNet32_(nn.Module):
    """
    LeNet-like architecture adapted for 32x32 images.

    Task:
        A compact CNN for image classification, suitable for CIFAR-10 or similar datasets.

    Args:
        n_classes (int): Number of output classes.
        num_input_channels (int): Input image channels.
        padding (int): Padding for conv layers.

    Input:
        Tensor of shape (N, C, 32, 32)

    Output:
        Tensor of shape (N, n_classes), after softmax.
    """
    def __init__(self, n_classes = 10, num_input_channels = 3, padding = 0):
        super(LeNet32_, self).__init__()
        self.n_classes = n_classes
        self.padding = padding
        self.num_input_channels = num_input_channels

        self.layers = nn.Sequential(
            nn.Conv2d(self.num_input_channels, 6, kernel_size=5, stride=1, padding=self.padding),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            View_((-1, 16*5*5)),
            nn.Linear(16*5*5, 120),
            nn.ReLU(inplace = True),
            nn.Linear(120, 84),
            nn.ReLU(inplace = True),
            nn.Linear(84, n_classes),
            nn.Softmax(dim=1))


    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx == 0:
                activation1 = x
            if idx == 3:
                activation2 = x

        return x#, activation1, activation2


class ResidualBlock_(nn.Module):
    """
    A residual block as defined by He et al. Residual block with optional downsampling (as in ResNet).

    Task:
        Applies two conv layers and adds input via skip connection. Downsamples 
        residual if stride ≠ 1.

    Args:
        in_channels (int): Input feature maps.
        out_channels (int): Output feature maps.
        kernel_size (int): Convolutional kernel size.
        stride (int): Stride for first conv.
        padding (int): Padding to maintain size.

    Input:
        Tensor of shape (N, C_in, H, W)

    Output:
        Tensor of shape (N, C_out, H_out, W_out)
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock_, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out = out + residual
        
        return out
    
    
    
class ResNet9_(nn.Module):
    """
    ResNet-9: Small ResNet variant for low-resolution image classification.

    Task:
        Employs residual blocks and downsampling for efficient classification.

    Args:
        n_classes (int): Output classes.
        num_input_channels (int): Input channels (1 for grayscale, 3 for RGB).
        padding (int): Optional padding for initial conv layer.

    Input:
        Tensor of shape (N, C, H, W)

    Output:
        Tensor of shape (N, n_classes)
    """
    def __init__(self, n_classes = 10, num_input_channels = 3, padding = 0):
        super(ResNet9_, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=num_input_channels, out_channels=64, kernel_size=3, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock_(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock_(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(in_features=256, out_features=n_classes, bias=True)

    def forward(self, x):
        for idx, layer in enumerate(self.conv):
            x = layer(x)
            if idx == 0:
                activation1 = x
            if idx == 3:
                activation2 = x
            if idx == 8:
                activation3 = x
            if idx == 12:
                activation4 = x
        
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc(x)

        return x#, activation1, activation2, activation3, activation4    
    



class MobileNet(nn.Module):
    """
    Custom MobileNetV3 Large model with activation trace outputs.

    Task:
        Feature extraction and classification using MobileNetV3 backbone.
        Returns intermediate activations for interpretability or auxiliary training.

    Input:
        x (Tensor): Image tensor of shape (N, C, H, W)

    Output:
        Tuple of:
            - Output logits (N, 1)
            - Intermediate activations from each major layer (actvn1 to actvn17)
    """
    def __init__(self):
        super().__init__()

        base = mobilenet_v3_large()
        base_layers = list(base.children())[0]  # features

        # First conv + norm
        self.conv_norm1 = nn.Sequential(base_layers[0])

        # Inverted residual blocks (layers 1 through 15)
        self.inverted_residuals = nn.ModuleList([base_layers[i] for i in range(1, 16)])

        # Final conv + norm layer
        self.conv_norm2 = nn.Sequential(base_layers[16])

        # Pooling layer and classifier
        self.pool1 = list(base.children())[1]  # AdaptiveAvgPool2d
        self.drop = nn.Dropout()
        self.final = nn.Linear(960, 1)

    def forward(self, x):
        activations = OrderedDict({()})

        x = self.conv_norm1(x)
        activations.append(x)

        for i, layer in enumerate(self.inverted_residuals):
            x = layer(x)
            activations[f"actvn_{i}"] = x

        x = self.conv_norm2(x)
        activations.append(x)

        x = self.pool1(x)
        x = x.view(-1, self.final.in_features)
        x = self.drop(x)
        out = self.final(x)

        return out, activations
    
    
class ResNet18(nn.Module):
    """
    Customized ResNet-18 architecture returning intermediate feature maps.

    Task:
        Image classification with feature extraction at various ResNet stages.

    Input:
        x (Tensor): Image tensor of shape (N, C, H, W)

    Output:
        Output logits (N, n_classes)
    """
    def __init__(self, n_classes = 10, num_input_channels = 3):
        super().__init__()
        base = resnet18(pretrained=False)
        in_features = base.fc.in_features
        base_list = [*list(base.children())[:-1]]
        self.layer1 = nn.Sequential(*base_list[0:3])
        self.pool1 = base_list[3]
        self.basic_block1 = base_list[4][0]
        self.basic_block2 = base_list[4][1]
        self.basic_block3 = base_list[5][0]
        self.basic_block4 = base_list[5][1]
        self.basic_block5 = base_list[6][0]
        self.basic_block6 = base_list[6][1]
        self.basic_block7 = base_list[7][0]
        self.basic_block8 = base_list[7][1]
        self.pool2 = base_list[8]
        self.drop = nn.Dropout()
        self.final = nn.Linear(512,n_classes)
        
    
    def forward(self,x):
        out = self.layer1(x)
        actvn1 = out
        
        out = self.pool1(out)
        
        out = self.basic_block1(out)
        actvn2 = out
        
        out = self.basic_block2(out)
        actvn3 = out
        
        out = self.basic_block3(out)
        actvn4 = out
        
        out = self.basic_block4(out)
        actvn5 = out
        
        out = self.basic_block5(out)
        actvn6 = out
        
        out = self.basic_block6(out)
        actvn7 = out
        
        out = self.basic_block7(out)
        actvn8 = out
        
        out = self.basic_block8(out)
        actvn9 = out
        
        out = self.pool2(out)
        out = out.view(-1,self.final.in_features)
            
        out = self.final(out)
        
        return out#, actvn1, actvn2, actvn3, actvn4, actvn5, actvn6, actvn7, actvn8, actvn9 

class TimeDistributed(nn.Module):
    """
    Applies a module over each time step of an input sequence.

    Task:
        Mimics Keras' TimeDistributed wrapper by reshaping input tensors.

    Input:
        x (Tensor): Shape (batch_size, time_steps, features)

    Output:
        Tensor with the same time structure, post-module application.
    """
    ## Takes any module and stacks the time dimension with the batch dimenison of inputs before apply the module
    ## From: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
    
class GLU(nn.Module):
    """
    Gated Linear Unit (GLU) for element-wise feature gating.

    Task:
        Applies a learned gate to modulate input features.

    Input:
        x (Tensor): Feature tensor

    Output:
        Tensor gated elementwise.
    """
    #Gated Linear Unit
    def __init__(self, input_size):
        super(GLU, self).__init__()
        
        self.fc1 = nn.Linear(input_size,input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)
    
class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) with context support and normalization.

    Task:
        Enhances feature transformations using gating and residual connections.

    Input:
        x (Tensor): Input feature sequence
        context (Tensor): Optional contextual input

    Output:
        Transformed and gated tensor with residual and normalization.
    """
    def __init__(self, input_size,hidden_state_size, output_size, dropout, hidden_context_size=None, batch_first=False):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_context_size = hidden_context_size
        self.hidden_state_size=hidden_state_size
        self.dropout = dropout
        
        if self.input_size!=self.output_size:
            self.skip_layer = TimeDistributed(nn.Linear(self.input_size, self.output_size))

        self.fc1 = TimeDistributed(nn.Linear(self.input_size, self.hidden_state_size), batch_first=batch_first)
        self.elu1 = nn.ELU()
        
        if self.hidden_context_size is not None:
            self.context = TimeDistributed(nn.Linear(self.hidden_context_size, self.hidden_state_size),batch_first=batch_first)
            
        self.fc2 = TimeDistributed(nn.Linear(self.hidden_state_size,  self.output_size), batch_first=batch_first)
        self.elu2 = nn.ELU()
        
        self.dropout = nn.Dropout(self.dropout)
        self.bn = TimeDistributed(nn.BatchNorm1d(self.output_size),batch_first=batch_first)
        self.gate = TimeDistributed(GLU(self.output_size), batch_first=batch_first)

    def forward(self, x, context=None):

        if self.input_size!=self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x
        
        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x+context
        x = self.elu1(x)
        
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)
        x = x+residual
        x = self.bn(x)
        
        return x

class PositionalEncoder(torch.nn.Module):
    """
    Fixed sinusoidal positional encoder for transformer inputs.

    Task:
        Injects position information into sequence embeddings.

    Input:
        x (Tensor): Sequence tensor (seq_len, batch, d_model)

    Output:
        Positionally encoded tensor.
    """
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(0)
            pe = self.pe[:, :seq_len].view(seq_len,1,self.d_model)
            x = x + pe
            return x

class VariableSelectionNetwork(nn.Module):
    """
    Task:
        Selects relevant input variables using gated residual networks and attention-based weighting.
    
    Inputs:
        - embedding (Tensor): Input tensor of shape [batch_size, time_steps, num_inputs * input_size].
        - context (Tensor, optional): Additional context tensor for conditioning, shape [batch_size, time_steps, context_size].
    
    Outputs:
        - outputs (Tensor): Weighted combination of transformed inputs, shape [batch_size, time_steps, hidden_size].
        - sparse_weights (Tensor): Attention weights for each input variable, shape [batch_size, num_inputs, 1].
    """
    def __init__(self, input_size, num_inputs, hidden_size, dropout, context=None):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.input_size =input_size
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context=context

        if self.context is not None:
            self.flattened_grn = GatedResidualNetwork(self.num_inputs*self.input_size, self.hidden_size, self.num_inputs, self.dropout, self.context)
        else:
            self.flattened_grn = GatedResidualNetwork(self.num_inputs*self.input_size, self.hidden_size, self.num_inputs, self.dropout)


        self.single_variable_grns = nn.ModuleList()
        for i in range(self.num_inputs):
            self.single_variable_grns.append(GatedResidualNetwork(self.input_size, self.hidden_size, self.hidden_size, self.dropout))

        self.softmax = nn.Softmax()

    def forward(self, embedding, context=None):
        if context is not None:
            sparse_weights = self.flattened_grn(embedding, context)
        else:
            sparse_weights = self.flattened_grn(embedding)

        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)

        var_outputs = []
        for i in range(self.num_inputs):
            ##select slice of embedding belonging to a single input
            var_outputs.append(self.single_variable_grns[i](embedding[:,:, (i*self.input_size) : (i+1)*self.input_size]))

        var_outputs = torch.stack(var_outputs, axis=-1)

        outputs = var_outputs*sparse_weights
        
        outputs = outputs.sum(axis=-1)

        return outputs, sparse_weights

class TFT(nn.Module):
    """
    Task:
        Implements the Temporal Fusion Transformer (TFT) for interpretable multi-horizon forecasting.

    Inputs:
        - x (Dict[str, Tensor]):
            - 'identifier': Static categorical inputs of shape [batch_size, time_steps, static_variables]
            - 'inputs': Dynamic inputs (real + categorical) of shape [batch_size, time_steps, input_dim]

    Outputs:
        - output (Tensor): Forecasted quantile values, shape [batch_size, decode_length, num_quantiles].
        - encoder_output (Tensor): LSTM encoder output.
        - decoder_output (Tensor): LSTM decoder output.
        - attn_output (Tensor): Output from attention mechanism.
        - attn_output_weights (Tensor): Weights from multi-head attention.
        - encoder_sparse_weights (Tensor): Variable importance weights for encoder inputs.
        - decoder_sparse_weights (Tensor): Variable importance weights for decoder inputs.
    """
    def __init__(self, config):
        """
        Initializes the full TFT model with embeddings, LSTMs, attention, and feedforward layers.
        """
        super(TFT, self).__init__()
        self.device = config['device']
        self.batch_size=config['batch_size']
        self.static_variables = config['static_variables']
        self.encode_length = config['encode_length']
        self.time_varying_categoical_variables =  config['time_varying_categoical_variables']
        self.time_varying_real_variables_encoder =  config['time_varying_real_variables_encoder']
        self.time_varying_real_variables_decoder =  config['time_varying_real_variables_decoder']
        self.num_input_series_to_mask = config['num_masked_series']
        self.hidden_size = config['lstm_hidden_dimension']
        self.lstm_layers = config['lstm_layers']
        self.dropout = config['dropout']
        self.embedding_dim = config['embedding_dim']
        self.attn_heads = config['attn_heads']
        self.num_quantiles = config['num_quantiles']
        self.valid_quantiles = config['valid_quantiles']
        self.seq_length = config['seq_length']
        
        self.static_embedding_layers = nn.ModuleList()
        for i in range(self.static_variables):
            emb = nn.Embedding(config['static_embedding_vocab_sizes'][i], config['embedding_dim']).to(self.device)
            self.static_embedding_layers.append(emb)
        
        
        
        self.time_varying_embedding_layers = nn.ModuleList()
        for i in range(self.time_varying_categoical_variables):
            emb = TimeDistributed(nn.Embedding(config['time_varying_embedding_vocab_sizes'][i], config['embedding_dim']), batch_first=True).to(self.device)
            self.time_varying_embedding_layers.append(emb)
            
        self.time_varying_linear_layers = nn.ModuleList()
        for i in range(self.time_varying_real_variables_encoder):
            emb = TimeDistributed(nn.Linear(1, config['embedding_dim']), batch_first=True).to(self.device)
            self.time_varying_linear_layers.append(emb)

        self.encoder_variable_selection = VariableSelectionNetwork(config['embedding_dim'],
                                (config['time_varying_real_variables_encoder'] +  config['time_varying_categoical_variables']),
                                self.hidden_size,
                                self.dropout,
                                config['embedding_dim']*config['static_variables']).to(self.device)

        self.decoder_variable_selection = VariableSelectionNetwork(config['embedding_dim'],
                                (config['time_varying_real_variables_decoder'] +  config['time_varying_categoical_variables']),
                                self.hidden_size,
                                self.dropout,
                                config['embedding_dim']*config['static_variables']).to(self.device)

        
        self.lstm_encoder_input_size = config['embedding_dim']*(config['time_varying_real_variables_encoder'] +  
                                                        config['time_varying_categoical_variables'] +
                                                        config['static_variables'])
        
        self.lstm_decoder_input_size = config['embedding_dim']*(config['time_varying_real_variables_decoder'] +  
                                                        config['time_varying_categoical_variables'] +
                                                        config['static_variables'])
                                      

        self.lstm_encoder = nn.LSTM(input_size=self.hidden_size, 
                            hidden_size=self.hidden_size,
                           num_layers=self.lstm_layers,
                           dropout=config['dropout']).to(self.device)
        
        self.lstm_decoder = nn.LSTM(input_size=self.hidden_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.lstm_layers,
                                   dropout=config['dropout']).to(self.device)

        self.post_lstm_gate = TimeDistributed(GLU(self.hidden_size)).to(self.device)
        self.post_lstm_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size)).to(self.device)

        self.static_enrichment = GatedResidualNetwork(self.hidden_size,self.hidden_size, self.hidden_size, self.dropout, config['embedding_dim']*self.static_variables).to(self.device)
        
        self.position_encoding = PositionalEncoder(self.hidden_size, self.seq_length).to(self.device)

        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.attn_heads).to(self.device)
        self.post_attn_gate = TimeDistributed(GLU(self.hidden_size)).to(self.device)

        self.post_attn_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size, self.hidden_size)).to(self.device)
        self.pos_wise_ff = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout).to(self.device)

        self.pre_output_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size, self.hidden_size)).to(self.device)
        self.pre_output_gate = TimeDistributed(GLU(self.hidden_size)).to(self.device)

        self.output_layer = TimeDistributed(nn.Linear(self.hidden_size, self.num_quantiles), batch_first=True).to(self.device)
        
    def __repr__(self):
        return auto_repr(self)
        
    def init_hidden(self):
        """Initializes LSTM hidden state."""
        return torch.zeros(self.lstm_layers, self.batch_size, self.hidden_size, device=self.device)
        
    def apply_embedding(self, x, static_embedding, apply_masking):
        """
        Applies linear and categorical embeddings to time-varying inputs.

        Inputs:
            - x (Tensor): Input tensor, shape [batch_size, time_steps, input_features].
            - static_embedding (Tensor): Static embeddings to broadcast, shape [batch_size, embed_dim * static_variables].
            - apply_masking (bool): Whether to mask inputs (used in decoder).

        Outputs:
            - embeddings (Tensor): Embedded and concatenated inputs, reshaped for sequence modeling.
        """
        ###x should have dimensions (batch_size, timesteps, input_size)
        ## Apply masking is used to mask variables that should not be accessed after the encoding steps
        #Time-varying real embeddings 
        if apply_masking:
            time_varying_real_vectors = []
            for i in range(self.time_varying_real_variables_decoder):
                emb = self.time_varying_linear_layers[i+self.num_input_series_to_mask](x[:,:,i+self.num_input_series_to_mask].view(x.size(0), -1, 1))
                time_varying_real_vectors.append(emb)
            time_varying_real_embedding = torch.cat(time_varying_real_vectors, dim=2)

        else: 
            time_varying_real_vectors = []
            for i in range(self.time_varying_real_variables_encoder):
                emb = self.time_varying_linear_layers[i](x[:,:,i].view(x.size(0), -1, 1))
                time_varying_real_vectors.append(emb)
            time_varying_real_embedding = torch.cat(time_varying_real_vectors, dim=2)
        
        
         ##Time-varying categorical embeddings (ie hour)
        time_varying_categoical_vectors = []
        for i in range(self.time_varying_categoical_variables):
            emb = self.time_varying_embedding_layers[i](x[:, :,self.time_varying_real_variables_encoder+i].view(x.size(0), -1, 1).long())
            time_varying_categoical_vectors.append(emb)
        time_varying_categoical_embedding = torch.cat(time_varying_categoical_vectors, dim=2)  

        ##repeat static_embedding for all timesteps
        static_embedding = torch.cat(time_varying_categoical_embedding.size(1)*[static_embedding])
        static_embedding = static_embedding.view(time_varying_categoical_embedding.size(0),time_varying_categoical_embedding.size(1),-1 )
        
        ##concatenate all embeddings
        embeddings = torch.cat([static_embedding,time_varying_categoical_embedding,time_varying_real_embedding], dim=2)
        
        return embeddings.view(-1,x.size(0),embeddings.size(2))
    
    def encode(self, x, hidden=None):
        """
        Encodes the input sequence using LSTM.

        Inputs:
            - x (Tensor): Input embeddings.
            - hidden (Tensor): Optional hidden state.

        Outputs:
            - output (Tensor): Encoded sequence.
            - hidden (Tensor): Final hidden state.
        """    
        if hidden is None:
            hidden = self.init_hidden()
            
        output, (hidden, cell) = self.lstm_encoder(x, (hidden, hidden))
        
        return output, hidden
        
    def decode(self, x, hidden=None):
        """
        Decodes the output sequence using LSTM.

        Inputs:
            - x (Tensor): Decoder input embeddings.
            - hidden (Tensor): Initial hidden state from encoder.

        Outputs:
            - output (Tensor): Decoded sequence.
            - hidden (Tensor): Final decoder state.
        """        
        if hidden is None:
            hidden = self.init_hidden()
            
        output, (hidden, cell) = self.lstm_decoder(x, (hidden,hidden))
        
        return output, hidden
    

    def forward(self, x):
        """
        Performs full forward pass through embedding, variable selection, LSTM, attention, and output head.

        Inputs:
            - x (dict): Dictionary containing static and dynamic input sequences.

        Outputs:
            - See class docstring.
        """
        ##inputs should be in this order
            # static
            # time_varying_categorical
            # time_varying_real

        embedding_vectors = []
        for i in range(self.static_variables):
            #only need static variable from the first timestep
            emb = self.static_embedding_layers[i](x['identifier'][:,0, i].long().to(self.device))
            embedding_vectors.append(emb)

        ##Embedding and variable selection
        static_embedding = torch.cat(embedding_vectors, dim=1)
        embeddings_encoder = self.apply_embedding(x['inputs'][:,:self.encode_length,:].float().to(self.device), static_embedding, apply_masking=False)
        embeddings_decoder = self.apply_embedding(x['inputs'][:,self.encode_length:,:].float().to(self.device), static_embedding, apply_masking=True)
        embeddings_encoder, encoder_sparse_weights = self.encoder_variable_selection(embeddings_encoder[:,:,:-(self.embedding_dim*self.static_variables)],embeddings_encoder[:,:,-(self.embedding_dim*self.static_variables):])
        embeddings_decoder, decoder_sparse_weights = self.decoder_variable_selection(embeddings_decoder[:,:,:-(self.embedding_dim*self.static_variables)],embeddings_decoder[:,:,-(self.embedding_dim*self.static_variables):])

        
        pe = self.position_encoding(torch.zeros(self.seq_length, 1, embeddings_encoder.size(2)).to(self.device)).to(self.device)
        
        embeddings_encoder = embeddings_encoder+pe[:self.encode_length,:,:]
        embeddings_decoder = embeddings_decoder+pe[self.encode_length:,:,:]

        ##LSTM
        lstm_input = torch.cat([embeddings_encoder,embeddings_decoder], dim=0)
        encoder_output, hidden = self.encode(embeddings_encoder)
        decoder_output, _ = self.decode(embeddings_decoder, hidden)
        lstm_output = torch.cat([encoder_output, decoder_output], dim=0)

        ##skip connection over lstm
        lstm_output = self.post_lstm_gate(lstm_output+lstm_input)

        ##static enrichment
        static_embedding = torch.cat(lstm_output.size(0)*[static_embedding]).view(lstm_output.size(0), lstm_output.size(1), -1)
        #print(lstm_output.device)
        #print(static_embedding.device)
        #print(self.static_enrichment.device)
        attn_input = self.static_enrichment(lstm_output, static_embedding)

        ##skip connection over lstm
        attn_input = self.post_lstm_norm(lstm_output)

        #attn_input = self.position_encoding(attn_input)

        ##Attention
        attn_output, attn_output_weights = self.multihead_attn(attn_input[self.encode_length:,:,:], attn_input[:self.encode_length,:,:], attn_input[:self.encode_length,:,:])

        ##skip connection over attention
        attn_output = self.post_attn_gate(attn_output) + attn_input[self.encode_length:,:,:]
        attn_output = self.post_attn_norm(attn_output)

        output = self.pos_wise_ff(attn_output) #[self.encode_length:,:,:])

        ##skip connection over Decoder
        output = self.pre_output_gate(output) + lstm_output[self.encode_length:,:,:]

        #Final output layers
        output = self.pre_output_norm(output)
        output = self.output_layer(output.view(self.batch_size, -1, self.hidden_size))
        
        
        return  output, encoder_output, decoder_output, attn_output, attn_output_weights, encoder_sparse_weights, decoder_sparse_weights
    
class LSTMnetwork(nn.Module):
    """
    Task:
        Compares two sequences using LSTM for similarity or regression scoring.

    Inputs:
        - sent1 (Tensor): First input sequence, shape [batch_size, seq_len, embed_dim].
        - sent2 (Tensor): Second input sequence, shape [batch_size, seq_len, embed_dim].

    Outputs:
        - output (Tensor): Regression or similarity score.
        - actv1, actv2, actv3, actv4 (Tensor): Intermediate activations for analysis or interpretability.
    """
    def __init__(self, text_embedding_dimension):
        """Initializes the LSTM and feedforward layers."""
        super().__init__()
        self.hidden_size = 64
        self.input_size = text_embedding_dimension
        self.num_layers = 1
        self.bidirectional = False
        self.num_directions = 1
        self.dropout1 = nn.Dropout(p=0.3)

        if self.bidirectional:
            self.num_directions = 2
 
        self.lstm = nn.LSTM( self.input_size, self.hidden_size, self.num_layers, 
                             bidirectional=self.bidirectional, batch_first=True)
        
        self.linear1 = nn.Linear(self.hidden_size*self.num_directions*2, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        
    def __repr__(self):
        return auto_repr(self)

    def forward(self, sent1, sent2):
        """Processes two input sequences and outputs a similarity score."""
        
        lstm_out1, _ = self.lstm( sent1)

        x1 = self.dropout1( lstm_out1)
        
        actv1 = x1
        
        lstm_out2, _ = self.lstm( sent2)
        
        x2 = self.dropout1( lstm_out2)
        
        actv2 = x2
        
        output = self.linear1(torch.cat([x1[:, -1, :], x2[:, -1, :]], axis = 1))
        actv3 = output
        output = self.relu(output)
        
        
        output = self.linear2(output)
        actv4 = output
        output = self.relu(output)
        
        
        output = self.linear3(output)
        output = self.relu(output)
        output = self.linear4(output)
        
        return torch.squeeze(output), actv1, actv2, actv3, actv4
    
class ViT(nn.Module):
    """
    Task:
        Performs image classification using pretrained Vision Transformer.

    Inputs:
        - pixel_values (Tensor): Input image batch, shape [batch_size, channels, height, width].

    Outputs:
        - logits (Tensor): Class logits, shape [batch_size, num_classes].
    """
    def __init__(self, num_classes=20):
        """Initializes pretrained ViT and a final classification head."""
        super(ViT, self).__init__()
        self.base = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.final = nn.Linear(self.base.config.hidden_size, num_classes)
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        
    def __repr__(self):
        return auto_repr(self)

    def forward(self, pixel_values):
        """Processes image data and returns classification logits."""
        outputs = self.base(pixel_values=pixel_values)
        logits = self.final(outputs.last_hidden_state[:,0])

        return logits
    
    
class SMILESLinearNet(nn.Module):
    """
    Task:
        Feedforward network for regression or classification on SMILES-based vector inputs.

    Inputs:
        - list_dims (List[int]): List of dimensions for each linear layer.
        - dropout (float): Dropout probability applied after each layer except the last.

    Outputs:
        - Tensor of shape [batch_size, output_dim] after passing through linear layers.
    """
    def __init__(self, list_dims, dropout=0.2, act_func='relu', norm_type='layer'):
        super().__init__()
        
        self.class_def = inspect.getsource(self.__class__)
        self.list_dims = list_dims
        self.dropout = dropout
        self.act_func = act_func
        self.norm_type = norm_type
        self.network_pyfile = network_pyfile
        self.MLP = make_mlp(list_dims=list_dims,
                            dropout=dropout,
                            act_func=act_func,
                            norm_type=norm_type
                            )
        
        # Weight Initialization
        for layer in self.MLP:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def __repr__(self):
        return auto_repr(self)
    
    def forward(self, x):
        """
        Task:
            Forward pass through MLP with ReLU, BatchNorm, and Dropout.

        Inputs:
            - x (Tensor): Input tensor of shape [batch_size, list_dims[0]]

        Outputs:
            - Tensor: Output tensor after final linear layer.
        """
        return self.MLP(x)

class FusionNet(nn.Module):
    def __init__(self,
                 input_dims,        # list or tuple of input sizes, e.g., [91, 384, 300, 256]
                 fusion_dim=256,
                 mlp_params=None):
        """
        input_dims: list of input feature dimensions for each modality.
        fusion_dim: projection size for each input modality.
        mlp_params: parameters to build final MLP (must include output dim).
        """
        super().__init__()

        self.input_dims = input_dims
        self.fusion_dim = fusion_dim
        self.mlp_params = mlp_params
        
        # Create projection layers dynamically
        self.projections = nn.ModuleList([nn.Linear(dim, fusion_dim) for dim in input_dims])

        # Final projection after concatenation
        self.final_proj = nn.Linear(fusion_dim * len(input_dims), fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim * len(input_dims))
        self.dropout = nn.Dropout(p=0.3)

        # MLP for prediction
        self.mlp = make_mlp(**mlp_params)

    def __repr__(self):
        return auto_repr(self)

    def feature_repr(self, modalities):
        """
        modalities: variable number of tensors.
                    Each tensor shape -> [batch, input_dim_i]
        """
        assert len(modalities) == len(self.projections), \
            f"Expected {len(self.projections)} inputs but received {len(modalities)}."

        # Project each modality to fusion dimension
        projected = [proj(mod) for proj, mod in zip(self.projections, modalities)]

        # Concatenate all projected features
        fused = torch.cat(projected, dim=-1)
        fused = self.norm(fused)
        fused = F.tanh(fused)
        fused = self.dropout(fused)

        # Final fusion projection + dropout
        fused = self.final_proj(fused)

        return fused

    def forward(self, modalities):
        x = self.feature_repr(modalities)
        return self.mlp(x)
    
class GCNLayer(nn.Module):
    """
    Task:
        Implements a single Graph Convolutional Layer with matrix multiplication.

    Inputs:
        - in_features (int): Number of input features per node.
        - out_features (int): Number of output features per node.

    Outputs:
        - Updated node features after applying graph convolution.
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def __repr__(self):
        return auto_repr(self)

    def forward(self, x, adjacency_matrix):
        """
        Task:
            Apply graph convolution using the adjacency matrix.

        Inputs:
            - x (Tensor): Node features [batch_size, num_nodes, in_features]
            - adjacency_matrix (Tensor): Normalized adjacency matrix [batch_size, num_nodes, num_nodes]

        Outputs:
            - Tensor: Updated features [batch_size, num_nodes, out_features]
        """
        out = torch.bmm(adjacency_matrix, x)
        out = self.linear(out)
        
        return out

class GCN(nn.Module):
    """
    Task:
        Graph Convolutional Network with dense layers for molecular representation learning.

    Inputs:
        - list_dims_gcn (List[int]): Dimensions for GCN layers.
        - list_dims_fc (List[int]): Dimensions for FC layers.
        - dropout (float): Dropout probability.
        - max_num_atom (int): Maximum number of atoms per graph for weighted pooling.

    Outputs:
        - Tensor: Graph-level representation [batch_size, output_dim]
    """
    def __init__(self, list_dims_gcn, list_dims_fc, dropout, max_num_atom):
        super(GCN, self).__init__()
        self.class_def = inspect.getsource(self.__class__)
        self.network_pyfile = network_pyfile
        self.max_num_atom = max_num_atom
        self.gcn_layers = nn.ModuleList([GCNLayer(list_dims_gcn[i], list_dims_gcn[i+1]) for i in range(len(list_dims_gcn) - 1)])
        self.gcn_layer_norms = nn.ModuleList([nn.LayerNorm(list_dims_gcn[i+1]) for i in range(len(list_dims_gcn) - 2)])
        
        self.fc_conn = nn.Linear(list_dims_gcn[-1], list_dims_fc[0])
        self.weighted_average = nn.Linear(max_num_atom, 1)
        
        self.fc_layers = nn.ModuleList([nn.Linear(list_dims_fc[i], list_dims_fc[i+1]) for i in range(len(list_dims_fc) - 1)])
        self.fc_layer_norms = nn.ModuleList([nn.LayerNorm(list_dims_fc[i+1]) for i in range(len(list_dims_fc) - 2)])
        self.dropout = nn.Dropout(dropout)
        
    def __repr__(self):
        return auto_repr(self)
        
    # ------------------------------------------------------------------
    # GCN feature extraction
    # ------------------------------------------------------------------
    def get_node_embeddings(self, x, adjacency_matrices, degree_matrices):
        """
        Task:
            Extract GCN-based molecular features using adjacency and degree info.

        Inputs:
            - x (Tensor): Node features [batch_size, num_nodes, in_features]
            - adjacency_matrices (Tensor): [batch_size, num_nodes, num_nodes]
            - degree_matrices (Tensor): [batch_size, num_nodes, num_nodes]

        Outputs:
            - Tensor: Pooled graph-level features [batch_size, gcn_out_dim]
        """
        adjacency_matrices = normalize_adjacency(adjacency_matrices, degree_matrices)
        
        for i in range(len(self.gcn_layers)):
            if i < len(self.gcn_layers)-1:
                x = F.relu(self.gcn_layers[i](x, adjacency_matrices))
                x = self.gcn_layer_norms[i](x)
                x = self.dropout(x)
            else:
                x = self.gcn_layers[i](x, adjacency_matrices)
        
        x = x.mean(dim=1)#self.weighted_average(x.transpose(-1, -2)).squeeze(-1)#
        
        return x
    
    def apply_fc_layers(self, x, layers, layer_norms, dropout):
        """
        Task:
            Apply a series of linear layers with normalization and dropout.

        Inputs:
            - x (Tensor): Input to FC stack.
            - layers, layer_norms, dropout: FC architecture components.

        Outputs:
            - Tensor: Final representation.
        """
        if len(layers):
            for i in range(len(layers)):
                if i < len(layers)-1:
                    x = F.relu(layers[i](x))
                    x = layer_norms[i](x)
                    x = dropout(x)
                else:
                    x = layers[i](x)
                
        return x
    
    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------
    def pool_graph(self, node_embeddings, mask=None):
        """
        Mean pooling with optional masking.

        node_embeddings : Tensor [B, N, F]
        mask : Tensor [B, N] (1 for valid nodes, 0 for padding)
        """
        if mask is None:
            return node_embeddings.mean(dim=1)

        mask = mask.unsqueeze(-1).float()
        summed = (node_embeddings * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)

        return summed / denom

    def forward(self, x, adjacency_matrices, degree_matrices):
        """
        Task:
            Forward pass through GCN and FC layers.

        Inputs:
            - x: Node features
            - adjacency_matrices, degree_matrices

        Outputs:
            - Tensor: Graph-level prediction [batch_size, output_dim]
        """
        x = self.get_node_embeddings(x, adjacency_matrices, degree_matrices)
        x = self.fc_conn(x)
        x = self.apply_fc_layers(x, self.fc_layers, self.fc_layer_norms, self.dropout)
        
        return x.squeeze()
        
    
class GCN_Connected(nn.Module):
    """
    Task:
        Augmented GCN that merges graph and auxiliary input `y` using learnable weighting.

    Inputs:
        - list_dims_gcn, list_dims_fc: GCN/FC architecture
        - dropout (float): Dropout probability
        - type_ (str): Merge type, default is 'concat'

    Outputs:
        - Tensor: Combined representation
    """
    def __init__(self,
                 list_dims_gcn,
                 list_dims_fc,
                 dropout,
                 external_input_dim,
                 n_fc_external = 1,
                 type_='concat'):
        super().__init__()
        self.class_def = inspect.getsource(self.__class__)
        self.network_pyfile = network_pyfile
        self.type_ = type_
        
        # GCN layers with LayerNorms for stability, except on last GCN layer
        self.gcn_layers = nn.ModuleList([GCNLayer(list_dims_gcn[i], list_dims_gcn[i+1]) for i in range(len(list_dims_gcn) - 1)])
        self.gcn_layer_norms = nn.ModuleList([nn.LayerNorm(list_dims_gcn[i+1]) for i in range(len(list_dims_gcn) - 2)])
        self.fc_conn = nn.Linear(list_dims_gcn[-1], list_dims_fc[0])
        
        self.fusion_dim = list_dims_fc[0]
        list_dims_y = list(np.linspace(external_input_dim, self.fusion_dim, n_fc_external+1, endpoint=True).astype(int))
        self.external_fc = make_mlp(list_dims_y, dropout=0.2, act_func='relu', norm_type='layer')
        
        # Fully Connected layers with LayerNorms for stability, except on last FC layer
        self.MLP = make_mlp(list_dims=list_dims_fc, dropout=dropout, act_func='relu', norm_type='layer')
        
        # Connected fusion parameters
        self.fusion_fc = nn.Linear(2*self.fusion_dim, self.fusion_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.param1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.param2 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
    def __repr__(self):
        return auto_repr(self)
        
    def get_features(self, x, adjacency_matrices, degree_matrices, y):
        """
        Task:
            Extract graph features and merge with auxiliary `y`.

        Inputs:
            - x: Node features
            - adjacency_matrices: Adjacency matrices
            - degree_matrices: Degree matrices
            - y: External input [batch_size, feature_dim]

        Outputs:
            - Tensor: Combined features [batch_size, fusion_dim]
        """
        adjacency_matrices = normalize_adjacency(adjacency_matrices, degree_matrices)
        
        for i in range(len(self.gcn_layers)):
            if i < len(self.gcn_layers)-1:
                x = F.relu(self.gcn_layers[i](x, adjacency_matrices))
                x = self.gcn_layer_norms[i](x)
                x = self.dropout(x)
            else:
                x = self.gcn_layers[i](x, adjacency_matrices)
        
        x = x.mean(dim=1)#self.weighted_average(x.transpose(-1, -2)).squeeze(-1)#
        x = self.fc_conn(x)
        y = self.external_fc(y)
        z = torch.concat((self.param1*x, self.param2*y), dim=-1)#torch.concat((x, y), dim=-1)#
        z = self.fusion_fc(z)
        
        return z
    
    def forward(self, x, adjacency_matrix, degree_matrix, y):
        """
        Task:
            Forward pass through GCN and FC layers, integrating auxiliary input.

        Inputs:
            - x: Node features
            - adjacency_matrix, degree_matrix
            - y: External feature

        Outputs:
            - Tensor: Final output [batch_size]
        """
        x = self.get_features(x, adjacency_matrix, degree_matrix, y)
        x = self.MLP(x, self.fc_layers, self.fc_layer_norms, self.dropout)
        
        return x.squeeze()  
    
    

## Multi Layer GCN Network
class RGCNConv(nn.Module):
    """
    Task:
        Relation-aware Graph Convolution Layer for multi-relational graphs.

    Inputs:
        - in_channels (int): Input node feature dim.
        - out_channels (int): Output node feature dim.
        - num_relations (int): Number of bond types or edge relations.

    Outputs:
        - Tensor: Transformed node features with all relation contributions.
    """
    def __init__(self, in_channels, out_channels, num_relations=4, bias=True):
        super(RGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.weights = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        self.self_loop_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()
        
    def __repr__(self):
        return auto_repr(self)

    def reset_parameters(self):
        """Task: Initialize parameters using Xavier initialization."""
        nn.init.xavier_uniform_(self.weights)
        nn.init.xavier_uniform_(self.self_loop_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adjacency_tensor):
        """
        Task:
            Perform multi-relational message passing.

        Inputs:
            - x (Tensor): Node features [batch_size, num_nodes, in_channels]
            - adjacency_tensor (Tensor): [batch_size, num_relations, num_nodes, num_nodes]

        Outputs:
            - Tensor: Updated node features [batch_size, num_nodes, out_channels]
        """
        batch_size, num_relations, num_nodes, _ = adjacency_tensor.size()
        out = torch.zeros(batch_size, num_nodes, self.out_channels, device=x.device)

        # Option - A (for loop)
        # ********************************************************************************
        for rel in range(num_relations):
            adj = adjacency_tensor[:, rel]  # Shape: [batch_size, num_nodes, num_nodes]

            h_rel = torch.matmul(x, self.weights[rel])  # Shape: [batch_size, num_nodes, out_channels]
            out += torch.matmul(adj, h_rel)
        # ********************************************************************************
            
        # # Option - B (vectorised) 
        # # ********************************************************************************
        # h = torch.matmul(x.unsqueeze(1),          # [B,1,N,Fin]
        #                  self.weights.unsqueeze(0) # [1,R,Fin,Fout]
        #                  )  # -> [B,R,N,Fout]
        # out = torch.matmul(adjacency_tensor, h)  # [B,R,N,Fout]
        # out = out.sum(dim=1)
        # # ********************************************************************************
            
        # ********************************************************************************
        out += torch.matmul(x, self.self_loop_weight)
        # out = out/num_relations
        # ********************************************************************************

        if self.bias is not None:
            out += self.bias

        return out

# class MRGCN(nn.Module):
#     """
#     Multi-Relational Graph Convolutional Network with optional attention pooling.
#     """
#     def __init__(self,
#                  list_dims_gcn,
#                  list_dims_fc,
#                  dropout,
#                  max_num_atom,
#                  num_relations=4,
#                  act_func='relu',
#                  act_func_gcn='relu',
#                  norm_type='layer',
#                  alpha=1.0,
#                  negative_slope=1e-2,
#                  use_attention=False):   # <--- new flag
#         super().__init__()
#         self.max_num_atom = max_num_atom
#         self.list_dims_gcn = list_dims_gcn
#         self.list_dims_fc = list_dims_fc
#         self.num_relations = num_relations
#         self.act_func = act_func
#         self.act_func_gcn = act_func_gcn
#         self.norm_type = norm_type
#         self.use_attention = use_attention

#         # --- GCN Layers ---
#         self.gcn_layers = nn.ModuleList([
#             RGCNConv(list_dims_gcn[i], list_dims_gcn[i + 1], num_relations=self.num_relations)
#             for i in range(len(list_dims_gcn) - 1)
#         ])
#         self.gcn_layer_norms = nn.ModuleList([
#             nn.LayerNorm(list_dims_gcn[i + 1]) for i in range(len(list_dims_gcn) - 2)
#         ])

#         self.dropout = nn.Dropout(dropout)
#         self.activation_func_gcn = activation_func(self.act_func_gcn, alpha=alpha, negative_slope=negative_slope)

#         # --- Attention Pooling (only used if flag=True) ---
#         self.att_pool = nn.Linear(list_dims_gcn[-1], 1)

#         # --- Fully connected head ---
#         if len(self.list_dims_fc) > 0:
#             self.fc_conn = nn.Linear(list_dims_gcn[-1], list_dims_fc[0])
#             self.input2repr = make_mlp(list_dims=list_dims_fc, dropout=dropout, act_func=act_func, norm_type=norm_type)
#         else:
#             self.fc_conn = nn.Identity()
#             self.input2repr = nn.Identity()

#     def __repr__(self):
#         return auto_repr(self)

#     def get_features(self, x, adjacency_tensor, degree_tensor):
#         """
#         Extract graph-level features with mean or attention pooling.
#         """
#         adjacency_tensor = normalize_adjacency(adjacency=adjacency_tensor,
#                                                degree=degree_tensor)

#         for i in range(len(self.gcn_layers)):
#             if i < len(self.gcn_layers) - 1:
#                 x = self.activation_func_gcn(self.gcn_layers[i](x, adjacency_tensor))
#                 x = self.gcn_layer_norms[i](x)
#                 x = self.dropout(x)
#             else:
#                 x = self.gcn_layers[i](x, adjacency_tensor)

#         # --- Conditional pooling ---
#         if self.use_attention:
#             att_weights = torch.softmax(self.att_pool(x), dim=1)  # [batch, num_nodes, 1]
#             x = torch.sum(att_weights * x, dim=1)                 # weighted sum
#         else:
#             x = x.mean(dim=1)                                     # simple mean pooling

#         self.feature_dim = x.shape[-1]
#         return x

#     def forward(self, feature_vector, adjacency_tensor, degree_tensor):
#         """
#         Forward pass from graph input to final prediction.
#         """
#         x = self.get_features(feature_vector, adjacency_tensor, degree_tensor)
#         x = self.fc_conn(x)
#         x = self.input2repr(x)
#         return x


class MRGCN(nn.Module):
    """
    Multi-Relational Graph Convolutional Network with optional attention-based pooling and dense prediction head.

    Args:
        list_dims_gcn (list[int]): Dimensions for GCN layers.
        list_dims_fc (list[int]): Dimensions for fully connected layers.
        dropout (float): Dropout rate.
        max_num_atom (int): Max number of atoms (used for pooling logic externally).
        num_relations (int): Number of edge types in the graph.
        act_func (str): Activation function for FC layers.
        act_func_gcn (str): Activation function for GCN layers.
        norm_type (str): Type of normalization ('layer', 'batch', etc.).
        alpha (float): Parameter for certain activation functions (e.g., ELU).
        negative_slope (float): Slope for LeakyReLU.
        use_attention (bool): Whether to use attention-based pooling.
        attn_type (str): Type of attention pooling ('self attention' or 'attention pool').
        attn_dim (int): Attention dimension.
        num_heads (int): Number of attention heads (for multi-head attention).

    Outputs:
        - Tensor: Final prediction or representation [batch_size, output_dim]
    """
    def __init__(self,
                 list_dims_gcn,
                 list_dims_fc,
                 max_num_atom,
                 dropout = 0.2,
                 num_relations=4,
                 act_func='relu',
                 act_func_gcn='relu',
                 norm_type='layer',
                 alpha=1.0,
                 negative_slope=1e-2,
                 use_attention=False,       # <--- flag for QK^T V
                 attn_type='self attention',# 'attention pool',#           
                 attn_dim=None,            # <--- dimension of attention projection
                 num_heads=1):             # <--- optional multi-head attention
        super().__init__()
        self.max_num_atom = max_num_atom
        self.list_dims_gcn = list_dims_gcn
        self.list_dims_fc = list_dims_fc
        self.num_relations = num_relations
        self.class_def = inspect.getsource(self.__class__)
        self.network_pyfile = network_pyfile
        self.act_func = act_func
        self.act_func_gcn = act_func_gcn
        self.norm_type = norm_type
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.alpha = alpha
        self.negative_slope = negative_slope
        self.use_attention = use_attention
        self.attn_type = attn_type

        # GCN Layers
        self.gcn_layers = nn.ModuleList([RGCNConv(list_dims_gcn[i], list_dims_gcn[i + 1], num_relations=self.num_relations) for i in range(len(list_dims_gcn) - 1)])
        self.gcn_layer_norms = nn.ModuleList([nn.LayerNorm(list_dims_gcn[i + 1]) for i in range(len(list_dims_gcn) - 2)])
        
        # --- QK^T V Attention Pooling ---
        hidden_dim = list_dims_gcn[-1]
        attn_dim = attn_dim or hidden_dim
        if self.use_attention and self.attn_type=='self attention':
            self.W_Q = nn.Linear(hidden_dim, attn_dim * num_heads)
            self.W_K = nn.Linear(hidden_dim, attn_dim * num_heads)
            self.W_V = nn.Linear(hidden_dim, attn_dim * num_heads)
            self.W_out = nn.Linear(attn_dim * num_heads, hidden_dim)
        if self.use_attention and self.attn_type=='attention pool':
            # self.att_pool = nn.Linear(hidden_dim, 1)
            self.att_pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                          nn.Tanh(), nn.Linear(hidden_dim // 2, 1)
                                          )

        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.activation_func_gcn = activation_func(self.act_func_gcn, alpha=alpha, negative_slope=negative_slope)
        if len(self.list_dims_fc) >0:
            self.fc_conn = nn.Linear(list_dims_gcn[-1], list_dims_fc[0])
            self.input2repr = make_mlp(list_dims=list_dims_fc, dropout=dropout, act_func=act_func, norm_type=norm_type)
        else:
            self.fc_conn = nn.Identity()
            self.input2repr = nn.Identity()
        
    def __repr__(self):
        return auto_repr(self)

    def get_features(self, feature_vector, adjacency_tensor, degree_tensor, feature_mask=1.0, adjacency_mask=1.0):
        """
        Task:
            Extract graph-level features from multi-relational input.

        Inputs:
            - x (Tensor): Node features
            - adjacency_tensor (Tensor): Edge-type-specific adjacency matrices
            - degree_tensor (Tensor): Node degree matrices

        Outputs:
            - Tensor: Aggregated graph features [batch_size, gcn_out_dim]
        """
        feature_vector = feature_vector * feature_mask
        adjacency_tensor = adjacency_tensor * adjacency_mask
        degree_tensor = degree_tensor * adjacency_mask

        # import time
        # time.sleep(1)
        # check_tensor(feature_vector, "feature_vector")
        # check_tensor(adjacency_tensor, "adjacency_tensor")
        # check_tensor(degree_tensor, "degree_tensor")

        adjacency_tensor = normalize_adjacency(adjacency=adjacency_tensor,
                                               degree=degree_tensor
                                               )
        x = feature_vector
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, adjacency_tensor)
            if i < len(self.gcn_layers) - 1:
                x = self.activation_func_gcn(x)
                x = self.gcn_layer_norms[i](x)
                x = self.dropout(x)

        # --- Conditional pooling ---
        if self.use_attention and self.attn_type=='self attention':
            B, N, D = x.shape  # batch, num_nodes, hidden_dim
            H = self.num_heads
            d_k = D // H

            Q = self.W_Q(x).view(B, N, H, -1).transpose(1, 2)  # [B, H, N, d_k]
            K = self.W_K(x).view(B, N, H, -1).transpose(1, 2)  # [B, H, N, d_k]
            V = self.W_V(x).view(B, N, H, -1).transpose(1, 2)  # [B, H, N, d_k]

            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # [B, H, N, N]
            attn_weights = torch.softmax(attn_scores, dim=-1)                  # attention matrix
            x = torch.matmul(attn_weights, V)                                  # [B, H, N, d_k]

            x = x.mean(dim=2)                                                  # pool over nodes
            x = x.transpose(1, 2).contiguous().view(B, -1)                     # concat heads
            x = x.reshape(B, -1)
            x = self.W_out(x)                                                  # [B, D]
        elif self.use_attention and self.attn_type=='attention pool':
            att_weights = torch.softmax(self.att_pool(x), dim=1)  # [batch, num_nodes, 1]
            x = torch.sum(att_weights * x, dim=1)
        else:
            x = x.mean(dim=1)  # simple mean pooling
            
        self.feature_dim = x.shape[-1]
        return x

    def forward(self, feature_vector, adjacency_tensor, degree_tensor, feature_mask=1.0, adjacency_mask=1.0):
        """
        Task:
            Forward pass from graph input to final prediction via GCN and FC layers.

        Inputs:
            - x: Node features
            - adjacency_tensor: Multi-relational adjacency
            - degree_tensor: Node degree tensor
            - feature_mask (float or Tensor): Mask to apply on features
            - adjacency_mask (float or Tensor): Mask to apply on adjacency

        Outputs:
            - Tensor: Final prediction/representation
        """
        x = self.get_features(feature_vector, adjacency_tensor, degree_tensor,
                              feature_mask=feature_mask, adjacency_mask=adjacency_mask)
        # x = x.mean(dim=1).unsqueeze(1)
        x = self.fc_conn(x)
        x = self.input2repr(x)
        return x#.squeeze()
    
    
class GINConv(nn.Module):
    """
    Task:
        Graph Isomorphism Network (GIN) convolution layer.

    Inputs:
        - in_channels (int): Input node feature dimension.
        - out_channels (int): Output node feature dimension.
        - eps (float): Initial epsilon value.
        - train_eps (bool): Whether epsilon is learnable.

    Outputs:
        - Tensor: Updated node features.
    """
    def __init__(self, in_channels, out_channels, eps=0.0, train_eps=True):
        super(GINConv, self).__init__()

        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.mlp = nn.Sequential(nn.Linear(in_channels, out_channels),
                                 nn.ReLU(),
                                 nn.Linear(out_channels, out_channels)
                                 )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, adjacency):
        """
        Task:
            Perform GIN message passing.

        Inputs:
            - x (Tensor): Node features [batch_size, num_nodes, in_channels]
            - adjacency (Tensor): Adjacency matrix [batch_size, num_nodes, num_nodes]

        Outputs:
            - Tensor: Updated node features [batch_size, num_nodes, out_channels]
        """
        
        # Sum aggregation of neighbors
        neigh_agg = torch.matmul(adjacency, x)

        # (1 + eps) * x + sum_j x_j
        out = (1.0 + self.eps) * x + neigh_agg

        # Apply MLP
        out = self.mlp(out)

        return out
    

class GIN(nn.Module):
    """
    Graph Isomorphism Network (GIN) for graph-level representation learning.

    Args:
        list_dims_gin (list[int]): Dimensions of GIN layers.
            Example: [in_dim, hidden1, hidden2]
        list_dims_fc (list[int]): Dimensions for graph-level MLP head.
            Example: [hidden2, 128, out_dim]
        dropout (float): Dropout probability.
        max_num_atom (int): Maximum number of nodes (kept for compatibility).
        act_func (str): Activation for FC layers.
        act_func_gin (str): Activation for GIN layers.
        norm_type (str): Normalization type used in MLP.
        eps (float): Initial epsilon for GIN aggregation.
        train_eps (bool): Whether epsilon is learnable.
    """

    def __init__(
        self,
        list_dims_gin,
        list_dims_fc,
        dropout=0.0,
        max_num_atom=None,
        act_func='relu',
        act_func_gin='relu',
        norm_type='layer',
        eps=torch.pi,
        train_eps=False,
    ):
        super().__init__()

        self.class_def = inspect.getsource(self.__class__)
        self.max_num_atom = max_num_atom
        self.act_func = act_func
        self.act_func_gin = act_func_gin
        self.norm_type = norm_type
        self.eps = eps
        self.train_eps = train_eps
        self.dropout = nn.Dropout(dropout)
        self.activation_func_gin = activation_func(act_func_gin)
        self.readout_dim = sum(list_dims_gin[1:])
        # list_dims_fc = [self.readout_dim] + list_dims_fc if list_dims_fc else []

        # --------------------------------------------------
        # GIN layers
        # --------------------------------------------------
        self.gin_layers = nn.ModuleList([GINConv(list_dims_gin[i], list_dims_gin[i + 1], eps=eps, train_eps=train_eps) for i in range(len(list_dims_gin) - 1)])
        self.gin_norms = nn.ModuleList([nn.LayerNorm(list_dims_gin[i + 1]) for i in range(len(list_dims_gin) - 2)])
        self.mixer_fc = nn.Linear(self.readout_dim, self.readout_dim)
        self.fc_conn = nn.Linear(self.readout_dim, list_dims_fc[0])
        # --------------------------------------------------
        # Graph-level prediction head
        # --------------------------------------------------
        if list_dims_fc and len(list_dims_fc) > 1:
            self.head = make_mlp(list_dims=list_dims_fc, dropout=dropout, act_func=act_func, norm_type=norm_type)
        else:
            self.head = nn.Identity()

    # ------------------------------------------------------
    # Representation
    # ------------------------------------------------------
    def __repr__(self):
        return auto_repr(self)

    # ------------------------------------------------------
    # Node embeddings -> graph embedding
    # ------------------------------------------------------
    def get_features( self,
                     feature_vector: torch.Tensor,
                     adjacency_tensor: torch.Tensor,
                     degree_tensor: torch.Tensor = None,
                     feature_mask=1.0,
                     adjacency_mask=1.0
                     ) -> torch.Tensor:
        """
        Task:
            Extract GIN-based node embeddings.

        Inputs:
            - feature_vector (Tensor): Node features [batch_size, num_nodes, in_features]
            - adjacency_tensor (Tensor): Adjacency matrices [batch_size, num_nodes, num_nodes]
            - degree_tensor (Tensor): Degree matrices [batch_size, num_nodes, num_nodes]

        Outputs:
            - Tensor: Graph-level features after pooling [batch_size, gin_out_dim]
        Inputs:
            feature_vector: [B, N, F]
            adjacency_tensor: [B, R, N, N] or [B, N, N]
        """
        
        feature_vector = feature_vector * feature_mask
        adjacency_tensor = adjacency_tensor * adjacency_mask
        degree_tensor = degree_tensor * adjacency_mask

        # If multi-relational adjacency → collapse relations
        if adjacency_tensor.dim() == 4:
            adjacency_matrices = adjacency_tensor.sum(dim=1)
        else:
            adjacency_matrices = adjacency_tensor
        x = feature_vector
        
        layer_outputs = []
        for i, layer in enumerate(self.gin_layers):

            x = layer(x, adjacency_matrices)
            # hidden layers only
            if i < len(self.gin_layers) - 1:
                x = self.activation_func_gin(x)
                x = self.gin_norms[i](x)
                x = self.dropout(x)
            layer_outputs.append(x)
            
        pooled = [h.sum(dim=1) for h in layer_outputs]
        graph_repr = torch.cat(pooled, dim=-1)
        x = F.relu(self.mixer_fc(graph_repr))
        x = self.fc_conn(x)
        
        self.feature_dim = x.shape[-1]

        return x

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------
    def forward(
                self,
                feature_vector: torch.Tensor,
                adjacency_tensor: torch.Tensor,
                degree_tensor: torch.Tensor = None,
                feature_mask=1.0,
                adjacency_mask=1.0
            ) -> torch.Tensor:

        x = self.get_features(feature_vector, adjacency_tensor, degree_tensor, feature_mask, adjacency_mask)
        x = self.head(x)

        return x


class GIN(nn.Module):
    """
    Graph Isomorphism Network (GIN) for graph-level representation learning.

    Args:
        list_dims_gin (list[int]): Dimensions of GIN layers.
            Example: [in_dim, hidden1, hidden2]
        list_dims_fc (list[int]): Dimensions for graph-level MLP head.
            Example: [hidden2, 128, out_dim]
        dropout (float): Dropout probability.
        max_num_atom (int): Maximum number of nodes (kept for compatibility).
        act_func (str): Activation for FC layers.
        act_func_gin (str): Activation for GIN layers.
        norm_type (str): Normalization type used in MLP.
        eps (float): Initial epsilon for GIN aggregation.
        train_eps (bool): Whether epsilon is learnable.
    """

    def __init__(
        self,
        list_dims_gin,
        list_dims_fc,
        dropout=0.0,
        max_num_atom=None,
        act_func='relu',
        act_func_gin='relu',
        norm_type='layer',
        eps=torch.pi,
        train_eps=False,
    ):
        super().__init__()

        self.class_def = inspect.getsource(self.__class__)
        self.list_dims_gin = list_dims_gin
        self.list_dims_fc = list_dims_fc
        self.max_num_atom = max_num_atom
        self.act_func = act_func
        self.act_func_gin = act_func_gin
        self.norm_type = norm_type
        self.eps = eps
        self.train_eps = train_eps
        self.dropout = nn.Dropout(dropout)
        self.activation_func_gin = activation_func(act_func_gin)
        self.readout_dim = sum(list_dims_gin[1:])
        # list_dims_fc = [self.readout_dim] + list_dims_fc if list_dims_fc else []

        # --------------------------------------------------
        # GIN layers
        # --------------------------------------------------
        self.gin_layers = nn.ModuleList([GINConv(list_dims_gin[i], list_dims_gin[i + 1], eps=eps, train_eps=train_eps) for i in range(len(list_dims_gin) - 1)])
        self.gin_norms = nn.ModuleList([nn.LayerNorm(list_dims_gin[i + 1]) for i in range(len(list_dims_gin) - 2)])
        self.mixer_fc = nn.Sequential(nn.Linear(self.readout_dim, self.readout_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.readout_dim, list_dims_gin[-1])
                                      )
        # --------------------------------------------------
        # Graph-level prediction head
        # --------------------------------------------------
        self.fc_conn = nn.Linear(list_dims_gin[-1], list_dims_fc[0])
        if list_dims_fc and len(list_dims_fc) > 1:
            self.head = make_mlp(list_dims=list_dims_fc, dropout=dropout, act_func=act_func, norm_type=norm_type)
        else:
            self.head = nn.Identity()

    # ------------------------------------------------------
    # Representation
    # ------------------------------------------------------
    def __repr__(self):
        return auto_repr(self)
    
    # ------------------------------------------------------
    # Node embeddings -> graph embedding
    # ------------------------------------------------------
    def get_features(self,
                     feature_vector: torch.Tensor,
                     adjacency_tensor: torch.Tensor,
                     degree_tensor: torch.Tensor = None,
                     feature_mask=1.0,
                     adjacency_mask=1.0
                     ) -> torch.Tensor:
        """
        Task:
            Extract GIN-based node embeddings.

        Inputs:
            - feature_vector (Tensor): Node features [batch_size, num_nodes, in_features]
            - adjacency_tensor (Tensor): Adjacency matrices [batch_size, num_nodes, num_nodes]
            - degree_tensor (Tensor): Degree matrices [batch_size, num_nodes, num_nodes]

        Outputs:
            - Tensor: Graph-level features after pooling [batch_size, gin_out_dim]
        Inputs:
            feature_vector: [B, N, F]
            adjacency_tensor: [B, R, N, N] or [B, N, N]
        """
        
        feature_vector = feature_vector * feature_mask
        adjacency_tensor = adjacency_tensor * adjacency_mask
        degree_tensor = degree_tensor * adjacency_mask

        # If multi-relational adjacency → collapse relations
        if adjacency_tensor.dim() == 4:
            adjacency_matrices = adjacency_tensor.sum(dim=1)
        else:
            adjacency_matrices = adjacency_tensor
        x = feature_vector
        
        layer_outputs = []
        for i, layer in enumerate(self.gin_layers):

            x = layer(x, adjacency_matrices)
            # hidden layers only
            if i < len(self.gin_layers) - 1:
                x = self.activation_func_gin(x)
                x = self.gin_norms[i](x)
                x = self.dropout(x)
                # print(f'layer {i}: ', x)
            layer_outputs.append(x)
            
        pooled = [h.sum(dim=1) for h in layer_outputs]
        graph_repr = torch.cat(pooled, dim=-1)
        x = self.mixer_fc(graph_repr)
        
        self.feature_dim = x.shape[-1]

        return x

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------
    def forward(
                self,
                feature_vector: torch.Tensor,
                adjacency_tensor: torch.Tensor,
                degree_tensor: torch.Tensor = None,
                feature_mask=1.0,
                adjacency_mask=1.0
            ) -> torch.Tensor:

        x = self.get_features(feature_vector, adjacency_tensor, degree_tensor, feature_mask, adjacency_mask)
        x = self.fc_conn(x)
        x = self.head(x)

        return x
    
class GIN_FU(GIN):
    """
    Graph Isomorphism Network (GIN) for graph-level representation learning.

    Args:
        list_dims_gin (list[int]): Dimensions of GIN layers.
            Example: [in_dim, hidden1, hidden2]
        list_dims_fc (list[int]): Dimensions for graph-level MLP head.
            Example: [hidden2, 128, out_dim]
        dropout (float): Dropout probability.
        max_num_atom (int): Maximum number of nodes (kept for compatibility).
        act_func (str): Activation for FC layers.
        act_func_gin (str): Activation for GIN layers.
        norm_type (str): Normalization type used in MLP.
        eps (float): Initial epsilon for GIN aggregation.
        train_eps (bool): Whether epsilon is learnable.
    """

    def __init__(
        self,
        list_dims_gin,
        list_dims_fc,
        dropout=0.0,
        max_num_atom=None,
        act_func='relu',
        act_func_gin='relu',
        norm_type='layer',
        eps=torch.pi,
        train_eps=False,
    ):
        super().__init__(list_dims_gin=list_dims_gin,
                         list_dims_fc=list_dims_fc,
                         dropout=dropout,
                         max_num_atom=max_num_atom,
                         act_func=act_func,
                         act_func_gin=act_func_gin,
                         norm_type=norm_type,
                         eps=eps,
                         train_eps=train_eps
                        )
        
    def forward(self,
                feature_vector: torch.Tensor,
                adjacency_tensor: torch.Tensor,
                degree_tensor: torch.Tensor = None,
                feature_mask=1.0,
                adjacency_mask=1.0
            ) -> torch.Tensor:
        
        x = super().forward(feature_vector, adjacency_tensor, degree_tensor, feature_mask, adjacency_mask)
        x = torch.sigmoid(x)
        
        return x


class GIN_SINKHORN(nn.Module):
    """
    GIN with:
        - Residual connections per layer
        - Dense layer-wise readout (JK-style)
        - Sinkhorn-normalized transport mixing
    """

    def __init__(
        self,
        list_dims_gin,
        list_dims_fc,
        dropout=0.0,
        act_func='relu',
        act_func_gin='relu',
        norm_type='layer',
        eps=0.0,
        train_eps=False,
        sinkhorn_iters=20,
        use_sinkhorn=False,
        transport_residual=True,
    ):
        super().__init__()

        self.list_dims_gin = list_dims_gin
        self.list_dims_fc = list_dims_fc
        self.act_func = act_func
        self.act_func_gin = act_func_gin
        self.norm_type = norm_type
        self.eps = eps
        self.train_eps = train_eps
        self.dropout = nn.Dropout(dropout)
        self.sinkhorn_iters = sinkhorn_iters
        self.use_sinkhorn = use_sinkhorn
        self.transport_residual = transport_residual

        # --------------------------------------------------
        # GIN Layers
        # --------------------------------------------------
        self.gin_layers = nn.ModuleList()
        self.gin_norms = nn.ModuleList()
        self.res_projections = nn.ModuleList()

        for i in range(len(list_dims_gin) - 1):
            in_dim = list_dims_gin[i]
            out_dim = list_dims_gin[i + 1]

            self.gin_layers.append(GINConv(in_dim, out_dim, eps=eps, train_eps=train_eps))
            if i < len(list_dims_gin) - 2:
                self.gin_norms.append(nn.LayerNorm(out_dim))

            # Residual projection
            if in_dim != out_dim:
                self.res_projections.append(nn.Linear(in_dim, out_dim))
            else:
                self.res_projections.append(nn.Identity())

        self.activation_func_gin = activation_func(act_func_gin)

        # --------------------------------------------------
        # Dense Readout
        # --------------------------------------------------
        self.readout_dim = sum(list_dims_gin[1:])

        # Sinkhorn transport weight (log-space parameter)
        if use_sinkhorn:
            self.transport_weight = nn.Parameter(torch.randn(self.readout_dim, self.readout_dim) * 0.1)
            
        self.post_transport = nn.Sequential(nn.Linear(self.readout_dim, list_dims_gin[-1]), nn.SiLU())
        # self.post_transport = nn.Sequential(nn.Linear(self.readout_dim, list_dims_gin[-1]),
        #                                     nn.ReLU(),
        #                                     nn.Dropout(dropout)
        #                                     )

        # --------------------------------------------------
        # Graph-level Head
        # --------------------------------------------------
        self.fc_conn = nn.Linear(list_dims_gin[-1], list_dims_fc[0])

        if list_dims_fc and len(list_dims_fc) > 1:
            self.head = make_mlp(list_dims=list_dims_fc, dropout=dropout, act_func=act_func, norm_type=norm_type)
        else:
            self.head = nn.Identity()
            
            
    def __repr__(self):
        return auto_repr(self)

    # ------------------------------------------------------
    # Sinkhorn normalization (weights)
    # ------------------------------------------------------
    def sinkhorn_normalization(self, log_alpha, eps=1e-6):
        Q = torch.exp(log_alpha)

        for _ in range(self.sinkhorn_iters):
            Q = Q / (Q.sum(dim=1, keepdim=True) + eps)  # row norm
            Q = Q / (Q.sum(dim=0, keepdim=True) + eps)  # col norm

        return Q

    # ------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------
    def get_features(self,
                     feature_vector: torch.Tensor,
                     adjacency_tensor: torch.Tensor,
                     degree_tensor: torch.Tensor = None,
                     feature_mask=1.0,
                     adjacency_mask=1.0
                     ):

        if adjacency_tensor.dim() == 4:
            adjacency_matrices = adjacency_tensor.sum(dim=1)
        else:
            adjacency_matrices = adjacency_tensor

        x = feature_vector
        layer_outputs = []

        for i, layer in enumerate(self.gin_layers):

            identity = x
            out = layer(x, adjacency_matrices)

            # Residual connection
            identity = self.res_projections[i](identity)
            x = out + identity

            if i < len(self.gin_layers) - 1:
                x = self.activation_func_gin(x)
                x = self.gin_norms[i](x)
                x = self.dropout(x)

            layer_outputs.append(x)

        # JK-style dense pooling
        pooled = [h.sum(dim=1) for h in layer_outputs]
        graph_repr = torch.cat(pooled, dim=-1)

        # --------------------------------------------------
        # Sinkhorn transport mixing
        # --------------------------------------------------
        if self.use_sinkhorn:
            W_ds = self.sinkhorn_normalization(self.transport_weight)
            transported = graph_repr @ W_ds

            if self.transport_residual:
                transported = transported + graph_repr

            x = self.post_transport(transported)
            
        else:
            x = self.post_transport(graph_repr)
            
        self.feature_dim = x.shape[-1]

        return x

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------
    def forward(self,
                feature_vector: torch.Tensor,
                adjacency_tensor: torch.Tensor,
                degree_tensor: torch.Tensor = None,
                feature_mask=1.0,
                adjacency_mask=1.0):

        x = self.get_features(feature_vector, adjacency_tensor)
        x = self.fc_conn(x)
        x = self.head(x)

        return x
    

class GIN_PyG(nn.Module):
    def __init__(
        self,
        list_dims_gin,
        list_dims_fc,
        dropout=0.0,
        eps=0.0,
        train_eps=False,
    ):
        """
        list_dims_gin: [in_dim, hidden1, hidden2, ...]
        list_dims_fc:  [hidden_last, fc1, ..., out_dim]
        """

        super().__init__()

        self.list_dims_gin = list_dims_gin
        self.list_dims_fc = list_dims_fc
        self.eps = eps
        self.train_eps = train_eps
        self.class_def = inspect.getsource(self.__class__)
        self.dropout = nn.Dropout(dropout)

        # --------------------------------------------------
        # GIN Layers (PyG implementation)
        # --------------------------------------------------
        self.gin_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(len(list_dims_gin) - 1):

            mlp = nn.Sequential(
                nn.Linear(list_dims_gin[i], list_dims_gin[i + 1]),
                nn.Tanh(),
                nn.Linear(list_dims_gin[i + 1], list_dims_gin[i + 1]),
            )

            conv = PyG_GINConv(nn=mlp, eps=eps, train_eps=train_eps)
            self.gin_layers.append(conv)

            if i < len(list_dims_gin) - 2:
                self.norm_layers.append(LayerNorm(list_dims_gin[i + 1]))

        # --------------------------------------------------
        # Graph-level head
        # --------------------------------------------------
        self.fc_head = nn.Sequential()
        list_dims_fc = [list_dims_gin[-1], ] + list_dims_fc

        if len(list_dims_fc) > 1:
            layers = []
            for i in range(len(list_dims_fc) - 1):
                layers.append(nn.Linear(list_dims_fc[i], list_dims_fc[i + 1]))
                if i < len(list_dims_fc) - 2:
                    layers.append(LayerNorm(list_dims_fc[i + 1]))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
            self.fc_head = nn.Sequential(*layers)
        else:
            self.fc_head = nn.Identity()
            
    def __repr__(self):
        return auto_repr(self)
    
    def get_features(self, x, edge_index, batch, **kwargs):
        for i, conv in enumerate(self.gin_layers):

            x = conv(x, edge_index)

            if i < len(self.gin_layers) - 1:
                x = F.relu(x)
                x = self.norm_layers[i](x)
                x = self.dropout(x)

        # Graph-level pooling
        x = global_add_pool(x, batch)
        self.feature_dim = x.shape[-1]

        return x

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------
    def forward(self, x, edge_index, batch, **kwargs):

        x = self.get_features(x, edge_index, batch)
        # FC head
        x = self.fc_head(x)

        return x
    
    
class ChemBERTaRegressorroberta(RobertaPreTrainedModel):
    """
    Task:
        A regression model built on top of ChemBERTa (RobertaModel) for predicting molecular properties.
    """
    def __init__(self, config, list_dims, dropout=0.2):
        """
        Task:
            Initializes the ChemBERTaRegressorroberta model with a Roberta backbone and MLP head.

        Input:
            config (PretrainedConfig): Configuration object for RobertaModel.
            list_dims (list[int]): Dimensions for the MLP regressor head.
            dropout (float): Dropout probability for regularization.

        Output:
            None
        """
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.regressor = self.make_mlp(list_dims=list_dims, dropout=dropout)
        self.init_weights()
        
    def make_mlp(self, list_dims, dropout):
        """
        Task:
            Constructs a sequential MLP regressor.

        Input:
            list_dims (list[int]): Dimensions of the layers in the MLP.
            dropout (float): Dropout probability between layers.

        Output:
            nn.Sequential: An MLP with linear, normalization, activation, and dropout layers.
        """
        layers = nn.ModuleList([])
        for i in range(len((list_dims)) - 1):
            layers.append(nn.Linear(list_dims[i], list_dims[i+1]))
            if i < len(list_dims) - 2:
                layers.extend([nn.LayerNorm(list_dims[i+1]), nn.ReLU(), nn.Dropout(dropout)])
                
        return nn.Sequential(*layers)
    
    def embeddings(self, input_ids, attention_mask=None):
        """
        Task:
            Extracts the CLS embedding from Roberta's output.

        Input:
            input_ids (Tensor): Tokenized input IDs of shape [batch_size, seq_len].
            attention_mask (Tensor, optional): Attention mask indicating valid tokens.

        Output:
            Tensor: CLS token embedding of shape [batch_size, hidden_size].
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0]  # CLS token
        
        return x

    def forward(self, input_ids, attention_mask=None):
        """
        Task:
            Performs a forward pass through the model to predict the regression value.

        Input:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor, optional): Attention mask.

        Output:
            Tensor: Predicted output of shape [batch_size, output_dim].
        """
        x = self.embeddings(input_ids=input_ids, attention_mask=attention_mask)
        x = self.regressor(x)
            
        return x
    

class ChemBERTa(nn.Module):
    """
    Task:
        ChemBERTa-based regressor for molecular property prediction using AutoModel.
    """
    def __init__(self,
                 model_name: str,
                 list_dims: list[int],
                 dropout: float = 0.2,
                 act_func: str = 'relu',
                 norm_type: str = 'layer',
                 ):
        """
        Task:
            Initialize model with a pretrained transformer and a regression MLP.

        Input:
            model_name (str): Huggingface model identifier (e.g., 'seyonec/ChemBERTa').
            list_dims (list[int]): Dimensions for the MLP head.
            dropout (float): Dropout probability.

        Output:
            None
        """
        super().__init__()
        self.class_def = collect_class_definitions(self)
        self.network_pyfile=network_pyfile
        
        self.model_name = model_name
        self.list_dims = list_dims
        self.dropout = dropout
        self.act_func = act_func
        self.norm_type = norm_type
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True
        self.config.output_attentions = True
        self.chemberta = AutoModel.from_pretrained(model_name, config=self.config)
        
        self.hidden_size = self.config.hidden_size
        self.list_dims = [self.hidden_size, ] + list_dims
        self.regressor = make_mlp(list_dims=self.list_dims, dropout=self.dropout, act_func=self.act_func, norm_type=self.norm_type)

        
    def __repr__(self):
        """
        Task:
            String representation of the model including its name and configuration.

        Output:
            str: Model name and configuration.
        """
        return auto_repr(self)
    
    def token_embeddings(self, input_ids, attention_mask, **kwargs):
        
        outputs = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        token_embed = outputs.last_hidden_state  # [B, T, H]

        return token_embed, outputs

    def embeddings(self, input_ids, attention_mask, **kwargs):
        """
        Task:
            Compute mean-pooled embedding over valid token positions.

        Input:
            input_ids (Tensor): Input token IDs of shape [B, T].
            attention_mask (Tensor): Attention mask of shape [B, T].

        Output:
            Tensor: Mean pooled embedding of shape [B, H].
        """
        token_embed, _ = self.token_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        x = token_embed[:, 0]  # CLS token
        
        return x
    
    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Task:
            Perform forward pass for regression prediction.

        Input:
            input_ids (Tensor): Tokenized inputs.
            attention_mask (Tensor): Valid attention positions.

        Output:
            Tensor: Regression prediction.
        """
        x = self.embeddings(input_ids=input_ids, attention_mask=attention_mask)
        x = self.regressor(x)
        
        return x
    

class ChemBERTaRegressor(ChemBERTa):
    """
    Task:
        ChemBERTa-based regressor for molecular property prediction using AutoModel.
    """
    def __init__(self,
                 model_name: str,
                 list_dims: list[int],
                 dropout: float = 0.2,
                 act_func: str = 'relu',
                 norm_type: str = 'layer',
                 ):
        """
        Task:
            Initialize model with a pretrained transformer and a regression MLP.

        Input:
            model_name (str): Huggingface model identifier (e.g., 'seyonec/ChemBERTa').
            list_dims (list[int]): Dimensions for the MLP head.
            dropout (float): Dropout probability.

        Output:
            None
        """
        super().__init__(model_name=model_name,
                         list_dims=list_dims,
                         dropout=dropout,
                         act_func=act_func,
                         norm_type=norm_type
                         )
        self.model_name = model_name
        self.dropout = dropout
        self.act_func = act_func
        self.list_dims = list_dims
        self.norm_type = norm_type

        self.hidden_size = self.config.hidden_size
        list_dims = [self.hidden_size, ] + list_dims
        self.regressor = make_mlp(list_dims=list_dims, dropout=self.dropout, act_func=self.act_func, norm_type=self.norm_type)
        
    
    def embeddings(self, input_ids, attention_mask, **kwargs):
        """
        Task:
            Compute mean-pooled embedding over valid token positions.

        Input:
            input_ids (Tensor): Input token IDs of shape [B, T].
            attention_mask (Tensor): Attention mask of shape [B, T].

        Output:
            Tensor: Mean pooled embedding of shape [B, H].
        """
        token_embed, _ = self.token_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        masked_embed = (token_embed * attention_mask.unsqueeze(-1)).sum(1)
        denom = attention_mask.sum(1, keepdim=True).clamp(min=1e-6)
        x = masked_embed / denom  # mean pooling
        
        return x
    

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Task:
            Perform forward pass for regression prediction.

        Input:
            input_ids (Tensor): Tokenized inputs.
            attention_mask (Tensor): Valid attention positions.

        Output:
            Tensor: Regression prediction.
        """
        x = self.embeddings(input_ids=input_ids, attention_mask=attention_mask)
        x = self.regressor(x)
        return x
    


class ChemBERTaRegressorwithAttention(ChemBERTaRegressor):
    """
    Task:
        ChemBERTa-based regressor with PyTorch's native multi-head attention for molecular property prediction.
    """
    def __init__(self,
                 model_name: str,
                 list_dims: list[int],
                 dropout: float = 0.2,
                 act_func: str = 'relu',
                 norm_type: str = 'layer',
                 num_heads: int = 4,
                 mode: str = 'dot',  # Ignored for native attention
                 max_position_embeddings: int = 0
                 ):
        """
        Initialize model with pretrained ChemBERTa and native PyTorch multi-head attention.
        """
        self.model_name = model_name
        self.list_dims = list_dims
        self.dropout = dropout
        self.act_func = act_func
        self.norm_type = norm_type
        self.num_heads = num_heads
        self.mode = mode
        self.max_position_embeddings = max_position_embeddings
        
        super().__init__(
            model_name=model_name,
            list_dims=list_dims,
            dropout=dropout,
            act_func=act_func,
            norm_type=norm_type
        )

        hidden_dim = self.config.hidden_size
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)

    def token_embeddings(self, input_ids, attention_mask):
        token_embed, _ = super().token_embeddings(input_ids=input_ids, attention_mask=attention_mask)  # [B, T, H]

        if self.max_position_embeddings > 0:
            B, T = input_ids.size()
            position_ids = torch.arange(T, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(B, T)
            pos_embed = self.position_embeddings(position_ids)
            token_embed = token_embed + pos_embed  # [B, T, H]

        return token_embed

    def embeddings(self, input_ids, attention_mask, **kwargs):
        embed, _ = self.token_embeddings(input_ids=input_ids, attention_mask=attention_mask)  # [B, T, H]

        # Convert attention_mask to key_padding_mask (True = ignore, False = attend)
        key_padding_mask = ~attention_mask.bool()  # [B, T]

        # Apply multi-head self-attention (query=key=value=embed)
        attn_output, _ = self.attn(embed, embed, embed, key_padding_mask=key_padding_mask)  # [B, T, H]

        # Pool over the sequence dimension (mean pooling over valid tokens)
        attn_output = attn_output.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)  # [B, T, H]
        lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(1)  # [B, 1]
        pooled = attn_output.sum(dim=1) / lengths  # [B, H]

        return pooled

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Perform forward pass for regression prediction using native multi-head attention.
        """
        x = self.embeddings(input_ids=input_ids, attention_mask=attention_mask)
        x = self.regressor(x)
        return x
    
    
class ChemBERTaRegressorwithMultiheadAttention(ChemBERTaRegressor):
    """
    Task:
        ChemBERTa-based regressor with multi-head attention for molecular property prediction.
    """
    def __init__(self,
                 model_name: str,
                 list_dims: list[int],
                 dropout: float = 0.2,
                 act_func: str = 'relu',
                 norm_type: str = 'layer',
                 num_heads: int = 4,
                 mode: str = 'dot',
                 max_position_embeddings: int = 0
                 ):
        """
        Task:
            Initialize model with a pretrained transformer and a regression MLP with multi-head attention.

        Input:
            model_name (str): Huggingface model identifier (e.g., 'seyonec/ChemBERTa').
            list_dims (list[int]): Dimensions for the MLP head.
            dropout (float): Dropout probability.
            act_func (str): Activation function type.
            norm_type (str): Normalization type.
            num_heads (int): Number of attention heads.

        Output:
            None
        """
        self.model_name = model_name
        self.list_dims = list_dims
        self.dropout = dropout
        self.act_func = act_func
        self.norm_type = norm_type
        self.num_heads = num_heads
        self.mode = mode
        self.max_position_embeddings = max_position_embeddings
        
        if mode not in ['dot', 'additive']:
            raise ValueError("mode must be either 'dot' or 'additive'")
        super().__init__(model_name=model_name,
                        list_dims=list_dims,
                        dropout=dropout,
                        act_func=act_func,
                        norm_type=norm_type
                        )
        # self.multihead_attn = nn.MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=num_heads, dropout=dropout)
        
        if num_heads == 1:
            self.multihead_attn = UnifiedAttention(input_dim=self.config.hidden_size, attn_dim=self.config.hidden_size, mode=self.mode)
        elif num_heads > 1:
            self.multihead_attn = MultiHeadAttention(embed_dim=self.config.hidden_size, num_heads=num_heads, mode=self.mode)
        else:
            print('-'*80)
            print("No multi-head attention applied. Using only pooling attention layer instead.")
            print('-'*80)
        self.attn_layer = nn.Linear(self.config.hidden_size, 1)
        
    def token_embeddings(self, input_ids, attention_mask):
        token_embed, _ = super().token_embeddings(input_ids=input_ids, attention_mask=attention_mask)# [B, T, H]

        if self.max_position_embeddings>0:
            # Add positional embeddings
            B, T = input_ids.size()
            position_ids = torch.arange(T, dtype=torch.long, device=input_ids.device)  # [T]
            position_ids = position_ids.unsqueeze(0).expand(B, T)  # [B, T]
            pos_embed = self.position_embeddings(position_ids)  # [B, T, H]
            token_embed = token_embed + pos_embed  # [B, T, H]

        return token_embed
        
    def embeddings(self, input_ids, attention_mask, **kwargs):
        
        embed, _ = self.token_embeddings(input_ids=input_ids, attention_mask=attention_mask)# [B, T, H]
        # print(f'embed.shape = {embed.shape}')
        if isinstance(self.num_heads, int) and self.num_heads > 0:
            embed, attn_weights = self.multihead_attn(x=embed, mask=attention_mask)
        # print(f'embed.shape after attention = {embed.shape}')
        attn_weights = torch.tanh(self.attn_layer(embed))  # [B, T, 1]
        # print(f'attn_weights.shape = {attn_weights.shape}')
        attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf'))
        # print(f'attn_weights.shape after masking = {attn_weights.shape}')
        attn_weights = F.softmax(attn_weights, dim=1)
        # print(f'attn_weights.shape after softmax = {attn_weights.shape}')
        pooled = torch.sum(attn_weights * embed, dim=1)  # [B, H]
        # print(f'pooled.shape = {pooled.shape}')
        
        return pooled
    
    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Task:
            Perform forward pass for regression prediction with attention.

        Input:
            input_ids (Tensor): Tokenized inputs.
            attention_mask (Tensor): Valid attention positions.

        Output:
            Tensor: Regression prediction.
        """
        x = self.embeddings(input_ids=input_ids, attention_mask=attention_mask)
        x = self.regressor(x)
        return x
    
 

class ChemBERTaRegressorwithLSTM(ChemBERTa):
    """
    Task:
        ChemBERTa-based regressor enhanced with LSTM for capturing sequential molecular token patterns.
    """

    def __init__(self,
                 model_name: str,
                 list_dims: list[int],
                 lstm_hidden: int = 256,
                 lstm_layers: int = 1,
                 bidirectional: bool = True,
                 dropout: float = 0.2,
                 act_func: str = 'relu',
                 norm_type: str = 'layer'):
        """
        Initialize model components: ChemBERTa, LSTM, and MLP regressor.
        """
        super().__init__(model_name=model_name,
                         list_dims=list_dims,
                         dropout=dropout,
                         act_func=act_func,
                         norm_type=norm_type)
        self.model_name = model_name
        self.dropout = dropout
        self.act_func = act_func
        self.lstm_layers = lstm_layers
        self.norm_type = norm_type
        self.bidirectional = bidirectional
        self.lstm_hidden = lstm_hidden

        self.lstm = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.list_dims = [lstm_out_dim] + list_dims
        self.regressor = make_mlp(list_dims=self.list_dims,
                                  dropout=self.dropout,
                                  act_func=self.act_func,
                                  norm_type=self.norm_type)
        
    def embeddings(self, input_ids, attention_mask, **kwargs):
        """
        Task:
            Compute LSTM-based embedding from token embeddings of ChemBERTa.

        Input:
            input_ids (Tensor): Tokenized input IDs of shape [B, T].
            attention_mask (Tensor): Corresponding attention mask [B, T].

        Output:
            Tensor: LSTM output embedding [B, H] where H depends on bidirectionality.
        """
        token_embed, _ = self.token_embeddings(input_ids=input_ids, attention_mask=attention_mask)  # [B, T, H]
        lengths = attention_mask.sum(dim=1).cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(token_embed, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        
        if self.bidirectional:
            final_embedding = torch.cat([hn[-2], hn[-1]], dim=1)  # [B, 2*H]
        else:
            final_embedding = hn[-1]  # [B, H]

        return final_embedding

    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Task:
            Perform forward pass: LSTM-based embedding + MLP regression.

        Input:
            input_ids (Tensor): Tokenized input sequences [B, T].
            attention_mask (Tensor): Attention mask indicating valid tokens [B, T].

        Output:
            Tensor: Regression prediction [B, output_dim].
        """
        x = self.embeddings(input_ids=input_ids, attention_mask=attention_mask)
        x = self.regressor(x)
        
        return x
    

class ChemBERTaRegressorRNN(ChemBERTa):
    """
    Task:
        ChemBERTa-based regressor using a configurable RNN block (LSTM or GRU) for sequential modeling.
    """

    def __init__(self,
                 model_name: str,
                 list_dims: list[int],
                 rnn_type: str = 'lstm',
                 rnn_hidden: int = 256,
                 rnn_layers: int = 1,
                 bidirectional: bool = True,
                 dropout: float = 0.2,
                 act_func: str = 'relu',
                 norm_type: str = 'layer'):
        """
        Initialize ChemBERTa, RNN (LSTM or GRU), and MLP regressor.

        Input:
            model_name (str): Huggingface ChemBERTa model name.
            list_dims (list[int]): List of MLP hidden/output dims.
            rnn_type (str): 'lstm' or 'gru'.
            rnn_hidden (int): Hidden size of the RNN.
            rnn_layers (int): Number of RNN layers.
            bidirectional (bool): Use bidirectional RNN.
            dropout (float): Dropout rate.
            act_func (str): Activation function in MLP.
            norm_type (str): Normalization type in MLP.
        """
        super().__init__(model_name=model_name,
                         list_dims=list_dims,
                         dropout=dropout,
                         act_func=act_func,
                         norm_type=norm_type)
        self.model_name = model_name
        self.rnn_type = rnn_type.lower()
        self.dropout = dropout
        self.act_func = act_func
        self.rnn_layers = rnn_layers
        self.norm_type = norm_type
        self.bidirectional = bidirectional
        self.rnn_hidden = rnn_hidden

        rnn_cls = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU

        self.lstm = rnn_cls(
            input_size=self.config.hidden_size,
            hidden_size=rnn_hidden,
            num_layers=self.rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        rnn_out_dim = rnn_hidden * (2 if bidirectional else 1)
        self.list_dims = [rnn_out_dim] + list_dims

        self.regressor = make_mlp(list_dims=self.list_dims,
                                  dropout=self.dropout,
                                  act_func=self.act_func,
                                  norm_type=self.norm_type)

    def embeddings(self, input_ids, attention_mask, **kwargs):
        """
        Compute embedding from RNN output based on token-level ChemBERTa embeddings.

        Output:
            Tensor: [B, H]
        """
        token_embed, _ = self.token_embeddings(input_ids=input_ids, attention_mask=attention_mask)
        lengths = attention_mask.sum(dim=1).cpu()

        packed_input = nn.utils.rnn.pack_padded_sequence(token_embed, lengths, batch_first=True, enforce_sorted=False)
        if self.rnn_type == 'lstm':
            packed_output, (hn, _) = self.lstm(packed_input)
        else:
            packed_output, hn = self.lstm(packed_input)

        if self.bidirectional:
            final_embedding = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            final_embedding = hn[-1]

        return final_embedding

    def forward(self, input_ids, attention_mask, **kwargs):
        
        x = self.embeddings(input_ids=input_ids, attention_mask=attention_mask)
        x =  self.regressor(x)
        
        return x


    
class AdditiveAttention(nn.Module):
    """
    Task:
        Implements a simple additive attention mechanism for sequence summarization.
    """
    def __init__(self, input_dim, hidden_dim):
        """
        Task:
            Initialize layers for additive attention.

        Input:
            input_dim (int): Dimensionality of input vectors.
            hidden_dim (int): Dimensionality of attention space.

        Output:
            None
        """
        super(AdditiveAttention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Task:
            Compute weighted context vector using additive attention.

        Input:
            x (Tensor): Input sequence tensor of shape [B, S, D].

        Output:
            context (Tensor): Weighted context vector [B, H].
            attn_weights (Tensor): Attention weights [B, S, 1].
        """
        # x: (batch_size, seq_len, input_dim)
        Q = self.query(x)   # (B, S, H)
        K = self.key(x)     # (B, S, H)
        V = self.value(x)   # (B, S, H)

        scores = torch.tanh(Q + K)                # (B, S, H)
        attn_weights = F.softmax(self.attn_score(scores), dim=1)  # (B, S, 1)
        context = torch.sum(attn_weights * V, dim=1)  # (B, H)
        
        return context, attn_weights
    
    
class DotProductAttention(nn.Module):
    """
    Implements single-head scaled dot-product attention.

    Inputs:
        embed_dim (int): Input embedding dimension.
        head_dim (int): Dimension for query, key, and value projections.
    """
    def __init__(self, embed_dim, head_dim):
        super(DotProductAttention, self).__init__()
        self.q_proj = nn.Linear(embed_dim, head_dim)
        self.k_proj = nn.Linear(embed_dim, head_dim)
        self.v_proj = nn.Linear(embed_dim, head_dim)
        self.scale = head_dim ** 0.5

    def forward(self, x, mask=None):
        """
        Apply scaled dot-product attention.

        Input:
            x (Tensor): [B, T, D] input embeddings
            mask (Tensor, optional): [B, T] attention mask

        Output:
            out (Tensor): [B, T, head_dim]
            weights (Tensor): [B, T, T] attention weights
        """
        Q = self.q_proj(x)  # [B, T, head_dim]
        K = self.k_proj(x)
        V = self.v_proj(x)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, T, T]

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, T]
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, T, T]
        out = torch.matmul(attn_weights, V)  # [B, T, head_dim]

        return out, attn_weights

class UnifiedAttention(nn.Module):
    """
    Unified implementation of Additive and Scaled Dot-Product Attention,
    using AdditiveAttention and DotProductAttention classes internally.
    """
    def __init__(self, input_dim: int, attn_dim: int, mode: str = 'dot'):
        super(UnifiedAttention, self).__init__()
        assert mode in ['dot', 'additive'], "mode must be either 'dot' or 'additive'"
        self.mode = mode

        if self.mode == 'additive':
            ## Use AdditiveAttention which takes input_dim, hidden_dim
            self.attn = AdditiveAttention(input_dim, attn_dim)
        else:
            # Use DotProductAttention which takes embed_dim, head_dim
            self.attn = DotProductAttention(input_dim, attn_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x (Tensor): Input sequence tensor of shape [B, T, D].
            mask (Tensor, optional): Binary mask of shape [B, T], 1 for valid, 0 for pad.

        Returns:
            out (Tensor): Output tensor.
            attn_weights (Tensor): Attention weights.
        """
        if self.mode == 'additive':
            # AdditiveAttention returns context: [B, H], attn_weights: [B, S, 1]
            context, attn_weights = self.attn(x)
            # context shape [B, H], expand to [B, T, H] to be consistent
            out = context.unsqueeze(1).expand(-1, x.size(1), -1)
        else:
            # DotProductAttention expects mask for masking, which is [B, T]
            out, attn_weights = self.attn(x, mask)
        
        return out, attn_weights

    
class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention using multiple DotProductAttention blocks.

    Inputs:
        embed_dim (int): Input embedding dimension.
        num_heads (int): Number of attention heads.

    Output:
        out (Tensor): [B, T, embed_dim]
        attn_weights (Tensor): [B, num_heads, T, T]
    """
    def __init__(self, embed_dim, num_heads, head_dim: int = None, mode: str = 'dot'):
        super(MultiHeadAttention, self).__init__()
        # assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mode = mode
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads
        self.concat_dim = self.head_dim*num_heads

        # Create a list of attention heads
        self.heads = nn.ModuleList([UnifiedAttention(input_dim=embed_dim, attn_dim=self.head_dim, mode=self.mode) for _ in range(num_heads)])

        # Final linear projection to merge all heads
        self.out_proj = nn.Linear(self.concat_dim, embed_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): [B, T, D]
                - B: Batch size (number of samples)
                - T: Sequence length (number of tokens)
                - D: Embedding dimension (hidden size)
            
            mask (Tensor, optional): [B, T]
                - Attention mask. 1 for valid tokens, 0 for padding.
                - Used to prevent attending to padded positions.

        Returns:
            out (Tensor): [B, T, D]
                - Output after applying multi-head attention.
                - Same shape as input, but now each token has contextualized information.
 
            attn_weights (Tensor): [B, H, T, T]
                - Attention weights for each head.
                - H: Number of attention heads.
                - Each [T, T] matrix shows how each token attends to every other token.
        """
        head_outputs = []
        head_weights = []

        for head in self.heads:
            h_out, h_weights = head(x, mask)
            head_outputs.append(h_out)         # [B, T, head_dim]
            head_weights.append(h_weights)     # [B, T, T]

        # Concatenate head outputs along last dim
        concat = torch.cat(head_outputs, dim=-1)  # [B, T, embed_dim]
        out = self.out_proj(concat)               # [B, T, embed_dim]

        # Stack attention weights
        attn_weights = torch.stack(head_weights, dim=1)  # [B, num_heads, T, T]

        return out, attn_weights 

    

class GeneralizedAdditiveAttention(nn.Module):
    """
    Generalizedized additive attention mechanism using customizable MLP for scoring.

    Capable of summarizing sequences by computing attention over input features
    with optional Q/K/V projections and MLP-based attention scoring.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mlp_hidden: list = [128],
        proj_qkv: bool = True,
        reduce: str = 'sum',
        dropout: float = 0.0,
        norm_type: str = 'layer',
        act_func: str = 'tanh',
        proj_dims: list = [128],
        proj_dropout: float = 0.0,
        proj_act: str = None,
        proj_norm: str = 'layer'
    ):
        """
        Args:
            input_dim (int): Dimensionality of input features.
            hidden_dim (int): Dimension for Q, K, V after projection.
            mlp_hidden (list of int): Hidden layer sizes for attention scoring MLP.
            proj_qkv (bool): Whether to project Q, K, V through MLPs.
            reduce (str): Reduction strategy over sequence: 'sum', 'mean', or 'none'.
            dropout (float): Dropout used inside the attention MLP.
            norm_type (str): Normalization used inside attention MLP.
            act_func (str): Activation function used in attention MLP.
            proj_dims (list of int): Layer dimensions for QKV projection ending in `hidden_dim`.
            proj_dropout (float): Dropout in QKV projection MLPs.
            proj_act (str): Activation function in QKV projection MLPs.
            proj_norm (str): Normalization used in QKV projection MLPs.
        """
        super().__init__()

        self.reduce = reduce.lower()
        assert self.reduce in ['sum', 'mean', 'none'], "reduce must be 'sum', 'mean', or 'none'"

        if proj_qkv:
            qkv_proj_dims = proj_dims + [hidden_dim]
            self.query = make_mlp([input_dim] + qkv_proj_dims, dropout=proj_dropout, act_func=proj_act, norm_type=proj_norm)
            self.key   = make_mlp([input_dim] + qkv_proj_dims, dropout=proj_dropout, act_func=proj_act, norm_type=proj_norm)
            self.value = make_mlp([input_dim] + qkv_proj_dims, dropout=proj_dropout, act_func=proj_act, norm_type=proj_norm)
        else:
            self.query = nn.Identity()
            self.key   = nn.Identity()
            self.value = nn.Identity()

        # Attention scoring MLP: combines Q+K to scalar attention logit
        self.attn_score_mlp = make_mlp([hidden_dim] + mlp_hidden + [1], dropout=dropout, act_func=act_func, norm_type=norm_type)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x (Tensor): Input tensor of shape [B, S, D].
            mask (Tensor, optional): Binary mask of shape [B, S], with 1 for valid tokens.

        Returns:
            context (Tensor): Context vector [B, H] or [B, S, H] if reduce='none'.
            attn_weights (Tensor): Attention weights [B, S, 1].
        """
        B, S, D = x.shape

        Q = self.query(x)  # [B, S, H]
        K = self.key(x)    # [B, S, H]
        V = self.value(x)  # [B, S, H]

        score_input = torch.tanh(Q + K)  # [B, S, H]
        attn_logits = self.attn_score_mlp(score_input)  # [B, S, 1]

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=1)  # [B, S, 1]
        weighted = attn_weights * V  # [B, S, H]

        if self.reduce == 'sum':
            context = torch.sum(weighted, dim=1)  # [B, H]
        elif self.reduce == 'mean':
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1) if mask is not None else S
            context = torch.sum(weighted, dim=1) / denom  # [B, H]
        else:  # 'none'
            context = weighted  # [B, S, H]

        return context, attn_weights
    
class GeneralizedDotProductAttention(nn.Module):
    def __init__(self, input_dim,
                 hidden_dim,
                 proj_dims=[128],
                 proj_qkv=True,
                 proj_qkv_act_func: str = None,
                 proj_qkv_norm_type: str = None,
                 dropout=0.0,
                 reduce_='none'
                 ):
        super().__init__()
        self.reduce = reduce_
        self.scale = hidden_dim ** 0.5

        if proj_qkv:
            proj_layers = proj_dims + [hidden_dim]
            self.q_proj = make_mlp([input_dim] + proj_layers, dropout=dropout, act_func=proj_qkv_act_func, norm_type=proj_qkv_norm_type)
            self.k_proj = make_mlp([input_dim] + proj_layers, dropout=dropout, act_func=proj_qkv_act_func, norm_type=proj_qkv_norm_type)
            self.v_proj = make_mlp([input_dim] + proj_layers, dropout=dropout, act_func=proj_qkv_act_func, norm_type=proj_qkv_norm_type)
        else:
            self.q_proj = nn.Identity()
            self.k_proj = nn.Identity()
            self.v_proj = nn.Identity()

    def forward(self, x, mask=None):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)

        if self.reduce == 'sum':
            out = out.sum(dim=1)
        elif self.reduce == 'mean':
            denom = mask.sum(dim=2, keepdim=True).clamp(min=1) if mask is not None else x.size(1)
            out = out.sum(dim=1) / denom
        # else: reduce='none', keep as is

        return out, attn_weights


class GeneralizedUnifiedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, mode='dot', **kwargs):
        super().__init__()
        assert mode in ['dot', 'additive'], "mode must be 'dot' or 'additive'"
        self.mode = mode
        if mode == 'dot':
            self.attn = GeneralizedDotProductAttention(input_dim, hidden_dim, **kwargs)
        else:
            self.attn = GeneralizedAdditiveAttention(input_dim, hidden_dim, **kwargs)

    def forward(self, x, mask=None):
        out, attn_weights = self.attn(x, mask)
        if isinstance(out, torch.Tensor) and out.dim() == 2:  # [B, H]
            out = out.unsqueeze(1).expand(-1, x.size(1), -1)
        return out, attn_weights

    

class GeneralizedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, mode='dot', **kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.heads = nn.ModuleList([GeneralizedUnifiedAttention(input_dim=embed_dim,
                                                            hidden_dim=self.head_dim,
                                                            mode=mode,
                                                            **kwargs
                                                            )for _ in range(num_heads)
                                    ])

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        head_outputs = []
        head_weights = []

        for head in self.heads:
            h_out, h_weights = head(x, mask)
            head_outputs.append(h_out)
            head_weights.append(h_weights)

        concat = torch.cat(head_outputs, dim=-1)
        out = self.out_proj(concat)

        attn_weights = torch.stack(head_weights, dim=1)
        return out, attn_weights