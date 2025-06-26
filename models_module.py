#!/usr/bin/env python3
import numpy as np
import math, os, pickle
from collections import OrderedDict

import torch
from torch import nn
from torchinfo import summary
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large
from torchvision.models import resnet18
from transformers import ViTModel, ViTFeatureExtractor
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers import AutoModel, AutoConfig



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
    'GCN',
    'GCN_Connected',
    'MRGCN',
    'ChemBERTaRegressorroberta',
    'ChemBERTaRegressor',
    'SimpleAttention',
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
        return nn.ReLU(inplace=True)
    elif name == 'selu':
        return nn.SELU(inplace=True)
    elif name == 'elu':
        return nn.ELU(alpha, inplace=True)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(negative_slope, inplace=True)
    elif name == 'celu':
        return nn.CELU(alpha, inplace=True)
    else:
        raise ValueError(f'Unsupported activation function: "{name}"')


def make_mlp(list_dims, dropout=0.0, act_func='relu', norm_type='layer', alpha=1.0, negative_slope=1e-2):
    """
    Construct a multi-layer perceptron with normalization, activation, and dropout.

    Args:
        list_dims (list of int): List of layer dimensions.
        dropout (float): Dropout rate.
        act_func (str): Activation function name.
        norm_type (str): Normalization type: 'batch', 'layer', or None.
        alpha (float): Alpha parameter for ELU/CELU.
        negative_slope (float): Negative slope for LeakyReLU.

    Returns:
        nn.Sequential: MLP model.
    """
    layers = []
    act_layer = activation_func(act_func, alpha=alpha, negative_slope=negative_slope)
    
    for i in range(len(list_dims) - 1):
        in_dim = list_dims[i]
        out_dim = list_dims[i + 1]
        layers.append(nn.Linear(in_dim, out_dim))
        
        if i < len(list_dims) - 2:  # Skip last layer for normalization and activation
            if norm_type == 'batch':
                layers.append(nn.BatchNorm1d(out_dim))
            elif norm_type == 'layer':
                layers.append(nn.LayerNorm(out_dim))
            elif norm_type is not None:
                raise ValueError(f"Unsupported normalization type: {norm_type}")

            layers.append(act_layer)
            layers.append(nn.Dropout(dropout))

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
    def __init__(self, list_dims, dropout):
        super(SMILESLinearNet, self).__init__()
        
        self.model_script = None
        self.layers = nn.ModuleList([nn.Linear(list_dims[i], list_dims[i+1]) for i in range(len(list_dims) - 1)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(list_dims[i+1]) for i in range(len(list_dims) - 2)])
        self.dropout = nn.Dropout(dropout)
        
        # Weight Initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        Task:
            Forward pass through MLP with ReLU, BatchNorm, and Dropout.

        Inputs:
            - x (Tensor): Input tensor of shape [batch_size, list_dims[0]]

        Outputs:
            - Tensor: Output tensor after final linear layer.
        """
        for i in range(len(self.layers)):
            if i < len(self.layers) - 1:
                x = F.relu(self.layers[i](x)) 
                x = self.batch_norms[i](x) 
                x = self.dropout(x) 
            else:
                x = self.layers[i](x) 
        
        return x


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
        self.model_script = network_pyfile
        self.max_num_atom = max_num_atom
        self.gcn_layers = nn.ModuleList([GCNLayer(list_dims_gcn[i], list_dims_gcn[i+1]) for i in range(len(list_dims_gcn) - 1)])
        self.gcn_layer_norms = nn.ModuleList([nn.LayerNorm(list_dims_gcn[i+1]) for i in range(len(list_dims_gcn) - 2)])
        
        self.fc_conn = nn.Linear(list_dims_gcn[-1], list_dims_fc[0])
        self.weighted_average = nn.Linear(max_num_atom, 1)
        
        self.fc_layers = nn.ModuleList([nn.Linear(list_dims_fc[i], list_dims_fc[i+1]) for i in range(len(list_dims_fc) - 1)])
        self.fc_layer_norms = nn.ModuleList([nn.LayerNorm(list_dims_fc[i+1]) for i in range(len(list_dims_fc) - 2)])
        self.dropout = nn.Dropout(dropout)
        
    def get_features(self, x, adjacency_matrices, degree_matrices):
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
    
    def input2repr(self, x, layers, layer_norms, dropout):
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
        x = self.get_features(x, adjacency_matrices, degree_matrices)
        x = self.fc_conn(x)
        x = self.input2repr(x, self.fc_layers, self.fc_layer_norms, self.dropout)
        
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
    def __init__(self, list_dims_gcn, list_dims_fc, dropout, type_='concat'):
        super().__init__()
        self.model_script = network_pyfile
        self.type_ = type_
        
        # GCN layers with LayerNorms for stability, except on last GCN layer
        self.gcn_layers = nn.ModuleList([GCNLayer(list_dims_gcn[i], list_dims_gcn[i+1]) for i in range(len(list_dims_gcn) - 1)])
        self.gcn_layer_norms = nn.ModuleList([nn.LayerNorm(list_dims_gcn[i+1]) for i in range(len(list_dims_gcn) - 2)])
        
        self.fc_conn = nn.Linear(list_dims_gcn[-1], list_dims_fc[0])
        
        # Fully Connected layers with LayerNorms for stability, except on last FC layer
        self.fc_layers = nn.ModuleList([nn.Linear(list_dims_fc[i], list_dims_fc[i+1]) for i in range(len(list_dims_fc) - 1)])
        self.fc_layer_norms = nn.ModuleList([nn.LayerNorm(list_dims_fc[i+1]) for i in range(len(list_dims_fc) - 2)])
        
        self.dropout = nn.Dropout(dropout)
        self.param1 = nn.Parameter(torch.tensor(1.0))
        self.param2 = nn.Parameter(torch.tensor(1.0)) 
        
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
        x = torch.concat((self.param1*x, self.param2*y), dim=-1)#torch.concat((x, y), dim=-1)#
        
        return x
    
    def input2repr(self, x, layers, layer_norms, dropout):
        
        if len(layers):
            for i in range(len(layers)):
                if i < len(layers)-1:
                    x = F.relu(layers[i](x))
                    x = layer_norms[i](x)
                    x = dropout(x)
                else:
                    x = layers[i](x)
                
        return x

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
        x = self.input2repr(x, self.fc_layers, self.fc_layer_norms, self.dropout)
        
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

        for rel in range(num_relations):
            adj = adjacency_tensor[:, rel]  # Shape: [batch_size, num_nodes, num_nodes]

            h_rel = torch.matmul(x, self.weights[rel])  # Shape: [batch_size, num_nodes, out_channels]
            out += torch.matmul(adj, h_rel)
            
        # ********************************************************************************
        ## Changed (To reverse uncomment the next line and comment the line after. Also change in get_multirelational_bond_matrix function in utils_.py file)
        out += torch.matmul(x, self.self_loop_weight)
        # out = out/num_relations
        # ********************************************************************************

        if self.bias is not None:
            out += self.bias

        return out


class MRGCN(nn.Module):
    """
    Task:
        Multi-Relational Graph Convolutional Network with relation-aware aggregation and dense prediction head.

    Inputs:
        - list_dims_gcn: Dimensions of GCN layers
        - list_dims_fc: Dimensions of FC layers
        - dropout: Dropout probability
        - max_num_atom: For weighted pooling (not used directly here)
        - num_relations: Number of edge types

    Outputs:
        - Tensor: Final prediction or representation [batch_size, output_dim]
    """
    def __init__(self, list_dims_gcn, list_dims_fc, dropout, max_num_atom, num_relations=4):#):
        super().__init__()
        self.max_num_atom = max_num_atom
        self.num_relations = num_relations
        self.model_script = network_pyfile

        # GCN Layers
        self.gcn_layers = nn.ModuleList([
            RGCNConv(list_dims_gcn[i], list_dims_gcn[i + 1], num_relations=self.num_relations)
            for i in range(len(list_dims_gcn) - 1)
        ])
        self.gcn_layer_norms = nn.ModuleList([
            nn.LayerNorm(list_dims_gcn[i + 1]) for i in range(len(list_dims_gcn) - 2)
        ])
        
        ## GCN output to Feature Representation
        # self.fc_layers_feat = nn.ModuleList([
        #     nn.Linear(in_features=list_dims_gcn[-1],out_features=list_dims_gcn[-1], bias=False) for _ in range(2)
        # ])
        # self.fc_layer_norms_feat = nn.ModuleList([
        #     nn.LayerNorm(list_dims_gcn[-1]) for _ in range(len(self.fc_layers_feat) - 1)
        # ])

        # Fully connected layers
        self.fc_conn = nn.Linear(list_dims_gcn[-1], list_dims_fc[0])
        self.fc_layers = nn.ModuleList([
            nn.Linear(list_dims_fc[i], list_dims_fc[i + 1]) for i in range(len(list_dims_fc) - 1)
        ])
        self.fc_layer_norms = nn.ModuleList([
            nn.LayerNorm(list_dims_fc[i + 1]) for i in range(len(list_dims_fc) - 2)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def input2repr(self, x, layers, layer_norms, dropout):
        for i in range(len(layers)):
            if i < len(layers) - 1:
                x = F.relu(layers[i](x))
                x = layer_norms[i](x)
                x = dropout(x)
            else:
                x = layers[i](x)
        return x

    def get_features(self, x, adjacency_tensor, degree_tensor):
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
        adjacency_tensor = normalize_adjacency(adjacency_tensor=adjacency_tensor,
                                                            degree_tensor=degree_tensor
                                                            )
        for i in range(len(self.gcn_layers)):
            if i < len(self.gcn_layers) - 1:
                x = F.relu(self.gcn_layers[i](x, adjacency_tensor))
                x = self.gcn_layer_norms[i](x)
                x = self.dropout(x)
            else:
                x = self.gcn_layers[i](x, adjacency_tensor)

        # Average over nodes for each graph in the batch
        x = x.mean(dim=1)
        # x = self.input2repr(x, layers=self.fc_layers_feat, layer_norms=self.fc_layer_norms_feat, dropout=self.dropout)
        self.feature_dim = x.shape[-1]
        return x

    def forward(self, x, adjacency_tensor, degree_tensor):
        """
        Task:
            Forward pass from graph input to final prediction via GCN and FC layers.

        Inputs:
            - x: Node features
            - adjacency_tensor: Multi-relational adjacency
            - degree_tensor: Node degree tensor

        Outputs:
            - Tensor: Final prediction/representation
        """
        x = self.get_features(x, adjacency_tensor, degree_tensor)
        # x = x.mean(dim=1).unsqueeze(1)
        x = self.fc_conn(x)
        x = self.input2repr(x, self.fc_layers, self.fc_layer_norms, self.dropout)
        return x#.squeeze()
    
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
    
    
class ChemBERTaRegressor(nn.Module):
    """
    Task:
        ChemBERTa-based regressor for molecular property prediction using AutoModel.
    """
    def __init__(self, model_name: str, list_dims: list[int], dropout: float = 0.2):
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
        self.model_name = model_name
        self.dropout = dropout
        self.config = AutoConfig.from_pretrained(model_name)
        self.chemberta = AutoModel.from_pretrained(model_name, config=self.config)

        hidden_size = self.config.hidden_size
        list_dims = [hidden_size, ] + list_dims

        self.regressor = self.make_mlp(list_dims=list_dims, dropout=dropout)
        
    def __repr__(self):
        """
        Task:
            String representation of the model including its name and configuration.

        Output:
            str: Model name and configuration.
        """
        return f"{self.__class__.__name__}(model_name={self.model_name}, list_dims={self.list_dims}, dropout={self.dropout})"

    def make_mlp(self, list_dims, dropout):
        """
        Task:
            Build a feedforward regressor with optional dropout and activation.

        Input:
            list_dims (list[int]): List of layer dimensions.
            dropout (float): Dropout rate.

        Output:
            nn.Sequential: Multi-layer perceptron for regression.
        """
        layers = nn.ModuleList([])
        for i in range(len(list_dims) - 1):
            layers.append(nn.Linear(list_dims[i], list_dims[i + 1]))
            if i < len(list_dims) - 2:
                layers.extend([
                    nn.LayerNorm(list_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
        return nn.Sequential(*layers)

    def embeddings(self, input_ids, attention_mask):
        """
        Task:
            Compute mean-pooled embedding over valid token positions.

        Input:
            input_ids (Tensor): Input token IDs of shape [B, T].
            attention_mask (Tensor): Attention mask of shape [B, T].

        Output:
            Tensor: Mean pooled embedding of shape [B, H].
        """
        outputs = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        masked_embed = (last_hidden * attention_mask.unsqueeze(-1)).sum(1)
        denom = attention_mask.sum(1, keepdim=True).clamp(min=1e-6)
        return masked_embed / denom  # mean pooling

    def forward(self, input_ids, attention_mask):
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
    
    
class SimpleAttention(nn.Module):
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
        super(SimpleAttention, self).__init__()
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