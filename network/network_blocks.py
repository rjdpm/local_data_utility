#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##############################################################################
"""
-----------------------------------------------------------------------------
FUNCTION:
-----------------------------------------------------------------------------
@AUTHOR: Soumitra Samanta                DATE: Tue Dec  1 23:31:42 2020
For bug and others mail me at soumitramath39@gmail.com
-----------------------------------------------------------------------------
INPUT:
OUTPUT:
-----------------------------------------------------------------------------
EXAMPLE:
-----------------------------------------------------------------------------
"""
##############################################################################


import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = [
    'networks_blocks',
    'weights_initilizer'
]

class networks_blocks():
    
    def __init__(self):
        super(networks_blocks, self).__init__()
    
    #calculate conv1d output dims
    @staticmethod
    def conv1d_output(h_in, padding=0, dilation=1, \
                      kernel_size=9, stride=1
                     ):
        h_out = int(np.floor(1+((h_in + 2 * padding - dilation*(kernel_size-1)-1)/float(stride))))

        return h_out

    #calculate conv2d output dims
    @staticmethod
    def conv2d_output(h_in, w_in, padding=(1,1), dilation=(1, 1), \
                      kernel_size=(3, 3), stride=(1,1)
                     ):
        h_out = int(np.floor(1+((h_in + 2 * padding[0] - dilation[0]*(kernel_size[0]-1)-1)/float(stride[0]))))
        w_out = int(np.floor(1+((w_in + 2 * padding[1] - dilation[1]*(kernel_size[1]-1)-1)/float(stride[1]))))

        return h_out, w_out
    
    #calculate max-pool output dims
    @staticmethod
    def maxpool_output(h_in, w_in, padding=(0,0), \
                       dilation=(1, 1), kernel_size=(3, 3), stride=(1,1)
                      ):
        h_out = int(np.floor(1+((h_in + 2 * padding[0] - dilation[0]*(kernel_size[0]-1)-1)/float(stride[0]))))
        w_out = int(np.floor(1+((w_in + 2 * padding[1] - dilation[1]*(kernel_size[1]-1)-1)/float(stride[1]))))

        return h_out, w_out
    
    # acitivation function
    @staticmethod
    def _activation_func(name='relu', param={'alpha':1.0, 'negative_slope':1e-2}):
        
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
        
    # linear blocks
    @staticmethod
    def _fc_layers(fc_in, fc_out, \
                   num_layers=0, \
                   dropout_prob=0.0, \
                   act_func='', \
                   param_act_func={'alpha':1.0, 'negative_slope':1e-2}, \
                   name='encoder'
                  ):

        _layers = OrderedDict({})

        if num_layers:
            # calculate intermediate layer's dimension
            dims_gap = fc_in - fc_out
            dims_step = math.floor(float(dims_gap)/(num_layers+1))
            dims_fc = np.zeros(num_layers, dtype=int)
            for i in range(num_layers):
                dims_fc[i] = fc_in - (i + 1)*dims_step

            # define input and intermediate layers 
            for i in range(num_layers):
                _layers[name+'_fc'+str(i+1)] = nn.Linear(fc_in, dims_fc[i])
                if len(act_func):
                    _layers[name+'_fc_'+act_func+str(i+1)] = networks_blocks._activation_func(act_func, param_act_func)
                fc_in = dims_fc[i]
                if dropout_prob:
                    _layers[name+'_fc'+str(i+1)+'_dropout'] = nn.Dropout(dropout_prob)
            # define last-fc layer
            _layers[name+'_fc'+str(num_layers+1)] = nn.Linear(dims_fc[-1], fc_out)
                
        else:
            _layers[name+'_fc'+str(num_layers+1)] = nn.Linear(fc_in, fc_out)

        return nn.Sequential(_layers)
    #------------------------------------------------------
    
    # 1d-conv blocks
    @staticmethod
    def _conv1d_layers(dims_input_data, num_kernels=[32], size_kernels=[3], \
                       act_func='relu', \
                       param_act_func={'alpha':1.0, 'negative_slope':1e-2}, \
                       name='encoder'
                      ):
    
        _layers = OrderedDict({})
        
        num_layers = len(num_kernels)
        for i in range(num_layers):
            _layers[name+'_conv1d'+str(i+1)] = nn.Conv1d(dims_input_data, num_kernels[i], kernel_size=size_kernels[i])
#             _layers[name+'_relu'+str(i+1)] = nn.ReLU(inplace=True)
            _layers[name+'_'+act_func+str(i+1)] = networks_blocks._activation_func(act_func, param_act_func)
            dims_input_data = num_kernels[i]

        return nn.Sequential(_layers)

    # 2d-conv unet blocks
    @staticmethod
    def _unet_blocks(in_channels, num_kernel, size_kernel, \
                     stride, padding, layer_idx=1, \
                     act_func='relu', \
                     param_act_func={'alpha':1.0, 'negative_slope':1e-2}, \
                     name='encoder'
                    ):

        _blocks = OrderedDict({})

        for i in range(2):
            _blocks[name+'_conv2d'+str(layer_idx+i)] = nn.Conv2d(in_channels=in_channels, \
                                                           out_channels=num_kernel, \
                                                           kernel_size=size_kernel, \
                                                           stride=stride, \
                                                           padding=padding, \
#                                                            bias=False\
                                                          )
            _blocks[name+'_bnorm'+str(layer_idx+i)] = nn.BatchNorm2d(num_features=num_kernel)
#             _blocks[name+'_relu'+str(layer_idx+i)] = nn.ReLU(inplace=True)
            _blocks[name+'_'+act_func+str(layer_idx+i)] = networks_blocks._activation_func(act_func, param_act_func)

            in_channels = num_kernel

        return _blocks

    # 2d-conv unet encoder
    @staticmethod
    def _conv2d_unet_layers(in_channels, num_kernels_conv=[32, 64, 128, 265, 512], \
                            size_kernels_conv=[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3]], \
                            strides_conv=[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], \
                            paddings_conv=[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], \
                            size_kernels_pool=[[2, 2, 2, 2, 2],[2, 2, 2, 2, 2]], \
                            strides_pool=[[2, 2, 2, 2, 2],[2, 2, 2, 2, 2]], \
                            act_func='relu', \
                            param_act_func={'alpha':1.0, 'negative_slope':1e-2}, \
                            name='encoder'):

        _layers = OrderedDict({})

        num_layers = len(num_kernels_conv)
        for i in range(num_layers):
            # unet block
            _blocks = networks_blocks._unet_blocks(in_channels, \
                                                   num_kernels_conv[i], \
                                                   (size_kernels_conv[0][i], size_kernels_conv[1][i]), \
                                                   (strides_conv[0][i], strides_conv[1][i]), \
                                                   (paddings_conv[0][i], paddings_conv[1][i]), \
                                                   layer_idx=2*i+1, \
                                                   act_func=act_func, \
                                                   param_act_func=param_act_func, \
                                                   name=name
                                                  )
            _layers.update(_blocks)
            # max-pool layer
            if i != num_layers-1:
                _layers[name+'_maxpool'+str(2*i+1)] = nn.MaxPool2d(kernel_size=(size_kernels_pool[0][i], size_kernels_pool[1][i]), \
                                                                   stride=(strides_pool[0][i], strides_pool[1][i]))

            in_channels = num_kernels_conv[i]

        return nn.Sequential(_layers)


# network weight inilializer
class weights_initilizer():
    
    def __init__(self, network, initializer='xvr_unifrm'):
        
        if initializer=='default':
            pass
        elif initializer=='zeros':
            network.apply(self.weights_init_zeros)
        elif initializer=='ones':
            network.apply(self.weights_init_ones)
        elif initializer=='unifrm':
            network.apply(self.weights_init_uniform)
        elif initializer=='nrmal':
            network.apply(self.weights_init_normal)
        elif initializer=='xvr_unifrm':
            network.apply(self.weights_init_xavior_uniform)
        elif initializer=='xvr_nrmal':
            network.apply(self.weights_init_xavior_normal)
        else:
            raise ValueError('Define your parameters initializer: "{}"' .format(initializer))
    #------------------------------------------------------
    
    # weight initializer help function
    
    def weights_init_zeros(self, m):
        if((isinstance(m, nn.Linear)) or \
           (isinstance(m, nn.Conv1d)) or (isinstance(m, nn.ConvTranspose1d)) or \
           (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.ConvTranspose2d)) or \
           (isinstance(m, nn.Conv3d)) or (isinstance(m, nn.ConvTranspose3d)) or \
           (isinstance(m, nn.BatchNorm1d)) or \
           (isinstance(m, nn.BatchNorm2d)) or \
           (isinstance(m, nn.BatchNorm3d))):
            torch.nn.init.zeros_(m.weight)
            torch.nn.init.zeros_(m.bias)
            print('Weights "{}" initialized by "zeros" scheme' .format(m))
    #------------------------------------------------------
    
    def weights_init_ones(self, m):
        if((isinstance(m, nn.Linear)) or \
           (isinstance(m, nn.Conv1d)) or (isinstance(m, nn.ConvTranspose1d)) or \
           (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.ConvTranspose2d)) or \
           (isinstance(m, nn.Conv3d)) or (isinstance(m, nn.ConvTranspose3d)) or \
           (isinstance(m, nn.BatchNorm1d)) or \
           (isinstance(m, nn.BatchNorm2d)) or \
           (isinstance(m, nn.BatchNorm3d))):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
            print('Weights "{}" initialized by "ones" scheme' .format(m))
    #------------------------------------------------------
    
    def weights_init_uniform(self, m):
        if((isinstance(m, nn.Linear)) or \
           (isinstance(m, nn.Conv1d)) or (isinstance(m, nn.ConvTranspose1d)) or \
           (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.ConvTranspose2d)) or \
           (isinstance(m, nn.Conv3d)) or (isinstance(m, nn.ConvTranspose3d)) or \
           (isinstance(m, nn.BatchNorm1d)) or \
           (isinstance(m, nn.BatchNorm2d)) or \
           (isinstance(m, nn.BatchNorm3d))):
            torch.nn.init.uniform_(m.weight)
            torch.nn.init.uniform_(m.bias)
            print('Weights "{}" initialized by "uniform" scheme' .format(m))
    #------------------------------------------------------
    
    def weights_init_normal(self, m):
        if((isinstance(m, nn.Linear)) or \
           (isinstance(m, nn.Conv1d)) or (isinstance(m, nn.ConvTranspose1d)) or \
           (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.ConvTranspose2d)) or \
           (isinstance(m, nn.Conv3d)) or (isinstance(m, nn.ConvTranspose3d)) or \
           (isinstance(m, nn.BatchNorm1d)) or \
           (isinstance(m, nn.BatchNorm2d)) or \
           (isinstance(m, nn.BatchNorm3d))):
            torch.nn.init.normal_(m.weight)
            torch.nn.init.normal_(m.bias)
            print('Weights "{}" initialized by "normal_dist" scheme' .format(m))
    #------------------------------------------------------
    
    def weights_init_xavior_uniform(self, m):
        if((isinstance(m, nn.Linear)) or \
           (isinstance(m, nn.Conv1d)) or (isinstance(m, nn.ConvTranspose1d)) or \
           (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.ConvTranspose2d)) or \
           (isinstance(m, nn.Conv3d)) or (isinstance(m, nn.ConvTranspose3d)) or \
           (isinstance(m, nn.BatchNorm1d)) or \
           (isinstance(m, nn.BatchNorm2d)) or \
           (isinstance(m, nn.BatchNorm3d))):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
            print('Weights "{}" initialized by "xavior_uniform" scheme' .format(m))
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    torch.nn.init.orthogonal_(param.data)
                else:
                    torch.nn.init.normal_(param.data)
            print('Weights "{}" initialized by "orthogonal" scheme' .format(m))
    #------------------------------------------------------
    
    def weights_init_xavior_normal(self, m):
        if((isinstance(m, nn.Linear)) or \
           (isinstance(m, nn.Conv1d)) or (isinstance(m, nn.ConvTranspose1d)) or \
           (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.ConvTranspose2d)) or \
           (isinstance(m, nn.Conv3d)) or (isinstance(m, nn.ConvTranspose3d)) or \
           (isinstance(m, nn.BatchNorm1d)) or \
           (isinstance(m, nn.BatchNorm2d)) or \
           (isinstance(m, nn.BatchNorm3d))):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
            print('Weights "{}" initialized by "xavior_normal" scheme' .format(m))
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    torch.nn.init.orthogonal_(param.data)
                else:
                    torch.nn.init.normal_(param.data)
            print('Weights "{}" initialized by "orthogonal" scheme' .format(m))
            
