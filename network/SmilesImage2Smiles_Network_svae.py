#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##############################################################################
"""
-----------------------------------------------------------------------------
FUNCTION:
-----------------------------------------------------------------------------
@AUTHOR: Soumitra Samanta                DATE: Mon Mar  1303231330
For bug and others mail me at soumitra.samanta@gm.rkmvu.ac.in
-----------------------------------------------------------------------------
INPUT:
OUTPUT:
-----------------------------------------------------------------------------
EXAMPLE:
-----------------------------------------------------------------------------
"""
##############################################################################
from tqdm import tqdm
# import shutil
import numpy as np
import pickle
import os
import sys
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from typing import Any, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
# from torchviz import make_dot

# to import parent package
sys.path.append('../')
from data_utils import *
from .network_blocks import *
from input_output import *

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform

___all__ = [
    'CNN1D_GRU_SVAE',
    'SmIm2Sm_Network'
]


class CNN1D_GRU_SVAE(nn.Module):
    def __init__(
        self, 
        dims_input_data: Tuple = (248, 40), 
        dims_output_data: Tuple =(248, 40), 
        num_kernels: List = [9, 9, 10], 
        size_kernels: List =[9, 9, 11],
        num_fc_layer_encoder: int = 0,
        dropout_prob: float = 0.0, 
        dims_latent: int = 100, 
        act_func: str = 'relu', 
        param_act_func: Dict ={'alpha':1.0, 'negative_slope':1e-2}, 
        scale_latent_space: float = 1e-2, 
        num_fc_layer_decoder: int = 0, 
        gru_hidden_size: int = 500, 
        num_gru: int = 4, 
        distribution: str = 'vmf',
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> None:
        super(CNN1D_GRU_SVAE, self).__init__()
        
        
        E_last_conv1d_out = dims_input_data[1]
        for i in range(len(size_kernels)):
            E_last_conv1d_out = networks_blocks.conv1d_output(
                E_last_conv1d_out, 
                kernel_size=size_kernels[i]
            )
        E_dims_first_fc = E_last_conv1d_out*num_kernels[-1]
        
        # define encoder layers
        # cnn1d layers
        self.E_conv1d = networks_blocks._conv1d_layers(
            dims_input_data[0], 
            num_kernels, 
            size_kernels, 
            act_func=act_func, 
            param_act_func=param_act_func, 
            name='encoder' 
        )
        
        # fc layers 
        self.E_fc = networks_blocks._fc_layers(
            E_dims_first_fc,
            dims_latent, 
            num_fc_layer_encoder, 
            dropout_prob, 
            act_func=act_func, 
            param_act_func=param_act_func, 
            name='encoder' 
        )
        
        # Distribution parameters
        self.E_mean = networks_blocks._fc_layers(
            dims_latent,
            dims_latent, 
            name='mean' 
        )
        # compute concentration of the von Mises-Fisher
        if distribution == 'vmf':
            self.E_sigma = networks_blocks._fc_layers(
                dims_latent,
                1, 
                name='sigma'
            )
        else:
            raise ValueError('Define your parameters for the distribution: "{}"' .format(distribution))
            
        # define decoder layers
        # fc layers 
        self.D_fc1 = networks_blocks._fc_layers(
            dims_latent,
            dims_latent, 
            num_fc_layer_decoder, 
            name='decoder' 
        )
        
        # rnn layers
        self.D_gru = nn.GRU(dims_latent, gru_hidden_size, num_gru, batch_first=True)
        
        # output layer
        self.D_fc2 = networks_blocks._fc_layers(
            gru_hidden_size,
            dims_output_data[1], 
            name='decoder' 
        )
    
        
        self.dims_input_data = dims_input_data
        self.dims_output_data = dims_output_data
        self.E_last_conv1d_out = E_last_conv1d_out
        self.E_dims_first_fc = E_dims_first_fc
        self.dims_latent = dims_latent
        self.dropout_prob = dropout_prob
        self.act_func = act_func
        self.param_act_func = param_act_func
        self.num_fc_layer_decoder = num_fc_layer_decoder
        self.gru_hidden_size = gru_hidden_size
        self.num_gru = num_gru
        self.distribution = distribution
        self.scale_latent_space = scale_latent_space
        self.device = device        
        
    #------------------------------------------------------
    
    # define encoder
    def encoder(self, x):
#         print('----------ENCODER---------------')
        
        x = self.E_conv1d(x)
        x = x.view(x.size(0), -1)
        x = F.selu(self.E_fc(x))
        mu_z = self.E_mean(x)
        mu_z = mu_z / mu_z.norm(dim=-1, keepdim=True)
        sigma_z = F.softplus(self.E_sigma(x)) + 1
#         print(mu_z.shape, sigma_z.shape)
        
        
        return mu_z, sigma_z
    #------------------------------------------------------
    
    # define reparametarization trick
    def reparametarization(self, mu, sigma):
        
        q_z = VonMisesFisher(mu, sigma)
        p_z = HypersphericalUniform(self.dims_latent - 1, device=self.device)
       
        return q_z, p_z
    #------------------------------------------------------
    
    # define decoder
    def decoder(self, z):
#         print('----------DECODER---------------')
        
        z = F.selu(self.D_fc1(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.dims_output_data[0], 1)
        z, h = self.D_gru(z)
        z_reshape = z.contiguous().view(-1, z.size(-1))
        x_bar = F.softmax(self.D_fc2(z_reshape), dim=1)
        x_bar = x_bar.contiguous().view(z.size(0), -1, x_bar.size(-1))
        
        return x_bar
    #------------------------------------------------------
    
    # define forward pass
    def forward(self, x):
        
        mu, sigma = self.encoder(x)
        q_z, p_z = self.reparametarization(mu, sigma)
        z = q_z.rsample()
        x_bar = self.decoder(z)
        
        return x_bar, mu, sigma, z, q_z, p_z
#################################################################################################
    
    

# train, val and text network
class SmIm2Sm_Network():

    def __init__(
        self, 
        dims_input_data: Tuple, 
        dims_output_data: Tuple, 
        smiles_char_map: List = [], 
        max_string_len: int = 250, 
        padding: str = 'right', 
        num_kernels: List[int] = [9, 9, 10],
        size_kernels: List[int] = [9, 9, 11], 
        num_fc_layer_encoder: int = 0, 
        dropout_prob: float = 0.0, 
        dims_latent: int = 100, 
        act_func: str = 'relu', 
        param_act_func: Dict = {'alpha':1.0, 'negative_slope':1e-2}, 
        scale_latent_space: float = 1e-2, 
        num_fc_layer_decoder: int = 0, 
        gru_hidden_size: int = 500, 
        num_gru: int = 4, 
        layer_type: int = '1dcnn_gru', 
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weight_init: str = 'xvr_unifrm', 
        loss_type: str = 'bce_kld', 
        solver_type: str = 'adam', 
        num_epoch: int = 2, 
        batch_size: int = 128, 
        learning_rate: float = 1e-3, 
        num_train_test_samples: int = 1000,    
        save_result_ateach_epoch: int = 1, 
        result_savepath: str = 'cache/',
        # plot_network = True,
        # smiles_char_filename = '',
        # smiles_max_length_filename=''
    ) -> None:
        
        torch.manual_seed(1)# to control the random number generator
        np.random.seed(1)
        # define network
        if layer_type=='1dcnn_gru':# for encode-1dcnn svae and decoder-gru
            network = CNN1D_GRU_SVAE(
                dims_input_data, 
                dims_output_data, 
                num_kernels=num_kernels, 
                size_kernels=size_kernels, 
                num_fc_layer_encoder=num_fc_layer_encoder, 
                dropout_prob=dropout_prob, 
                dims_latent=dims_latent, 
                act_func=act_func, 
                param_act_func=param_act_func, 
                scale_latent_space=scale_latent_space, 
                num_fc_layer_decoder=num_fc_layer_decoder, 
                gru_hidden_size=gru_hidden_size, 
                num_gru=num_gru, 
                device=device # Example condition 
            )# initialize the network
        else:
            raise ValueError('Define your network with layer: "{}"' .format(layer_type))
        network = network.to(device)# move network to the gpu (if available)
    #     if torch.cuda.device_count() > 1:
    #         print("Let's use", torch.cuda.device_count(), "GPUs!")
    #         network = nn.DataParallel(network)

        # print the network summary
        print('-'*70)
        print('Network summary')
        summary(network, input_size=(batch_size, dims_input_data[0], dims_input_data[1]), device=device)
        print('-'*70)
        
        self.device = device
        self.dims_input_data= dims_input_data
        self.dims_output_data = dims_output_data
        self.smiles_char_map = smiles_char_map
        self.max_string_len = max_string_len
        self.padding = padding
        self.num_kernels = num_kernels
        self.size_kernels = size_kernels
        self.num_fc_layer_encoder = num_fc_layer_encoder
        self.dims_latent = dims_latent
        self.act_func = act_func
        self.param_act_func = param_act_func
        self.scale_latent_space = scale_latent_space
        self.num_fc_layer_decoder = num_fc_layer_decoder
        self.gru_hidden_size = gru_hidden_size
        self.num_gru = num_gru
        # self.plot_network = plot_network
        # self.smiles_char_filename = smiles_char_filename
        # self.smiles_max_length_filename = smiles_max_length_filename
        
        self.network = network
        self.solver_type = solver_type
        self.learning_rate = learning_rate
        
        self.layer_type = layer_type
        self.weight_init = weight_init
        self.loss_type = loss_type
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        
        self.num_train_test_samples = num_train_test_samples
        self.save_result_ateach_epoch = save_result_ateach_epoch
        
        # weight initialization and optimizer
        weights_initilizer(self.network, self.weight_init)# parameter initialization
        self.optimization_solver()
        
        # if self.plot_network:
        #     outputs = self.network(torch.Tensor(*((1,) + dims_input_data)).to(self.device))
        #     dot = make_dot(outputs[:3] , params=dict(list(self.network.named_parameters())))
        #     dot.format = 'pdf'  # Set the desired format ('pdf', 'png', etc.)
        #     dot.render("Full_Network")
        
        # intermediate results save path
        self.result_save_filename = ''.join(['smim2sm_net_', self.layer_type, 
             '_nkrnl_', str(self.num_kernels).replace(' ','_').strip('[]').replace(',',''), 
             '_szkrnl_', str(self.size_kernels).replace(' ','_').strip('[]').replace(',',''), 
             '_nefclr_', str(self.num_fc_layer_encoder), 
             '_drput_', str(dropout_prob), 
             '_nlntdms_', str(self.dims_latent), 
             '_afunc_', self.act_func, 
             '_ndfclr_', str(self.num_fc_layer_decoder), 
             '_szgruhdn_', str(self.gru_hidden_size), 
             '_ngru_', str(self.num_gru),
             '_wint_', self.weight_init, 
             '_los_', self.loss_type, 
             '_slvr_', self.solver_type, 
             '_npoch_', str(self.num_epoch), 
             '_bsz_', str(self.batch_size), 
             '_lr_', str(self.learning_rate)]).replace('.','_')
        
        self.result_savepath = ''.join([result_savepath, self.result_save_filename, '/']).replace('.','_')
        self.result_savepath = create_folder(self.result_savepath)
        # to save best network    
        self.best_network_save_filename = ''.join(
            ['best_', 
             self.result_savepath.split('/')[-2] 
            ]
        ).replace('.','_')
            
       #------------------------------------------------------
    
    # define loss function
    def loss_function(
        self, 
        x: torch.Tensor, 
        x_recon: torch.Tensor, 
        mu: torch.Tensor, 
        sigma: torch.Tensor, 
        q_z: Any, 
        p_z: Any
        ) -> torch.Tensor:
        
        if self.loss_type=='bce_kld':
            BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
            if self.network.distribution == 'vmf':
                KLD = torch.distributions.kl.kl_divergence(q_z, p_z).sum()
            else:
                raise ValueError('Define your loss function for the distribution: "{}"' .format(self.network.distribution))
            loss = BCE + KLD
#             print('loss (BCE): {}' .format(BCE))
#             print('loss (KLD): {}' .format(KLD))
        else:
            raise ValueError('Define your loss function: "{}"' .format(self.loss_type))

        return loss, KLD, BCE
    
    #------------------------------------------------------
    
    # define optimizer/solver
    def optimization_solver(self) -> Any:
        
        if self.solver_type=='adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        else:
            raise ValueError('Define your optimization solver: "{}"' .format(self.solver_type))
    #------------------------------------------------------
    
    # train function for an epoch
    def train_1epoch(self) -> float:

        self.network.train()# switch to train mode

        # data shuffle and minibatch settings
        num_train_samples = self.X_train.shape[0]
        idx = np.random.permutation(num_train_samples)
        num_iteration = int(np.ceil(float(num_train_samples)/self.batch_size)) 

        epoch_train_loss = 0# loss accumulation
        epoch_train_kld_loss = 0
        epoch_train_bce_loss = 0
        
        for i in tqdm(range(num_iteration)):# iteration over each minibatch

            start_idx = (i*self.batch_size)%num_train_samples
            _x = self.X_train[idx[start_idx:start_idx+self.batch_size], :].type(torch.float32).to(self.device)# take minibatch and move data to gpu (if available)
        
            self.optimizer.zero_grad()# set parameter gradients as zeros
            
            _x_bar, _mu, _sigma, _z, _q_z, _p_z = self.network(_x)# forward
            loss, kld_loss, bce_loss = self.loss_function(_x, _x_bar, _mu, _sigma, _q_z, _p_z)# loss calculation

            loss.backward()# backword
            self.optimizer.step()# parameter update
            epoch_train_loss += loss.item()# record minibatch loss
            epoch_train_kld_loss += kld_loss
            epoch_train_bce_loss += bce_loss

#         epoch_train_loss /= num_iteration# average loss
        epoch_train_loss /= num_train_samples# average loss
        epoch_train_kld_loss /= num_train_samples
        epoch_train_bce_loss /= num_train_samples

        return epoch_train_loss, epoch_train_kld_loss, epoch_train_bce_loss
    #------------------------------------------------------
    
    # test function
    def test(
        self, 
        X: torch.Tensor
    ) -> Tuple:

        self.network.eval()# switch to test mode

        # minibatch settings (don't need data shuffle)
        num_test_samples = X.shape[0]
        num_iteration = int(np.ceil(float(num_test_samples)/self.batch_size)) 

        X_bar = torch.zeros(X.shape)# for reconstruct data (output)
        epoch_test_loss = 0# loss accumulation
        epoch_test_kld_loss = 0
        epoch_test_bce_loss = 0
        
        with torch.no_grad():# no need to calculate gradient as its for test only
            for i in tqdm(range(num_iteration)):# iteration over each minibatch
                start_idx = (i*self.batch_size)%num_test_samples
                _x = X[start_idx:start_idx+self.batch_size, :].type(torch.float32).to(self.device)# take minibatch and move data to gpu (if available)
                
                _x_bar, _mu, _sigma, _z, _q_z, _p_z = self.network(_x)# forward
                loss, kld_loss, bce_loss = self.loss_function(_x, _x_bar, _mu, _sigma, _q_z, _p_z)# loss calculation
                X_bar[start_idx:start_idx+self.batch_size, :] = _x_bar.to("cpu") # move the resulkt to cpu for gpu memeory management
                
                epoch_test_loss += loss.item()# record minibatch loss
                epoch_test_kld_loss += kld_loss
                epoch_test_bce_loss += bce_loss
                
#         epoch_test_loss /= num_iteration# average loss
        epoch_test_loss /= num_test_samples# average loss
        epoch_test_kld_loss /= num_test_samples
        epoch_test_bce_loss /= num_test_samples
        
        return epoch_test_loss, X_bar, epoch_test_kld_loss, epoch_test_bce_loss
    #------------------------------------------------------
    
    def train(self) -> None:
        
        # to record avg. losses
        if self.count_epoch == 0:
            self.train_loss = np.zeros(self.num_epoch)
            self.val_loss = np.zeros(self.num_epoch)
            self.test_loss = np.zeros(self.num_epoch)
            
            self.epoch_test_train_kld_loss = np.zeros(self.num_epoch)
            self.epoch_test_train_bce_loss = np.zeros(self.num_epoch)
            
            self.epoch_test_val_kld_loss = np.zeros(self.num_epoch)
            self.epoch_test_val_bce_loss = np.zeros(self.num_epoch)
            
            self.epoch_test_test_kld_loss = np.zeros(self.num_epoch)
            self.epoch_test_test_bce_loss = np.zeros(self.num_epoch)
            
            
        else:
            self.train_loss = np.concatenate((self.train_loss, np.zeros(self.num_epoch-self.count_epoch)))
            self.val_loss = np.concatenate((self.val_loss, np.zeros(self.num_epoch-self.count_epoch)))
            self.test_loss = np.concatenate((self.test_loss, np.zeros(self.num_epoch-self.count_epoch)))
            
            self.epoch_test_train_kld_loss = np.concatenate((self.epoch_test_train_kld_loss, np.zeros(self.num_epoch-self.count_epoch)))
            self.epoch_test_train_bce_loss = np.concatenate((self.epoch_test_train_bce_loss, np.zeros(self.num_epoch-self.count_epoch)))
            
            self.epoch_test_val_kld_loss = np.concatenate((self.epoch_test_val_kld_loss, np.zeros(self.num_epoch-self.count_epoch)))
            self.epoch_test_val_bce_loss = np.concatenate((self.epoch_test_val_bce_loss, np.zeros(self.num_epoch-self.count_epoch)))
            
            self.epoch_test_test_kld_loss = np.concatenate((self.epoch_test_test_kld_loss, np.zeros(self.num_epoch-self.count_epoch)))
            self.epoch_test_test_bce_loss = np.concatenate((self.epoch_test_test_bce_loss, np.zeros(self.num_epoch-self.count_epoch)))

        
        for epoch in np.arange(self.count_epoch, self.num_epoch):#range(self.num_mini_epoch):# iterate over epoch

            # if self.count_epoch < self.num_epoch:
            print('Train epoch {}/({})' \
                    .format(self.count_epoch+1, self.num_epoch))
            # call train script for each epoch
            epoch_train_loss, epoch_train_kld_loss, epoch_train_bce_loss = self.train_1epoch()
            print('Avg. train loss: {:.6f}'.format(float(epoch_train_loss)))
            print('Avg. train kld loss: {:.6f}'.format(float(epoch_train_kld_loss)))
            print('Avg. train bce loss: {:.6f}'.format(float(epoch_train_bce_loss)))
            #+++++++++++++++++++++++++++++++++++++++++++++++++++

            # test on train data
            print('Test on train data epoch {}/({})' \
                    .format(self.count_epoch+1, self.num_epoch))
            # call test script
#             idx = np.random.permutation(self.X_train.shape[0])# take random samples from train set to test (As train set generally huge and its will be time consuming to test on whole train data)
            epoch_test_train_loss, _, epoch_test_train_kld_loss, epoch_test_train_bce_loss = self.test(self.X_train_test)
            self.train_loss[self.count_epoch] = epoch_test_train_loss
            self.epoch_test_train_kld_loss[self.count_epoch] = epoch_test_train_kld_loss
            self.epoch_test_train_bce_loss[self.count_epoch] = epoch_test_train_bce_loss
            print('Avg. test loss on train data: {:.6f}' \
                    .format(float(epoch_test_train_loss)))
            #+++++++++++++++++++++++++++++++++++++++++++++++++++

            # test on val data
            print('Test on val data epoch {}/({})' \
                    .format(self.count_epoch+1, self.num_epoch))
            # call test script
            epoch_test_val_loss, _, epoch_test_val_kld_loss, epoch_test_val_bce_loss = self.test(self.X_val)
            self.val_loss[self.count_epoch] = epoch_test_val_loss
            self.epoch_test_val_kld_loss[self.count_epoch] = epoch_test_val_kld_loss
            self.epoch_test_val_bce_loss[self.count_epoch] = epoch_test_val_bce_loss
            print('Avg. test loss on val data: {:.6f}' \
                    .format(float(epoch_test_val_loss)))
            #+++++++++++++++++++++++++++++++++++++++++++++++++++

            # test on test data
            print('Test on test data epoch {}/({})' \
                    .format(self.count_epoch+1, self.num_epoch))
            # call test script
            epoch_test_test_loss, _, epoch_test_test_kld_loss, epoch_test_test_bce_loss = self.test(self.X_test)
            self.test_loss[self.count_epoch] = epoch_test_test_loss
            self.epoch_test_test_kld_loss[self.count_epoch] = epoch_test_test_kld_loss
            self.epoch_test_test_bce_loss[self.count_epoch] = epoch_test_test_bce_loss
            print('Avg. test loss on test data: {:.6f}' \
                    .format(float(epoch_test_test_loss)))
            #+++++++++++++++++++++++++++++++++++++++++++++++++++

            # save different losses
            if self.count_epoch%min(1, self.save_result_ateach_epoch)==0:
                # save as plot image
                loss_save_filename = ''.join([self.result_savepath, \
                                                'all_loss_epoch_', str(self.num_epoch), \
                                                '_', self.result_savepath.split('/')[-2] \
                                                ]).replace('.','_')
                self.plot_loss_save_images(self.count_epoch, \
                                            loss_save_filename, \
                                            title=self.layer_type)
                # save as csv file
                self.save_loss_csv(self.count_epoch+1, \
                                    loss_save_filename)
            #+++++++++++++++++++++++++++++++++++++++++++++++++++

            # save the best trained network
            if((self.count_epoch==0) or (self.val_loss_best_network>epoch_test_val_loss)):             
                
                self.epoch_best_network = self.count_epoch
                self.train_loss_best_network = epoch_test_train_loss
                self.val_loss_best_network = epoch_test_val_loss
                self.test_loss_best_network = epoch_test_test_loss
                
                # note that we save the train loss based on the best val loss
                self.save_network(epoch=self.count_epoch+1, \
                                    network_save_filename=self.best_network_save_filename)

            # mandatory network save after fixed epoch
            if self.count_epoch%self.save_result_ateach_epoch==0:
                network_save_filename = ''.join(['network_epoch_', str(self.count_epoch), \
                                                    '_', self.result_savepath.split('/')[-2] \
                                                ]).replace('.','_')
                self.save_network(epoch=self.count_epoch+1,
                                    network_save_filename=network_save_filename)
            #+++++++++++++++++++++++++++++++++++++++++++++++++++
            
            # Plot Network Flowchart using Make Dot
            if self.count_epoch > 1:
                self.plot_network = False

            self.count_epoch += 1
            #+++++++++++++++++++++++++++++++++++++++++++++++++++
            # else:
                # save different losses (final)
        loss_save_filename = ''.join([self.result_savepath, \
                                        'final_all_loss_epoch_', str(self.num_epoch), \
                                        '_', self.result_savepath.split('/')[-2] \
                                        ]).replace('.','_')
        self.plot_loss_save_images(self.count_epoch-1, \
                                    loss_save_filename, \
                                    title=self.layer_type)
        # save as csv file
        self.save_loss_csv(self.count_epoch, \
                            loss_save_filename)
    #------------------------------------------------------ 
    
 
    
    # trained network save function
    def save_network(self, epoch=0,
                     acc=0.0, \
                     network_save_filename='network_best'):

        network_save_filename = ''.join([network_save_filename, '.pth'])
        print('Saving network in: "{}" as filename: "{}"' \
              .format(self.result_savepath, network_save_filename))
        
        source_file = os.path.abspath(__file__)
        # destination_file = ''.join([self.result_savepath, '/SmilesImage2Smiles_Network.py'])
        # shutil.copy(source_file, destination_file)
        with open(source_file, 'rb') as fp:
            file_ = fp.read()
            self.network_pyfile = pickle.dumps(file_)
            
        torch.save({\
                    'epoch': epoch,
                    'network_pyfile': self.network_pyfile,
                    'train_script' : self.train_script,
                    'size_kernels': self.size_kernels,
                    'num_kernels' : self.num_kernels,
                    'padding' : self.padding,
                    'num_fc_layer_encoder' : self.num_fc_layer_encoder,
                    'dims_latent' : self.dims_latent,
                    'act_func' : self.act_func,
                    'scale_latent_space' : self.scale_latent_space,
                    'num_fc_layer_decoder' : self.num_fc_layer_decoder,
                    'gru_hidden_size' : self.gru_hidden_size,
                    'num_gru' : self.num_gru,
                    'solver_type' : self.solver_type,
                    'learning_rate' : self.learning_rate,
                    'weight_init' : self.weight_init,
                    'batch_size' : self.batch_size,
                    'train_loss': self.train_loss[:epoch+1],
                    'train_loss_best_network': self.train_loss_best_network, 
                    'val_loss': self.val_loss[:epoch+1], 
                    'val_loss_best_network': self.val_loss_best_network,
                    'test_loss': self.test_loss[:epoch+1], 
                    'test_loss_best_network': self.test_loss_best_network,
                    'epoch_test_train_bce_loss':self.epoch_test_train_bce_loss[:epoch+1],
                    'epoch_test_train_kld_loss':self.epoch_test_train_kld_loss[:epoch+1], 
                    'epoch_test_val_bce_loss':self.epoch_test_val_bce_loss[:epoch+1],
                    'epoch_test_val_kld_loss':self.epoch_test_val_kld_loss[:epoch+1],
                    'epoch_test_test_bce_loss':self.epoch_test_test_bce_loss[:epoch+1],
                    'epoch_test_test_kld_loss':self.epoch_test_test_kld_loss[:epoch+1],
                    'acc': acc,
            #please take the the multiple gpu carefully using network.module
    #         'state_dict': self.network.module.state_dict() if torch.cuda.device_count() > 1 else self.network.state_dict(),
                    'state_dict': self.network.state_dict(), \
                    'optimizer': self.optimizer.state_dict() \
                   }, \
                   '' .join([self.result_savepath, \
                             network_save_filename]))
    #------------------------------------------------------
    
    # pre-trained network load function for test
    def load_test_network(self, temp_network_path=[]):
        
        if len(temp_network_path)==0:
            temp_network_path = '' .join([self.result_savepath, \
                                          self.best_network_save_filename, \
                                          '.pth'])
        if os.path.isfile(temp_network_path):
            print('Loading pre-trained network checkpoint from: "{}"' \
                  .format(temp_network_path))
            checkpoint = torch.load(temp_network_path, \
                                    map_location=self.device)
            #--------------------------------------------------
            
            if 'epoch' in checkpoint.keys():
                self.count_epoch = checkpoint['epoch']
                self.epoch_best_network = checkpoint['epoch']
            else:
                self.count_epoch = 0
                self.epoch_best_network = 0
            #+++++++++++++++++++++++++++++++++++++++++++++++++++
                
            if 'train_loss' in checkpoint.keys():
                self.train_loss = checkpoint['train_loss']
            else:
                self.train_loss = 'NA'
                
            if 'train_loss_best_network' in checkpoint.keys():
                self.train_loss_best_network = checkpoint['train_loss_best_network']
            else:
                self.train_loss_best_network = 'NA'
                
            if 'epoch_test_train_bce_loss' in checkpoint.keys():
                self.epoch_test_train_bce_loss = checkpoint['epoch_test_train_bce_loss']
            else:
                self.epoch_test_train_bce_loss = 'NA'
                
            if 'epoch_test_train_kld_loss' in checkpoint.keys():
                self.epoch_test_train_kld_loss = checkpoint['epoch_test_train_kld_loss']
            else:
                self.epoch_test_train_kld_loss = 'NA'
            #+++++++++++++++++++++++++++++++++++++++++++++++++++
                
            if 'val_loss' in checkpoint.keys():
                self.val_loss = checkpoint['val_loss']
            else:
                self.val_loss = 'NA'
                
            if 'val_loss_best_network' in checkpoint.keys():
                self.val_loss_best_network = checkpoint['val_loss_best_network']
            else:
                self.val_loss_best_network = 'NA'
                
            if 'epoch_test_val_bce_loss' in checkpoint.keys():
                self.epoch_test_val_bce_loss = checkpoint['epoch_test_val_bce_loss']
            else:
                self.epoch_test_train_bce_loss = 'NA'
                
            if 'epoch_test_val_kld_loss' in checkpoint.keys():
                self.epoch_test_val_kld_loss = checkpoint['epoch_test_val_kld_loss']
            else:
                self.epoch_test_val_kld_loss = 'NA'
            #+++++++++++++++++++++++++++++++++++++++++++++++++++
            
            if 'test_loss' in checkpoint.keys():
                self.test_loss = checkpoint['test_loss']
            else:
                self.test_loss = 'NA'
                
            if 'test_loss_best_network' in checkpoint.keys():
                self.test_loss_best_network = checkpoint['test_loss_best_network']
            else:
                self.test_loss_best_network = 'NA'
                
            if 'epoch_test_test_bce_loss' in checkpoint.keys():
                self.epoch_test_test_bce_loss = checkpoint['epoch_test_test_bce_loss']
            else:
                self.epoch_test_test_bce_loss = 'NA'
                
            if 'epoch_test_test_kld_loss' in checkpoint.keys():
                self.epoch_test_test_kld_loss = checkpoint['epoch_test_test_kld_loss']
            else:
                self.epoch_test_test_kld_loss = 'NA'
            #+++++++++++++++++++++++++++++++++++++++++++++++++++
            
            if 'network_pyfile' in checkpoint.keys():
                self.network_pyfile = checkpoint['network_pyfile']
            else:
                self.network_pyfile = 'NA'
            #+++++++++++++++++++++++++++++++++++++++++++++++++++
                
            self.network.load_state_dict(checkpoint['state_dict'])
            
            print('Loaded pre-trained network checkpoint from "{}"\nepoch: {} train loss: {} val loss: {} test loss: {}' \
                  .format(temp_network_path, self.epoch_best_network, self.train_loss[-1], self.val_loss[-1], self.test_loss[-1]) \
                 )
            print('Best models till "{}"\nepoch: {} best train loss: {} best val loss: {} best test loss: {}' \
                  .format(temp_network_path, self.epoch_best_network, self.train_loss_best_network, self.val_loss_best_network, self.test_loss_best_network) \
                 )

        else:
            print('No pre-trained network checkpoint found at "{}"' \
                  .format(temp_network_path) \
                 )
        print('------------------------------------------------------')
    
    
    # #------------------------------------------------------
    
    # plot and save (image) different losses
    def plot_loss_save_images(self, epoch, \
                              save_image_filename=[], \
                              title='VAE loss', \
                              marker='None', \
                              markersize=10, \
                              linewidth=2, \
                              linestyle='-', \
                              title_fontsize=25, \
                              xyticks_fontsize=20 \
                             ):

        """
        Plot the different train, val and test loss against epochs. 
        Input: 
        Output:
        """
        X = range(epoch+1)

        plt.plot(X, self.train_loss[:epoch+1], \
                 marker=marker, \
                 markersize=markersize, \
                 linewidth=linewidth, \
                 linestyle=linestyle, \
                 label='loss: train' \
                )
        plt.plot(X, self.val_loss[:epoch+1], \
                 marker=marker, \
                 markersize=markersize, \
                 linewidth=linewidth, \
                 linestyle=linestyle, \
                 label='loss: val' \
                )
        plt.plot(X, self.test_loss[:epoch+1], \
                 marker=marker, \
                 markersize=markersize, \
                 linewidth=linewidth, \
                 linestyle=linestyle, \
                 label='loss: test' \
                )
        plt.xticks(fontsize=xyticks_fontsize)
        plt.yticks(fontsize=xyticks_fontsize)
        plt.xlabel('epoch', fontsize=title_fontsize)
        plt.ylabel('avg. loss', fontsize=title_fontsize)
        plt.grid(linestyle='--')
        plt.legend(loc='upper right')
        plt.title(title, fontsize=title_fontsize)   

         # save the plot    
        if len(save_image_filename):
            plt.savefig('' .join([save_image_filename,'.png']), \
                        bbox_inches='tight' \
                       )#save the plot in png form

        plt.show(block=False)
        plt.close()
    #------------------------------------------------------
    
    #save (csv) different losses
    def save_loss_csv(self, epoch, \
                      save_csv_filename='temp' \
                     ):
        
        dict_loss = OrderedDict({'epoch': list(range(1, epoch+1)), 
                                 'train_loss': self.train_loss[ :epoch], 
                                 'val_loss': self.val_loss[:epoch], 
                                 'test_loss': self.test_loss[:epoch], 
                                 'train_bce_loss': self.epoch_test_train_bce_loss[:epoch],
                                 'train_kld_loss': self.epoch_test_train_kld_loss[:epoch], 
                                 'val_bce_loss': self.epoch_test_val_bce_loss[:epoch],
                                 'val_kld_loss': self.epoch_test_val_kld_loss[:epoch],
                                 'test_bce_loss': self.epoch_test_test_bce_loss[:epoch],
                                 'test_kld_loss': self.epoch_test_test_kld_loss[:epoch]
                                })
        # print(dict_loss)
        

        df_loss = pd.DataFrame.from_dict(dict_loss)
        loss_csv_filepath = ''.join([save_csv_filename, '.csv'])
        
        # # Concatenate DataFrames vertically
        # if os.path.isfile(loss_csv_filepath):
        #     existing_losses = pd.read_csv(loss_csv_filepath)
        #     df_loss = pd.concat([existing_losses, df_loss], axis=0)
        #     # Reset the index if needed
        #     df_loss.reset_index(drop=True, inplace=True)
            
        df_loss.to_csv(loss_csv_filepath, index=False)
        #----------------------------------------------------
    
    #+++++++++++++++++++++++++++++++++++++++++++++++++++ 
    
    # reconstruc smiles 
    def smiles_reconstruct_molecule(self, smiles, tokenize='', draw_num_samples=100, plot_nrows_module=10, save_path='temp_results/', dataset_name='temp'):

        self.network.eval()# switch to test mode

        # minibatch settings (don't need data shuffle)
        num_test_samples = len(smiles)
        num_iteration = int(np.ceil(float(num_test_samples)/self.batch_size))

        # to store the reconstructed smiles and their validation flag
        smiles_recon = [None]*num_test_samples
        check_ids_smiles_recon = [None]*num_test_samples

        print('<===smiles reconstruction===>')
        # to store the latent space representation
        with torch.no_grad(): 
            for i in tqdm(range(num_iteration)):
                start_idx = (i*self.batch_size)%num_test_samples
                _smiles = smiles[start_idx:start_idx+self.batch_size]# take minibatch
                # smiles to onehot representation
                _onehot_rep = PARSE_SMILES.smiles2onehot_vector(_smiles, self.smiles_char_map['domain'], max_string_len=self.max_string_len, padding=self.padding, tokenize=tokenize)
                _onehot_rep = torch.from_numpy(_onehot_rep).type(torch.float32).to(self.device)# data type conversion and move data to gpu (if available)

                # reconstruct smiles
#                 _onehot_rep_bar, _, _ = self.network(_onehot_rep)# forward
                _onehot_rep_bar, _, _, _, _, _ = self.network(_onehot_rep)# forward                
                _onehot_rep_bar = _onehot_rep_bar.to("cpu").numpy()
                smiles_recon[start_idx:start_idx+self.batch_size] = PARSE_SMILES.onehot2smiles(_onehot_rep_bar, self.smiles_char_map['co-domain'])
                # validity checking of reconstructed smiles
                _, check_ids_smiles_recon[start_idx:start_idx+self.batch_size] = PARSE_SMILES.check_smiles_validity(smiles_recon[start_idx:start_idx+self.batch_size])
        #+++++++++++++++++++++++++++++++++++++++++++++++++++

        num_valid_smiles = len(np.asarray(check_ids_smiles_recon)[check_ids_smiles_recon])
        recon_acc = round(100.0*(float(num_valid_smiles)/num_test_samples), 2)
        # find exact reconstruction
        num_exact_reconst_smiles = 0
        check_ids_smiles_exact_recon = [False]*num_test_samples
        for i in range(num_test_samples):
            if smiles[i] == smiles_recon[i]:
                num_exact_reconst_smiles += 1
                check_ids_smiles_exact_recon[i] = True
        exact_recon_acc = round(100.0*(float(num_exact_reconst_smiles)/num_test_samples), 2)    
        
        print('Valid reconstructed smiles: total: {} ({})\nacc: {}(%) ' .format(num_valid_smiles, num_test_samples, recon_acc))
        print('Valid exact reconstructed smiles: total: {} ({})\nacc: {}(%) ' .format(num_exact_reconst_smiles, num_test_samples, exact_recon_acc))

        dict_result = OrderedDict({'dataset_name': dataset_name, 'total_number_of_samples':num_test_samples, 'total_valid_reconstructed_samples':num_valid_smiles, 'acc': recon_acc, 'total_exact_reconst_smiles':num_exact_reconst_smiles, 'exact_recon_acc':exact_recon_acc})
        dict_smiles = OrderedDict({'smiles':smiles, 'smiles_recon':smiles_recon, 'smiles_valid':check_ids_smiles_recon,'smiles_exact_recon':check_ids_smiles_exact_recon})

        # save smiles, reconstructed smiles in CSV and pkl file
        save_path = create_folder(save_path)
        save_filename = ''.join([save_path, dataset_name, '_reconstructed_molecules_tsample_', str(num_test_samples), '_vsamples_', str(num_valid_smiles), '_acc_', str(recon_acc)])
        csv_save_filename = ''.join([save_filename,'.csv'])
        pd.DataFrame.from_dict(dict_smiles).to_csv(csv_save_filename, header=True, encoding='utf-8', index=False)
        plk_save_filename = ''.join([save_filename,'.pkl'])
        save_dict_pickle(dict_smiles, plk_save_filename)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++

        # draw smiles & reconstructed ones and save
        if num_valid_smiles:
            # take the valid reconstructed smiles
            valid_smiles = np.asarray(smiles)[check_ids_smiles_recon]
            valid_smiles_recon = np.asarray(smiles_recon)[check_ids_smiles_recon]

            # take radom smiles to draw
            random_idx = np.random.choice(min(valid_smiles.shape[0], draw_num_samples), draw_num_samples, replace=False)
            valid_smiles = valid_smiles[random_idx]
            valid_smiles_recon = valid_smiles_recon[random_idx]
            ids = list(np.asarray(np.arange(len(smiles)))[check_ids_smiles_recon][random_idx])

            all_smiles = [None]*(2*draw_num_samples) 
            legends = [None]*(2*draw_num_samples) 

            for i in range(draw_num_samples):
                all_smiles[2*i] = valid_smiles[i]
                legends[2*i] = '' .join(['mol_id_', str(ids[i]), '_original'])
                all_smiles[2*i+1] = valid_smiles_recon[i]
                legends[2*i+1] = '' .join(['mol_id_', str(ids[i]), '_reconstructed'])

            img = PARSE_SMILES.draw_molecules(all_smiles, plot_nrows_module, legends=legends)
            img.save(''.join([save_filename, '.png']))

        return dict_result, dict_smiles
    #------------------------------------------------------


#################################################################################################