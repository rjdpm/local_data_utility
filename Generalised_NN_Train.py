#!/usr/bin/env python3

from typing import Union, Tuple, Dict, List, Any
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from collections import OrderedDict
import seaborn as sns

import torch
import tarfile
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as tt
from torchvision import datasets, transforms
from torchvision.models import resnet18, mobilenet_v2
from torchvision.models import mobilenet_v3_large
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.datasets import CIFAR100
from transformers import ViTModel, ViTFeatureExtractor


__all__ = [
    'utils_',
    'accuracy',
    'plot_sample_with_label',
    'unlearn',
    'accuracy_multiclass_unlearn',
    'subset_accuracy',
    'load_cifar10',
    'load_svhn',
    'load_mnist',
    'load_fashionmnist',
    'load_cifar100',
    'View_',
    'LeNet32_',
    'ResidualBlock_',
    'ResNet9_',
    'AllCNN_',
    'MobileNet',
    'ResNet18',
    'TimeDistributed',
    'GLU',
    'GatedResidualNetwork',
    'PositionalEncoder',
    'VariableSelectionNetwork',
    'TFT',
    'LSTMnetwork'
]

class utils_:
    
    def __init__(self,
                 network: Any = None,#mobilenet_v2(weights='IMAGENET1K_V1'),
                 input_size: tuple = (3, 32, 32),#(Channel, Height, Width) or (Channel, Length)
                 num_classes: int = 10,
                 padding: tuple | int = 0,
                 solver_type: str = 'adam',
                 learning_rate: float = 1e-3,
                 batch_size: int = 128,
                 num_epochs: int = 100,
                 model_save_name: str = '',
                 model_name: str = ''
                 ):
        self.input_size = input_size
        self.num_input_channels = input_size[0]
        self.num_classes = num_classes
        self.padding = padding
        self.solver_type = solver_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.result_savepath = './results/'
        self.model_save_name = model_save_name
        # self.network = network
        self.result_savepath = self.create_folder(self.result_savepath)
        self.test_loss_best_network = 1e10
        self.val_data = None
        self.model_name = model_name
        self.best_model_save_path = ''.join([self.result_savepath,
                                             'best_network',
                                             '_lr_', str(self.learning_rate).replace('.', '_').replace('-', '_'),
                                             '_batsz_', str(self.batch_size),
                                             '_solver_', str(self.solver_type),
                                             self.model_save_name,
                                             '.pth']
                                            )
        print('='*100)
        print('-'*100)
        
        #=======================================================================
        if network:
            self.network = network
        else:
            if self.model_name == 'LeNet32':    
                self.network = LeNet32_(n_classes=self.num_classes,
                                        num_input_channels=self.num_input_channels,
                                        padding=self.padding
                                        )
                self.network = self.network.to(self.device)   
            #=======================================================================
            elif self.model_name == 'ResNet9':
                self.network = ResNet9_(n_classes=self.num_classes,
                                        num_input_channels=self.num_input_channels,
                                        padding=self.padding)
                self.network = self.network.to(self.device)
            #=======================================================================
            elif self.model_name == 'AllCNN':
                self.network = AllCNN_(n_channels = self.num_input_channels,
                                    num_classes = self.num_classes,
                                    padding = self.padding
                                    )
                self.network = self.network.to(self.device)
            #=======================================================================
            elif self.model_name == 'ResNet18':
                # self.network = ResNet18(n_classes=self.num_classes)
                self.network = resnet18(weights='IMAGENET1K_V1')
                # conv_weight = self.network.conv1.weight
                # fc_weight = self.network.fc.weight
                # fc_bias = self.network.fc.bias
                # conv_weight = torch.nn.Parameter((conv_weight.sum(dim = 1).unsqueeze(dim = 1))/3)
                    
                self.network.conv1 = nn.Conv2d(in_channels = self.num_input_channels,
                                            out_channels = self.network.conv1.out_channels,
                                            kernel_size=self.network.conv1.kernel_size,
                                            stride=self.network.conv1.stride,
                                            padding=self.network.conv1.padding,
                                            bias=False
                                            )
                # self.network.conv1.weight = conv_weight
                self.network.fc = nn.Linear(in_features=self.network.fc.in_features,
                                            out_features=self.num_classes,
                                            bias=True)
                # self.network.fc.weight = torch.nn.Parameter(fc_weight[:self.num_classes])
                # self.network.fc.bias = torch.nn.Parameter(fc_bias[:self.num_classes])
            #========================================================================
                
            elif self.model_name == 'MobileNet_v2':
                self.network = mobilenet_v2(weights='IMAGENET1K_V1')
                # conv_weight = self.network.features[0][0].weight
                # fc_weight = self.network.classifier[-1].weight
                # fc_bias = self.network.classifier[-1].bias
                # if data_name == 'mnist' or data_name == 'fashionMNIST':
                #     conv_weight = torch.nn.Parameter((conv_weight.sum(dim = 1).unsqueeze(dim = 1))/3)
                self.network.features[0][0] = nn.Conv2d(in_channels = self.num_input_channels,
                                                        out_channels = self.network.features[0][0].out_channels,
                                                        kernel_size=self.network.features[0][0].kernel_size,
                                                        stride=self.network.features[0][0].stride,
                                                        padding=self.network.features[0][0].padding,
                                                        bias=False)
                # self.network.features[0][0].weight = conv_weight
                self.network.classifier[-1] = nn.Linear(in_features=self.network.classifier[-1].in_features,
                                                        out_features=self.num_classes,
                                                        bias=True)
                # self.network.classifier[-1].weight = torch.nn.Parameter(fc_weight[:self.num_classes])
                # self.network.classifier[-1].bias = torch.nn.Parameter(fc_bias[:self.num_classes])
            #========================================================================        
        
        total_input_size = (self.batch_size, ) + self.input_size
        print(f'Input Size per Batch = {total_input_size}') 
        print('-'*100)
        print('\nModel Architecture:\n', )
        summary(self.network, total_input_size, device=str("cpu"))
        self.optimization_solver()
    #-------------------------------------------------------------------------------------------------------------------------
        
        
    def data_subset(self, dataloader, subset_size = 0.2):
        
        original_dataset = dataloader.dataset
        subset_size = int(subset_size*len(original_dataset))
        perm = np.random.permutation(np.arange(len(original_dataset)))
        subset_indices = list(perm[:subset_size])
        complement_indices = list(perm[subset_size:])
        subset_dataset = Subset(original_dataset, subset_indices)
        complement_subset_dataset = Subset(original_dataset, complement_indices)
        subset_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=self.batch_size, shuffle=True)
        complement_loader = torch.utils.data.DataLoader(complement_subset_dataset, batch_size=self.batch_size, shuffle=True)
        
        return subset_loader, complement_loader
    
    #-------------------------------------------------------------------------------------------------------------------------
    
    def create_folder(self, folder_name):
        if len(folder_name):
            if not os.path.isdir(folder_name):
                os.makedirs(folder_name)
            
        return folder_name
    
    #-------------------------------------------------------------------------------------------------------------------------
        
    # define optimizer/solver
    def optimization_solver(self):
        
        if self.solver_type=='adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        elif self.solver_type=='SGD':
            self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)
        elif self.solver_type=='Rprop':
            self.optimizer = optim.Rprop(self.network.parameters(), lr=self.learning_rate)
        elif self.solver_type=='RMSprop':
            self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate)
        elif self.solver_type=='RAdam':
            self.optimizer = optim.RAdam(self.network.parameters(), lr=self.learning_rate)
        else:
            raise ValueError('Define your optimization solver: "{}"\n' .format(self.solver_type))
    #-------------------------------------------------------------------------------------------------------------------------
    
    def loss_function(self, y_bar, y):
        
        loss = F.cross_entropy(y_bar, y, reduction = 'sum')
        
        return loss

    #-------------------------------------------------------------------------------------------------------------------------
    
    def train_1epoch(self, model, optimizer, data_loader):
        
        model.train()# switch to train model
        model = model.to(self.device)
        epoch_train_loss = 0
        with torch.autograd.set_detect_anomaly(True):
            for batch, label in data_loader:
                batch, label = batch.to(self.device), label.to(self.device)
                optimizer.zero_grad()
                y_pred = model(batch.type(torch.float))
                loss = self.loss_function(y_pred, label)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            epoch_train_loss /= len(data_loader.dataset)
        
        return epoch_train_loss
    
    #-------------------------------------------------------------------------------------------------------------------------
    
    def test(self, model, dataloader, aprox_ndigits = 2):
        
        model.eval()
        epoch_test_loss = 0# loss accumulation
        actual_labels = []#*len(dataloader.dataset)
        predicted_labels = []#*len(dataloader.dataset)
        model.to(self.device)
        
        with torch.no_grad():# no need to calculate gradient as its for test
            for batch, labels in dataloader:
                data, labels = batch.to(self.device), labels.to(self.device)
                predicted = model(data.type(torch.float))
                loss = self.loss_function(predicted, labels)
                epoch_test_loss += loss.item()# record minibatch loss
                predicted = predicted.cpu().detach().numpy()
                predicted = np.argmax(predicted, axis=1)
                predicted_labels.extend(predicted)
                actual_labels.extend(labels.cpu().detach().numpy())
            epoch_test_loss /= len(dataloader.dataset)# record minibatch loss
            
            confusion_matrix = metrics.confusion_matrix(actual_labels, predicted_labels)#Computing the confusion matrix
            acc = round((sum(confusion_matrix.diagonal())/np.sum(confusion_matrix))*100, aprox_ndigits)
            classwise_accuracy = [round((confusion_matrix[i,i]/np.sum(confusion_matrix, axis=1)[i])*100, aprox_ndigits) if np.sum(confusion_matrix, axis=1)[i] != 0 else 'NA' for i in range(confusion_matrix.shape[0])]
            
        return epoch_test_loss, confusion_matrix, acc, classwise_accuracy, actual_labels, predicted_labels

    #-------------------------------------------------------------------------------------------------------------------------
    
    def train(self):
        
        # to record avg. losses  
        if self.count_epoch == 0:
            self.train_loss = np.zeros(self.num_epochs)
            self.test_loss = np.zeros(self.num_epochs)
            self.val_loss = np.zeros(self.num_epochs)
            self.train_acc = np.zeros(self.num_epochs)
            self.test_acc = np.zeros(self.num_epochs)
            self.val_acc = np.zeros(self.num_epochs)
        else:
            self.train_loss = np.concatenate((self.train_loss, np.zeros(self.num_epochs-self.count_epoch)))
            self.test_loss = np.concatenate((self.test_loss, np.zeros(self.num_epochs-self.count_epoch)))
            self.val_loss = np.concatenate((self.val_loss, np.zeros(self.num_epochs-self.count_epoch)))
            self.train_acc = np.concatenate((self.train_acc, np.zeros(self.num_epochs-self.count_epoch)))
            self.test_acc = np.concatenate((self.test_acc, np.zeros(self.num_epochs-self.count_epoch)))
            self.val_acc = np.concatenate((self.val_acc, np.zeros(self.num_epochs-self.count_epoch)))
            
        self.train_loader = DataLoader(self.train_data, self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, self.batch_size, shuffle=True)
        if self.val_data:
            self.val_loader = DataLoader(self.val_data, self.batch_size, shuffle=True)
        else:
            print('Validation set not found. Separating Test set equally into test & val set.')
            self.val_loader, self.test_loader = self.data_subset(dataloader=self.test_loader, subset_size=0.50)
        
        for epoch in tqdm(np.arange(self.count_epoch, self.num_epochs)):
            
            print('\nTrain epoch {}/({})\n' .format(epoch+1, self.num_epochs))
            # epoch_train_loss = self.train_1epoch()
            epoch_train_loss = self.train_1epoch(self.network, self.optimizer, self.train_loader)
            print('Average train loss = {}\n'.format(epoch_train_loss))
            
            # test on train data
            print('Test on train data epoch {}/({})\n' .format(epoch+1, self.num_epochs))
            self.test_train_loader, _ = self.data_subset(self.train_loader, 0.20)
            epoch_test_train_loss, train_confusion_matrix, epoch_test_train_acc, self.train_classwise_accuracy, _, _ = self.test(self.network, self.test_train_loader)
            self.train_loss[epoch] = epoch_test_train_loss
            self.train_acc[epoch] = epoch_test_train_acc
            print('Avg. test loss on train data: {:.6f} and acc: {:.6f}\n' .format(float(epoch_test_train_loss), float(epoch_test_train_acc)))
            
            # test on val data
            print('Test on validation data epoch {}/({})\n' .format(epoch+1, self.num_epochs))
            self.test_val_loader, _ = self.data_subset(self.val_loader, 0.20)
            epoch_test_val_loss, val_confusion_matrix, epoch_test_val_acc, self.val_classwise_accuracy, _, _ = self.test(self.network, self.test_val_loader)
            self.val_loss[epoch] = epoch_test_val_loss
            self.val_acc[epoch] = epoch_test_val_acc
            print('Avg. test loss on validation data: {:.6f} and acc: {:.6f}\n' .format(float(epoch_test_val_loss), float(epoch_test_val_acc)))
            
            # test on test data
            print('Test on test data epoch {}/({})\n' .format(epoch+1, self.num_epochs))
            self.test_test_loader, _ = self.data_subset(self.test_loader, 0.20)
            epoch_test_test_loss, test_confusion_matrix, epoch_test_test_acc, self.test_classwise_accuracy, _, _ = self.test(self.network, self.test_test_loader)
            self.test_loss[epoch] = epoch_test_test_loss
            self.test_acc[epoch] = epoch_test_test_acc
            print('Avg. test loss on test data: {:.6f} and acc: {:.6f}\n' .format(float(epoch_test_test_loss), float(epoch_test_test_acc)))
            
            # save the best trained network
            if((self.count_epoch==0) or (epoch_test_val_acc > self.val_acc_best_network)):             
                self.epoch_best_network = self.count_epoch
                self.train_loss_best_network = epoch_test_train_loss
                self.val_acc_best_network = epoch_test_val_acc
                self.test_acc_best_network = epoch_test_test_acc
                self.train_confusion_matrix = train_confusion_matrix
                self.test_confusion_matrix = test_confusion_matrix
                self.val_confusion_matrix = val_confusion_matrix
                self.save_model(epoch = self.epoch_best_network, model_save_path=self.best_model_save_path)
                print(f'Saving model in path: {self.best_model_save_path}\n')
                
            self.save_loss_csv(self.count_epoch,
                               save_csv_filename = ''.join([self.result_savepath,
                                                            'Training_Loss']
                                                           )
                               )
            self.plot_loss_save_images(epoch=self.count_epoch,
                                       loss_type='loss',
                                       save_image_filename=''.join([self.result_savepath,
                                                                    'training_time_loss_plot']
                                                                   )
                                       )
            self.plot_loss_save_images(epoch=self.count_epoch,
                                       loss_type='acc',
                                       save_image_filename=''.join([self.result_savepath,
                                                                    'training_time_acc_plot']
                                                                   )
                                       )
            self.count_epoch = epoch + 1
            
        self.save_confusion_matrix(confusion_matrix=train_confusion_matrix,
                                   name = ''.join([self.result_savepath,
                                                   'train_confusion_matrix']
                                                  )
                                   )
        self.save_confusion_matrix(confusion_matrix=test_confusion_matrix,
                                   name = ''.join([self.result_savepath,
                                                   'test_confusion_matrix']
                                                  )
                                   )
        self.save_confusion_matrix(confusion_matrix=test_confusion_matrix,
                                   name = ''.join([self.result_savepath,
                                                   'val_confusion_matrix']
                                                  )
                                   )
    #-------------------------------------------------------------------------------------------------------------------------
    
        
    def save_model(self, epoch, model_save_path = './'):
        
        torch.save({
            'epoch':epoch+1,
            'train_loss':self.train_loss[:epoch+1],
            'test_loss':self.test_loss[:epoch+1],
            'val_loss':self.val_loss[:epoch+1],
            
            'epoch_train_loss':self.train_loss[epoch],
            'epoch_test_loss':self.test_loss[epoch],
            'epoch_val_loss':self.val_loss[epoch],
            
            'train_acc':self.train_acc[:epoch+1],
            'test_acc':self.test_acc[:epoch+1],
            'val_acc':self.val_acc[:epoch+1],
            
            'epoch_train_acc':self.train_acc[epoch],
            'epoch_test_acc':self.test_acc[epoch],
            'epoch_val_acc':self.val_acc[epoch],
            
            'train_classwise_accuracy':self.train_classwise_accuracy,
            'test_classwise_accuracy':self.test_classwise_accuracy,
            'val_classwise_accuracy':self.val_classwise_accuracy,
            
            'state_dict':self.network.state_dict(),
            'optimizer':self.optimizer.state_dict()
        }, model_save_path)
        
    #-------------------------------------------------------------------------------------------------------------------------
        
        
    def load_network(self, network_path=''):
        
        if len(network_path)==0:
            network_path = self.best_model_save_path
        
        if os.path.isfile(network_path):
            print('Loading pre-trained network checkpoint from: "{}"\n'.format(network_path))
            checkpoint = torch.load(network_path, map_location=self.device)
            
            self.count_epoch = checkpoint['epoch']
            
            self.train_acc = checkpoint['train_acc']
            self.test_acc = checkpoint['test_acc']
            self.val_acc = checkpoint['val_acc']
            
            self.train_classwise_accuracy = checkpoint['train_classwise_accuracy']
            self.test_classwise_accuracy = checkpoint['test_classwise_accuracy']
            self.val_classwise_accuracy = checkpoint['val_classwise_accuracy']
            
            self.train_loss = checkpoint['train_loss']
            self.test_loss = checkpoint['test_loss']
            self.val_loss = checkpoint['val_loss']
            
            self.epoch_train_loss = checkpoint['epoch_train_loss']
            self.epoch_test_loss = checkpoint['epoch_test_loss']
            self.epoch_val_loss = checkpoint['epoch_val_loss']
            
            self.epoch_train_acc = checkpoint['epoch_train_acc']
            self.epoch_test_acc = checkpoint['epoch_test_acc']
            self.epoch_val_acc = checkpoint['epoch_val_acc']
            
            self.optimizer_state_dict = checkpoint['optimizer']
            self.network.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            print('Loaded pre-trained network checkpoint from "{}"\nepoch: {} train loss: {} test loss: {} val loss: {}\n' 
                  .format(network_path, self.count_epoch, self.epoch_train_loss, self.epoch_test_loss, self.epoch_val_loss))
            
        else:
            print('No pre-trained network checkpoint found at "{}"\n'.format(network_path))
        print('-'*80)
        print('='*80)
    #-------------------------------------------------------------------------------------------------------------------------
    # plot and save (image) different losses

    def plot_loss_save_images(
        self, 
        epoch: int, 
        loss_type: str = 'loss',
        save_image_filename: str = '', 
        marker: str = 'None', 
        markersize: int = 10, 
        linewidth: int = 2, 
        linestyle: str = '-', 
        title_fontsize: int = 25, 
        xyticks_fontsize: int = 20 
    ) -> None:
        """
        Plot the different train, val and test loss against epochs. 
        """
        
        X = range(epoch+1)
        
        plt.plot(
            X, self.train_loss[:epoch+1] if loss_type=='loss' else self.train_acc[:epoch+1], 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='train' 
        )
        plt.plot(
            X, self.test_loss[:epoch+1] if loss_type=='loss' else self.test_acc[:epoch+1], 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='test' 
        )
        plt.plot(
            X, self.val_loss[:epoch+1] if loss_type=='loss' else self.val_acc[:epoch+1], 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='val' 
        )
        plt.xticks(fontsize=xyticks_fontsize)
        plt.yticks(fontsize=xyticks_fontsize)
        plt.xlabel('epoch', fontsize=title_fontsize)
        plt.ylabel('avg. loss' if loss_type=='loss' else 'avg. acc.', fontsize=title_fontsize)
        plt.grid(linestyle='--')
        plt.legend(loc='upper right')
        title = 'Loss Plot' if loss_type=='loss' else 'Accuracy Plot'
        plt.title(title , fontsize=title_fontsize)   

            # save the plot    
        if len(save_image_filename):
            plt.savefig(
                '' .join([save_image_filename,'.png']), 
                bbox_inches='tight' 
            )#save the plot in png form

        plt.show(block=False)
        plt.close()
    #------------------------------------------------------
    
    #save (csv) different losses
    def save_loss_csv(self,
                      epoch,
                      save_csv_filename='temp'
                      ):
        
        dict_loss = OrderedDict({'epoch':list(range(1, epoch+2)),
                                 'train_loss':self.train_loss[:epoch+1],
                                 'test_loss':self.test_loss[:epoch+1],
                                 'val_loss':self.val_loss[:epoch+1],
                                 'train_acc':self.train_acc[:epoch+1],
                                 'test_acc':self.test_acc[:epoch+1],
                                 'val_acc':self.val_acc[:epoch+1],
                                })
        df_loss = pd.DataFrame.from_dict(dict_loss)
        df_loss.to_csv(''.join([save_csv_filename, '.csv']), index=False)
        
        
    def save_confusion_matrix(self, confusion_matrix, name):
        
        # Plotting the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=True)
        plt.title(f"Confusion Matrix(Epoch - {self.count_epoch})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        # Save the confusion matrix as a PNG file
        plt.savefig(name + '.png')
        plt.close()
        # plt.show(block = False)
        
        
    # def plot_random_sample(self, data, labels):
    #     idx = np.random.choice(len(data))
    #     plt.imshow(data[idx].permute(1, 2, 0), cmap='gray')
    #     plt.title('class-'+str(labels[idx].item()))
    #     plt.show()
#-------------------------------------------------------------------------------------
######################################################################################

    @staticmethod  
    def accuracy(model, dataloader, unlearn_cls, aprox_ndigits = 2, num_classes = 10):

        actual_labels = []#*len(dataloader.dataset)
        predicted_labels = []#*len(dataloader.dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for batch, labels in dataloader:
                data = batch.to(device)
                predicted = model(data.type(torch.float))
                predicted = predicted.cpu().detach().numpy()
                predicted = np.argmax(predicted, axis=1)
                predicted_labels.extend(predicted)
                actual_labels.extend(labels)
        confusion_matrix = metrics.confusion_matrix(actual_labels, predicted_labels)#Computing the confusion matrix
        acc = round((sum(confusion_matrix.diagonal())/np.sum(confusion_matrix))*100, aprox_ndigits)
        retain_acc = (sum(confusion_matrix.diagonal()) - confusion_matrix[unlearn_cls, unlearn_cls])/(np.sum(confusion_matrix) - np.sum(confusion_matrix[unlearn_cls]))
        retain_acc = round(retain_acc*100, aprox_ndigits)
        unlearn_acc = confusion_matrix[unlearn_cls, unlearn_cls]/np.sum(confusion_matrix[unlearn_cls])
        unlearn_acc = round(unlearn_acc*100, aprox_ndigits)
        classwise_accuracy = [round((confusion_matrix[i,i]/np.sum(confusion_matrix, axis=1)[i])*100, aprox_ndigits) for i in range(confusion_matrix.shape[0])]
        # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = list(range(num_classes)))
        # cm_display.plot()
        # plt.close()
        
        return confusion_matrix, acc, classwise_accuracy, retain_acc, unlearn_acc#actual_labels, predicted_labels


    @staticmethod
    def accuracy_multiclass_unlearn(model, dataloader, unlearn_classes, aprox_ndigits = 2):#, num_classes = 10):

        actual_labels = []#*len(dataloader.dataset)
        predicted_labels = []#*len(dataloader.dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for batch, labels in dataloader:
                data = batch.to(device)
                predicted = model(data.type(torch.float))
                predicted = predicted.cpu().detach().numpy()
                predicted = np.argmax(predicted, axis=1)
                predicted_labels.extend(predicted)
                actual_labels.extend(labels)
        confusion_matrix = metrics.confusion_matrix(actual_labels, predicted_labels)#Computing the confusion matrix
        acc = round((sum(confusion_matrix.diagonal())/np.sum(confusion_matrix))*100, aprox_ndigits)
        retain_acc = (sum(confusion_matrix.diagonal()) - sum(confusion_matrix[i, i] for i in unlearn_classes))/(np.sum(confusion_matrix) - sum(sum(confusion_matrix[i]) for i in unlearn_classes))
        retain_acc = round(retain_acc*100, aprox_ndigits)
        unlearn_acc = sum(confusion_matrix[i, i] for i in unlearn_classes)/sum(sum(confusion_matrix[i]) for i in unlearn_classes)
        unlearn_acc = round(unlearn_acc*100, aprox_ndigits)
        classwise_accuracy = [round((confusion_matrix[i,i]/np.sum(confusion_matrix, axis=1)[i])*100, aprox_ndigits) for i in range(confusion_matrix.shape[0])]
        # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = list(range(num_classes)))
        # cm_display.plot()
        # plt.close()
        
        return confusion_matrix, acc, classwise_accuracy, retain_acc, unlearn_acc#actual_labels, predicted_labels


    @staticmethod
    def subset_accuracy(model, dataloader, aprox_ndigits = 2):#, num_classes = 10):

        actual_labels = []#*len(dataloader.dataset)
        predicted_labels = []#*len(dataloader.dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for batch, labels in dataloader:
                data = batch.to(device)
                predicted = model(data.type(torch.float))
                predicted = predicted.cpu().detach().numpy()
                predicted = np.argmax(predicted, axis=1)
                predicted_labels.extend(predicted)
                actual_labels.extend(labels)
        confusion_matrix = metrics.confusion_matrix(actual_labels, predicted_labels)#Computing the confusion matrix
        acc = round((np.trace(confusion_matrix)/np.sum(confusion_matrix))*100, aprox_ndigits)
        classwise_accuracy = [round((confusion_matrix[i,i]/np.sum(confusion_matrix[i]))*100, aprox_ndigits) for i in range(confusion_matrix.shape[0])]
        # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = list(range(num_classes)))
        # cm_display.plot()
        # plt.show(block = False)
            
        return confusion_matrix, acc, classwise_accuracy, actual_labels, predicted_labels
    

    @staticmethod
    def plot_sample_with_label(data, labels, figsize=(8, 6)):

        plt.figure(figsize=figsize)
        plt.imshow(data, cmap='gray')
        plt.title('class-'+str(labels))
        plt.show()


#---------------------------------------------------------------------------------------------------------------------------
#===========================================================================================================================
#---------------------------------------------------------------------------------------------------------------------------

class CustomCIFAR100(CIFAR100):
    def __init__(self, root, train, download, transform):
        super().__init__(root = root, train = train, download = download, transform = transform)
        self.coarse_map = {
            0:[4, 30, 55, 72, 95],
            1:[1, 32, 67, 73, 91],
            2:[54, 62, 70, 82, 92],
            3:[9, 10, 16, 28, 61],
            4:[0, 51, 53, 57, 83],
            5:[22, 39, 40, 86, 87],
            6:[5, 20, 25, 84, 94],
            7:[6, 7, 14, 18, 24],
            8:[3, 42, 43, 88, 97],
            9:[12, 17, 37, 68, 76],
            10:[23, 33, 49, 60, 71],
            11:[15, 19, 21, 31, 38],
            12:[34, 63, 64, 66, 75],
            13:[26, 45, 77, 79, 99],
            14:[2, 11, 35, 46, 98],
            15:[27, 29, 44, 78, 93],
            16:[36, 50, 65, 74, 80],
            17:[47, 52, 56, 59, 96],
            18:[8, 13, 48, 58, 90],
            19:[41, 69, 81, 85, 89]
        }
        
    #def __len__(self):
    #    len(self.main_dataset)
        
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        coarse_y = None
        for i in range(20):
            for j in self.coarse_map[i]:
                if y == j:
                    coarse_y = i
                    break
            if coarse_y != None:
                break
        if coarse_y == None:
            print(y)
            assert coarse_y != None
        return x, coarse_y# y, coarse_y
    
    
    
          
class load_datasets:
    
    def __init__(self,
                 name: str = 'mnist',
                 root: str = '/home/rajdeep/Codes/Datasets/Torchvision_Data/'
                 ):

        self.root = root
        if name == 'mnist':
            self.dataset = load_datasets.load_mnist(self.root)
        elif name == 'cifar10':
            self.dataset = load_datasets.load_cifar10(self.root)
        elif name == 'fashionMNIST':
            self.dataset = load_datasets.load_fashionmnist(self.root)
        elif name == 'svhn':
            self.root = ''.join([self.root, 'SVHN_Data/'])
            self.dataset = load_datasets.load_svhn(self.root)
        elif name == 'cifar100':
            self.root = ''.join([self.root, 'CIFAR100/'])
            self.dataset = load_datasets.load_cifar100(self.root)
        

    @staticmethod
    def load_cifar10(root = './'):
        transform = tt.Compose([
            tt.ToTensor(),
            tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Look into the data directory
        data_dir = os.path.join(root, 'cifar10')
        
        dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
        download_url(dataset_url, data_dir+'/')

        
        # Extract from archive
        with tarfile.open(data_dir+'/cifar10.tgz', 'r:gz') as tar:
            tar.extractall(path=root)
        
        
        # Look into the data directory
        # data_dir = os.path.join(root, 'cifar10')
        #print(os.listdir(data_dir))
        #classes = os.listdir(data_dir + "/train")
        
        #train_ds = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
        #valid_ds = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform)
        train_ds = ImageFolder(data_dir+'/train', transform)
        valid_ds = ImageFolder(data_dir+'/test', transform)
        return train_ds, valid_ds

    @staticmethod
    def load_svhn(root = './'):
        transform = tt.Compose([
            tt.ToTensor(),
            tt.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])
        
        train_ds = torchvision.datasets.SVHN(root=root, split='train', download=True, transform=transform)
        valid_ds = torchvision.datasets.SVHN(root=root, split='test', download=True, transform=transform)
        
        return train_ds, valid_ds

    @staticmethod
    def load_mnist(root = './'):
        transform = tt.Compose([
            tt.ToTensor()])#,
            #tt.Normalize((0.5, ), (0.5,))
    # ])
        
        train_ds = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
        valid_ds = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)

        return train_ds, valid_ds

    @staticmethod
    def load_fashionmnist(root = './'):
        
        # Define the transformation to be applied to the data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        # Download and load the training dataset
        train_ds = datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)

        # Download and load the test dataset
        valid_ds = datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)
        
        return train_ds, valid_ds


    @staticmethod
    def load_cifar100(root = './'):
        
        # Define the transformation to be applied to the data
        
        transform = transforms.Compose([
            # transforms.Resize(224),
            tt.ToTensor(),
            tt.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
        
        # Download and load the training dataset
        train_ds = CustomCIFAR100(root=root, train=True, transform=transform, download=True)

        # Download and load the test dataset
        test_ds = CustomCIFAR100(root=root, train=False, transform=transform, download=True)
        
        return train_ds, test_ds

    # @staticmethod
    # def load_cifar100(root = './'):
        
    #     # Define the transformation to be applied to the data
    #     transform = transforms.Compose([
    #         tt.ToTensor(),
    #         tt.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
    #                      (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
    #     #(0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))])
    #     # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    #     # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])#
    #     # (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])#

    #     # Download and load the training dataset
    #     train_ds = datasets.CIFAR100(root=root, train=True, transform=transform, download=True)

    #     # Download and load the test dataset
    #     test_ds = datasets.CIFAR100(root=root, train=False, transform=transform, download=True)
        
    #     return train_ds, test_ds


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)
    
class View_(nn.Module):
    def __init__(self, size):
        super(View_, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    

class ConvStandard(nn.Conv2d): 
    
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
    A residual block as defined by He et al.
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
    A Residual network.
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
    def __init__(self):
        super().__init__()
        base = mobilenet_v3_large()
        base_list = [*list(base.children())[:-1]]
        self.conv_norm1 = nn.Sequential(*base_list[0][0])
        for i in range(1, 16):
            exec(f"self.inverted_residual_{i} = base_list[0][{i}]")
        self.conv_norm2 = nn.Sequential(*base_list[0][16])
        self.pool1 = base_list[1]
        self.drop = nn.Dropout()
        self.final = nn.Linear(960,1)
    
    def forward(self,x):
        actvn1 = self.conv_norm1(x)
        
        for i in range(1, 16):
            exec(f"actvn{i+1} = self.inverted_residual_{i}(actvn{i})", locals(), globals())
        
        actvn17 = self.conv_norm2(actvn16)
        out = self.pool1(actvn17)
        
        out = self.drop(out.view(-1,self.final.in_features))
        return self.final(out), actvn1, actvn2, actvn3, actvn4, actvn5, actvn6, actvn7,\
                actvn8, actvn9, actvn10, actvn11, actvn12, actvn13, actvn14, actvn15,\
                actvn16, actvn17

    
class ResNet18(nn.Module):
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
    def __init__(self, config):
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
        return torch.zeros(self.lstm_layers, self.batch_size, self.hidden_size, device=self.device)
        
    def apply_embedding(self, x, static_embedding, apply_masking):
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
    
        if hidden is None:
            hidden = self.init_hidden()
            
        output, (hidden, cell) = self.lstm_encoder(x, (hidden, hidden))
        
        return output, hidden
        
    def decode(self, x, hidden=None):
        
        if hidden is None:
            hidden = self.init_hidden()
            
        output, (hidden, cell) = self.lstm_decoder(x, (hidden,hidden))
        
        return output, hidden
    

    def forward(self, x):

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
    def __init__(self, text_embedding_dimension):
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
    def __init__(self, num_classes=20):
        super(ViT, self).__init__()
        self.base = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.final = nn.Linear(self.base.config.hidden_size, num_classes)
        self.num_classes = num_classes
        self.relu = nn.ReLU()

    def forward(self, pixel_values):
        outputs = self.base(pixel_values=pixel_values)
        logits = self.final(outputs.last_hidden_state[:,0])

        return logits
    
    
    
# # test network
# if __name__ == '__main__':
    
#     num_chanel = 3
#     H, W = 32, 32#28, 28
#     num_class = 10#3
#     batch_size = 128
#     model = AllCNN_(n_channels=num_chanel, num_classes=num_class, padding = 0)
    
#     print('-'*70)
#     print('Network architechture (num chanel-{}, num class- {}) as follows:' .format(num_chanel, num_class))
#     print('-'*70)
#     print(model)
#     print('-'*70)
    
#     print('Network summary:')
#     print('-'*70)
#     summary(model, (batch_size, num_chanel, H, W ), device=str("cpu"))
#     print('-'*70)
    
#     print('Network input output dims check')
#     print('-'*70)
#     x = torch.randn(batch_size, num_chanel, H, W)
#     print(x.shape)
#     y = model(x)
#     # print('input shape(batch_size x num_chanels x height x width): {}\noutput shape(batch_size x num_class): {}' .format(x.shape, y.shape))
#     # print('-'*70)
#     print(y[0].shape)
    
    



if __name__ == '__main__':
    dataset_obj = load_datasets(name = 'mnist',
                                root = '/home/rajdeep/Codes/Datasets/Torchvision_Data/'
                                )
    train_data, test_data = dataset_obj.dataset
    input_size = train_data[0][0].shape
    print(input_size)
    train_utils = utils_(network = '',#mobilenet_v2(weights='IMAGENET1K_V1'),
                 input_size = input_size,#(1, 28, 28),#(3, 32, 32),# (Channel, Height, Width) or (Channel, Length)
                 num_classes = 10,
                 padding = 2,# 0,
                 solver_type = 'adam',
                 learning_rate = 1e-3,
                 batch_size = 128,
                 num_epochs = 1,
                 model_save_name = '',
                 model_name = 'LeNet32'
                 )
    train_utils.count_epoch = 0
    train_utils.train_data = train_data
    train_utils.test_data = test_data
    train_utils.train()
