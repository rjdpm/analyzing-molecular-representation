"""
-----------------------------------------------------------------------------
For deep learning architechture and training scripts for euclidean space
-----------------------------------------------------------------------------
AUTHOR: Soumitra Samanta                            DATE: Friday, 12012241030
For bug and others mail me at (soumitra.samanta@gm.rkmvu.ac.in)
-----------------------------------------------------------------------------
"""


from typing import Union, Tuple, Dict, List, Any
from tqdm import tqdm
import numpy as np
import os
import sys
# import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchsummary import summary
from torchinfo import summary

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from input_output import *

# to import parent package
sys.path.append('../')
from data_utils import *
from .network_blocks import *



__all__ = [
    'CNN1D_GRU_VAE',
    'net_selfies2selfies',
    
]

class CNN1D_GRU_VAE(nn.Module):
    """
    VAE for small molecule:
    """
    
    def __init__(
        self, 
        dims_input_data: Tuple = (130, 60), 
        dims_output_data: Tuple = (130, 60), 
        num_kernels: List = [9, 9, 10], 
        size_kernels: List = [9, 9, 11], 
        num_fc_layer_encoder: int = 0, 
        dropout_prob: float = 0.0, 
        dims_latent: int = 100, 
        act_func: str = 'relu', 
        param_act_func: Dict = {'alpha':1.0, 'negative_slope':1e-2}, 
        scale_latent_space: float = 1e-2, 
        num_fc_layer_decoder: int = 0, 
        gru_hidden_size: int = 500, 
        num_gru: int = 4 
    ) -> None:
        super(CNN1D_GRU_VAE, self).__init__()
        
        
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
        self.E_mean = networks_blocks._fc_layers(
            dims_latent, 
            dims_latent, 
            name='mean' 
        )
        self.E_sigma = networks_blocks._fc_layers(
            dims_latent, 
            dims_latent, 
            name='sigma' 
        )
            
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
        self.scale_latent_space = scale_latent_space
    #------------------------------------------------------
    
    # define encoder
    def encoder(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Encoder network
        """
        
        x = self.E_conv1d(x)
        x = x.view(x.size(0), -1)
        x = F.selu(self.E_fc(x))
        mu_z = self.E_mean(x)
        sigma_z = self.E_sigma(x)
        
        return mu_z, sigma_z
    #------------------------------------------------------
    
    # define reparametarization trick
    def reparametarization(
        self, 
        mu: torch.Tensor, 
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Re-parameterization tricks
        """
        
        std = torch.exp(0.5*sigma)
        eps = self.scale_latent_space * torch.rand_like(std)
        z = mu + eps*std
        
        return z
    #------------------------------------------------------
    
    # define decoder
    def decoder(
        self, 
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Decoder network
        """
        
        z = F.selu(self.D_fc1(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.dims_output_data[0], 1)
        z, h = self.D_gru(z)
        z_reshape = z.contiguous().view(-1, z.size(-1))
        x_bar = F.softmax(self.D_fc2(z_reshape), dim=1)
        x_bar = x_bar.contiguous().view(z.size(0), -1, x_bar.size(-1))
        
        return x_bar
    #------------------------------------------------------
    
    # define forward pass
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        
        mu, sigma = self.encoder(x)
        z = self.reparametarization(mu, sigma)
        x_bar = self.decoder(z)
        
        return x_bar, mu, sigma
#======================================================


# train, val and text network
class net_selfies2selfies():
    """
    Selfies network
    """

    def __init__(
        self, 
        dims_input_data: Tuple, 
        dims_output_data: Tuple, 
        max_string_len: int = 130, 
        num_kernels: List = [9, 9, 10], 
        size_kernels: List = [9, 9, 11], 
#         num_hidden_nodes_ffnn=2048, \
#         num_layers_encoder=4, \
        num_fc_layer_encoder: int = 0, 
        dropout_prob: float = 0.0, 
        dims_latent: int = 100, 
        act_func: str = 'relu', 
        param_act_func: Dict = {'alpha':1.0, 'negative_slope':1e-2}, 
        scale_latent_space: float = 1e-2, 
        num_fc_layer_decoder: int =0, 
        gru_hidden_size: int = 500, 
        num_gru: int = 4, 
        layer_type: str = '1dcnn_gru', 
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
        random_seed: int = 1
    ) -> None:
        
        # define network
        torch.manual_seed(random_seed)# to control the random number generator
        
        if layer_type=='1dcnn_gru':# for encode-1dcnn vae and decoder-gru
            network = CNN1D_GRU_VAE(
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
                num_gru=num_gru
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
        print(summary(network, input_size=(batch_size, dims_input_data[0], dims_input_data[1]), device=device))
        print('-'*70)
        
        self.device = device
        
        self.dims_input_data= dims_input_data
        self.dims_output_data = dims_output_data
#         self.smiles_char_map = smiles_char_map
        self.max_string_len = max_string_len
#         self.padding = padding
        self.num_kernels = num_kernels
        self.size_kernels = size_kernels
#         self.num_hidden_nodes_ffnn = num_hidden_nodes_ffnn
#         self.num_layers_encoder = num_layers_encoder
        self.num_fc_layer_encoder = num_fc_layer_encoder
        self.dims_latent = dims_latent
        self.act_func = act_func
        self.param_act_func = param_act_func
        self.scale_latent_space = scale_latent_space
        self.num_fc_layer_decoder = num_fc_layer_decoder
        self.gru_hidden_size = gru_hidden_size
        self.num_gru = num_gru
        
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
        
        # intermediate results save path
        self.result_savepath = ''.join([
            result_savepath, 
            'selfies_net_', self.layer_type
        ])
        if layer_type=='1dcnn_gru':# for encode-1dcnn vae and decoder-gru
            self.result_savepath = ''.join([
                self.result_savepath, 
                '_nkrnl_', str(self.num_kernels).replace(' ','_').strip('[]').replace(',',''),
                '_szkrnl_', str(self.size_kernels).replace(' ','_').strip('[]').replace(',','')
            ])
            
        self.result_savepath = ''.join([
            self.result_savepath, 
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
            '_lr_', str(self.learning_rate), '/' 
        ]).replace('.','_')
        self.result_savepath = create_folder(self.result_savepath)
        # to save best network    
        self.best_network_save_filename = ''.join([
            'best_', 
            self.result_savepath.split('/')[-2] 
        ]).replace('.','_')
            
    #------------------------------------------------------
    
    # define loss function
    def loss_function(
        self, 
        x: torch.Tensor, 
        x_recon: torch.Tensor, 
        mu: torch.Tensor, 
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """Loss function"""
        
        if self.loss_type=='bce_kld':
            BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
            KLD = -0.5*torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
            loss = BCE + KLD
        elif self.loss_type=='bce_kld_uniform':
            BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
            KLD = -0.5*torch.sum(1 + sigma)
            loss = BCE + KLD
        else:
            raise ValueError('Define your loss function: "{}"' .format(self.loss_type))

        return loss
    #------------------------------------------------------
    
    # define optimizer/solver
    def optimization_solver(self) ->None:
        """Optimizer"""
        
        if self.solver_type=='adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        else:
            raise ValueError('Define your optimization solver: "{}"' .format(self.solver_type))
    #------------------------------------------------------
    
    # train function for an epoch
    def train_1epoch(self) ->float:
        """Training one epoch"""

        self.network.train()# switch to train mode

        # data shuffle and minibatch settings
        num_train_samples = self.X_train.shape[0]
        idx = np.random.permutation(num_train_samples)
        num_iteration = int(np.ceil(float(num_train_samples)/self.batch_size)) 

        epoch_train_loss = 0# loss accumulation

        for i in tqdm(range(num_iteration)):# iteration over each minibatch

            start_idx = (i*self.batch_size)%num_train_samples
            _x = self.X_train[idx[start_idx:start_idx+self.batch_size], :].type(torch.float32).to(self.device)# take minibatch and move data to gpu (if available)
        
            self.optimizer.zero_grad()# set parameter gradients as zeros
            
            _x_bar, _mu, _sigma = self.network(_x)# forward
            loss = self.loss_function(_x, _x_bar, _mu, _sigma)# loss calculation

            loss.backward()# backword
            self.optimizer.step()# parameter update
            epoch_train_loss += loss.item()# record minibatch loss

#         epoch_train_loss /= num_iteration# average loss
        epoch_train_loss /= num_train_samples# average loss

        return epoch_train_loss
    #------------------------------------------------------
    
    # test function
    def test(
        self, 
        X: torch.Tensor
    ) -> Tuple:
        """Testing"""

        self.network.eval()# switch to test mode

        # minibatch settings (don't need data shuffle)
        num_test_samples = X.shape[0]
        num_iteration = int(np.ceil(float(num_test_samples)/self.batch_size)) 

        X_bar = torch.zeros(X.shape)# for reconstruct data (output)
        epoch_test_loss = 0# loss accumulation

        with torch.no_grad():# no need to calculate gradient as its for test only
            for i in tqdm(range(num_iteration)):# iteration over each minibatch
                start_idx = (i*self.batch_size)%num_test_samples
                _x = X[start_idx:start_idx+self.batch_size, :].type(torch.float32).to(self.device)# take minibatch and move data to gpu (if available)
                
                _x_bar, _mu, _sigma = self.network(_x)# forward
                loss = self.loss_function(_x, _x_bar, _mu, _sigma)# loss calculation
                X_bar[start_idx:start_idx+self.batch_size, :] = _x_bar.to("cpu") # move the resulkt to cpu for gpu memeory management
                
                epoch_test_loss += loss.item()# record minibatch loss

#         epoch_test_loss /= num_iteration# average loss
        epoch_test_loss /= num_test_samples# average loss

        return epoch_test_loss, X_bar
    #------------------------------------------------------
    
    def train(self) -> None:
        """Training block"""
        
        # to record avg. losses  
        if self.count_epoch == 0:
            self.train_loss = np.zeros(self.num_epoch)
            self.val_loss = np.zeros(self.num_epoch)
            self.test_loss = np.zeros(self.num_epoch)
        else:
            self.train_loss = np.concatenate((self.train_loss, np.zeros(self.num_epoch-self.count_epoch)))
            self.val_loss = np.concatenate((self.val_loss, np.zeros(self.num_epoch-self.count_epoch)))
            self.test_loss = np.concatenate((self.test_loss, np.zeros(self.num_epoch-self.count_epoch)))
        in_epoch = self.count_epoch
        
        for epoch in range(in_epoch, self.num_epoch):# iterate over epoch
            print('='*70)
            print('Train epoch {}/({})' .format(epoch+1, self.num_epoch))
            print('-'*70)
            # call train script for each epoch
            epoch_train_loss = self.train_1epoch()
            print('Avg. train loss: {:.6f}' .format(float(epoch_train_loss)))
            print('-'*70)
            #+++++++++++++++++++++++++++++++++++++++++++++++++++

            # test on train data
            print('-'*70)
            print('Test on train data epoch {}/({})' .format(epoch+1, self.num_epoch))
            # call test script
#             idx = np.random.permutation(self.X_train.shape[0])# take random samples from train set to test (As train set generally huge and its will be time consuming to test on whole train data)
            epoch_test_train_loss, _ = self.test(self.X_train_test)
            self.train_loss[epoch] = epoch_test_train_loss
            print('Avg. test loss on train data: {:.6f}' .format(float(epoch_test_train_loss)))
            print('-'*70)
            #+++++++++++++++++++++++++++++++++++++++++++++++++++

            # test on val data
            print('-'*70)
            print('Test on val data epoch {}/({})' .format(epoch+1, self.num_epoch))
            # call test script
            epoch_test_val_loss, _ = self.test(self.X_val)
            self.val_loss[epoch] = epoch_test_val_loss
            print('Avg. test loss on val data: {:.6f}' .format(float(epoch_test_val_loss)))
            print('-'*70)
            #+++++++++++++++++++++++++++++++++++++++++++++++++++

            # test on test data
            print('-'*70)
            print('Test on test data epoch {}/({})' .format(epoch+1, self.num_epoch))
            # call test script
            epoch_test_test_loss, _ = self.test(self.X_test)
            self.test_loss[epoch] = epoch_test_test_loss
            print('Avg. test loss on test data: {:.6f}' .format(float(epoch_test_test_loss)))
            print('-'*70)
            #+++++++++++++++++++++++++++++++++++++++++++++++++++

            # save different losses
            if epoch%min(1, self.save_result_ateach_epoch)==0:
                # save as plot image
                loss_save_filename = ''.join([
                    self.result_savepath, 
                    'all_loss_epoch_', str(self.num_epoch), 
                    '_', self.result_savepath.split('/')[-2] 
                ]).replace('.','_')
                self.plot_loss_save_images(
                    epoch, 
                    loss_save_filename, 
                    title=self.layer_type
                )
                # save as csv file
                self.save_loss_csv(
                    epoch, 
                    loss_save_filename
                )
            #+++++++++++++++++++++++++++++++++++++++++++++++++++

            # save the best trained network
            if((epoch==0) or (self.val_loss_best_network>epoch_test_val_loss)):             
                
                self.epoch_best_network = epoch
                self.train_loss_best_network = epoch_test_train_loss
                self.val_loss_best_network = epoch_test_val_loss
                self.test_loss_best_network = epoch_test_test_loss
                
                # note that we save the train loss based on the best val loss
                self.save_network(
                    epoch=epoch, 
                    network_save_filename=self.best_network_save_filename
                )
                # save the best model on that particular epoch
                self.save_network(
                    epoch=epoch, 
                    network_save_filename=''.join([self.best_network_save_filename, '_epoch_', str(epoch)])
                )

            # mandatory network save after fixed epoch
            if epoch%self.save_result_ateach_epoch==0:
                network_save_filename = ''.join([
                    'network_epoch_', str(epoch), 
                    '_', self.result_savepath.split('/')[-2]
                ]).replace('.','_')
                self.save_network(
                    epoch=epoch, 
                    network_save_filename=network_save_filename
                )
            #+++++++++++++++++++++++++++++++++++++++++++++++++++

            self.count_epoch += 1
            #+++++++++++++++++++++++++++++++++++++++++++++++++++
            
        # save different losses (final)
        loss_save_filename = ''.join([
            self.result_savepath, 
            'final_all_loss_epoch_', str(self.num_epoch), 
            '_', self.result_savepath.split('/')[-2] 
        ]).replace('.','_')
        self.plot_loss_save_images(
            self.count_epoch-1, 
            loss_save_filename, 
            title=self.layer_type
        )
        # save as csv file
        self.save_loss_csv(
            self.count_epoch-1, 
            loss_save_filename
        )
    #------------------------------------------------------     
    
    
    # trained network save function
    def save_network(
        self, 
        epoch: int = 0,
        network_save_filename: str = 'network_best'
    ) -> None:
        """Save losses (train, val, test)"""

        network_save_filename = ''.join([network_save_filename, '.pth'])
        print('Saving network in: "{}" as filename: "{}"' .format(self.result_savepath, network_save_filename))
        torch.save({
            'epoch': epoch, 
            'train_loss': self.train_loss[:epoch+1],
            'train_loss_best_network': self.train_loss_best_network, 
            'val_loss': self.val_loss[:epoch+1], 
            'val_loss_best_network': self.val_loss_best_network,
            'test_loss': self.test_loss[:epoch+1], 
            'test_loss_best_network': self.test_loss_best_network, 
            #please take the the multiple gpu carefully using network.module
    #         'state_dict': self.network.module.state_dict() if torch.cuda.device_count() > 1 else self.network.state_dict(),
            'state_dict': self.network.state_dict(), 
            'optimizer': self.optimizer.state_dict() 
        }, 
            '' .join([
                self.result_savepath,
                network_save_filename
            ])
        )
    #------------------------------------------------------
    
    # pre-trained network load function for test
    def load_test_network(
        self, 
        temp_network_path: str = ''
    ) -> None:
        """Load pre-trained network"""
        
        
        if len(temp_network_path)==0:
            temp_network_path = '' .join([
                self.result_savepath, 
                self.best_network_save_filename, 
                '.pth'
            ])
            
        print('='*70)
        print('Loading pre-trained network checkpoint from: "{}"' .format(temp_network_path))
        print('-'*70)
            
        if os.path.isfile(temp_network_path):            
            checkpoint = torch.load(temp_network_path, map_location=self.device)
            if 'epoch' in checkpoint.keys():
                self.epoch_best_network = checkpoint['epoch']
            else:
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
            #+++++++++++++++++++++++++++++++++++++++++++++++++++
            
            if 'val_loss' in checkpoint.keys():    
                self.val_loss = checkpoint['val_loss']
            else:
                self.val_loss_best_network = 'NA'
            if 'val_loss_best_network' in checkpoint.keys():    
                self.val_loss_best_network = checkpoint['val_loss_best_network']
            else:
                self.val_loss_best_network = 'NA'
            #+++++++++++++++++++++++++++++++++++++++++++++++++++
                
            if 'test_loss' in checkpoint.keys():
                self.test_loss = checkpoint['test_loss']
            else:
                self.test_loss = 'NA'
            if 'test_loss_best_network' in checkpoint.keys():
                self.test_loss_best_network = checkpoint['test_loss_best_network']
            else:
                self.test_loss_best_network = 'NA'
            #+++++++++++++++++++++++++++++++++++++++++++++++++++
            
            self.network.load_state_dict(checkpoint['state_dict'])
            

            print('Loaded pre-trained network checkpoint from "{}"\nepoch: {} train loss: {} val loss: {} test loss: {}' \
                  .format(temp_network_path, self.epoch_best_network, self.train_loss_best_network, self.val_loss_best_network, self.test_loss_best_network) 
                 )

        else:
            print('No pre-trained network checkpoint found at "{}"' .format(temp_network_path) 
                 )
            self.epoch_best_network = 0
        print('='*70)
    #------------------------------------------------------
    
    
    # plot and save (image) different losses
    def plot_loss_save_images(
        self, 
        epoch: int, 
        save_image_filename: str = '', 
        title: str = 'VAE loss', 
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
            X, self.train_loss[:epoch+1], 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='loss: train' 
        )
        plt.plot(
            X, self.val_loss[:epoch+1], 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='loss: val' 
        )
        plt.plot(
            X, self.test_loss[:epoch+1], 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='loss: test' 
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
            plt.savefig(
                '' .join([save_image_filename,'.png']), 
                bbox_inches='tight' 
            )#save the plot in png form

        plt.show(block=False)
        plt.close()
    #------------------------------------------------------
    
    
    #save (csv) different losses
    def save_loss_csv(
        self, 
        epoch: int, 
        save_csv_filename: str = 'temp' 
    ) -> None:
        """Save loses as a .csv"""
        
        dict_loss = OrderedDict({
            'epoch':list(range(1, epoch+2)), 
            'train_loss':self.train_loss[:epoch+1], 
            'val_loss':self.val_loss[:epoch+1], 
            'test_loss':self.test_loss[:epoch+1] 
        })
        save_dict_csv_pandas(
            dict_loss,
            save_filename=''.join([save_csv_filename, '.csv']),
        )
        

    # reconstruct selfies 
    def reconstruct_selfies(
        self, 
        selfies: List, 
        dict_selfies_tokens: Dict,
        padded_token: str = 'eos',
        flag_decode_smiles: bool = True, 
        save_path: str = 'temp_results/', 
        dataset_name: str = 'temp'
    ) -> Tuple[Dict]:
        """Reconstruct selfies"""

        print('='*70)
        print('<===doing selfies reconstruction===>')
        print('-'*70)
        
        self.network.eval()# switch to test mode

        # minibatch settings (don't need data shuffle)
        num_test_samples = len(selfies)
        num_iteration = int(np.ceil(float(num_test_samples)/self.batch_size))

        # to store the reconstructed smiles and their validation flag
        selfies_recon = [None]*num_test_samples
        if flag_decode_smiles:
            smiles_decode = [None]*num_test_samples
        else:
            smiles_decode = []
        check_ids_selfies_recon = [None]*num_test_samples

        # to store the latent space representation
        with torch.no_grad(): 
            for i in tqdm(range(num_iteration)):
                start_idx = (i*self.batch_size)%num_test_samples
                _selfies = selfies[start_idx:start_idx+self.batch_size]# take minibatch
                # selfies to one-hot representation
                _onehot_rep, _ = _selfies2onehot(
                    _selfies,
                    max_selfies_len=self.max_string_len,
                    dict_selfies=dict_selfies_tokens,
                    padded_token=padded_token,
                    verbose=False
                )
                _onehot_rep = torch.from_numpy(_onehot_rep).type(torch.float32).to(self.device)# data type conversion and move data to gpu (if available)

                # reconstruct selfies
                _onehot_rep_bar, _, _ = self.network(_onehot_rep)# forward
                _onehot_rep_bar = _onehot_rep_bar.to("cpu").numpy()
                selfies_recon[start_idx:start_idx+self.batch_size] = _onehot2selfies(
                    _onehot_rep_bar, 
                    dict_selfies_tokens['codomain'],
                    verbose=False
                )
                # validity checking of reconstructed smiles
                _, check_ids_selfies_recon[start_idx:start_idx+self.batch_size], temp_smiles_decode = _check_validity_selfies(
                    selfies_recon[start_idx:start_idx+self.batch_size],
                    flag_decode_smiles=flag_decode_smiles,
                    verbose=False
                )
                if flag_decode_smiles:
                    smiles_decode[start_idx:start_idx+self.batch_size] = temp_smiles_decode
        #+++++++++++++++++++++++++++++++++++++++++++++++++++

        num_valid_selfies = len(np.asarray(check_ids_selfies_recon)[check_ids_selfies_recon])
        recon_acc = round(100.0*(float(num_valid_selfies)/num_test_samples), 2)
        # find exact reconstruction
        num_exact_reconst_selfies = 0
        check_ids_selfies_exact_recon = [False]*num_test_samples
        for i in range(num_test_samples):
            if selfies[i] == selfies_recon[i]:
                num_exact_reconst_selfies += 1
                check_ids_selfies_exact_recon[i] = True
        exact_recon_acc = round(100.0*(float(num_exact_reconst_selfies)/num_test_samples), 2)    
        
        print('-'*70)
        print('Valid reconstructed selfies: total: {} ({})\nacc: {}(%) ' .format(num_valid_selfies, num_test_samples, recon_acc))
        print('-'*70)
        print('Valid exact reconstructed selfies: total: {} ({})\nacc: {}(%) ' .format(num_exact_reconst_selfies, num_test_samples, exact_recon_acc))

        dict_result = OrderedDict({
            'dataset_name': dataset_name, 
            'total_number_of_samples':num_test_samples, 
            'total_valid_reconstructed_samples':num_valid_selfies, 
            'acc': recon_acc, 
            'total_exact_reconst_selfies':num_exact_reconst_selfies, 
            'exact_recon_acc_selfies':exact_recon_acc
        })
        dict_selfies = OrderedDict({
            'selfies':selfies, 
            'selfies_recon':selfies_recon, 
            'selfies_valid':check_ids_selfies_recon,
            'selfies_exact_recon':check_ids_selfies_exact_recon
        })
        if flag_decode_smiles:
            dict_selfies['smiles_decode'] = smiles_decode
            
            
        # save selfies, reconstructed smiles in CSV and pkl file
        save_path = create_folder(save_path)
        save_filename = ''.join([
            save_path, 
            dataset_name, 
            '_reconstructed_selfies_tsample_', str(num_test_samples), 
            '_vsamples_', str(num_valid_selfies), 
            '_acc_', str(recon_acc).replace('.', '_')
        ])
        save_dict_parquet_pandas(
            dict_selfies,
            save_filename=''.join([save_filename,'.parquet'])
        )
        save_dict_csv_pandas(
            dict_selfies,
            save_filename=''.join([save_filename,'.csv'])
        )
        save_dict_pickle(
            dict_selfies, 
            save_filename=''.join([save_filename,'.pkl'])
        )
        print('='*70)

        return dict_result, dict_selfies
    
            
            
#======================================================


if __name__ == '__main__':
    
    # create a random dataset
    input_data_dims = (130, 60)
    output_data_dims = (130, 60)
    num_kernels = [9, 9, 10]
    size_kernels = [9, 9, 11]
    num_fc_layer_encoder = 0
    dims_reduce = 100
    scale_latent_space = 1e-2
    gru_hidden_size = 500
    num_gru = 4
    
    num_samples = 1000
    
    X_train = np.random.random((num_samples, input_data_dims[0], input_data_dims[1]))
    X_val = np.random.random((num_samples, input_data_dims[0], input_data_dims[1]))
    X_test = np.random.random((num_samples, input_data_dims[0], input_data_dims[1]))
    
    X_train[X_train<0.5] = 0
    X_train[X_train>=0.5] = 1
    X_train = X_train.astype(np.bool_)  
    
    X_val[X_val<0.5] = 0
    X_val[X_val>=0.5] = 1
    X_val = X_val.astype(np.bool_)  
    
    X_test[X_test<0.5] = 0
    X_test[X_test>=0.5] = 1
    X_test = X_test.astype(np.bool_)  
    
    # network parameter
    weight_init = 'xvr_unifrm'
    loss_type = 'bce_kld'
    solver_type = 'adam'
    num_epoch = 5
    
    dims_input_data = (X_train.shape[-2], X_train.shape[-1]) 
    dims_output_data = (X_train.shape[-2], X_train.shape[-1])
    # network initializer
    network = net_selfies2selfies(
        dims_input_data, 
        dims_output_data,  
        num_kernels=num_kernels, 
        size_kernels=size_kernels, 
        num_fc_layer_encoder=num_fc_layer_encoder, 
        dims_latent=dims_reduce, 
        scale_latent_space=scale_latent_space, 
        gru_hidden_size=gru_hidden_size, 
        num_gru=num_gru, 
        weight_init=weight_init, 
        loss_type=loss_type, 
        solver_type=solver_type, 
        num_epoch=num_epoch
    )
    
    network.X_train = torch.from_numpy(X_train)
    network.X_train_test = torch.from_numpy(X_train)
    network.X_val = torch.from_numpy(X_val)
    network.X_test = torch.from_numpy(X_test)
    network.count_epoch = 0
    network.num_mini_epoch = num_epoch
    
    # load pre-trained network
    network.load_test_network()# load network
    network.count_epoch = 0
    # train
    network.train()
    
    # test on best trained network
    network.load_test_network()# load network
    
    # on train data
    print('Final test on "train" data of size:\n X: {}' .format(X_train.shape))
    test_train_loss, X_train_bar = network.test(torch.from_numpy(X_train).type(torch.float32))
    print('Avg. test loss on train data: {:.6f}'.format(float(test_train_loss)))
    
    # on val data
    print('Final test on "val" data of size:\n X: {}' .format(X_val.shape))
    test_val_loss, X_val_bar = network.test(torch.from_numpy(X_val).type(torch.float32))
    print('Avg. test loss on val data: {:.6f}'.format(float(test_val_loss)))
    
    # on test data
    print('Final test on "test" data of size:\n X: {}' .format(X_test.shape))
    test_test_loss, X_test_bar = network.test(torch.from_numpy(X_test).type(torch.float32))
    print('Avg. test loss on test data: {:.6f}'.format(float(test_test_loss)))
          



