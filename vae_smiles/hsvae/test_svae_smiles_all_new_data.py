#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##############################################################################
"""
-----------------------------------------------------------------------------
FUNCTION:
-----------------------------------------------------------------------------
@AUTHOR: Soumitra Samanta                DATE: Mon Jan  4 12:24:18 2021
For bug and others mail me at soumitramath39@gmail.com
-----------------------------------------------------------------------------
INPUT:
OUTPUT:
-----------------------------------------------------------------------------
EXAMPLE:
-----------------------------------------------------------------------------
"""
##############################################################################

from datetime import datetime
import pickle
import gzip
import os, sys
import wget
import numpy as np
import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings('ignore')

import copy
import time
from collections import OrderedDict
import pandas as pd

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from network.SmilesImage2Smiles_Network_svae import *

from data_utils import *

##############################################################################################
# for my macbook-pro
path_init = '/Users/soumitra/Documents/SS_DATASET/SS_SUSCORD/SS_MOLECULES/all_data_partion/canonical/05012021/'
# for my iib-system
path_init = '/media/large/SOUMITRA/SS_DATASET/SS_SUSCORD/SS_MOLECULES/all_data_partion/canonical/05012021/'
# for my mbio-system
path_init = '/home/rajdeep/Codes/Datasets/SS_DATASET/SS_MOLECULES/all_data_partion/canonical/05012021/'
#on Brahma
path_init = '/home/rajdeep/Codes/Datasets/SS_DATASET/SS_MOLECULES/all_data_partion/canonical/05012021/'
#For new data
path_init = '/home/rajdeep/workspace/Codes/Datasets/selfies/zinc/'

## Dataset prepare 
gpu_device_id = 0# GPU number for multiple GPUs (pytorch takes default 0 or which is availabe next)
device = torch.device("cuda:"+str(gpu_device_id) if torch.cuda.is_available() else "cpu")

# dataset parameters
num_train_samples = 12941195#1000#10500000#100000#10#
num_train_test_samples = 'all'#100000#10#
train_ratio = 0.5
val_ratio = 0.2
test_ratio = 1 - train_ratio - val_ratio
# dataset_name_prefix = 'zinc6m_all4'#'zinc6m_all4_mona'
smiles_char_filename = 'token_smiles_train_0_5_0_2_test_0_3_52.json'
smiles_max_length_filename = 'max_len_smiles_train_0_5_val_0_2_test_0_3_150.txt'
dataset_name = 'canon_smiles_selfies_train_0_5_val_0_2_test_0_3.parquet'
padding = 'right'

train_data_name = ''.join(['train_', str(train_ratio), '_']).replace('.', '_')
test_data_name = ''.join(['test_', str(test_ratio), '_']).replace('.', '_')
val_data_name = ''.join(['val_', str(val_ratio), '_']).replace('.', '_')

smiles_char_filepath = ''.join(['./req_files/', smiles_char_filename])
smiles_max_length_filepath = ''.join(['./req_files/', smiles_max_length_filename])
#+++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++

# read smiles,  vocabulary and smiles max-length
smiles = PARSE_SMILES([])

smiles.padding = padding
# load smiles char vocabulary
smiles.smiles_char, smiles.smiles_char_map = smiles.load_smiles_char(smiles_char_filepath)
# load smiles max-length
smiles.max_smiles_length = smiles.load_smiles_max_length(smiles_max_length_filepath)

# load train smiles
smiles.smiles_train = smiles.load_smiles_from_parquet(''.join([path_init, train_data_name, dataset_name]), smiles_column_ids='smiles', max_string_len=smiles.max_smiles_length)
    
# load val smiles
smiles.smiles_val = smiles.load_smiles_from_parquet(''.join([path_init, test_data_name, dataset_name]), smiles_column_ids='smiles', max_string_len=smiles.max_smiles_length)

# load test smiles
smiles.smiles_test = smiles.load_smiles_from_parquet(''.join([path_init, val_data_name, dataset_name]), smiles_column_ids='smiles', max_string_len=smiles.max_smiles_length)

## sVAE_GRU parameters
dims_input_data = (smiles.max_smiles_length, len(smiles.smiles_char_map['domain'].keys())) 
dims_output_data = (smiles.max_smiles_length, len(smiles.smiles_char_map['domain'].keys()))
num_kernels = [9, 9, 10]##[11, 13, 15]#[15, 17, 19]#[11, 13, 15]#[9, 9, 10]
size_kernels = [9, 9, 11]#[9, 9, 11]#[15, 13, 11]#[9, 9, 11]#[9, 9, 11]
num_fc_layer_encoder = 0
dropout_prob = 0.0
dims_latent = 150#200#100#75#
act_func = 'relu'
scale_latent_space = 1e-2
num_fc_layer_decoder = 0
gru_hidden_size = 500
num_gru = 5#4#3#
layer_type = '1dcnn_gru'
weight_init = 'xvr_unifrm'
loss_type = 'bce_kld'
solver_type = 'adam'
num_epoch = 500
batch_size = 128
learning_rate = 1e-4
save_result_ateach_epoch = 50
result_savepath = ''.join(['cache/smi2smi_svae/', path_init.split('/')[-2], '_', dataset_name, '_', str(num_train_samples), '/']).replace('.','_')
#+++++++++++++++++++++++++++++++++++++++++++++++++++

smiles_svae_model = SmIm2Sm_Network(dims_input_data, dims_output_data, 
                                   smiles_char_map=smiles.smiles_char_map, 
                                   max_string_len=smiles.max_smiles_length, 
                                   padding=smiles.padding, 
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
                                   result_savepath=result_savepath)
#+++++++++++++++++++++++++++++++++++++++++++++++++++


# test on best trained network
smiles_svae_model.load_test_network()# load pre-trained best network

save_path = ''.join(['result/', smiles_svae_model.result_savepath.split('/', 1)[-1],'reconst/', datetime.today().strftime('%d%m%Y'), '/'])
save_path = create_folder(save_path)
#+++++++++++++++++++++++++++++++++++++++++++++++++++

# on train data
print('Test on "train" data of size: {}' .format(len(smiles.smiles_train)))
dict_result_train, dict_smiles_recon_train = smiles_svae_model.smiles_reconstruct_molecule(smiles.smiles_train, tokenize='spacial', save_path=save_path, dataset_name='train')
#+++++++++++++++++++++++++++++++++++++++++++++++++++

# on val data
print('Test on "val" data of size: {}' .format(len(smiles.smiles_val)))
dict_result_val, dict_smiles_recon_val = smiles_svae_model.smiles_reconstruct_molecule(smiles.smiles_val, tokenize='spacial', save_path=save_path, dataset_name='val')
#+++++++++++++++++++++++++++++++++++++++++++++++++++

# on test data
print('Test on "test" data of size: {}' .format(len(smiles.smiles_test)))
dict_result_test, dict_smiles_recon_test = smiles_svae_model.smiles_reconstruct_molecule(smiles.smiles_test, tokenize='spacial', save_path=save_path, dataset_name='test')
#+++++++++++++++++++++++++++++++++++++++++++++++++++

# combine all results and save as csv
# for accuraccy
dict_result_all = OrderedDict({})
for k in dict_result_train.keys():
    dict_result_all[k] = [dict_result_train[k], dict_result_val[k], dict_result_test[k]]
save_filename = ''.join([save_path, 'all_reconstructed_smiles_accuracy'])
pd.DataFrame.from_dict(dict_result_all).to_csv(''.join([save_filename,'.csv']), header=True, encoding='utf-8', index=False)
#+++++++++++++++++++++++++++++++++++++++++++++++++++

# for reconstructed smiles
dict_smiles_recon_all = OrderedDict({})
for k in dict_smiles_recon_train.keys():
    dict_smiles_recon_all[k] = dict_smiles_recon_train[k] + dict_smiles_recon_val[k] + dict_smiles_recon_test[k]
save_filename = ''.join([save_path, 'all_reconstructed_smiles'])
pd.DataFrame.from_dict(dict_smiles_recon_all).to_csv(''.join([save_filename,'.csv']), header=True, encoding='utf-8', index=False)

# # to save RAM
# del smiles_train, smiles_val, smiles_test
# del dict_result_train, dict_smiles_recon_train
# del dict_result_val, dict_smiles_recon_val
# del dict_result_test, dict_smiles_recon_test
# del dict_result_all, dict_smiles_recon_all                                               
