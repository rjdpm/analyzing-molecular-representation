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
import random
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
from network.SmilesImage2Smiles_Network_vae import *
from data_utils import *
from data_utils_smiles import *

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

# # for auto-reloading external modules
# # see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2

##############################################################################################
# for my macbook-pro
path_init = '/Users/soumitra/Documents/SS_DATASET/SS_SUSCORD/SS_MOLECULES/all_data_partion/canonical/05012021/'
# # for my iib-system
# path_init = '/media/large/SOUMITRA/SS_DATASET/SS_SUSCORD/SS_MOLECULES/all_data_partion/canonical/05012021/'
# for my mbio-system
path_init = '/home/soumitra/Documents/SOUMITRA/SS_DATASET/SS_SUSCORD/SS_MOLECULES/all_data_partion/canonical/05012021/'

# for rkmveri-gpu2
path_init = '/home/rajdeep/Codes/Datasets/SS_DATASET/SS_MOLECULES/all_data_partion/canonical/05012021/'

# rajdeep local
path_init = '/home/rajdeep/Codes/Datasets/selfies/zinc/'
path_init_doglas = '/home/rajdeep/Codes/Datasets/selfies/douglas/smiles_selfies/'

# rajdeep brahma
path_init = '/home/rajdeep/Codes/Datasets/selfies/zinc/'
path_init_doglas = '/home/rajdeep/Codes/Datasets/selfies/douglas/smiles_selfies/'

# Liverpool
path_init = '/home/rkmvu/Dataset/selfies/zinc/'
path_init_doglas = '/home/rkmvu/Dataset/selfies/douglas/smiles_selfies/'

#Caco2 Data
caco2_data_path_init = '/home/rkmvu/Dataset/Coca-2/Experimental_Data_P_app/train_test_partition_literature/'

## Dataset prepare 
gpu_device_id = 0# GPU number for multiple GPUs (pytorch takes default 0 or which is availabe next)
device = torch.device("cuda:"+str(gpu_device_id) if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('='*70)
print('Model will use: {}' .format(device))
print('='*70)


# dataset parameters
# num_total_samples = 3963360#3970176
num_train_samples = 1000#12941195#10500000#100000#10#
num_train_test_samples = 1000#100000#'all'#10#
train_ratio = 0.5
val_ratio = 0.2
test_ratio = 1 - train_ratio - val_ratio

selected_data_frac = 0.7
ticks = [-1, 0, 1, 2, 3, 4, 5]
# dataset_name_prefix = 'zinc6m_all4'#'zinc6m_all4_mona'
smiles_char_filename = 'tokens_smiles_train_0_5_val_0_2_test_0_3_63.json'
smiles_max_length_filename = 'max_len_smiles_train_0_5_val_0_2_test_0_3_241.txt'
dataset_name = 'canon_smiles_selfies_with_descriptors_train_0_5_val_0_2_test_0_3.parquet'
caco2_all_data_path_filename = 'all_data_features_modified_wrt_independent_set.csv'
caco2_independent_data_path_filename = 'independent_test_set.csv'
padding = 'right'

train_data_name = ''.join(['train_', str(train_ratio), '_']).replace('.', '_')
test_data_name = ''.join(['test_', str(test_ratio), '_']).replace('.', '_')
val_data_name = ''.join(['val_', str(val_ratio), '_']).replace('.', '_')

smiles_char_filepath = ''.join([path_init, '/', smiles_char_filename])
smiles_max_length_filepath = ''.join([path_init, '/', smiles_max_length_filename])
#+++++++++++++++++++++++++++++++++++++++++++++++++++

# read smiles,  vocabulary and smiles max-length
smiles = PARSE_SMILES([])

padding = padding
# load smiles char vocabulary
smiles.smiles_char, smiles.smiles_char_map = smiles.load_smiles_char(smiles_char_filepath)
# load smiles max-length
smiles.max_smiles_length = smiles.load_smiles_max_length(smiles_max_length_filepath)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# load train smiles
smiles_train_zinc_df = pd.read_parquet(''.join([path_init, train_data_name, dataset_name]))
selected_df, _ = smiles.selective_range_data_sampling(df=smiles_train_zinc_df,
                                                   ticks = ticks,
                                                   column='logp',
                                                   frac=selected_data_frac,
                                                   random_state=1
                                                   )
smiles_train_zinc = selected_df['smiles'].tolist()
total_train_sample = len(smiles_train_zinc)
smiles_train_zinc = smiles_train_zinc[:min(num_train_samples, total_train_sample)]

# Using caco-2 data for training
caco2_data_all = pd.read_csv(''.join([caco2_data_path_init, caco2_all_data_path_filename]))
caco2_data_independent = pd.read_csv(''.join([caco2_data_path_init, caco2_independent_data_path_filename]))
caco2_data = caco2_data_all['SMILES'].tolist()
caco2_data_ind = caco2_data_independent['SMILES'].tolist()

smiles_train = smiles_train_zinc + caco2_data + caco2_data_ind
#------------------------------------------------------------
  
# load val smiles
smiles_val = pd.read_parquet(''.join([path_init, test_data_name, dataset_name]))['smiles']
total_val_sample = len(smiles_val)
smiles_val = smiles_val.tolist()#[:num_train_test_samples]
#------------------------------------------------------------

# load test smiles
smiles_test = pd.read_parquet(''.join([path_init, val_data_name, dataset_name]))['smiles']
total_test_sample = len(smiles_test)
smiles_test = smiles_test.tolist()#[:num_train_test_samples]
print('Data Loading Complete.')
print('-'*80)
#------------------------------------------------------------
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


## VAE_GRU parameters
dims_input_data = (smiles.max_smiles_length, len(smiles.smiles_char_map['domain'].keys())+1) 
dims_output_data = (smiles.max_smiles_length, len(smiles.smiles_char_map['domain'].keys())+1)
num_kernels = [9, 9, 10]#[[11, 13, 15]#11, 13, 15]#[15, 17, 19]#[11, 13, 15]#[9, 9, 10]
size_kernels = [9, 9, 11]#[[13, 11, 9]#9, 9, 11]#[15, 13, 11]#[9, 9, 11]#[9, 9, 11]
num_fc_layer_encoder = 0#2#
dropout_prob = 0.0
dims_latent = 300#200#100#50#25#
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
learning_rate = 1e-4#1e-3#
save_result_ateach_epoch = 50
result_savepath = ''.join(['cache/smi2smi_vae/', path_init.split('/')[-2], '_', dataset_name, '_', str(num_train_samples), '/']).replace('.','_')
#+++++++++++++++++++++++++++++++++++++++++++++++++++

smiles_vae_model = SmIm2Sm_Network(dims_input_data, dims_output_data, 
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
                                   result_savepath=result_savepath)
#+++++++++++++++++++++++++++++++++++++++++++++++++++


# test on best trained network
smiles_vae_model.load_test_network()# load pre-trained best network

save_path = ''.join(['result/', smiles_vae_model.result_savepath.split('/', 1)[-1],'reconst/', datetime.today().strftime('%d%m%Y'), '/'])
save_path = create_folder(save_path)
#+++++++++++++++++++++++++++++++++++++++++++++++++++

# on train data
print('Test on "train" data of size: {}' .format(len(smiles_train)))
dict_result_train, dict_smiles_recon_train = smiles_vae_model.smiles_reconstruct_molecule(smiles_train, tokenize='spacial', save_path=save_path, dataset_name='train')
# dict_result_train, dict_smiles_recon_train = {}, {}
#+++++++++++++++++++++++++++++++++++++++++++++++++++

# on val data
print('Test on "val" data of size: {}' .format(len(smiles_val)))
dict_result_val, dict_smiles_recon_val = smiles_vae_model.smiles_reconstruct_molecule(smiles_val, tokenize='spacial', save_path=save_path, dataset_name='val')
# dict_result_val, dict_smiles_recon_val = {}, {}
#+++++++++++++++++++++++++++++++++++++++++++++++++++

# on test data
print('Test on "test" data of size: {}' .format(len(smiles_test)))
dict_result_test, dict_smiles_recon_test = smiles_vae_model.smiles_reconstruct_molecule(smiles_test, tokenize='spacial', save_path=save_path, dataset_name='test')
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
