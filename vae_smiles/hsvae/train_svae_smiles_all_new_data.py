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


import pickle
import os, sys
import numpy as np
# import shutil
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from network.SmilesImage2Smiles_Network_svae import *
from data_utils import *
from data_utils_smiles import *
from input_output import *

torch.manual_seed(1)
np.random.seed(1)

# # for auto-reloading external modules
# # see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2

##############################################################################################
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
# dataset_name_prefix = 'zinc6m_all4'#'zinc6m_all4_mona'
smiles_char_filename = 'tokens_smiles_train_0_5_val_0_2_test_0_3_63.json'
smiles_max_length_filename = 'max_len_smiles_train_0_5_val_0_2_test_0_3_241.txt'
dataset_name = 'canon_smiles_selfies_train_0_5_val_0_2_test_0_3.parquet'
caco2_all_data_path_filename = 'all_data_features_modified.csv'
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
smiles_char, smiles_char_map = smiles.load_smiles_char(smiles_char_filepath)
# load smiles max-length
max_smiles_length = smiles.load_smiles_max_length(smiles_max_length_filepath)

# load train smiles
smiles_train_zinc = smiles.load_smiles_from_parquet(''.join([path_init, train_data_name, dataset_name]),
                                                      smiles_column_ids='smiles',
                                                      max_string_len=max_smiles_length,
                                                      smiles_check_flag=False,
                                                      cannonical_check=False
                                                      )
total_train_sample = len(smiles_train_zinc)
smiles_train_zinc = smiles_train_zinc[:min(num_train_samples, total_train_sample)]
    
# load val smiles
smiles_val = smiles.load_smiles_from_parquet(''.join([path_init, test_data_name, dataset_name]),
                                                    smiles_column_ids='smiles',
                                                    max_string_len=max_smiles_length,
                                                    smiles_check_flag=False,
                                                    cannonical_check=False,
                                                    n_data=num_train_test_samples
                                                    )
total_val_sample = len(smiles_val)

# load test smiles
smiles_test = smiles.load_smiles_from_parquet(''.join([path_init, val_data_name, dataset_name]),
                                                     smiles_column_ids='smiles',
                                                     max_string_len=max_smiles_length,
                                                     smiles_check_flag=False,
                                                     cannonical_check=False,
                                                     n_data=num_train_test_samples
                                                     )
total_test_sample = len(smiles_test)
smiles_train = smiles_train_zinc
#+++++++++++++++++++++++++++++++++++++++++++++++++++

## VAE_GRU parameters
dims_input_data = (max_smiles_length, len(smiles_char_map['domain'].keys())+1) 
dims_output_data = (max_smiles_length, len(smiles_char_map['domain'].keys())+1)
# dims_input_data = (max_smiles_length, len(smiles_char_map['domain'].keys())) 
# dims_output_data = (max_smiles_length, len(smiles_char_map['domain'].keys()))
num_kernels = [9, 9, 10]#[[11, 13, 15]#11, 13, 15]#[15, 17, 19]#[11, 13, 15]#[9, 9, 10]
size_kernels = [9, 9, 11]#[[13, 11, 9]#9, 9, 11]#[15, 13, 11]#[9, 9, 11]#[9, 9, 11]
num_fc_layer_encoder = 0#2#
dropout_prob = 0.0
dims_latent = 50#25#
act_func = 'relu'
scale_latent_space = 1e-2
num_fc_layer_decoder = 0
gru_hidden_size = 500
num_gru = 4#5#3#
layer_type = '1dcnn_gru'
weight_init = 'xvr_unifrm'
loss_type = 'bce_kld'
solver_type = 'adam'
num_epoch = 5# 500
batch_size = 128
learning_rate = 1e-4#1e-3#
save_result_ateach_epoch = 50
result_savepath = ''.join(['cache/smi2smi_svae/', path_init.split('/')[-2], '_', dataset_name, '_', str(num_train_samples), '/']).replace('.','_')
#+++++++++++++++++++++++++++++++++++++++++++++++++++

smiles_svae_model = SmIm2Sm_Network(dims_input_data, dims_output_data, 
                                   smiles_char_map=smiles_char_map, 
                                   max_string_len=max_smiles_length, 
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

print('='*70, '\n')
print(f'Number of Train Data = {len(smiles_train)}')
print(f'Number of Test Data = {len(smiles_test)}')
print(f'Number of Val Data = {len(smiles_val)}')
print(f'Dimension of Input Data = {dims_input_data}')
print(f'Dimension of Output Data = {dims_output_data}')
print(f'Maximum Smiles Length = {max_smiles_length}')
print(f'Number of Train Epochs = {num_epoch}')
print(f'Number of Dictionary Chracters = {len(smiles_char)}')
print(f'Dictionary Characters = {smiles_char}')
print(f'Number of kernels = {num_kernels}')
print(f'Size of the kernels = {size_kernels}')
print(f'Number of Fully Connected layer in Encoder = {num_fc_layer_encoder}')
print(f'Latent Dimension = {dims_latent}\nNumber of GRU = {num_gru}')
print(f'Batch Size = {batch_size}\nGRU Hidden Dimension = {gru_hidden_size}')
print(f'Learning Rate = {learning_rate}\nWeight Initializer: {weight_init}')
print(f'Optimizer: {solver_type}\n')
print('='*70, '\n')

# smiles to one-hot image 
# for train data                           
smiles_svae_model.X_train = torch.from_numpy(listsmi2onehot(smiles_list=smiles_train,
                                                           char_list=smiles_char,
                                                           max_str_len=max_smiles_length
                                                           ))

smiles_svae_model.X_train_test = smiles_svae_model.X_train[:num_train_test_samples,:]
# for val data 
smiles_svae_model.X_val = torch.from_numpy(listsmi2onehot(smiles_list=smiles_val,
                                                         char_list=smiles_char,
                                                         max_str_len=max_smiles_length
                                                         )
                                          )
# for test data 
smiles_svae_model.X_test = torch.from_numpy(listsmi2onehot(smiles_list=smiles_test,
                                                          char_list=smiles_char,
                                                          max_str_len=max_smiles_length
                                                          )
                                           )

# model train
smiles_svae_model.count_epoch = 0
smiles_svae_model.num_mini_epoch = num_epoch

smiles_svae_model.load_test_network()# load pre-trained best network

#Saving the Network File
source_file = os.path.abspath(__file__)
# destination_file = ''.join([smiles_svae_model.result_savepath, '/train_script_', smiles_svae_model.result_save_filename,'.py'])
# shutil.copy(source_file, destination_file)
with open(source_file, 'rb') as fp:
    train_script = fp.read()
    smiles_svae_model.train_script = pickle.dumps(train_script)

# train
smiles_svae_model.train()                   