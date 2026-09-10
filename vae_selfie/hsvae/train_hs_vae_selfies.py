#!/usr/bin/env python3
"""
-----------------------------------------------------------------------------
Train a variational autoencoder for selfies for hyperspherical space
-----------------------------------------------------------------------------
AUTHOR: Soumitra Samanta                           DATE: Tuesday, 23012241130
For bug and others mail me at (soumitra.samanta@gm.rkmvu.ac.in)
-----------------------------------------------------------------------------
"""
import os
import sys
import torch
import selfies as sf

from input_output import *

from network.selfies2selfies_netwrok_hs_vae import *
from data_utils import *
 

# # for auto-reloading external modules
# # see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2


# for my macbook-pro
# path_init = '/Users/soumitra/Documents/SS_DATASET/SS_SUSCORD/SS_MOLECULES/05012024/zinc/soumitra/data/partions/'
# # for my suscord-system
# path_init = '/home/soumitra/Documents/SOUMITRA/SS_DATASET/SS_SUSCORD/SS_MOLECULES/05012024/zinc/soumitra/data/partions/'
# for my mbio-system
# path_init = '/home/soumitra/Documents/SOUMITRA/SS_DATASET/SS_SUSCORD/SS_MOLECULES/05012024/zinc/soumitra/data/partions/'
# # For dgx system
# path_init = '/mnt/SOUMITRA/SS_DATASET/SS_SUSCORD/SS_MOLECULES/05012024/zinc/soumitra/data/partions/'

# # for rkmveri-gpu2
# path_init = '/home/soumitra/Documents/SS_DATASET/SS_SUSCORD/SS_MOLECULES/05012024/zinc/soumitra/data/partions/'

## For Liverpool
# path_init = '/home/rkmvu/Dataset/selfies/douglas/smiles_selfies/'
path_init = '/home/rkmvu/Dataset/selfies/zinc/'

# GPU device parameter
gpu_device_id = 0# GPU number for multiple GPUs (pytorch takes default 0 or which is availabe next)
device = torch.device("cuda:"+str(gpu_device_id) if torch.cuda.is_available() else "cpu")

# dataset parameters
train_ratio = 0.5
val_ratio = 0.2
test_ratio = 1 - train_ratio - val_ratio
num_tokens = 58
selfies_max_len = 128
flag_selfies_tokens = True
flag_extra_data = False#True
flag_extra_tokens = True
num_extra_tokens = 105
req_selfies_max_len = 128#150
num_train_samples = 150000#3000000#10000#12941195#100000#18000000#10000
num_train_test_samples = 100000#10000#100000#

# read datasets 
# load train data from .parquet file    
input_filename = ''.join([
    path_init,
    'train_', str(train_ratio).replace('.', '_'),
    '_canon_smiles_selfies_',
    'train_', str(train_ratio).replace('.', '_'),
    '_val_', str(val_ratio).replace('.', '_'),
    '_test_', str(test_ratio).replace('.', '_'), 
    '.parquet'
])
dict_train_data = read_parquet_file_to_dict_pandas(input_filename)
need_num_extra_train_samples = num_train_samples - len(dict_train_data['selfies'])

if need_num_extra_train_samples < 0:
    for k in dict_train_data.keys():
        dict_train_data[k] = dict_train_data[k][:num_train_samples]

# load val data from .parquet file    
input_filename = ''.join([
    path_init,
    'val_', str(val_ratio).replace('.', '_'),
    '_canon_smiles_selfies_',
    'train_', str(train_ratio).replace('.', '_'),
    '_val_', str(val_ratio).replace('.', '_'),
    '_test_', str(test_ratio).replace('.', '_'), 
    '.parquet'
])
dict_val_data = read_parquet_file_to_dict_pandas(input_filename)
num_val_samples = len(dict_val_data['selfies'])
                                                            
for k in dict_val_data.keys():
    if need_num_extra_train_samples > 0:
        dict_train_data[k] += dict_val_data[k][num_train_test_samples:min(num_train_test_samples+need_num_extra_train_samples, num_val_samples)]
    dict_val_data[k] = dict_val_data[k][:num_train_test_samples]
    
# load test data from .parquet file    
input_filename = ''.join([
    path_init,
    'test_', str(test_ratio).replace('.', '_'),
    '_canon_smiles_selfies_',
    'train_', str(train_ratio).replace('.', '_'),
    '_val_', str(val_ratio).replace('.', '_'),
    '_test_', str(test_ratio).replace('.', '_'), 
    '.parquet'
])
dict_test_data = read_parquet_file_to_dict_pandas(input_filename)
for k in dict_test_data.keys():
    dict_test_data[k] = dict_test_data[k][:num_train_test_samples]

#read selfies tokens
input_filename = ''.join([
    path_init,
    'tokens_selfies_',
    'train_', str(train_ratio).replace('.', '_'),
    '_val_', str(val_ratio).replace('.', '_'),
    '_test_', str(test_ratio).replace('.', '_'), 
    '_', str(num_tokens),
    '.csv'
])
tokens_selfies = _read_tokens_selfies(input_filename)
if flag_extra_tokens:
    tokens_selfies = list(set(tokens_selfies).union(sf.get_semantic_robust_alphabet()))
dict_selfies_tokens = _get_dict_selfies(tokens_selfies)

#read selfies max length
input_filename = ''.join([
    path_init,
    'max_len_selfies_',
    'train_', str(train_ratio).replace('.', '_'),
    '_val_', str(val_ratio).replace('.', '_'),
    '_test_', str(test_ratio).replace('.', '_'), 
    '_', str(selfies_max_len),
    '.txt'
])
selfies_max_len = _read_max_len_selfies(input_filename)

#+++++++++++++++++++++++++++++++++++++++++++++++++++
# Extra data (M-drug, NP, Fluro, Recon)
if flag_extra_data:
    extra_tokens = []
    extra_selfies = []
    # train data
    input_filename = ''.join([
        path_init,
        '/douglas/revised/smiles_selfies/',
        'train_', str(train_ratio).replace('.', '_'),
        '_canon_smiles_selfies_',
        'train_', str(train_ratio).replace('.', '_'),
        '_val_', str(val_ratio).replace('.', '_'),
        '_test_', str(test_ratio).replace('.', '_'), 
        '.parquet'
    ])
    dict_data = read_parquet_file_to_dict_pandas(input_filename)
    check_ids = [True]*len(dict_data['selfies'])
    for i, s in enumerate(dict_data['selfies']):
        tokens = sf.split_selfies(s)
        if sf.len_selfies(s) > req_selfies_max_len:
            check_ids[i] = False
        if not flag_extra_tokens:
            for t in tokens:
                if t not in tokens_selfies:
                    check_ids[i] = False
    for k in dict_data.keys():
        dict_data[k] = [dict_data[k][j] for j, flag in enumerate(check_ids) if flag]
    dict_train_data['selfies'] += dict_data['selfies']
    dict_train_data['smiles'] += dict_data['smiles']
    extra_selfies += dict_data['selfies']
    extra_tokens += sf.get_alphabet_from_selfies(dict_data['selfies'])
    
    #Val data
    input_filename = ''.join([
        path_init,
        '/douglas/revised/smiles_selfies/',
        'val_', str(val_ratio).replace('.', '_'),
        '_canon_smiles_selfies_',
        'train_', str(train_ratio).replace('.', '_'),
        '_val_', str(val_ratio).replace('.', '_'),
        '_test_', str(test_ratio).replace('.', '_'), 
        '.parquet'
    ])
    dict_data = read_parquet_file_to_dict_pandas(input_filename)
    check_ids = [True]*len(dict_data['selfies'])
    for i, s in enumerate(dict_data['selfies']):
        tokens = sf.split_selfies(s)
        if sf.len_selfies(s) > req_selfies_max_len:
            check_ids[i] = False
        if not flag_extra_tokens:
            for t in tokens:
                if t not in tokens_selfies:
                    check_ids[i] = False
    for k in dict_data.keys():
        dict_data[k] = [dict_data[k][j] for j, flag in enumerate(check_ids) if flag]
    dict_val_data['selfies'] += dict_data['selfies']
    dict_val_data['smiles'] += dict_data['smiles']
    extra_selfies += dict_data['selfies']
    extra_tokens += sf.get_alphabet_from_selfies(dict_data['selfies'])
    
    #Test data
    input_filename = ''.join([
        path_init,
        '/douglas/revised/smiles_selfies/',
        'test_', str(test_ratio).replace('.', '_'),
        '_canon_smiles_selfies_',
        'train_', str(train_ratio).replace('.', '_'),
        '_val_', str(val_ratio).replace('.', '_'),
        '_test_', str(test_ratio).replace('.', '_'), 
        '.parquet'
    ])
    dict_data = read_parquet_file_to_dict_pandas(input_filename)
    check_ids = [True]*len(dict_data['selfies'])
    for i, s in enumerate(dict_data['selfies']):
        tokens = sf.split_selfies(s)
        if sf.len_selfies(s) > req_selfies_max_len:
            check_ids[i] = False
        if not flag_extra_tokens:
            for t in tokens:
                if t not in tokens_selfies:
                    check_ids[i] = False
    for k in dict_data.keys():
        dict_data[k] = [dict_data[k][j] for j, flag in enumerate(check_ids) if flag]
    dict_test_data['selfies'] += dict_data['selfies']
    dict_test_data['smiles'] += dict_data['smiles']
    extra_selfies += dict_data['selfies']
    extra_tokens += sf.get_alphabet_from_selfies(dict_data['selfies'])
    
#     #read extra tokens
#     if flag_extra_tokens:
#         input_filename = ''.join([
#             path_init,
#             '../../../../douglas/revised/smiles_selfies/',
#             'tokens_selfies_',
#             'train_', str(train_ratio).replace('.', '_'),
#             '_val_', str(val_ratio).replace('.', '_'),
#             '_test_', str(test_ratio).replace('.', '_'), 
#             '_', str(num_extra_tokens),
#             '.csv'
#         ])
#         extra_tokens += _read_tokens_selfies(input_filename)
    tokens_selfies = list(set(dict_selfies_tokens['domain'].keys()).union(set(extra_tokens)))
    dict_selfies_tokens = _get_dict_selfies(tokens_selfies)
        
    selfies_max_len = max(selfies_max_len, req_selfies_max_len)
#+++++++++++++++++++++++++++++++++++++++++++++++++++    
    

## VAE_GRU parameters
dims_input_data = (selfies_max_len, len(tokens_selfies)) 
dims_output_data = (selfies_max_len, len(tokens_selfies))
num_kernels = [25, 21, 17]#[9, 9, 10]#[11, 13, 15]#[15, 17, 19]#[11, 13, 15]#
size_kernels = [15, 13, 11]#[13, 11, 9]#[9, 9, 11]#[9, 9, 11]#[9, 9, 11]#
num_fc_layer_encoder = 0
dropout_prob = 0.2
dims_latent = 50#7
act_func = 'relu'
scale_latent_space = 1e-2
num_fc_layer_decoder = 0
gru_hidden_size = 500
num_gru = 4
distribution = 'vmf'
space_transform = False
layer_type = '1dcnn_gru'
weight_init = 'xvr_unifrm'
loss_type = 'bce_kld'#'bce_kld_uniform'#
solver_type = 'adam'
num_epoch = 500
batch_size = 512#256#128
learning_rate = 1e-4
save_result_ateach_epoch = 50
result_savepath = ''.join([
    'cache/selfie2selfies_hs_vae/', 
    'train_', str(train_ratio).replace('.', '_'),
    '_val_', str(val_ratio).replace('.', '_'),
    '_test_', str(test_ratio).replace('.', '_'), 
    '_nt_', str(num_tokens),
    '_ml_', str(selfies_max_len),
    '_slft_', str(flag_selfies_tokens),
    '_exd_'+str(flag_extra_data)+'_ext_'+str(flag_extra_tokens)+'_nt_'+str(num_extra_tokens)+'_rsl_'+str(req_selfies_max_len) if flag_extra_data else '_exd_'+str(flag_extra_data),
    '_nts_', str(num_train_samples),
    '_ntts_', str(num_train_test_samples),
    '/'
]).replace('.','_')
#+++++++++++++++++++++++++++++++++++++++++++++++++++

selfies_hs_vae_model = net_selfies2selfies(
    dims_input_data, 
    dims_output_data, 
    max_string_len=selfies_max_len,
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
    distribution=distribution,
    space_transform=space_transform,
    layer_type=layer_type, 
    device=device, 
    weight_init=weight_init, 
    loss_type=loss_type, 
    solver_type=solver_type, 
    num_epoch=num_epoch, 
    batch_size=batch_size, 
    learning_rate=learning_rate, 
    save_result_ateach_epoch=save_result_ateach_epoch, 
    result_savepath=result_savepath
)
#+++++++++++++++++++++++++++++++++++++++++++++++++++

# selfies to one-hot image 
# for train data                           
selfies_hs_vae_model.X_train, _ = _selfies2onehot(
    dict_train_data['selfies'],
    selfies_max_len,
    dict_selfies_tokens,
)
selfies_hs_vae_model.X_train = torch.from_numpy(selfies_hs_vae_model.X_train)
selfies_hs_vae_model.X_train_test = selfies_hs_vae_model.X_train[:num_train_test_samples,:]

# for val data 
selfies_hs_vae_model.X_val, _ = _selfies2onehot(
    dict_val_data['selfies'],
    selfies_max_len,
    dict_selfies_tokens,
)
selfies_hs_vae_model.X_val = torch.from_numpy(selfies_hs_vae_model.X_val)

# for test data 
selfies_hs_vae_model.X_test, _ = _selfies2onehot(
    dict_test_data['selfies'],
    selfies_max_len,
    dict_selfies_tokens,
)
selfies_hs_vae_model.X_test = torch.from_numpy(selfies_hs_vae_model.X_test)

# model train
selfies_hs_vae_model.load_test_network()# load pre-trained best network
selfies_hs_vae_model.count_epoch = selfies_hs_vae_model.epoch_best_network
selfies_hs_vae_model.num_mini_epoch = num_epoch

# train
selfies_hs_vae_model.train()                                                 
    
