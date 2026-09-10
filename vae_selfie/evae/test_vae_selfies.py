"""
-----------------------------------------------------------------------------
Test variational autoencoder for selfies on ZINC data
-----------------------------------------------------------------------------
AUTHOR: Soumitra Samanta                            DATE: Monday, 15012240930
For bug and others mail me at (soumitra.samanta@gm.rkmvu.ac.in)
-----------------------------------------------------------------------------
"""


from datetime import datetime
from collections import OrderedDict

import torch
import selfies as sf

from input_output import *

from network.selfies2selfies_netwrok import *
from data_utils import *
 

# # for auto-reloading external modules
# # see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2


# for my macbook-pro
path_init = '/Users/soumitra/Documents/SS_DATASET/SS_SUSCORD/SS_MOLECULES/05012024/zinc/soumitra/data/partions/'
# # for my suscord-system
# path_init = '/home/soumitra/Documents/SOUMITRA/SS_DATASET/SS_SUSCORD/SS_MOLECULES/05012024/zinc/soumitra/data/partions/'
# for my mbio-system
# path_init = '/home/soumitra/Documents/SOUMITRA/SS_DATASET/SS_SUSCORD/SS_MOLECULES/05012024/zinc/soumitra/data/partions/'
# # For dgx system
# path_init = '/mnt/SOUMITRA/SS_DATASET/SS_SUSCORD/SS_MOLECULES/05012024/zinc/soumitra/data/partions/'

# # for rkmveri-gpu2
# path_init = '/home/soumitra/Documents/SS_DATASET/SS_SUSCORD/SS_MOLECULES/05012024/zinc/soumitra/data/partions/'

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
flag_extra_data = True
flag_extra_tokens = False
num_extra_tokens = 105
req_selfies_max_len = 128#150
num_train_samples = 18000000#12941195#10000
num_train_test_samples = 100000#1000#10000#

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
if flag_selfies_tokens:
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
        '../../../../douglas/revised/smiles_selfies/',
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
        '../../../../douglas/revised/smiles_selfies/',
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
        '../../../../douglas/revised/smiles_selfies/',
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
num_kernels = [9, 9, 10]#[25, 21, 17]#[11, 13, 15]#[15, 17, 19]#[11, 13, 15]#
size_kernels = [9, 9, 11]#[13, 11, 9]#[9, 9, 11]#[15, 13, 11]#[9, 9, 11]#
num_fc_layer_encoder = 0
dropout_prob = 0.0
dims_latent = 100
act_func = 'relu'
scale_latent_space = 1e-2
num_fc_layer_decoder = 0
gru_hidden_size = 500
num_gru = 4
layer_type = '1dcnn_gru'
weight_init = 'xvr_unifrm'
loss_type = 'bce_kld'#'bce_kld_uniform'#
solver_type = 'adam'
num_epoch = 500
batch_size = 128
learning_rate = 1e-4
save_result_ateach_epoch = 50
result_savepath = ''.join([
    'cache/selfie2selfies_vae/', 
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

selfies_vae_model = net_selfies2selfies(
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

selfies_vae_model.load_test_network()# load pre-trained best network

save_path = ''.join([
    'result/', 
    selfies_vae_model.result_savepath.split('/', 1)[-1],
    'reconst/', 
    datetime.today().strftime('%d%m%Y'), 
    '/'
])
save_path = create_folder(save_path)
#+++++++++++++++++++++++++++++++++++++++++++++++++++

# on train data
print('='*70)
print('Test on "train" data of size: {}' .format(len(dict_train_data['selfies'])))
print('-'*70)
dict_result_train, dict_selfies_recon_train = selfies_vae_model.reconstruct_selfies(
    dict_train_data['selfies'], 
    dict_selfies_tokens=dict_selfies_tokens,
    flag_decode_smiles=True,
    save_path=save_path, 
    dataset_name='train'
)
# find the exact smiles reconstructionm acc
dict_selfies_recon_train['org_smiles'] = dict_train_data['smiles'][:len(dict_selfies_recon_train['smiles_decode'])]
dict_selfies_recon_train['smiles_exact_recon'] = [True]*len(dict_selfies_recon_train['org_smiles'])
num_exact_reconst_smiles = 0
for i, sm in enumerate(dict_selfies_recon_train['org_smiles']):
    if sm == dict_selfies_recon_train['smiles_decode'][i]:
        num_exact_reconst_smiles += 1
    else:
        dict_selfies_recon_train['smiles_exact_recon'][i] = False
exact_recon_acc_smlies = round(100.0*(float(num_exact_reconst_smiles)/len(dict_selfies_recon_train['org_smiles'])), 2)
print('Valid exact reconstructed smiles: total: {} ({})\nacc: {}(%) ' .format(num_exact_reconst_smiles, len(dict_selfies_recon_train['org_smiles']), exact_recon_acc_smlies))    
dict_result_train['total_exact_reconst_smiles'] = num_exact_reconst_smiles
dict_result_train['exact_recon_acc_smiles'] = exact_recon_acc_smlies
print('='*70)
#+++++++++++++++++++++++++++++++++++++++++++++++++++

# on val data
print('Test on "val" data of size: {}' .format(len(dict_val_data['selfies'])))
dict_result_val, dict_selfies_recon_val = selfies_vae_model.reconstruct_selfies(
    dict_val_data['selfies'], 
    dict_selfies_tokens=dict_selfies_tokens,
    flag_decode_smiles=True,
    save_path=save_path, 
    dataset_name='val'
)
# find the exact smiles reconstructionm acc
dict_selfies_recon_val['org_smiles'] = dict_val_data['smiles'][:len(dict_selfies_recon_val['smiles_decode'])]
dict_selfies_recon_val['smiles_exact_recon'] = [True]*len(dict_selfies_recon_val['org_smiles'])
num_exact_reconst_smiles = 0
for i, sm in enumerate(dict_selfies_recon_val['org_smiles']):
    if sm == dict_selfies_recon_val['smiles_decode'][i]:
        num_exact_reconst_smiles += 1
    else:
        dict_selfies_recon_val['smiles_exact_recon'][i] = False
exact_recon_acc_smlies = round(100.0*(float(num_exact_reconst_smiles)/len(dict_selfies_recon_val['org_smiles'])), 2)
print('Valid exact reconstructed smiles: total: {} ({})\nacc: {}(%) ' .format(num_exact_reconst_smiles, len(dict_selfies_recon_val['org_smiles']), exact_recon_acc_smlies))  
dict_result_val['total_exact_reconst_smiles'] = num_exact_reconst_smiles
dict_result_val['exact_recon_acc_smiles'] = exact_recon_acc_smlies
print('='*70)
#+++++++++++++++++++++++++++++++++++++++++++++++++++

# on test data
print('Test on "test" data of size: {}' .format(len(dict_test_data['selfies'])))
dict_result_test, dict_selfies_recon_test = selfies_vae_model.reconstruct_selfies(
    dict_test_data['selfies'], 
    dict_selfies_tokens=dict_selfies_tokens,
    flag_decode_smiles=True,
    save_path=save_path, 
    dataset_name='test'
)
# find the exact smiles reconstructionm acc
dict_selfies_recon_test['org_smiles'] = dict_test_data['smiles'][:len(dict_selfies_recon_test['smiles_decode'])]
dict_selfies_recon_test['smiles_exact_recon'] = [True]*len(dict_selfies_recon_test['org_smiles'])
num_exact_reconst_smiles = 0
for i, sm in enumerate(dict_selfies_recon_test['org_smiles']):
    if sm == dict_selfies_recon_test['smiles_decode'][i]:
        num_exact_reconst_smiles += 1
    else:
        dict_selfies_recon_test['smiles_exact_recon'][i] = False
exact_recon_acc_smlies = round(100.0*(float(num_exact_reconst_smiles)/len(dict_selfies_recon_test['org_smiles'])), 2)
print('Valid exact reconstructed smiles: total: {} ({})\nacc: {}(%) ' .format(num_exact_reconst_smiles, len(dict_selfies_recon_test['org_smiles']), exact_recon_acc_smlies))   
dict_result_test['total_exact_reconst_smiles'] = num_exact_reconst_smiles
dict_result_test['exact_recon_acc_smiles'] = exact_recon_acc_smlies
print('='*70)
#+++++++++++++++++++++++++++++++++++++++++++++++++++

# combine all results and save as .csv and .parquet
# for accuraccy
dict_result_all = OrderedDict({})
for k in dict_result_train.keys():
    dict_result_all[k] = [dict_result_train[k], dict_result_val[k], dict_result_test[k]]
save_dict_parquet_pandas(
    dict_result_all,
    save_filename=''.join([save_path, 'all_reconstructed_selfies_accuracy.parquet'])
)
save_dict_csv_pandas(
    dict_result_all,
    save_filename=''.join([save_path, 'all_reconstructed_selfies_accuracy.csv'])
)
#+++++++++++++++++++++++++++++++++++++++++++++++++++

# for reconstructed smiles

dict_selfies_recon_all = OrderedDict({})
for k in dict_selfies_recon_train.keys():
    dict_selfies_recon_all[k] = dict_selfies_recon_train[k] + dict_selfies_recon_val[k] + dict_selfies_recon_test[k]
save_dict_parquet_pandas(
    dict_selfies_recon_all,
    save_filename=''.join([save_path, 'all_reconstructed_selfies_smiles.parquet'])
)
save_dict_csv_pandas(
    dict_selfies_recon_all,
    save_filename=''.join([save_path, 'all_reconstructed_selfies_smiles.csv'])
)


                               
                                          
    