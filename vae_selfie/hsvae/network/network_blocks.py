"""
-----------------------------------------------------------------------------
For deep neural network block helpers
-----------------------------------------------------------------------------
AUTHOR: Soumitra Samanta                            DATE: Friday, 12012241030
For bug and others mail me at (soumitra.samanta@gm.rkmvu.ac.in)
-----------------------------------------------------------------------------
"""


import numpy as np
import math
from typing import Union, Tuple, Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict

__all__ = [
    'networks_blocks',
    'self_attention_layer',
    'transformer_block_encoder',
    'transformer_encoder',
    'cross_attention_layer',
    'transformer_block_decoder',
    'transformer_decoder',
    'weights_initilizer',
    
]

class networks_blocks():
    
    def __init__(self):
        super(networks_blocks, self).__init__()
    #------------------------------------------------------
    
    
    #calculate conv1d output dims
    @staticmethod
    def conv1d_output(
        h_in: int, 
        padding: int = 0, 
        dilation: int = 1, 
        kernel_size: int = 9, 
        stride: int = 1
    ) -> int:
        """
        Calculate 1d-conv output size
        """
        
        h_out = int(np.floor(1+((h_in + 2 * padding - dilation*(kernel_size-1)-1)/float(stride))))

        return h_out
    #------------------------------------------------------
    

    #calculate conv2d output dims
    @staticmethod
    def conv2d_output(
        h_in: int, 
        w_in: int, 
        padding: Tuple = (1,1), 
        dilation: Tuple = (1, 1), 
        kernel_size: Tuple = (3, 3), 
        stride: Tuple = (1,1)
    ) -> Tuple:
        """
        Calculate 2d-conv output size
        """
        
        h_out = int(np.floor(1+((h_in + 2 * padding[0] - dilation[0]*(kernel_size[0]-1)-1)/float(stride[0]))))
        w_out = int(np.floor(1+((w_in + 2 * padding[1] - dilation[1]*(kernel_size[1]-1)-1)/float(stride[1]))))

        return h_out, w_out
    #------------------------------------------------------
    
    
    #calculate max-pool output dims
    @staticmethod
    def maxpool_output(
        h_in: int, 
        w_in: int,
        padding: Tuple = (0,0), 
        dilation: Tuple = (1, 1), 
        kernel_size: Tuple = (3, 3), 
        stride: Tuple = (1,1)
    ) -> Tuple:
        """
        Calculate 2d-soft output size
        """
        
        h_out = int(np.floor(1+((h_in + 2 * padding[0] - dilation[0]*(kernel_size[0]-1)-1)/float(stride[0]))))
        w_out = int(np.floor(1+((w_in + 2 * padding[1] - dilation[1]*(kernel_size[1]-1)-1)/float(stride[1]))))

        return h_out, w_out
    #------------------------------------------------------
    
    
    # acitivation function
    @staticmethod
    def _activation_func(
        name: str = 'relu', 
        param: Dict = {'alpha':1.0, 'negative_slope':1e-2}
    ) -> Any:
        """
        Activation function
        """
        
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
    #------------------------------------------------------
    
        
    # linear blocks
    @staticmethod
    def _fc_layers(
        fc_in: int, 
        fc_out: int, 
        num_layers: int = 0, 
        dropout_prob: int = 0.0, 
        act_func: str = '', 
        param_act_func: Dict = {'alpha':1.0, 'negative_slope':1e-2}, 
        name: str= 'encoder'
    ) -> Any:
        """
        Define FC layers
        """

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
    def _conv1d_layers(
        dims_input_data: int, 
        num_kernels: List = [32], 
        size_kernels: List = [3], 
        act_func: str = 'relu', 
        param_act_func: Dict = {'alpha':1.0, 'negative_slope':1e-2}, 
        name: str = 'encoder'
    ) -> Any:
        """
        Define 1d-conv layers
        """
        
        _layers = OrderedDict({})
        
        num_layers = len(num_kernels)
        for i in range(num_layers):
            _layers[name+'_conv1d'+str(i+1)] = nn.Conv1d(dims_input_data, num_kernels[i], kernel_size=size_kernels[i])
#             _layers[name+'_relu'+str(i+1)] = nn.ReLU(inplace=True)
            _layers[name+'_'+act_func+str(i+1)] = networks_blocks._activation_func(act_func, param_act_func)
            dims_input_data = num_kernels[i]

        return nn.Sequential(_layers)
    #------------------------------------------------------
    

class self_attention_layer(nn.Module):
    """
    Self attention layer
    """
    
    def __init__(
        self,
        dims_embd: int,
    )->None:
        """
        Self attention class initialization
        
        Inpout:
            - dims_embd (int): Embedding dimension
        """
        
        super().__init__()
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        self.W_q_ = nn.Linear(dims_embd, dims_embd)
        self.W_k_ = nn.Linear(dims_embd, dims_embd)
        self.W_v_ = nn.Linear(dims_embd, dims_embd)
        
        self.dims_embd_ = dims_embd
        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
    def forward(
        self, 
        x: Tensor 
    )->Tensor:
        """
        Forward pass for the self attention layer
        
        Imput:
            - x (torch tensor): Input data
            
        Output:
        
        """
        
        y = []
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        Q = self.W_q_(x)
        K = self.W_k_(x)
        V = self.W_v_(x)
        
        # scaled dot-product
        Q /= (self.dims_embd_ ** (1./4))
        K /= (self.dims_embd_ ** (1./4))
        temp = torch.bmm(Q, K.transpose(1, 2))
        temp = F.softmax(temp, dim=-1)
        y = temp.bmm(V)
        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
    
        return y
    
class transformer_block_encoder(nn.Module):
    """
    Transformer single block
    """
    
    def __init__(
        self,
        dims_embd: int,
        num_hidden_nodes_ffnn: int = 2048,
        dropout_prob: float = 0.0
    )->None:
        """
        Transformer single block class initialization
        
        Inpout:
            - dims_embd (int):             Embedding dimension
            - num_hidden_nodes_ffnn (int): Number of neurons in the fed-forward layer
            - dropout_prob (float):        Dropout probability in liner layers
        """
        
        super().__init__()
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        self.attention_ = self_attention_layer(dims_embd)
        
        self.layer_norm1_ = nn.LayerNorm(dims_embd)
        self.layer_norm2_ = nn.LayerNorm(dims_embd)
        
        self.ffnn_ = nn.Sequential(
            nn.Linear(dims_embd, num_hidden_nodes_ffnn),
            nn.ReLU(),
            nn.Linear(num_hidden_nodes_ffnn, dims_embd)
        )
        self.droput_ops_ = nn.Dropout(dropout_prob)
        
        self.dims_embd_ = dims_embd
        self.num_hidden_nodes_ffnn_ = num_hidden_nodes_ffnn
        self.dropout_prob_ = dropout_prob
        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
    def forward(
        self,
        x: Tensor,
    )->Tensor:
        """
        Forward pass for the transformer block
        
        Imput:
            - x (torch tensor): Input data
            
        Output:
        
        """
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        x = self.layer_norm1_(x + self.attention_(x))
        x = self.droput_ops_(x)
        x = self.layer_norm2_(x + self.ffnn_(x))
        x = self.droput_ops_(x)
    
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
        return x
        
class transformer_encoder(nn.Module):
    """
    Transformer encoder module
    """
    
    def __init__(
        self,
        dims_embd: int,
        num_hidden_nodes_ffnn: int = 2048,
        dropout_prob: float = 0.0,
        num_layers_encoder: int = 2
    )->None:
        """
        Transformer encoder class initialization
        
        Inpout:
            - dims_embd (int):             Embedding dimension
            - num_hidden_nodes_ffnn (int): Number of neurons in the fed-forward layer
            - dropout_prob (float):        Dropout probability in liner layers
            - num_layers_encoder (int):    Number encoder blocks
        """
        super().__init__()
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        self.trs_endr_blocks_ = nn.ModuleList(
            [
                transformer_block_encoder(dims_embd, num_hidden_nodes_ffnn, dropout_prob) for _ in range(num_layers_encoder)
            ]
        )
        
        self.num_layers_encoder_ = num_layers_encoder
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
    
    def forward(
        self,
        x: Tensor,
    )->Tensor:
        """
        Forward pass for the transformer encoder
        
        Imput:
            - x (torch tensor): Input data
            
        Output:
        
        """
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        for block in self.trs_endr_blocks_:
            x = block(x)
        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
        return x
    
    
class cross_attention_layer(nn.Module):
    """
    Cross attention layer
    """
    
    def __init__(
        self,
        dims_embd: int,
    )->None:
        """
        Cross attention class initialization
        
        Inpout:
            - dims_embd (int): Embedding dimension
        """
        
        super().__init__()
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        self.W_q_ = nn.Linear(dims_embd, dims_embd)
        self.W_k_ = nn.Linear(dims_embd, dims_embd)
        self.W_v_ = nn.Linear(dims_embd, dims_embd)
        
        self.dims_embd_ = dims_embd
        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
    def forward(
        self, 
        x: Tensor,
        y: Tensor
    )->Tensor:
        """
        Forward pass for the cross-attention layer
        
        Imput:
            - x (torch tensor): Input encoder data
            - y (torch tensor): Input decoder data
            
        Output:
        
        """
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        Q = self.W_q_(y)
        K = self.W_k_(x)
        V = self.W_v_(x)
        
        # scaled dot-product
        Q /= (self.dims_embd_ ** (1./4))
        K /= (self.dims_embd_ ** (1./4))
        temp = torch.bmm(Q, K.transpose(1, 2))
        temp = F.softmax(temp, dim=-1)
        y = temp.bmm(V)
        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
    
        return y
    

class transformer_block_decoder(nn.Module):
    """
    Transformer single decoder block
    """
    
    def __init__(
        self,
        dims_embd: int,
        num_hidden_nodes_ffnn: int = 2048,
        dropout_prob: float = 0.0
    )->None:
        """
        Transformer single block class initialization
        
        Inpout:
            - dims_embd (int):             Embedding dimension
            - num_hidden_nodes_ffnn (int): Number of neurons in the fed-forward layer
            - dropout_prob (float):        Dropout probability in liner layers
        """
        
        super().__init__()
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        self.attention_ = self_attention_layer(dims_embd)
        self.cross_attention_ = cross_attention_layer(dims_embd)
        
        self.layer_norm1_ = nn.LayerNorm(dims_embd)
        self.layer_norm2_ = nn.LayerNorm(dims_embd)
        self.layer_norm3_ = nn.LayerNorm(dims_embd)
        
        self.ffnn_ = nn.Sequential(
            nn.Linear(dims_embd, num_hidden_nodes_ffnn),
            nn.ReLU(),
            nn.Linear(num_hidden_nodes_ffnn, dims_embd)
        )
        self.droput_ops_ = nn.Dropout(dropout_prob)
        
        self.dims_embd_ = dims_embd
        self.num_hidden_nodes_ffnn_ = num_hidden_nodes_ffnn
        self.dropout_prob_ = dropout_prob
        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
    def forward(
        self,
        x: Tensor,
        y: Tensor
    )->Tensor:
        """
        Forward pass for the transformer block
        
        Imput:
            - x (torch tensor): Input encoder data
            - y (torch tensor): Input decoder data
            
        Output:
        
        """
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        x = self.layer_norm1_(x + self.attention_(x))
        x = self.droput_ops_(x)
        x = self.layer_norm2_(x + self.cross_attention_(x, y))
        x = self.droput_ops_(x)
        x = self.layer_norm3_(x + self.ffnn_(x))
        x = self.droput_ops_(x)
    
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
        return x
    
    
class transformer_decoder(nn.Module):
    """
    Transformer decoder module
    """
    
    def __init__(
        self,
        dims_embd: int,
        num_hidden_nodes_ffnn: int = 2048,
        dropout_prob: float = 0.0,
        num_layers_decoder: int = 2
    )->None:
        """
        Transformer decoder class initialization
        
        Inpout:
            - dims_embd (int):             Embedding dimension
            - num_hidden_nodes_ffnn (int): Number of neurons in the fed-forward layer
            - dropout_prob (float):        Dropout probability in liner layers
            - num_layers_decoder (int):    Number decoder blocks
        """
        super().__init__()
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        self.trs_dcdr_blocks_ = nn.ModuleList(
            [
                transformer_block_decoder(dims_embd, num_hidden_nodes_ffnn, dropout_prob) for _ in range(num_layers_decoder)
            ]
        )
        
        self.num_layers_decoder_ = num_layers_decoder
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
    def forward(
        self,
        x: Tensor,
        y: Tensor
    )->Tensor:
        """
        Forward pass for the transformer encoder
        
        Imput:
            - x (torch tensor): Input encoder data
            - y (torch tensor): Input decoder data
            
        Output:
        
        """
        
        ############################################################################
        #                             Your code will be here                       #
        #--------------------------------------------------------------------------#
        
        for block in self.trs_dcdr_blocks_:
            x = block(x, y)
        
        #--------------------------------------------------------------------------#
        #                             End of your code                             #
        ############################################################################
        
        return x
        
        
# network weight inilializer
class weights_initilizer():
    """Parameter initilizer"""
    
    def __init__(
        self, 
        network: Any, 
        initializer: str ='xvr_unifrm'
    ) -> None:
        
        print('='*70)
        print('Initializing network parameters using: "{}"' .format(initializer))
        print('-'*70)
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
        print('='*70)
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
            
