"""
Created on Sat Oct 15 09:57:27 2022
@author: beatrizlourenco
"""

import torch
import torch.nn as nn 
import math
from torch import nn, Tensor

class PositionalEncoder(nn.Module):

    """Transformer model for predicting mortality rates with age and gender. Follows the paper: N. Wu, B. Green, X. Ben, and S. O’Banion. Deep transformer models for time series forecasting: The influenza prevalence case.

    Note: The code of this class is adapted from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Attributes:
        d_model (int): The dimension of the output of sub-layers in the model.
        dropout (float): the dropout rate.
        batch_first (bool): True if the batch corresponds to the first position of the x Tensor. False otherwise.
        x_dim (int): The relative position of the time series time steps in the x Tensor.

        
    """

    def __init__(
        self, 
        dropout: float=0.1, 
        max_seq_len: int=5000, 
        d_model: int=512,
        batch_first: bool=True
        ):

        """
        Args:
            dropout (float): The dropout rate.
            max_seq_len (int): The maximum length of the input sequences.
            d_model (int): The dimension of the output of sub-layers in the model.
            batch_first (bool): Value that will set the batch_first attribute of the class.
            

        """

        super().__init__()

        self.d_model = d_model
        
        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        self.x_dim = 1 if self.batch_first else 0

        position = torch.arange(max_seq_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        if batch_first:
            pe = torch.zeros(1, max_seq_len , d_model)
            
            pe[0, :, 0::2] = torch.sin(position * div_term)
            
            pe[0, :, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe)
        
        else:
            pe = torch.zeros(1, max_seq_len , d_model)
            
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe)
        
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the positional encoding layer.

        Args:
            x (Tensor): Tensor to add positional information. Shape: [batch_size, enc_seq_len, dim_val] or [enc_seq_len, batch_size, dim_val]
        
        Returns:
            Tensor with the same shape as x with positional information.
        """
        x = x + self.pe[:x.size(self.x_dim)]

        return self.dropout(x)