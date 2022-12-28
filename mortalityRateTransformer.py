#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 09:58:22 2022

@author: beatrizlourenco

link: https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
"""

import torch.nn as nn 
from torch import nn, Tensor, triu, ones, concat
import positional_encoder as pe
import torch.nn.functional as F


class MortalityRateTransformer(nn.Module):
    """Transformer model for predicting mortality rates with age and gender. Follows the paper: N. Wu, B. Green, X. Ben, and S. O’Banion. Deep transformer models for time series forecasting: The influenza prevalence case.

    Attributes:
        both_gender_model (bool): True if the model forecasting 2 gender. False otherwise
        encoder_input_layer (nn.Linear): input linear layers needed for the encoder.
        decoder_input_layer (nn.Linear): input linear layers needed for the decoder.
        positional_encoding_layer (pe.PositionalEncoder): positional enconding layer for recording the sequence steps relative position.
        encoder (nn.TransformerEncoder): stack of multiple encoder layers in nn.TransformerEncoder.
        decoder (nn.TransformerDecoder): stack of multiple decoder layers in nn.TransformerEncoder.
        linear_mapping (nn.Linear): linear output for One Gender Model.
        linear_mapping_with_gender_ind (nn.Linear): linear output for two Gender Model.
        
    """

    def __init__(self, 
        input_size: int,
        batch_first: bool = True,
        n_heads: int = 8,
        d_model: int = 512,  
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        T_encoder: int = 3,
        T_decoder: int = 7,
        dropout_encoder: float = 0.2, 
        dropout_decoder: float = 0.2,
        dropout_pos_enc: float = 0.1,
        dim_feedforward_encoder: int = 2048,
        dim_feedforward_decoder: int = 2048,
        num_predicted_features: int = 1,
        both_gender_model: bool = False
        ): 

        """
        Args:
            input_size: The number of input variables. 1 if univariate forecasting.
            batch_first: Defines the shape of the received input. Shape: (T, B, E) if batch_first = False or (B, T, E) if batch_first = True, where T is the source sequence length, B is the batch size, and E is the number of features (1 if univariate)
            n_heads: The number of attention heads (aka parallel attention layers).
            d_model: Dimension of intermidiate layers. Can be any value divisible by n_heads. 
            n_decoder_layers: Number of times the decoder layer is stacked in the decoder.
            n_encoder_layers: Number of times the encoder layer is stacked in the encoder.
            T_encoder: Number of timesteps fed to the enconder.
            T_decoder: Number of timesteps fed to the decoder.
            dropout_encoder: Dropout probability used in the encoder layer.
            dropout_decoder: Dropout probability used in the decoder layer.
            dropout_pos_enc: Dropout probability used in the positional encoder layer.
            dim_feedforward_encoder: Dimension of feed forward layers in the encoder.
            dim_feedforward_decoder: Dimension of feed forward layers in the decoder.
            num_predicted_features: Dimension of each output timestep.
            both_gender_model: Setting the atribute both_gender_model.

        """


        super().__init__() 

        self.both_gender_model = both_gender_model

        self.encoder_input_layer = nn.Linear(
            in_features = input_size, 
            out_features = d_model 
            )

        self.decoder_input_layer = nn.Linear(
            in_features = num_predicted_features,
            out_features = d_model
            )  

        self.positional_encoding_layer_enc = pe.PositionalEncoder(
            d_model = d_model,
            dropout = dropout_pos_enc,
            max_seq_len = T_encoder
            )

        self.positional_encoding_layer_dec = pe.PositionalEncoder(
            d_model = d_model,
            dropout = dropout_pos_enc,
            max_seq_len = T_decoder
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = n_heads,
            dim_feedforward = dim_feedforward_encoder,
            dropout = dropout_encoder,
            batch_first = batch_first
            )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers, 
            norm=None
            )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first
            )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers, 
            norm=None
            )
        
        self.linear_mapping = nn.Linear(
            in_features = d_model,
            out_features = num_predicted_features
            )

        self.linear_mapping_with_gender_ind = nn.Linear(
            in_features = d_model + 1,
            out_features = num_predicted_features
            )

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None, 
                tgt_mask: Tensor=None, gender_index: Tensor=None) -> Tensor:
        """
        Returns a tensor of shape:
        [target_sequence_length, batch_size, num_predicted_features]
        
        Args:
            src: the encoder's output sequence. Shape: (S,E) for unbatched input, 
                 (S, N, E) if batch_first=False or (N, S, E) if 
                 batch_first=True, where S is the source sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)
            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input, 
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if 
                 batch_first=True, where T is the target sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)
            src_mask: the mask for the src sequence to prevent the model from 
                      using data points from the target sequence
            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence
        """


        # Pass throguh the input layer right before the encoder
        src = self.encoder_input_layer(src) # src shape: [batch_size, src length, d_model] regardless of number of input features

        # Pass through the positional encoding layer
        src = self.positional_encoding_layer_enc(src) # src shape: [batch_size, src length, d_model] regardless of number of input features

        # Pass through all the stacked encoder layers in the encoder
        src = self.encoder( # src shape: [batch_size, enc_seq_len, d_model]
            src=src
            )

        
        tgt = self.positional_encoding_layer_dec(tgt) # src shape: [batch_size, src length, d_model] regardless of number of input features

        # Pass decoder input through decoder input layer
        decoder_output = self.decoder_input_layer(tgt) # src shape: [target sequence length, batch_size, d_model] regardless of number of input features

        # Pass throguh decoder - output shape: [batch_size, target seq len, d_model]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            )

        output = decoder_output

        if self.both_gender_model:
            output = concat([decoder_output, gender_index], dim = 2)
            output = self.linear_mapping_with_gender_ind(output)

        else: 
            output = decoder_output
            output = self.linear_mapping(output) # shape [batch_size, target seq len]


        return output
    
    
def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.

    Source:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    
    Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 
    Return:
        A Tensor of shape [dim1, dim2]
    """
    return triu(ones(dim1, dim2) * float('-inf'), diagonal=1)