#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:23:31 2022

@author: beatrizlourenco
"""

from data_cleaning import preprocess_country_data, split_data
from preprocessing import preprocessing_with_both_genders, preprocessed_data
from preprocessing_transformer import preprocessed_data, data_to_logmat, reshape_logmat_agerange

import TimeSeriesTransformer as tst
from torch import utils, from_numpy, nn, optim
#from torch_training import train_model
from train import train, batchify

if __name__ == "__main__":
    
    # Control
    country = "PT"
    split_value = 2000
    raw_filenamePT = 'Dataset/Mx_1x1_alt.txt'
    raw_filenameSW = 'Dataset/CHE_mort.xlsx'
    T = 10
    T_encoder = 7
    T_decoder = 3
    tau0 = 5
    
    # Preprocessing
    data = preprocess_country_data(country, raw_filenamePT, raw_filenameSW)
    
    # Split Data
    training_data, validation_data  = split_data(data, split_value)
    
    data = data_to_logmat(training_data, 'Female')
    data = reshape_logmat_agerange(data, tau0)

    
    # preprocessing for the transformer
    xe, xd = preprocessed_data(data, training_data,'Female', (T_encoder, T_decoder), tau0, model = "transformer")
    xe = from_numpy(xe).float() 
    xd = from_numpy(xd).float()

    """
    ## Model parameters
    dim_val = 512 # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
    n_heads = 8 # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
    n_decoder_layers = 4 # Number of times the decoder layer is stacked in the decoder
    n_encoder_layers = 4 # Number of times the encoder layer is stacked in the encoder
    input_size = 5 # The number of input variables. 1 if univariate forecasting.
    dec_seq_len = T_decoder # length of input given to decoder. Can have any integer value.
    enc_seq_len = T_encoder # length of input given to encoder. Can have any integer value.
    output_sequence_length = 10 # Length of the target sequence, i.e. how many time steps should your forecast cover
    max_seq_len = enc_seq_len # What's the longest sequence the model will encounter? Used to make the positional encoder
    num_predicted_features = 5
    batch_size = 5
    

    xe = batchify(xe, batch_size)  # shape [seq_len, batch_size]
    xd = batchify(xd, batch_size)  # shape [seq_len, batch_size]
    yd = batchify(yd, batch_size)  # shape [seq_len, batch_size]



    model = tst.TimeSeriesTransformer(
        dim_val=dim_val,
        input_size=input_size, 
        dec_seq_len=dec_seq_len,
        out_seq_len=output_sequence_length, 
        n_decoder_layers=n_decoder_layers,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads,
        num_predicted_features=num_predicted_features)
    
    # Make xe mask for decoder with size:
    tgt_mask = tst.generate_square_subsequent_mask(
        dim1 = dec_seq_len,
        dim2 = dec_seq_len
       )
    
    xe_mask = tst.generate_square_subsequent_mask(
        dim1=dec_seq_len,
        dim2=enc_seq_len
    )
    
    # Defining loss function and optimizer
    loss = nn.MSELoss()
    lr = 0.05
    opt = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(opt, 1.0, gamma=0.95)
    
    #Losses
    all_losses = train(
        model = model, 
        src = xe,
        trg = xd,
        src_mask = xe_mask,
        tgt_mask = tgt_mask,
        num_epochs = 1, 
        optimizer = opt, 
        criterion = loss)
    
    # Mean Squared error
    output = model(
    src = xe, 
    tgt = xd,
    src_mask=xe_mask,
    tgt_mask=tgt_mask
    )
    
    error = loss(output, yd)
    print(f'The transformer error is at {error}.')
    

    """
    