#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:23:31 2022

@author: beatrizlourenco
"""

from data_cleaning import preprocess_country_data, split_data
from preprocessing import preprocessing_with_both_genders, preprocessed_data
import TimeSeriesTransformer as tst
from torch import utils, from_numpy, nn, optim
#from torch_training import train_model
from train import train

if __name__ == "__main__":
    # Control
    country = "PT"
    split_value = 2000
    raw_filenamePT = 'Dataset/Mx_1x1_alt.txt'
    raw_filenameSW = 'Dataset/CHE_mort.xlsx'
    T = 5
    tau0 = 5
    
    # Preprocessing
    data = preprocess_country_data(country, raw_filenamePT, raw_filenameSW)
    
    # Split Data
    training_data, validation_data  = split_data(data, split_value)
    
    
    # preprocessing for the transformer
    src, trg, trg_y = preprocessed_data(data, training_data,'Female', T, tau0, model = "transformer")
    src = from_numpy(src).float() 
    trg = from_numpy(trg).float()
    trg_y = from_numpy(trg_y).float()
    
    ## Model parameters
    dim_val = 512 # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
    n_heads = 8 # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
    n_decoder_layers = 4 # Number of times the decoder layer is stacked in the decoder
    n_encoder_layers = 4 # Number of times the encoder layer is stacked in the encoder
    input_size = 5 # The number of input variables. 1 if univariate forecasting.
    dec_seq_len = 5 # length of input given to decoder. Can have any integer value.
    enc_seq_len = 5 # length of input given to encoder. Can have any integer value.
    output_sequence_length = 5 # Length of the target sequence, i.e. how many time steps should your forecast cover
    max_seq_len = enc_seq_len # What's the longest sequence the model will encounter? Used to make the positional encoder
    num_predicted_features = 5
    
    model = tst.TimeSeriesTransformer(
        dim_val=dim_val,
        input_size=input_size, 
        dec_seq_len=dec_seq_len,
        out_seq_len=output_sequence_length, 
        n_decoder_layers=n_decoder_layers,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads,
        num_predicted_features=num_predicted_features)
    
    # Make src mask for decoder with size:
    tgt_mask = tst.generate_square_subsequent_mask(
        dim1=output_sequence_length,
        dim2=output_sequence_length
       )
    
    src_mask = tst.generate_square_subsequent_mask(
        dim1=output_sequence_length,
        dim2=enc_seq_len
    )
    
    # Defining loss function and optimizer
    loss = nn.MSELoss()
    opt = optim.SGD(model.parameters(), lr=0.01)
    
    
    #model.train()
    error = loss(model(
    src = src, 
    tgt = trg,
    src_mask=src_mask,
    tgt_mask=tgt_mask
    ), trg)
    print(f'The initial transformer error is at {error}.')
    
    #Losses
    all_losses = train(
        model = model, 
        src = src,
        trg = trg,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
        num_epochs = 1, 
        optimizer = opt, 
        criterion = loss)
    
    # Mean Squared error
    output = model(
    src = src, 
    tgt = trg,
    src_mask=src_mask,
    tgt_mask=tgt_mask
    )
    
    error = loss(output, trg)
    print(f'The transformer error is at {error}.')
    
    #train_loss_list = fit(model, opt, loss, batchify_data(src), 10)

"""
    train_model = train_model(model, epochs= 100, print_every=5)
    
    
    
    
    """
    
    
    
    

    
    
    #transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
    #out = transformer_model(src, from_numpy(trg).float())

    
    