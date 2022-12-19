#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:23:31 2022

@author: beatrizlourenco
"""

from data_cleaning import preprocess_country_data, split_data
from preprocessing import preprocessing_with_both_genders, preprocessed_data
from preprocessing_transformer import preprocessed_data, data_to_logmat
from scheduler import Scheduler
import TimeSeriesTransformer as tst
from torch import utils, from_numpy, nn, optim
#from torch_training import train_model
from train_transformer import train, evaluate
import math, time, copy

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
    split_value1 = 1989
    split_value2 = 2000
    
    # Preprocessing
    data = preprocess_country_data(country, raw_filenamePT, raw_filenameSW)
    
    # Split Data
    training_data, validation_test_data  = split_data(data, split_value1)
    validation_data, test_data  = split_data(validation_test_data, split_value2)    

    training_data = data_to_logmat(training_data, 'Female')
    validation_data = data_to_logmat(validation_data, 'Female')
    
    # preprocessing for the transformer
    xe, xd, yd = preprocessed_data(training_data,'Female', (T_encoder, T_decoder), tau0, model = "transformer")
    xe = from_numpy(xe).float() 
    xd = from_numpy(xd).float()
    yd = from_numpy(yd).float()


    xe_val, xd_val, yd_val = preprocessed_data( validation_data,'Female', (T_encoder, T_decoder), tau0, model = "transformer")
    xe_val = from_numpy(xe_val).float() 
    xd_val = from_numpy(xd_val).float()
    yd_val = from_numpy(yd_val).float()

    
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
    opt = optim.SGD(model.parameters(), lr = lr)
    #opt = optim.Adam(model.parameters(), betas = (0.9, 0.98), eps = 1e-9)
    scheduler = Scheduler(opt, dim_embed = dim_val, warmup_steps = 5)

    #opt = optim.SGD(model.parameters(), lr=lr)
    #scheduler = optim.lr_scheduler.StepLR(opt, 1.0, gamma=0.95, )
    
    #Losses
    best_val_loss = float('inf')
    epochs = 5
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(
        model = model, 
        train_data = (xe, xd, yd),
        src_mask = xe_mask,
        tgt_mask = tgt_mask,
        epoch = epoch, 
        optimizer = opt, 
        lr = lr,
        criterion = loss)
        eval_data = (xe_val, xd_val, yd)
        val_loss = evaluate( model, eval_data, tgt_mask,  xe_mask, loss)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        if scheduler is not None:
            scheduler.step()

test_loss = evaluate(best_model, test_data)
training_loss = evaluate(best_model, training_data)

print('=' * 89)
print('| End of training | training loss {:5.2f} | test loss {:5.2f} | test ppl {:8.2f}'.format(
    training_loss, test_loss, math.exp(test_loss)))
print('=' * 89)

    

    