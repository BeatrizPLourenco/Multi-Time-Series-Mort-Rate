#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:23:31 2022

@author: beatrizlourenco
"""

from data_cleaning import preprocess_country_data, split_data
from preprocessing import  preprocessed_data
from preprocessing_transformer import preprocessed_data, data_to_logmat, preprocessing_with_both_genders
from scheduler import Scheduler
import mortalityRateTransformer as mrt
from torch import from_numpy, nn, optim
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
    gender = 'both'
    both_gender_model = (gender == 'both')
    
    # Preprocessing
    data = preprocess_country_data(country, raw_filenamePT, raw_filenameSW)
    
    # Split Data
    training_data, validation_test_data  = split_data(data, split_value1)
    validation_data, test_data  = split_data(validation_test_data, split_value2)    

    
    # preprocessing for the transformer
    if gender == 'both':
        xe, xd, gender_idx, yd = preprocessing_with_both_genders(training_data, (T_encoder, T_decoder), tau0)
        xe, xd, gender_idx, yd  = from_numpy(xe).float(), from_numpy(xd).float(), from_numpy(gender_idx).float(), from_numpy(yd).float()

        xe_val, xd_val, gender_idx_val, yd_val = preprocessing_with_both_genders( validation_data, (T_encoder, T_decoder), tau0)
        xe_val, xd_val, gender_idx_val, yd_val  = from_numpy(xe_val).float(), from_numpy(xd_val).float(), from_numpy(gender_idx_val).float(), from_numpy(yd_val).float()

        xe_test, xd_test, gender_idx_test, yd_test = preprocessing_with_both_genders(test_data, (T_encoder, T_decoder), tau0)
        xe_test, xd_test, gender_idx_test, yd_test  = from_numpy(xe_test).float(), from_numpy(xd_test).float(), from_numpy(gender_idx_test).float(), from_numpy(yd_test).float()


    elif gender == 'Male' or gender == 'Female' :
        xe, xd, gender_idx, yd = preprocessed_data( training_data, gender, (T_encoder, T_decoder), tau0, model = "transformer")
        xe, xd, gender_idx, yd  = from_numpy(xe).float(), from_numpy(xd).float(), from_numpy(yd).float()

        xe_val, xd_val, gender_idx_val, yd_val= preprocessed_data( validation_data, gender, (T_encoder, T_decoder), tau0, model = "transformer")
        xe_val, xd_val, gender_idx_val, yd_val = from_numpy(xe_val).float(), from_numpy(xd_val).float(), from_numpy(yd_val).float()

        xe_test, xd_test, gender_idx_test, yd_test = preprocessed_data(test_data, gender, (T_encoder, T_decoder), tau0, model = "transformer")
        xe_test, xd_test, gender_idx_test, yd_test  = from_numpy(xe_test).float(), from_numpy(xd_test).float(), from_numpy(yd_test).float()

    train_data = (xe, xd, gender_idx, yd)
    eval_data = (xe_val, xd_val, gender_idx_val, yd_val )
    test_data = (xe_test, xd_test, gender_idx_test, yd_test)

    
    ## Model parameters
    d_model = 512 # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
    n_heads = 8 # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
    n_decoder_layers = 4 # Number of times the decoder layer is stacked in the decoder
    n_encoder_layers = 4 # Number of times the encoder layer is stacked in the encoder
    input_size = 5 # The number of input variables. 1 if univariate forecasting.
    output_sequence_length = 10 # Length of the target sequence, i.e. how many time steps should your forecast cover
    num_predicted_features = 1
    batch_size = 5


    model = mrt.MortalityRateTransformer(
        input_size = input_size,
        d_model = d_model, 
        n_decoder_layers = n_decoder_layers,
        n_encoder_layers = n_encoder_layers,
        n_heads = n_heads,
        num_predicted_features = num_predicted_features,
        both_gender_model = both_gender_model
    )
    
    # Make xe mask for decoder with size:
    tgt_mask = mrt.generate_square_subsequent_mask(
        dim1 = T_decoder,
        dim2 = T_decoder
    )
    
    xe_mask = mrt.generate_square_subsequent_mask(
        dim1 = T_decoder,
        dim2 = T_encoder
    )
    
    # Defining loss function and optimizer
    loss = nn.MSELoss()

    lr = 0.05
    opt = optim.SGD(model.parameters(), lr = lr)
    scheduler = Scheduler(opt, dim_embed = d_model, warmup_steps = 2)

    #opt = optim.Adam(model.parameters(), betas = (0.9, 0.98), eps = 1e-9)
    #scheduler = optim.lr_scheduler.StepLR(opt, 1.0, gamma=0.95, )
    
    #Losses
    best_val_loss = float('inf')
    epochs = 5
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(
        model = model, 
        train_data = train_data,
        src_mask = xe_mask,
        tgt_mask = tgt_mask,
        epoch = epoch, 
        optimizer = opt, 
        lr = lr,
        criterion = loss)
        eval_data = eval_data
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

test_loss = evaluate(best_model, test_data, tgt_mask,  xe_mask, loss)
training_loss = evaluate(best_model, train_data, tgt_mask,  xe_mask, loss)

print('=' * 89)
print('| End of training | training loss {:5.2f} | test loss {:5.2f} | test ppl {:8.2f}'.format(
    training_loss, test_loss, math.exp(test_loss)))
print('=' * 89)



    