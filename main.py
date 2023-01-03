#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:23:31 2022

@author: beatrizlourenco
"""

import data_cleaning as dtclean
import preprocessing_transformer as prt
import train_transformer as trt
from scheduler import Scheduler
import mortalityRateTransformer as mrt
from torch import nn, optim
import recursive_forecast as rf
import math

if __name__ == "__main__":

    # training
    training_mode = False
    resume_training = False

    # Check for control logic inconsistency 
    if not training_mode:
        assert not resume_training
     
    
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


    # Model hyperparameters  
    input_size = 5
    batch_first = True
    batch_size = 5
    d_model = 512
    n_decoder_layers = 4 
    n_encoder_layers = 4
    n_heads = 8
    T_encoder = 7
    T_decoder = 3
    dropout_encoder = 0.2 
    dropout_decoder = 0.2
    dropout_pos_enc = 0.1
    dim_feedforward_encoder = 2048
    dim_feedforward_decoder = 2048
    num_predicted_features = 1
    

    # Preprocessing
    data = dtclean.get_country_data(country)
    
    # Split Data
    training_data, validation_test_data  = dtclean.split_data(data, split_value1)
    validation_data, testing_data  = dtclean.split_data(validation_test_data, split_value2)    
    
    # preprocessing for the transformer
    if gender == 'both':
        train_data = prt.preprocessing_with_both_genders(training_data, (T_encoder, T_decoder), tau0, batch_size)
        val_data  = prt.preprocessing_with_both_genders(validation_data,  (T_encoder, T_decoder), tau0, batch_size)
        test_data = prt.preprocessing_with_both_genders(testing_data,  (T_encoder, T_decoder), tau0, batch_size)


    elif gender == 'Male' or gender == 'Female' :
        train_data = prt.preprocessed_data( training_data,  gender, (T_encoder, T_decoder), tau0, batch_size)
        val_data = prt.preprocessed_data( validation_data, gender, (T_encoder, T_decoder), tau0, batch_size)
        test_data = prt.preprocessed_data(testing_data, gender,  (T_encoder, T_decoder), tau0, batch_size)


    train_data, val_data, test_data = prt.from_numpy_to_torch(train_data), prt.from_numpy_to_torch(val_data), prt.from_numpy_to_torch(test_data)

    # Initializing model object
    model = mrt.MortalityRateTransformer(
        input_size = input_size,
        batch_first = batch_first,
        d_model = d_model,
        n_decoder_layers = n_decoder_layers,
        n_encoder_layers = n_encoder_layers,
        n_heads = n_heads,
        T_encoder = T_encoder,
        T_decoder = T_decoder,
        dropout_encoder = dropout_encoder,
        dropout_decoder = dropout_decoder,
        dropout_pos_enc = dropout_pos_enc,
        dim_feedforward_encoder = dim_feedforward_encoder,
        dim_feedforward_decoder = dim_feedforward_decoder,
        num_predicted_features = num_predicted_features,
        both_gender_model = both_gender_model
    )
    
    # Masking the encoder output and decoder input for decoder:
    tgt_mask = mrt.generate_square_subsequent_mask(
        dim1 = T_decoder,
        dim2 = T_decoder
    )
    
    xe_mask = mrt.generate_square_subsequent_mask(
        dim1 = T_decoder,
        dim2 = T_encoder
    )

    # Training hyperparameters
    criterion = nn.MSELoss()
    opt = optim.SGD(model.parameters(), lr = 0.005)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size = 30, gamma = 0.1)
    #scheduler = Scheduler(opt, dim_embed = d_model, warmup_steps = 50)
    epochs = 1

    # Training
    if training_mode == True:
        best_model, history = trt.fit(
            model = model,
            batch_size = batch_size,
            epochs = epochs,
            train_data = train_data,
            val_data = val_data,
            xe_mask = xe_mask,
            tgt_mask = tgt_mask,
            opt = opt, 
            criterion = criterion, 
            scheduler = scheduler,
            resume_training = resume_training
        )
    else:
        best_model, history = trt.load_best_model(model)
    
    first_year, last_year = 2000, 2020

    ######## IMPROVE THIS SLICE
    import numpy as np
    from sklearn.metrics import mean_squared_error
    real_test_male = (testing_data[testing_data['Gender'] == 'Male']).copy()['mx']
    real_test_female = (testing_data[testing_data['Gender'] == 'Female']).copy()['mx']
    recursive_prediction_male = rf.recursive_forecast_both_genders(data, first_year,last_year, (T_encoder, T_decoder), tau0, model, xe_mask, tgt_mask, gender = 'Male')
    recursive_prediction_loss_male = np.round(mean_squared_error(real_test_male.to_numpy(),recursive_prediction_male['mx'].to_numpy())*10**4,3)
    recursive_prediction_female = rf.recursive_forecast_both_genders(data, first_year,last_year, (T_encoder, T_decoder), tau0, model, xe_mask, tgt_mask, gender = 'Male')
    recursive_prediction_loss_female = np.round(mean_squared_error(real_test_male.to_numpy(),recursive_prediction_female['mx'].to_numpy())*10**4,3)
    #########

    trt.save_plots(history['train_loss_history'], history['val_loss_history'])

    # Evaluation
    test_loss = trt.evaluate(best_model, batch_size, test_data, tgt_mask,  xe_mask, criterion)

    print('=' * 89)
    print('| End of training | training loss {:5.2f} | validation loss {:5.2f} | test loss {:5.2f} | test ppl {:8.2f}'.format(
        history['train_loss_history'][-1], history['val_loss_history'][-1], test_loss, math.exp(test_loss)))
    print('-' * 89)
    print('| Evaluating 20 years of recursive data | Male loss {:5.2f} | Female loss {:5.2f} '.format(recursive_prediction_loss_male,recursive_prediction_loss_female))
    print('=' * 89)



    