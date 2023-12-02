#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:23:31 2022

@author: beatrizlourenco
"""
import data_cleaning as dtclean
import numpy as np
import preprocessing_transformer as prt
import train_transformer as trt
from scheduler import Scheduler
import mortalityRateTransformer as mrt
from torch import nn, optim, zeros
import recursive_forecast as rf
import explainability as xai
import math
import torch
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from skorch import NeuralNetRegressor
from hyperparameter_tuning import flatten_tensors, unflatten_tensors, get_original_shapes




if __name__ == "__main__":

    # training
    training_mode = False
    resume_training = False

    # Check for control logic inconsistency 
    if not training_mode:
        assert not resume_training
     
    
    # Control
    country = "PT"
    #split_value = 2000
    raw_filenamePT = 'Dataset/Mx_1x1_alt.txt'
    raw_filenameSW = 'Dataset/CHE_mort.xlsx'
    T = 10
    T_encoder = 7
    T_decoder = 3
    tau0 = 5
    split_value1 = 1993 # 1993 a 2005 corresponde a 13 anos (13/66 approx. 20%)
    split_value2 = 2006 # 2006 a 2022 corresponde a 17 anos (17/83 approx. 20%)
    gender = 'male'
    both_gender_model = (gender == 'both')
    checkpoint_dir = f'Saved_models/checkpoint_{gender}.pt'
    best_model_dir = f'Saved_models/best_model_{gender}.pt'


    # Model hyperparameters  
    input_size = 5
    batch_first = True
    batch_size = 5
    d_model = 16
    n_decoder_layers = 2
    n_encoder_layers = 2
    n_heads = 4
    T_encoder = 7
    T_decoder = 3
    dropout_encoder = 0.2 
    dropout_decoder = 0.2
    dropout_pos_enc = 0.1
    dim_feedforward_encoder = 64
    dim_feedforward_decoder = 64
    num_predicted_features = 1


    # Preprocessing
    data = dtclean.get_country_data(country)
    data_logmat = prt.data_to_logmat(data, gender)
    xmin, xmax = prt.min_max_from_dataframe(data_logmat)
    
    # Split Data
    training_data, validation_test_data  = dtclean.split_data(data, split_value1)
    validation_data, testing_data  = dtclean.split_data(validation_test_data, split_value2)
    training_val_data, testing_data  = dtclean.split_data(data, split_value2)    
    
    # preprocessing for the transformer
    if gender == 'both':
        train_data = prt.preprocessing_with_both_genders(training_data, (T_encoder, T_decoder), tau0, xmin, xmax, batch_size)
        val_data  = prt.preprocessing_with_both_genders(validation_data,  (T_encoder, T_decoder), tau0, xmin, xmax, batch_size)
        test_data = prt.preprocessing_with_both_genders(testing_data,  (T_encoder, T_decoder), tau0, xmin, xmax, batch_size)
        train_val_data = prt.preprocessing_with_both_genders(training_val_data,  (T_encoder, T_decoder), tau0, xmin, xmax, batch_size)


    elif gender == 'Male' or gender == 'Female' :
        train_data = prt.preprocessed_data( training_data,  gender, (T_encoder, T_decoder),tau0, xmin, xmax, batch_size)
        val_data = prt.preprocessed_data( validation_data, gender, (T_encoder, T_decoder),tau0, xmin, xmax, batch_size)
        test_data = prt.preprocessed_data(testing_data, gender,  (T_encoder, T_decoder),tau0, xmin, xmax, batch_size)
        train_val_data = prt.preprocessed_data(training_val_data, gender,  (T_encoder, T_decoder),tau0, xmin, xmax, batch_size)


    train_data, val_data, test_data, train_val_data = prt.from_numpy_to_torch(train_data), \
        prt.from_numpy_to_torch(val_data), prt.from_numpy_to_torch(test_data), prt.from_numpy_to_torch(train_val_data)
    
    num_train_val_patterns, num_train_patterns, num_val_patterns, num_test_patterns = train_val_data[0].size(0),train_data[0].size(0), val_data[0].size(0), test_data[0].size(0)
    
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
    #opt = optim.SGD(model.parameters(), lr = 0.05)
    #opt = optim.Adam(model.parameters(), lr=0.001, betas = (0.9,0.98), eps =10**(-9))
    #scheduler = optim.lr_scheduler.StepLR(opt, step_size = 500, gamma = 0.99)
    #scheduler = Scheduler(opt, dim_embed = d_model, warmup_steps = 4000)
    epochs = 500

    og_shape_X = get_original_shapes(train_val_data[:-1])
    og_shape_y = get_original_shapes(train_val_data[-1])
    f= flatten_tensors(train_val_data[:-1])
    f = unflatten_tensors(train_val_data[:-1])
    first_year, last_year = 1940, 1993
    class sklearnWrapper(BaseEstimator):
        def __init__(self, 
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
            both_gender_model = both_gender_model, 
            criterion = nn.MSELoss()):

            self.input_size = input_size
            self.batch_first = batch_first
            self.d_model = d_model
            self.n_decoder_layers = n_decoder_layers
            self.n_encoder_layers = n_encoder_layers
            self.n_heads = n_heads
            self.T_encoder = T_encoder
            self.T_decoder = T_decoder
            self.dropout_encoder = dropout_encoder
            self.dropout_decoder = dropout_decoder
            self.dropout_pos_enc = dropout_pos_enc
            self.dim_feedforward_encoder = dim_feedforward_encoder
            self.dim_feedforward_decoder = dim_feedforward_decoder
            self.num_predicted_features = num_predicted_features
            self.both_gender_model = both_gender_model 
            self.criterion = criterion

            self.model = mrt.MortalityRateTransformer(
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

        def fit(self, X, y):
            # Ensure X and y are tensors
            train_data = unflatten_tensors(X, og_shape_X), unflatten_tensors(y, og_shape_y)
            
            self.model, history = trt.fit(
                model = self.model,
                batch_size = batch_size,
                epochs = epochs,
                train_data = train_data,
                val_data = val_data,
                xe_mask = xe_mask,
                tgt_mask = tgt_mask,
                opt = optim.Adam(self.model.parameters(), lr=0.001, betas = (0.9,0.98), eps =10**(-9)), 
                criterion = criterion, 
                scheduler = Scheduler(self.opt, dim_embed = d_model, warmup_steps = 4000),
                resume_training = resume_training,
                checkpoint_dir= checkpoint_dir,
                best_model_dir= best_model_dir
            )

            return self

        def predict(self, X):
            return rf.recursive_forecast(val_data, first_year,last_year, (T_encoder, T_decoder), tau0, xmin, xmax, model, xe_mask, tgt_mask)

    model = sklearnWrapper(
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

    param_grid = {
    'dropout_encoder': [0.10, 0.20, 0.30]
    }
    
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    grid_search = GridSearchCV(model, param_grid, scoring=scorer, cv=2, #(np.arange(0,num_train_val_patterns-num_val_patterns), np.arange(num_train_val_patterns-num_val_patterns,num_train_val_patterns)), 
                               verbose=1)

    # Fit the model to find the best hyperparameters
    grid_search.fit(flatten_tensors(train_val_data[:-1]), flatten_tensors([train_val_data[-1]]))
    