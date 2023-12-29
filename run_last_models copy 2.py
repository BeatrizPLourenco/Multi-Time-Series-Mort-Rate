#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:23:31 2022

@author: beatrizlourenco
"""
from keras.models import load_model
from keras.layers import Dense, LSTM, Flatten, Concatenate, Bidirectional, GRU
from LSTM_Keras import rnn_model
import data_cleaning as dtclean
import preprocessing_transformer as prt
import train_transformer as trt
import mortalityRateTransformer as mrt
import recursive_forecast as rf
import torch
from sklearn.base import BaseEstimator
import mortalityRateTransformer as mrt
import train_transformer as trt
import data_cleaning as dtclean
import preprocessing_transformer as prt
import train_transformer as trt
from scheduler import Scheduler
import mortalityRateTransformer as mrt
from torch import nn, optim, zeros
import recursive_forecast as rf
import explainability as xai
import math
from sklearn.model_selection import GridSearchCV
from itertools import product
import random
import numpy as np
from LSTM_Keras import rnn_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, LSTM, Flatten, Concatenate, Bidirectional, GRU
import pandas as pd
import time
from datetime import datetime
import ast

def train_rnn(parameters : dict, 
                      split_value = 2006,
                      gender = 'both',
                      raw_filename = 'Dataset/Mx_1x1_alt.txt',
                      country = "PT", 
                      seed = 0):
    
    # Preprocessing
    data = dtclean.get_country_data(country, filedir = raw_filename)
    data_logmat = prt.data_to_logmat(data, gender)
    xmin, xmax = prt.min_max_from_dataframe(data_logmat)
    
    # Split Data
    training_data, testing_data  = dtclean.split_data(data, split_value)
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Control
    #split_value = 2000
    #split_value1 = split_value1 # 1993 a 2005 corresponde a 13 anos (13/66 approx. 20%)
    #split_value2 = split_value2 # 2006 a 2022 corresponde a 17 anos (17/83 approx. 20%)
    gender = gender
    both_gender_model = (gender == 'both')
    checkpoint_dir = f'Saved_models/checkpoint_lstm_{gender}.h5'
    best_model_dir = f'Saved_models/best_model_lstm_{gender}.h5'


    # Model hyperparameters  
    T = parameters['T']
    tau0 = parameters['tau0']
    units_per_layer = parameters['units_per_layer']
    rnn_func = LSTM
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']

    if gender == 'both':
        train_data = prt.preprocessing_with_both_gendersLSTM(training_data, T, tau0,xmin, xmax)
        val_data  = prt.preprocessing_with_both_gendersLSTM(testing_data, T, tau0,xmin, xmax)
        test_data = prt.preprocessing_with_both_gendersLSTM(testing_data, T, tau0,xmin, xmax)


    elif gender == 'Male' or gender == 'Female' :
        train_data = prt.preprocessed_dataLSTM( training_data,  gender, T, tau0,xmin, xmax)
        val_data = prt.preprocessed_dataLSTM( testing_data, gender, T, tau0,xmin, xmax)
        test_data = prt.preprocessed_dataLSTM(testing_data, gender, T, tau0,xmin, xmax)


    model = rnn_model(T,tau0,  units_per_layer, rnn_func, gender=gender)
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f")[:-3]
    filepath = "{epoch:02d}-{val_loss:.2f}.weights.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only = True)
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50) #addition of early stopping
    callbacks_list = [checkpoint,earlystop]
    if both_gender_model:
        model.fit(x = train_data[:2], y = train_data[2], validation_split=0.2, epochs = epochs, batch_size = batch_size, verbose = 0, callbacks=callbacks_list) # Model checkpoint CallBack
    else:
        model.fit(x = train_data[:1], y = train_data[2], validation_split=0.2, epochs = epochs, batch_size = batch_size, verbose = 0, callbacks=callbacks_list) # Model checkpoint CallBack

    first_year, last_year = split_value, 2022
    recursive_prediction = rf.recursive_forecast(data, first_year,last_year, T, tau0, xmin, xmax, model, batch_size=1, model_type = 'lstm', gender = gender)
    recursive_prediction_loss_male, recursive_prediction_loss_female = rf.loss_recursive_forecasting(testing_data, recursive_prediction, gender_model = gender)

    model.save(f'lstm_model_retrained_{time.time()}.h5')



    if gender == 'both':
        print(f'Male: {recursive_prediction_loss_male}, Female: {recursive_prediction_loss_female}')
        return (recursive_prediction_loss_male + recursive_prediction_loss_female)/2
    
    elif gender == 'Male':
        return recursive_prediction_loss_male

    elif gender == 'Female':
        return recursive_prediction_loss_female

if __name__ == "__main__":
    split_value1 = 1993
    split_value2 = 2006
    raw_filename = 'Dataset/Mx_1x1_alt_1940_2022.txt'

    country = "PT"
    gender = 'both'

    parameters_lstm = {
        'T': 8,
        'tau0': 5,
        'units_per_layer': [5, 10, 15],
        'rnn_func': LSTM,
        'batch_size': 50,
        'epochs': 500
    }

    train_rnn(parameters_lstm, 
                      split_value = 2006,
                      gender = 'both',
                      raw_filename = raw_filename,
                      country = "PT", 
                      seed = 0)
    
    