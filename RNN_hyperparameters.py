#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:23:31 2022

@author: beatrizlourenco
"""

from hyperparameter_tuning import train_transformer, gridSearch, train_rnn, train_gru
#from ray import tune
from functools import partial
from keras.layers import Dense, LSTM, Flatten, Concatenate, Bidirectional, GRU
import json
import time
import pandas as pd

if __name__ == "__main__":
    split_value1 = 1993
    split_value2 = 2006
    raw_filename = 'Dataset/Mx_1x1_alt_1940_2022.txt'

    country = "PT"

    gender = 'Female'

    parameters = {
        'T': [8, 10, 12],
        'tau0': [3, 5, 7],
        'units_per_layer': [[5, 10, 15], [10, 15, 20], [10, 15], [15, 20]],
        'rnn_func': [LSTM],
        'batch_size': [5, 50, 100],
        'epochs': [500]
    }
    """
    gridSearch(parameters, 
        func_args = (split_value1, split_value2, gender, raw_filename, country), 
        func = train_rnn, 
        model_name =f'{gender}_{time.time()}_rnn',
        csv_to_fill = 'hyperparameters\hyperparameter_tuning_Female_1702314751.876638_rnn.csv')"""

    gender = 'Male'

    parameters = {
        'T': [8, 10, 12],
        'tau0': [3, 5, 7],
        'units_per_layer': [[5, 10, 15], [10, 15, 20], [10, 15], [15, 20]],
        'rnn_func': [LSTM],
        'batch_size': [5, 50, 100],
        'epochs': [500]
    }

    #gridSearch(parameters, func_args = (split_value1, split_value2, gender, raw_filename, country ), func = train_rnn, model_name =f'{gender}_{time.time()}_rnn')

    gender = 'both'

    parameters = {
        'T': [8, 10, 12],
        'tau0': [3, 5, 7],
        'units_per_layer': [[5, 10, 15], [10, 15, 20], [10, 15], [15, 20]],
        'rnn_func': [LSTM],
        'batch_size': [5, 50, 100],
        'epochs': [500]
    }

    #gridSearch(parameters, func_args = (split_value1, split_value2, gender, raw_filename, country), func = train_rnn, model_name =f'{gender}_{time.time()}_rnn')

    gender = 'both'

    parameters = {
        'T': [8, 10, 12],
        'tau0': [3, 5, 7],
        'units_per_layer': [[5, 10, 15], [10, 15, 20], [10, 15], [15, 20]],
        'rnn_func': [LSTM],
        'batch_size': [5, 50, 100],
        'epochs': [500]
    }

    #gridSearch(parameters, func_args = (split_value1, split_value2, gender, raw_filename, country), func = train_rnn, model_name =f'{gender}_{time.time()}_rnn')

    ###### -------------------------------------------------------------------------------------------------------------------------------------------- ###########

    gender = 'Female'

    parameters = {
        'T': [8, 10, 12],
        'tau0': [3, 5, 7],
        'units_per_layer': [[5, 10, 15], [10, 15, 20], [10, 15], [15, 20]],
        'rnn_func': [GRU],
        'batch_size': [5, 50, 100],
        'epochs': [1]
    }

    #gridSearch(parameters, func_args = (split_value1, split_value2, gender, raw_filename, country), func = train_gru, model_name =f'{gender}_{time.time()}_gru')

    gender = 'Male'

    parameters = {
        'T': [8, 10],
        'tau0': [3, 5],
        'units_per_layer': [[5, 10, 15], [10, 15]],
        'gru_func': [GRU],
        'batch_size': [5, 50, 100],
        'epochs': [1]
    }

    gridSearch(parameters, func_args = (split_value1, split_value2, gender, raw_filename, country), func = train_gru, model_name =f'{gender}_{time.time()}_gru')

    gender = 'both'

    parameters = {
        'T': [8, 10],
        'tau0': [3, 5],
        'units_per_layer': [[5, 10, 15], [10, 15]],
        'gru_func': [GRU],
        'batch_size': [5, 50],
        'epochs': [1]
    }

    gridSearch(parameters, func_args = (split_value1, split_value2, gender, raw_filename, country), func = train_gru, model_name =f'{gender}_{time.time()}_gru')

    gender = 'both'
    config = {
        'T' : [(7,3), (9,1)],
        'tau0' : [3, 5],
        'batch_size' : [5],
        'd_model' : [4],
        'n_decoder_layers' : [2],
        'n_encoder_layers' : [1, 2],
        'n_heads' : [1],
        'dropout_encoder' : [0.1, 0.2],
        'dropout_decoder' : [0.1, 0.2],
        'dropout_pos_enc' : [0.1],
        'dim_feedforward_encoder' : [4],
        'dim_feedforward_decoder' : [4],
        'epochs' : [1]
    }
    best_hyperparameters_b, best_evaluation_b = gridSearch(config,  func_args = (split_value1, split_value2, gender, raw_filename, country), model_name ='both_transformer')



    



    