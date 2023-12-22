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
    ###### -------------------------------------------------------------------------------------------------------------------------------------------- ###########

    gender = 'Male'

    parameters= {
        'T': [8, 10],
        'tau0': [3, 5],
        'units_per_layer': [[5, 10, 15], [10, 15, 20], [10, 15]],
        'rnn_func': [GRU],
        'batch_size': [5, 50],
        'epochs': [500]
    }


    gridSearch(parameters, func_args = (split_value1, split_value2, gender, raw_filename, country), func = train_gru, model_name =f'{gender}_{time.time()}_gru_NEW')

    gender = 'both'

    gridSearch(parameters, func_args = (split_value1, split_value2, gender, raw_filename, country), func = train_gru, model_name =f'{gender}_{time.time()}_gru_NEW')

    ###### -------------------------------------------------------------------------------------------------------------------------------------------- ###########

    gender = 'both'
    config = {
        'T' : [(7,3), (9,1)],
        'tau0' : [3, 5],
        'batch_size' : [5],
        'd_model' : [4],
        'n_decoder_layers' : [2],
        'n_encoder_layers' : [2],
        'n_heads' : [1],
        'dropout_encoder' : [0.1, 0.2],
        'dropout_decoder' : [0.1, 0.2],
        'dropout_pos_enc' : [0.1],
        'dim_feedforward_encoder' : [4],
        'dim_feedforward_decoder' : [4],
        'epochs' : [200]
    }
    best_hyperparameters_b, best_evaluation_b = gridSearch(config,  func_args = (split_value1, split_value2, gender, raw_filename, country), model_name =f'{gender}_{time.time()}_transformer')

    gender = 'Male'
    best_hyperparameters_b, best_evaluation_b = gridSearch(config,  func_args = (split_value1, split_value2, gender, raw_filename, country), model_name =f'{gender}_{time.time()}_transformer')

    gender = 'Female'
    best_hyperparameters_b, best_evaluation_b = gridSearch(config,  func_args = (split_value1, split_value2, gender, raw_filename, country), model_name =f'{gender}_{time.time()}_transformer')



    



    