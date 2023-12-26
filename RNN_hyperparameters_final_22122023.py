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
import ast

if __name__ == "__main__":
    split_value1 = 1993
    split_value2 = 2006
    raw_filename = 'Dataset/Mx_1x1_alt_1940_2022.txt'

    country = "PT"

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

    file = 'hyperparameters/hyperparameter_tuning_Male_1703257618.0138235_transformer_NEW.csv'
    dataframe = pd.read_csv(file)
    dataframe['T'] = dataframe['T'].apply(ast.literal_eval)
    gender = 'Male'
    best_hyperparameters_b, best_evaluation_b = gridSearch(config,  func_args = (split_value1, split_value2, gender, raw_filename, country), model_name =f'{gender}_{time.time()}_transformer_NEW', csv_to_fill = dataframe)

    #gender = 'both'
    #best_hyperparameters_b, best_evaluation_b = gridSearch(config,  func_args = (split_value1, split_value2, gender, raw_filename, country), model_name =f'{gender}_{time.time()}_transformer_NEW')
    
    

    



    