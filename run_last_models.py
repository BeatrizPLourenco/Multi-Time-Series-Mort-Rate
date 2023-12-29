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
    gender = 'both'

    """parameters_gru = {
    'T': 10,
    'tau0': 5,
    'units_per_layer': [5, 10, 15],
    'gru_func': GRU,
    'batch_size': 50,
    'epochs': 500
    }

    train_gru(parameters_gru, 
                      split_value1 = 1993, 
                      split_value2 = 2006,
                      gender = 'both',
                      raw_filename = 'Dataset/Mx_1x1_alt.txt',
                      country = "PT", 
                      seed = 0)"""
    
    parameters_lstm = {
        'T': 8,
        'tau0': 5,
        'units_per_layer': [5, 10, 15],
        'rnn_func': LSTM,
        'batch_size': 50,
        'epochs': 500
    }

    train_rnn(parameters_lstm, 
                      split_value1 = 2006, 
                      split_value2 = None,
                      split_value2 = None,
                      gender = 'both',
                      raw_filename = raw_filename,
                      country = "PT", 
                      seed = 0)
    
    """config_tf = {
        'T' : (9,1),
        'tau0' : 3,
        'batch_size' : 10,
        'd_model' : 4,
        'n_decoder_layers' : 2,
        'n_encoder_layers' : 2,
        'n_heads' : 1,
        'dropout_encoder' : 0.2,
        'dropout_decoder' : 0.2,
        'dropout_pos_enc' : 0.1,
        'dim_feedforward_encoder' : 4,
        'dim_feedforward_decoder' : 4,
        'epochs' : 200
    }

    train_transformer(config_tf, 
                      split_value1 = 1993, 
                      split_value2 = 2006,
                      gender = 'both',
                      raw_filename = 'Dataset/Mx_1x1_alt.txt',
                      country = "PT", 
                      seed = 0)"""

    



    