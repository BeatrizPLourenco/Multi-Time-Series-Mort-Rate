#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:23:31 2022

@author: beatrizlourenco
"""

from hyperparameter_tuning import train_transformer, gridSearch, train_rnn
#from ray import tune
from functools import partial
from keras.layers import Dense, LSTM, Flatten, Concatenate, Bidirectional, GRU
import json
import time

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

    gridSearch(parameters, func_args = (split_value1, split_value2, gender, raw_filename, country), func = train_rnn, model_name =f'{gender}_{time.time()}_rnn')

    gender = 'Male'

    parameters = {
        'T': [8, 10, 12],
        'tau0': [3, 5, 7],
        'units_per_layer': [[5, 10, 15], [10, 15, 20], [10, 15], [15, 20]],
        'rnn_func': [LSTM],
        'batch_size': [5, 50, 100],
        'epochs': [500]
    }

    gridSearch(parameters, func_args = (split_value1, split_value2, gender, raw_filename, country), func = train_rnn, model_name =f'{gender}_{time.time()}_rnn')

    gender = 'both'

    parameters = {
        'T': [8, 10, 12],
        'tau0': [3, 5, 7],
        'units_per_layer': [[5, 10, 15], [10, 15, 20], [10, 15], [15, 20]],
        'rnn_func': [LSTM],
        'batch_size': [5, 50, 100],
        'epochs': [500]
    }

    gridSearch(parameters, func_args = (split_value1, split_value2, gender, raw_filename, country), func = train_rnn, model_name =f'{gender}_{time.time()}_rnn')

    



    