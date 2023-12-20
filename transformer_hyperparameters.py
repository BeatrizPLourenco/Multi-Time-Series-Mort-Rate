#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:23:31 2022

@author: beatrizlourenco
"""
from hyperparameter_tuning import train_transformer, gridSearch
from functools import partial
import json
from _utils import hyperparameter_to_csv

if __name__ == "__main__":
    split_value1 = 1993
    split_value2 = 2006 
    raw_filename = 'Dataset/Mx_1x1_alt_1940_2022.txt'
    country = "PT"

    gender = 'Male'
    config = {
        'T' : [(7,3), (9,1), (7,1)],
        'tau0' : [3, 5],
        'batch_size' : [5],
        'd_model' : [4],
        'n_decoder_layers' : [1, 2],
        'n_encoder_layers' : [1, 2],
        'n_heads' : [1],
        'dropout_encoder' : [0.1, 0.2],
        'dropout_decoder' : [0.1, 0.2],
        'dropout_pos_enc' : [0.1],
        'dim_feedforward_encoder' : [4],
        'dim_feedforward_decoder' : [4],
        'epochs' : [200]
    }

    #best_hyperparameters_m, best_evaluation_m = gridSearch(config, 
    #                                                   func_args = (split_value1, split_value2, gender, raw_filename, country), model_name ='male_transformer')

    #df = hyperparameter_to_csv('male', 'transformer', column_names =  list(config.keys()) + ['results'])
    #df.to_csv('hyperparameter_tuning_male_transformer.csv', index=False)


    gender = 'Female'
    config = {
        'T' : [(7,3), (9,1), (7,1)],
        'tau0' : [3, 5],
        'batch_size' : [5],
        'd_model' : [4],
        'n_decoder_layers' : [1, 2],
        'n_encoder_layers' : [1, 2],
        'n_heads' : [1],
        'dropout_encoder' : [0.1, 0.2],
        'dropout_decoder' : [0.1, 0.2],
        'dropout_pos_enc' : [0.1],
        'dim_feedforward_encoder' : [4],
        'dim_feedforward_decoder' : [4],
        'epochs' : [200]
    }
    #best_hyperparameters_f, best_evaluation_f = gridSearch(config, 
    #                                                   func_args = (split_value1, split_value2, gender, raw_filename, country), model_name ='female_transformer')


    
    gender = 'both'
    config = {
        'T' : [(7,3), (9,1), (7,1)],
        'tau0' : [3, 5],
        'batch_size' : [5],
        'd_model' : [4],
        'n_decoder_layers' : [1, 2],
        'n_encoder_layers' : [1, 2],
        'n_heads' : [1],
        'dropout_encoder' : [0.1, 0.2],
        'dropout_decoder' : [0.1, 0.2],
        'dropout_pos_enc' : [0.1],
        'dim_feedforward_encoder' : [4],
        'dim_feedforward_decoder' : [4],
        'epochs' : [200]
    }
    #best_hyperparameters_b, best_evaluation_b = gridSearch(config,  func_args = (split_value1, split_value2, gender, raw_filename, country), model_name ='both_transformer')




    