#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:23:31 2022

@author: beatrizlourenco
"""
from hyperparameter_tuning import train_transformer, gridSearch
from ray import tune
from functools import partial
import json



if __name__ == "__main__":
    split_value1 = 1993
    split_value2 = 2006 
    raw_filename = 'Dataset/Mx_1x1_alt_1940_2022.txt'
    country = "PT"

    gender = 'Male'
    config = {
        'T' : [(7,3), (9,1)],
        'tau0' : [3],
        'batch_size' : [5],
        'd_model' : [4],
        'n_decoder_layers' : [1],
        'n_encoder_layers' : [1],
        'n_heads' : [1],
        'dropout_encoder' : [0.2],
        'dropout_decoder' : [0.2],
        'dropout_pos_enc' : [0.2],
        'dim_feedforward_encoder' : [4],
        'dim_feedforward_decoder' : [4],
        'epochs' : [1]
    }
    best_hyperparameters_m, best_evaluation_m = gridSearch(config, 
                                                       func_args = (split_value1, split_value2, gender, raw_filename, country))

    with open('Dataset/hyperparameter_tuning_male_transformer.txt', 'w') as convert_file: 
        convert_file.write(json.dumps({'best_hyperparameters_m': best_hyperparameters_m, 
                                       'best_evaluation_m': best_evaluation_m}))

    gender = 'Female'
    config = {
        'T' : [(7,3), (9,1)],
        'tau0' : [3, 5, 7],
        'batch_size' : [5],
        'd_model' : [4],
        'n_decoder_layers' : [1],
        'n_encoder_layers' : [1],
        'n_heads' : [1],
        'dropout_encoder' : [0.2],
        'dropout_decoder' : [0.2],
        'dropout_pos_enc' : [0.2],
        'dim_feedforward_encoder' : [4],
        'dim_feedforward_decoder' : [4],
        'epochs' : [1]
    }
    best_hyperparameters_f, best_evaluation_f = gridSearch(config, 
                                                       func_args = (split_value1, split_value2, gender, raw_filename, country))
    with open('Dataset/hyperparameter_tuning_female_transformer.txt', 'w') as convert_file: 
        convert_file.write(json.dumps({'best_hyperparameters_f': best_hyperparameters_f, 
                                       'best_evaluation_f': best_evaluation_f}))

    
    gender = 'both'
    config = {
        'T' : [(7,3), (9,1)],
        'tau0' : [3, 5, 7],
        'batch_size' : [5],
        'd_model' : [4],
        'n_decoder_layers' : [1],
        'n_encoder_layers' : [1],
        'n_heads' : [1],
        'dropout_encoder' : [0.2, 0.1],
        'dropout_decoder' : [0.2, 0.1],
        'dropout_pos_enc' : [0.2, 0.1],
        'dim_feedforward_encoder' : [4, 16],
        'dim_feedforward_decoder' : [4, 16],
        'epochs' : [1]
    }
    best_hyperparameters_b, best_evaluation_b = gridSearch(config, 
                                                       func_args = (split_value1, split_value2, gender, raw_filename, country))
    
    with open('Dataset/hyperparameter_tuning_both_gender_transformer.txt', 'w') as convert_file: 
        convert_file.write(json.dumps({'best_hyperparameters_b': best_hyperparameters_b, 
                                    'best_evaluation_b': best_evaluation_b}))



    