#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:23:31 2022

@author: beatrizlourenco
"""
from hyperparameter_tuning import train_transformer, gridSearch, train_lstm
from ray import tune
from functools import partial
import json

if __name__ == "__main__":
    split_value1 = 1993
    split_value2 = 2006 
    raw_filename = 'Dataset/Mx_1x1_alt_1940_2022.txt'
    country = "PT"

    gender = 'Male'

    parameters = {
        'T': [10],
        'tau0': [5],
        'tau1': [10],
        'tau2': [20],
        'tau3': [5]
    }

    gridSearch(parameters, func_args = (split_value1, split_value2, gender, raw_filename, country), func = train_lstm)

    



    