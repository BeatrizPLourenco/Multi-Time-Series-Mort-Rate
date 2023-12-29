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

    from keras.models import load_model

    # Replace 'your_model.h5' with the path to your saved model file
    model_path = 'your_model.h5'

    # Load the model
    loaded_model = load_model(model_path)

    # Now, you can use the loaded_model for predictions or further training

    



    