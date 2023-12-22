from hyperparameter_tuning import train_transformer, gridSearch, train_rnn, train_gru
#from ray import tune
from functools import partial
from keras.layers import Dense, LSTM, Flatten, Concatenate, Bidirectional, GRU
import json
import time
import pandas as pd


parameters_gru = {
    'T': [8, 10],
    'tau0': [3, 5],
    'units_per_layer': [[5, 10, 15], [10, 15, 20], [10, 15]],
    'gru_func': [GRU],
    'batch_size': [5, 50],
    'epochs': [1]
}

parameters_lstm = {
        'T': [8, 10],
        'tau0': [3, 5],
        'units_per_layer': [[5, 10, 15], [10, 15, 20], [10, 15]],
    #    'rnn_func': [LSTM],
        'batch_size': [5, 50],
        'epochs': [500]
    }

config_tf = {
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
        'epochs' : [200]
    }
from itertools import product
parameters = config_tf
file = f'golden/hyperparameter_tuning_transformer_female_final_21122023.csv'
hyperparameter_names = list(parameters.keys())
hyperparameter_values = list(parameters.values())
hyperparameter_combinations = list(product(*hyperparameter_values))
initial_idx = 0
dataframe = pd.DataFrame(hyperparameter_combinations, columns = hyperparameter_names)
dataframe['T'] = dataframe['T'].apply(str)

#dataframe['results'] = None

#scv_results = pd.read_csv('hyperparameters/hyperparameter_tuning_both_1703088500.0224714_rnn.csv')

#scv_results = pd.read_csv('hyperparameters/hyperparameter_tuning_both_1703088500.0224714_rnn.csv')

col = ['T','batch_size', 'd_model','dim_feedforward_encoder', 'dim_feedforward_decoder','dropout_encoder', 'dropout_decoder',
       'dropout_pos_enc','epochs', 'n_decoder_layers',
       'n_encoder_layers', 'n_heads', 'tau0', 'results']
scv_results = pd.read_csv('hyperparameters/hyperparameter_tuning_female_transformer.csv', names =col, index_col=None)
scv_results['T'] = scv_results['T'].apply(str)


merge = dataframe.merge(scv_results, how='left')

merge.to_csv(file, index=False)