import pandas as pd
import os
from itertools import product

def read_hyperparameters_file(filename):
    df = pd.read_csv(filename)
    hl = df.hyperparameters.to_list()
    hl.append(df.evaluation[0].item())
    return hl

def hyperparameter_to_csv(gender:str, model_name:str, filepath:str = 'results_hyperparameter_opt', column_names = ['T','tau0','units_per_layer','rnn_func','batch_size','epochs','results']):
    prefix = f'hyperparameter_tuning_{gender}_{model_name}'

    # Get a list of all files in the directory
    all_files = os.listdir(filepath)

    # Filter files that start with the specified prefix
    filtered_files = [read_hyperparameters_file(filepath + '\\' + file) for file in all_files if file.startswith(prefix)]

    # Print the list of filtered files
    return pd.DataFrame(filtered_files, columns = column_names)

def add_remaining_hyperparameter_comb(config, df):

    hyperparameter_names = list(config.keys())
    hyperparameter_values = list(config.values())

    hyperparameter_combinations = list(product(*hyperparameter_values))

    hyperparameter_combinations = list(product(*hyperparameter_values))
    parameters_missing = [ parameters for parameters in hyperparameter_combinations if not df.isin([parameters]).all(axis=1).any()]




if __name__ == '__main__':
    hyperparameter_to_csv(gender = 'male', model_name='transformer')



