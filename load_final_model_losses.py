#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:23:31 2022

@author: beatrizlourenco
"""
from keras.models import load_model
from keras.layers import Dense, LSTM, Flatten, Concatenate, Bidirectional, GRU
from LSTM_Keras import rnn_model
import data_cleaning as dtclean
import preprocessing_transformer as prt
import train_transformer as trt
import mortalityRateTransformer as mrt
import recursive_forecast as rf
import LeeCarter as lc
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == "__main__":
    split_value1 = 1993
    split_value2 = 2006
    raw_filename = 'Dataset/Mx_1x1_alt_1940_2022.txt'
    #raw_filename = 'Dataset/Mx_1x1_alt.txt'
    gender = 'both'
    country = "PT"

    # Preprocessing
    data = dtclean.get_country_data(country, filedir = raw_filename)
    data_logmat = prt.data_to_logmat(data, gender)
    xmin, xmax = prt.min_max_from_dataframe(data_logmat)
    
    # Split Data
    training_data, validation_test_data  = dtclean.split_data(data, split_value1)
    validation_data, testing_data  = dtclean.split_data(validation_test_data, split_value2) 

    country = "PT"
    gender = 'both'

    

    # Replace 'your_model.h5' with the path to your saved model file
    model_path = 'lstm_model_1703860261.18526.h5'

    parameters_lstm = {
        'T': 8,
        'tau0': 5,
        'units_per_layer': [5, 10, 15],
        'batch_size': 50,
        'epochs': 500
    }
    T = parameters_lstm['T']
    tau0 = parameters_lstm['tau0']
    units_per_layer = parameters_lstm['units_per_layer']
    rnn_func = GRU
    batch_size = parameters_lstm['batch_size']
    epochs = parameters_lstm['epochs']

    model = rnn_model(8, 5,  [5, 10, 15], LSTM, gender=gender)

    # Load the model
    model.load_weights(model_path)

    first_year, last_year = 2006, 2022 
    recursive_prediction = rf.recursive_forecast(data, first_year, last_year, T, tau0, xmin, xmax, model, batch_size=1, model_type = 'lstm', gender = gender)
    recursive_prediction_loss_male, recursive_prediction_loss_female = rf.loss_recursive_forecasting(testing_data, recursive_prediction, gender_model = gender)

    gender = 'Female'
    real_test = pd.merge(testing_data, recursive_prediction, on = ['Year', 'Age', 'Gender'])
    real_test = real_test[real_test.Gender == gender]
    real_test['losses_mx'] = (real_test['mx_x'] - real_test['mx_y']).apply(lambda x: x**2)
    real_test['losses_logmx'] = (real_test['logmx_x'] - real_test['logmx_y']).apply(lambda x: x**2)
    

    mortality_var = 'losses_mx'
    agg_level = 'Year'
    
    real_test_by_year = real_test.groupby(by = [agg_level]).sum().reset_index()
    axis =real_test_by_year.plot(figsize = (10,10),x=agg_level, y= mortality_var, legend = False, kind = 'bar', fontsize=18)
    path_to_save_fig = f'losses_{gender}_{agg_level}_{mortality_var}' + str(datetime.now()) + '.pdf'
    axis.set_ylabel(ylabel='Losses', fontdict={'fontsize': 20, 'fontweight':'bold'})
    axis.set_xlabel(xlabel=agg_level, fontdict={'fontsize': 20, 'fontweight':'bold'})
    plt.savefig(path_to_save_fig)

    gender = 'Male'
    real_test = pd.merge(testing_data, recursive_prediction, on = ['Year', 'Age', 'Gender'])
    real_test = real_test[real_test.Gender == gender]
    real_test['losses_mx'] = (real_test['mx_x'] - real_test['mx_y']).apply(lambda x: x**2)
    real_test['losses_logmx'] = (real_test['logmx_x'] - real_test['logmx_y']).apply(lambda x: x**2)
    

    mortality_var = 'losses_mx'
    agg_level = 'Year'
    
    real_test_by_year = real_test.groupby(by = [agg_level]).sum().reset_index()
    axis = real_test_by_year.plot(figsize = (10,10), x=agg_level, y= mortality_var, legend = False, kind = 'bar',  fontsize=18)
    path_to_save_fig = f'losses_{gender}_{agg_level}_{mortality_var}' + str(datetime.now()) + '.pdf'
    axis.set_ylabel(ylabel='Losses', fontdict={'fontsize': 20, 'fontweight':'bold'})
    axis.set_xlabel(xlabel=agg_level, fontdict={'fontsize': 20, 'fontweight':'bold'})
    plt.savefig(path_to_save_fig)






    

    



    