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


    # Now, you can use the loaded_model for predictions or further training
    gender = 'Male'
    train_val_data = pd.concat([training_data, validation_data], axis = 0)
    year_range_train = train_val_data['Year'].unique()
    year_range_val = testing_data['Year'].unique()
    leecarter = lc.LeeCarter(gender)
    leecarter.fit(train_val_data)
    kt_forecast = leecarter.map_mx_to_kt(recursive_prediction)
    prediction = leecarter.predict(num_time_steps = len(year_range_val), return_only_prediction = True)
    pred_mx = prediction['mx']
    LC_new_kt = prediction['kt_forecast']
    validation_test_kt = leecarter.map_mx_to_kt(data = validation_test_data)
    train_prediction = leecarter.predict(num_time_steps = 0, return_only_prediction = False)
    train_pred_mx = train_prediction['mx']

    kt_forecasts = { ('r', 'LSTM') : kt_forecast , ('y', 'Lee Carter') : LC_new_kt}
    #kt_forecasts = { ('r', 'LSTM') : kt_forecast }
    

    lc.viz_kt_graph(leecarter.kt, kt_forecasts, x_axis_train_values = year_range_train, x_axis_val_values = year_range_val, path_to_save_fig = f'lstm_leecarter_{gender}_{country}_' + str(datetime.now()) + '.pdf')



    



    