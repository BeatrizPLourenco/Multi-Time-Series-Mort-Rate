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
import explainability_lstm as exai
import shap


if __name__ == "__main__":
    split_value1 = 1993
    split_value2 = 2006
    raw_filename = 'Dataset/Mx_1x1_alt_1940_2022.txt'
    #raw_filename = 'Dataset/Mx_1x1_alt.txt'
    gender = 'both'
    country = "PT"

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

    # Preprocessing
    data = dtclean.get_country_data(country, filedir = raw_filename)
    data_logmat = prt.data_to_logmat(data, gender)
    xmin, xmax = prt.min_max_from_dataframe(data_logmat)

    training_data, validation_test_data  = dtclean.split_data(data, split_value1)
    validation_data, testing_data  = dtclean.split_data(validation_test_data, split_value2) 

    if gender == 'both':
        train_data = prt.preprocessing_with_both_gendersLSTM(training_data, T, tau0,xmin, xmax)
        val_data  = prt.preprocessing_with_both_gendersLSTM(validation_data, T, tau0,xmin, xmax)
        test_data = prt.preprocessing_with_both_gendersLSTM(testing_data, T, tau0,xmin, xmax)

    # Replace 'your_model.h5' with the path to your saved model file
    model_path = 'lstm_model_1703860261.18526.h5'

    
    model = rnn_model(8,5,  [5, 10, 15], LSTM, gender=gender)

    # Load the model
    model.load_weights(model_path)

    # Now, you can use the loaded_model for predictions or further training
    yearmin, yearmax = 2006, 2022
    beg_year, middle_age = 2010, 13
    gender = 'Male'
    gender_idx = 1 if gender == 'Male' else 0
    instance_idx = ((beg_year - yearmin)*99 + middle_age ) * 2 + gender_idx
    
    feature_size = 5*10
    train_data_exp = train_data[:2]
    test_data_exp = test_data[:2]

    instance = exai.get_instance(test_data_exp, instance_idx)
    reshaped_instance_with_gender = exai.flat_input(instance)
    reshaped_train_with_gender = exai.flat_input(train_data_exp)
    reshaped_test_with_gender = exai.flat_input(test_data_exp)

    modelwrapped = exai.ModelWrapper(model)

    explainer = shap.KernelExplainer(modelwrapped.forward, shap.sample(reshaped_train_with_gender, 10))

    shap_values = explainer.shap_values(reshaped_instance_with_gender)

    filepath = f'shap_bar_plot_{beg_year}_{middle_age}_{gender}.pdf'
    """exai.shap_barplot(shap_values, 
                      reshaped_instance_with_gender, 
                      filepath = filepath, 
                      feature_names = exai.feature_names(beg_year, middle_age,gender))"""


    shap.force_plot(explainer.expected_value, 
                    shap_values, 
                    reshaped_test_with_gender, 
                    matplotlib=True, 
                    feature_names = exai.feature_names(beg_year, middle_age,gender), show = False)

    

    



    