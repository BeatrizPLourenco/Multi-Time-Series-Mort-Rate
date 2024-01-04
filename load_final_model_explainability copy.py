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
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
    sample = validation_test_data[validation_test_data['Year']>=1998] # 1998+8 = 2006

    if gender == 'both':
        train_data = prt.preprocessing_with_both_gendersLSTM(training_data, T, tau0,xmin, xmax)
        val_data  = prt.preprocessing_with_both_gendersLSTM(validation_data, T, tau0,xmin, xmax)
        test_data = prt.preprocessing_with_both_gendersLSTM(testing_data, T, tau0,xmin, xmax)
        sample_input = prt.preprocessing_with_both_gendersLSTM(validation_test_data, T, tau0,xmin, xmax) #1993+8 =2001

    # Replace 'your_model.h5' with the path to your saved model file
    model_path = 'lstm_model_1703860261.18526.h5'

    
    model = rnn_model(8,5,  [5, 10, 15], LSTM, gender=gender)

    # Load the model
    model.load_weights(model_path)

    # Now, you can use the loaded_model for predictions or further training
    explain_method = 'IntegratedGradients'
    train_data_exp = train_data[:2]
    test_data_exp = test_data[:2]
    val_data_exp = val_data[:2]
    #val_test_data_exp = val_test_data[:2]
    sample_input_exp = sample_input[:2]
    yearmax, yearmin = 2022, 2006
    beg_year, middle_age = 2006, 0
    gender = 'Male'
    gender_idx = 1 if gender == 'Male' else 0
    instance_idx = ((beg_year - yearmin)*99 + middle_age ) * 2 + gender_idx
    instance = [np.squeeze(x) for x in exai.get_instance(sample_input_exp, instance_idx)]

    #instance = exai.get_instance(test_data_exp, instance_idx)
    baseline = [np.zeros_like(input_data) for input_data in instance]

    ig = exai.integrated_gradients(model, baseline_inputs = baseline, inputs = instance)

    ig_mx, ig_gender = ig
    igmat = np.abs(np.transpose(ig_mx))
    filepath = f'{explain_method}_heatmap_plot_{beg_year}_{middle_age}_{gender}.pdf'
    """exai.explain_scores_heatmap(igmat, 
                                beg_year, 
                                middle_age,
                                vmin=0,vmax=np.max(igmat), explain_method ='IntegratedGradients', 
                                cmap = 'hot',
                                filepath = filepath)"""
    

    gender_scores = []
    selected_year = '2006_2022'
    igmat_sum = np.zeros_like(igmat)
    for gender_idx in [0,1]:
        scores = []
        for age in range(0, 99+1):
            ig_mx_per_year = []
            ig_gender_per_year =[]
            for year in range(yearmin, yearmax+1):
                instance_idx = ((year - yearmin)*99 + age ) * 2 + gender_idx
                print(year, age, instance_idx)
                instance = [np.squeeze(x) for x in exai.get_instance(sample_input_exp, instance_idx)]
                ig_mx, ig_gender = exai.integrated_gradients(model, baseline_inputs = baseline, inputs = instance)
                igmat_per_year_age_gender = np.abs(np.transpose(ig_mx))
                igmat_sum = igmat_sum + igmat_per_year_age_gender
                ig_gender_per_year.append(float(ig_gender))
            scores.append(np.array(ig_gender_per_year).mean())
        gender_scores.append(scores)

    #exai.explain_scores_per_age_plot(gender_scores, gender='both', year = selected_year, explain_method = 'Scores' )
    scaler = MinMaxScaler()
    igmat_norm = scaler.fit_transform(igmat)
    filepath = f'{explain_method}_heatmap_plot_{selected_year}.pdf'
    exai.explain_scores_heatmap(igmat_norm, 
                                xticklabels = ['T - 8', 'T - 7', 'T - 6', 'T - 5', 'T - 4', 'T - 3', 'T - 2', 'T - 1'],
                                yticklabels = ['Age - 2', 'Age - 1', 'Age','Age + 1', 'Age + 2'],
                                vmin = 0,vmax = np.max(igmat_norm), explain_method = 'IntegratedGradients', 
                                cmap = 'hot',
                                filepath = filepath, year=selected_year)
    print('end!')
        



    