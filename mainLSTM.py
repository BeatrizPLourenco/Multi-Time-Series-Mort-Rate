#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:23:31 2022

@author: beatrizlourenco
"""
import data_cleaning as dtclean
import preprocessing_transformer as prt
import train_lstm as trl
from scheduler import Scheduler
import train_transformer as trt
import mortalityRateLSTM as mrl
import mortalityRateTransformer as mrt

from torch import nn, optim, zeros
import recursive_forecast as rf
import explainability as xai
import math

if __name__ == "__main__":

    # training
    training_mode = True
    resume_training = False

    # Check for control logic inconsistency 
    if not training_mode:
        assert not resume_training
     
    
    # Control
    country = "PT"
    split_value = 2000
    raw_filenamePT = 'Dataset/Mx_1x1_alt.txt'
    raw_filenameSW = 'Dataset/CHE_mort.xlsx'
    T = 10
    T_encoder = 7
    T_decoder = 3
    tau0 = 5
    split_value1 = 1989
    split_value2 = 2000
    gender = 'both'
    both_gender_model = (gender == 'both')
    checkpoint_dir = f'Saved_models/checkpoint_LSTM_{gender}.pt'
    best_model_dir = f'Saved_models/best_model_LSTM_{gender}.pt'


    # Model hyperparameters  
    input_size = 5
    batch_first = True
    batch_size = 100

    # Define hyperparameters
    input_size = 5
    hidden_size1 = 20
    hidden_size2 = 15
    hidden_size3 = 10


    # Preprocessing
    data = dtclean.get_country_data(country)
    data_logmat = prt.data_to_logmat(data, gender)
    xmin, xmax = prt.min_max_from_dataframe(data_logmat)
    
    # Split Data
    training_data, validation_test_data  = dtclean.split_data(data, split_value1)
    validation_data, testing_data  = dtclean.split_data(validation_test_data, split_value2)    
    
    # preprocessing for the transformer
    if gender == 'both':
        train_data = prt.preprocessing_with_both_gendersLSTM(training_data, T, tau0, xmin, xmax)
        val_data  = prt.preprocessing_with_both_gendersLSTM(validation_data,  T, tau0, xmin, xmax)
        test_data = prt.preprocessing_with_both_gendersLSTM(testing_data,  T, tau0, xmin, xmax)


    elif gender == 'Male' or gender == 'Female' :
        train_data = prt.preprocessed_dataLSTM( training_data,  gender, T,tau0, xmin, xmax)
        val_data = prt.preprocessed_dataLSTM( validation_data, gender, T,tau0, xmin, xmax)
        test_data = prt.preprocessed_dataLSTM(testing_data, gender,  T,tau0, xmin, xmax)


    train_data, val_data, test_data = prt.from_numpy_to_torch(train_data), prt.from_numpy_to_torch(val_data), prt.from_numpy_to_torch(test_data)

    # Initializing model object
    model = mrl.MortalityRateLSTM(input_size, hidden_size1, hidden_size2, hidden_size3)
    best_model = mrl.MortalityRateLSTM(input_size, hidden_size1, hidden_size2, hidden_size3)

    # Training hyperparameters
    criterion = nn.MSELoss()
    #opt = optim.SGD(model.parameters(), lr = 0.05)
    opt = optim.Adam(model.parameters(), lr=0.001, betas = (0.9,0.999), eps =10**(-7))
    scheduler = optim.lr_scheduler.StepLR(opt, step_size = 500, gamma = 0.99)
    scheduler = None
    epochs = 600
    first_year, last_year = 2000, 2020

    # Training
    if training_mode == True:
        best_model, history = trl.fit(
            model = model,
            best_model = best_model,
            batch_size = batch_size,
            epochs = epochs,
            train_data = train_data,
            val_data = val_data,
            opt = opt, 
            criterion = criterion, 
            scheduler = scheduler,
            resume_training = resume_training,
            checkpoint_dir= checkpoint_dir,
            best_model_dir= best_model_dir
        )
    else:
        best_model, history = trt.load_best_model(model, best_model_dir = best_model_dir)
        
    
    forecast_dataframe = rf.recursive_forecast_both_genders(data, first_year,last_year, T, tau0, xmin, xmax, best_model)
    best_model.eval()
    recursive_prediction_loss_male, recursive_prediction_loss_female = rf.loss_recursive_forecasting(testing_data, forecast_dataframe, gender_model = gender)

    trt.save_plots(history['train_loss_history'], history['val_loss_history'], gender = gender)

    # Evaluation
    test_loss = trl.evaluate(best_model, batch_size,test_data ,criterion)
    val_loss = trl.evaluate(best_model, batch_size,val_data ,criterion)
    train_loss = trl.evaluate(best_model, batch_size,train_data, criterion)


print('=' * 100)
print('| End of training | training loss {:5.2f} | validation loss {:5.2f} | test loss {:5.2f} | test ppl {:8.2f}'.format(
    train_loss, val_loss, test_loss, math.exp(test_loss)))
print('-' * 100)

text_to_print = '| Evaluating 20 years of recursive data'
if recursive_prediction_loss_male is not None: text_to_print = text_to_print + '| Male loss {:5.2f}'.format(recursive_prediction_loss_male)
if recursive_prediction_loss_female is not None: text_to_print = text_to_print + '| female loss {:5.2f}'.format(recursive_prediction_loss_female)
print(text_to_print)

print('=' * 100)
    
"""
# Explain Model
unbatch_input = prt.unbatchify(test_data)
exp_model = xai.ExplainableMortalityRateTransformer('both gender Transformer', best_model)

sample_index = 1
dim_to_explain = 1
sample = prt.get_pattern(unbatch_input, sample_index)[:-1]
exp_model.explain(sample, batched_input = False, dim_to_explain = dim_to_explain)
print()

"""


    