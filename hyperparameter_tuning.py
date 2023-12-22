import torch
from sklearn.base import BaseEstimator
import mortalityRateTransformer as mrt
import train_transformer as trt
import data_cleaning as dtclean
import preprocessing_transformer as prt
import train_transformer as trt
from scheduler import Scheduler
import mortalityRateTransformer as mrt
from torch import nn, optim, zeros
import recursive_forecast as rf
import explainability as xai
import math
from sklearn.model_selection import GridSearchCV
from itertools import product
import random
import numpy as np
from LSTM_Keras import rnn_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, LSTM, Flatten, Concatenate, Bidirectional, GRU
import pandas as pd
import time
from datetime import datetime
import ast

seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)



def flatten_tensors(tensor_list):
    flattened_tensors = [tensor.view(tensor.size(0),-1) for tensor in tensor_list if tensor is not None]
    flattened_tensor = torch.cat(flattened_tensors, 1)
    return flattened_tensor

def unflatten_tensors(flattened_tensor, original_shapes):
    unflattened_tensors = []
    current_index = 0

    for shape in original_shapes:
        size = shape[0] * shape[1]
        unflattened_tensor = flattened_tensor[:, current_index:current_index + size].view(*shape)
        unflattened_tensors.append(unflattened_tensor)
        current_index += size

    return unflattened_tensors

def get_original_shapes(tensor_list):
    return [tensor.shape if tensor is not None else None for tensor in tensor_list]

def train_gru(parameters : dict, 
                      split_value1 = 1993, 
                      split_value2 = 2006,
                      gender = 'both',
                      raw_filename = 'Dataset/Mx_1x1_alt.txt',
                      country = "PT", 
                      seed = 0):
    
    # Preprocessing
    data = dtclean.get_country_data(country, filedir = raw_filename)
    data_logmat = prt.data_to_logmat(data, gender)
    xmin, xmax = prt.min_max_from_dataframe(data_logmat)
    
    # Split Data
    training_data, validation_test_data  = dtclean.split_data(data, split_value1)
    validation_data, testing_data  = dtclean.split_data(validation_test_data, split_value2) 
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Control
    #split_value = 2000
    split_value1 = split_value1 # 1993 a 2005 corresponde a 13 anos (13/66 approx. 20%)
    split_value2 = split_value2 # 2006 a 2022 corresponde a 17 anos (17/83 approx. 20%)
    gender = gender
    both_gender_model = (gender == 'both')
    checkpoint_dir = f'Saved_models/checkpoint_{gender}.pt'
    best_model_dir = f'Saved_models/best_model_{gender}.pt'


    # Model hyperparameters  
    T = parameters['T']
    tau0 = parameters['tau0']
    units_per_layer = parameters['units_per_layer']
    rnn_func = GRU
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']

    if gender == 'both':
        train_data = prt.preprocessing_with_both_gendersLSTM(training_data, T, tau0,xmin, xmax)
        val_data  = prt.preprocessing_with_both_gendersLSTM(validation_data, T, tau0,xmin, xmax)
        test_data = prt.preprocessing_with_both_gendersLSTM(testing_data, T, tau0,xmin, xmax)


    elif gender == 'Male' or gender == 'Female' :
        train_data = prt.preprocessed_dataLSTM( training_data,  gender, T, tau0,xmin, xmax)
        val_data = prt.preprocessed_dataLSTM( validation_data, gender, T, tau0,xmin, xmax)
        test_data = prt.preprocessed_dataLSTM(testing_data, gender, T, tau0,xmin, xmax)


    model = rnn_model(T,tau0,  units_per_layer, rnn_func, gender=gender)
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f")[:-3]
    filepath = "{epoch:02d}-{val_loss:.2f}.weights.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only = True)
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50) #addition of early stopping
    callbacks_list = [checkpoint,earlystop]
    if both_gender_model:
        model.fit(x = train_data[:2], y = train_data[2], validation_split=0.2, epochs = epochs, batch_size = batch_size, verbose = 0, callbacks=callbacks_list) # Model checkpoint CallBack
    else:
        model.fit(x = train_data[:1], y = train_data[2], validation_split=0.2, epochs = epochs, batch_size = batch_size, verbose = 0, callbacks=callbacks_list) # Model checkpoint CallBack

    first_year, last_year = split_value1, split_value2 -1
    recursive_prediction = rf.recursive_forecast(data, first_year,last_year, T, tau0, xmin, xmax, model, batch_size=1, model_type = 'lstm', gender = gender)
    recursive_prediction_loss_male, recursive_prediction_loss_female = rf.loss_recursive_forecasting(validation_data, recursive_prediction, gender_model = gender)

    if gender == 'both':
        return (recursive_prediction_loss_male + recursive_prediction_loss_female)/2
    
    elif gender == 'Male':
        return recursive_prediction_loss_male

    elif gender == 'Female':
        return recursive_prediction_loss_female

def train_rnn(parameters : dict, 
                      split_value1 = 1993, 
                      split_value2 = 2006,
                      gender = 'both',
                      raw_filename = 'Dataset/Mx_1x1_alt.txt',
                      country = "PT", 
                      seed = 0):
    
    # Preprocessing
    data = dtclean.get_country_data(country, filedir = raw_filename)
    data_logmat = prt.data_to_logmat(data, gender)
    xmin, xmax = prt.min_max_from_dataframe(data_logmat)
    
    # Split Data
    training_data, validation_test_data  = dtclean.split_data(data, split_value1)
    validation_data, testing_data  = dtclean.split_data(validation_test_data, split_value2) 
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Control
    #split_value = 2000
    split_value1 = split_value1 # 1993 a 2005 corresponde a 13 anos (13/66 approx. 20%)
    split_value2 = split_value2 # 2006 a 2022 corresponde a 17 anos (17/83 approx. 20%)
    gender = gender
    both_gender_model = (gender == 'both')
    checkpoint_dir = f'Saved_models/checkpoint_{gender}.pt'
    best_model_dir = f'Saved_models/best_model_{gender}.pt'


    # Model hyperparameters  
    T = parameters['T']
    tau0 = parameters['tau0']
    units_per_layer = parameters['units_per_layer']
    rnn_func = LSTM
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']

    if gender == 'both':
        train_data = prt.preprocessing_with_both_gendersLSTM(training_data, T, tau0,xmin, xmax)
        val_data  = prt.preprocessing_with_both_gendersLSTM(validation_data, T, tau0,xmin, xmax)
        test_data = prt.preprocessing_with_both_gendersLSTM(testing_data, T, tau0,xmin, xmax)


    elif gender == 'Male' or gender == 'Female' :
        train_data = prt.preprocessed_dataLSTM( training_data,  gender, T, tau0,xmin, xmax)
        val_data = prt.preprocessed_dataLSTM( validation_data, gender, T, tau0,xmin, xmax)
        test_data = prt.preprocessed_dataLSTM(testing_data, gender, T, tau0,xmin, xmax)


    model = rnn_model(T,tau0,  units_per_layer, rnn_func, gender=gender)
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f")[:-3]
    filepath = "{epoch:02d}-{val_loss:.2f}.weights.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only = True)
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50) #addition of early stopping
    callbacks_list = [checkpoint,earlystop]
    if both_gender_model:
        model.fit(x = train_data[:2], y = train_data[2], validation_split=0.2, epochs = epochs, batch_size = batch_size, verbose = 0, callbacks=callbacks_list) # Model checkpoint CallBack
    else:
        model.fit(x = train_data[:1], y = train_data[2], validation_split=0.2, epochs = epochs, batch_size = batch_size, verbose = 0, callbacks=callbacks_list) # Model checkpoint CallBack

    first_year, last_year = split_value1, split_value2 -1
    recursive_prediction = rf.recursive_forecast(data, first_year,last_year, T, tau0, xmin, xmax, model, batch_size=1, model_type = 'lstm', gender = gender)
    recursive_prediction_loss_male, recursive_prediction_loss_female = rf.loss_recursive_forecasting(validation_data, recursive_prediction, gender_model = gender)

    if gender == 'both':
        return (recursive_prediction_loss_male + recursive_prediction_loss_female)/2
    
    elif gender == 'Male':
        return recursive_prediction_loss_male

    elif gender == 'Female':
        return recursive_prediction_loss_female




def train_transformer(parameters : dict, 
                      split_value1 = 1993, 
                      split_value2 = 2006,
                      gender = 'both',
                      raw_filename = 'Dataset/Mx_1x1_alt.txt',
                      country = "PT", 
                      seed = 0):
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # training
    training_mode = True
    resume_training = False

    # Check for control logic inconsistency 
    if not training_mode:
        assert not resume_training
     
    
    # Control
    #split_value = 2000
    T_ = parameters['T']
    T_encoder = T_[0]
    T_decoder = T_[1]
    T = T_encoder + T_decoder
    tau0 = parameters['tau0']
    split_value1 = split_value1 # 1993 a 2005 corresponde a 13 anos (13/66 approx. 20%)
    split_value2 = split_value2 # 2006 a 2022 corresponde a 17 anos (17/83 approx. 20%)
    gender = gender
    both_gender_model = (gender == 'both')
    checkpoint_dir = f'Saved_models/checkpoint_{gender}.pt'
    best_model_dir = f'Saved_models/best_model_{gender}.pt'


    # Model hyperparameters  
    input_size = tau0
    batch_first = True  
    batch_size = parameters['batch_size']  
    epochs = parameters['epochs']  
    d_model = parameters['d_model']   
    n_decoder_layers = parameters['n_decoder_layers']    
    n_encoder_layers = parameters['n_encoder_layers']  
    n_heads = parameters['n_heads']  
    dropout_encoder = parameters['dropout_encoder'] 
    dropout_decoder = parameters['dropout_decoder'] 
    dropout_pos_enc = parameters['dropout_pos_enc'] 
    dim_feedforward_encoder = parameters['dim_feedforward_encoder'] 
    dim_feedforward_decoder = parameters['dim_feedforward_decoder']
    num_predicted_features = 1


    # Preprocessing
    data = dtclean.get_country_data(country, filedir = raw_filename)
    data_logmat = prt.data_to_logmat(data, gender)
    xmin, xmax = prt.min_max_from_dataframe(data_logmat)
    
    # Split Data
    training_data, validation_test_data  = dtclean.split_data(data, split_value1)
    validation_data, testing_data  = dtclean.split_data(validation_test_data, split_value2)    
    
    # preprocessing for the transformer
    if gender == 'both':
        train_data = prt.preprocessing_with_both_genders(training_data, (T_encoder, T_decoder), tau0, xmin, xmax, batch_size)
        val_data  = prt.preprocessing_with_both_genders(validation_data,  (T_encoder, T_decoder), tau0, xmin, xmax, batch_size)
        test_data = prt.preprocessing_with_both_genders(testing_data,  (T_encoder, T_decoder), tau0, xmin, xmax, batch_size)


    elif gender == 'Male' or gender == 'Female' :
        train_data = prt.preprocessed_data( training_data,  gender, (T_encoder, T_decoder),tau0, xmin, xmax, batch_size)
        val_data = prt.preprocessed_data( validation_data, gender, (T_encoder, T_decoder),tau0, xmin, xmax, batch_size)
        test_data = prt.preprocessed_data(testing_data, gender,  (T_encoder, T_decoder),tau0, xmin, xmax, batch_size)


    train_data, val_data, test_data = prt.from_numpy_to_torch(train_data), prt.from_numpy_to_torch(val_data), prt.from_numpy_to_torch(test_data)

    # Initializing model object
    model = mrt.MortalityRateTransformer(
        input_size = input_size,
        batch_first = batch_first,
        d_model = d_model,
        n_decoder_layers = n_decoder_layers,
        n_encoder_layers = n_encoder_layers,
        n_heads = n_heads,
        T_encoder = T_encoder,
        T_decoder = T_decoder,
        dropout_encoder = dropout_encoder,
        dropout_decoder = dropout_decoder,
        dropout_pos_enc = dropout_pos_enc,
        dim_feedforward_encoder = dim_feedforward_encoder,
        dim_feedforward_decoder = dim_feedforward_decoder,
        num_predicted_features = num_predicted_features,
        both_gender_model = both_gender_model
    )
    
    # Masking the encoder output and decoder input for decoder:
    tgt_mask = mrt.generate_square_subsequent_mask(
        dim1 = T_decoder,
        dim2 = T_decoder
    )
    
    xe_mask = mrt.generate_square_subsequent_mask(
        dim1 = T_decoder,
        dim2 = T_encoder
    )

    # Training hyperparameters
    criterion = nn.MSELoss()
    #opt = optim.SGD(model.parameters(), lr = 0.05)
    opt = optim.Adam(model.parameters(), lr=0.001, betas = (0.9,0.98), eps =10**(-9))
    #scheduler = optim.lr_scheduler.StepLR(opt, step_size = 500, gamma = 0.99)
    scheduler = Scheduler(opt, dim_embed = d_model, warmup_steps = 4000)
    

    # Training
    if training_mode == True:
        best_model, history = trt.fit(
            model = model,
            batch_size = batch_size,
            epochs = epochs,
            train_data = train_data,
            val_data = val_data,
            xe_mask = xe_mask,
            tgt_mask = tgt_mask,
            opt = opt, 
            criterion = criterion, 
            scheduler = scheduler,
            resume_training = resume_training,
            checkpoint_dir= checkpoint_dir,
            best_model_dir= best_model_dir,
            verbose = 0,
            patience=5
        )
    else:
        best_model, history = trt.load_best_model(model, best_model_dir= best_model_dir)
        
    
    first_year, last_year = split_value1, split_value2 -1
    recursive_prediction = rf.recursive_forecast(data, first_year,last_year, (T_encoder, T_decoder), tau0, xmin, xmax, model, batch_size, xe_mask, tgt_mask, gender=gender)
    recursive_prediction_loss_male, recursive_prediction_loss_female = rf.loss_recursive_forecasting(validation_data, recursive_prediction, gender_model = gender)

    trt.save_plots(history['train_loss_history'], history['val_loss_history'], gender = gender)

    # Evaluation
    test_loss = trt.evaluate(best_model, batch_size, test_data, tgt_mask,  xe_mask, criterion)
    val_loss = trt.evaluate(best_model, batch_size, val_data, tgt_mask,  xe_mask, criterion)
    train_loss = trt.evaluate(best_model, batch_size, train_data, tgt_mask,  xe_mask, criterion)


    print('=' * 100)
    print('| End of training | training loss {:5.2f} | validation loss {:5.2f} | test loss {:5.2f} | test ppl {:8.2f}'.format(
        train_loss, val_loss, test_loss, math.exp(test_loss)))
    print('-' * 100)

    text_to_print = '| Evaluating on recursive data'
    if recursive_prediction_loss_male is not None: text_to_print = text_to_print + '| Male loss {:5.2f}'.format(recursive_prediction_loss_male)
    if recursive_prediction_loss_female is not None: text_to_print = text_to_print + '| female loss {:5.2f}'.format(recursive_prediction_loss_female)
    print(text_to_print)
    
    print('=' * 100)

    if gender == 'both':
        return (recursive_prediction_loss_male + recursive_prediction_loss_female)/2
    
    elif gender == 'Male':
        return recursive_prediction_loss_male

    elif gender == 'Female':
        return recursive_prediction_loss_female
    

def gridSearch(parameters: dict, func_args: tuple, func: callable = train_transformer, csv_to_fill: str = None, model_name: str = 'model', folder = 'hyperparameters'):
    file = f'{folder}/hyperparameter_tuning_{model_name}.csv'

    # Get hyperparameter names and values
    hyperparameter_names = list(parameters.keys())
    hyperparameter_values = list(parameters.values())

    # Generate all combinations of hyperparameter values
    

    if csv_to_fill is None:
        hyperparameter_combinations = list(product(*hyperparameter_values))
        initial_idx = 0
        dataframe = pd.DataFrame(hyperparameter_combinations, columns = hyperparameter_names)
        dataframe['results'] = None
        dataframe.to_csv(file, index=False)
    
    else:
        file = csv_to_fill
        dataframe = pd.read_csv(csv_to_fill)
        dataframe['units_per_layer'] = dataframe['units_per_layer'].apply(ast.literal_eval)
        initial_idx = len(dataframe[dataframe.results.notna()])
        hyperparameter_combinations = dataframe[dataframe.results.isna()].drop(columns=['results']).values.tolist()
        



    print('\n')
    print('=' * 100)
    print('=' * 100)
    print(f'| Total Number of Trials: {len(hyperparameter_combinations)} |')
    print('=' * 100)
    print('=' * 100)
    print('\n')

    # Store the best hyperparameters and corresponding evaluation
    best_hyperparameters = None
    best_evaluation = float('inf')  # Assuming higher is better

    # Iterate through each hyperparameter combination
    for i, combo in enumerate(hyperparameter_combinations):
        j = i + initial_idx

        # Create a dictionary with current hyperparameter values
        current_hyperparameters = dict(zip(hyperparameter_names, combo))  

        # Train the model with current hyperparameters and get the evaluation on the validation set
        print('-' * 100)
        print(f'| Training combo number: {i + initial_idx+1}/{len(hyperparameter_combinations) + initial_idx} |')
        print(f'| Hyperparameters:{combo} |')
        current_evaluation = func(current_hyperparameters, *func_args)
        print(f'| Current Avg. evaluation: {current_evaluation}')
        print('-' * 100)
        print('\n')


        dataframe.at[i + initial_idx, 'results'] = current_evaluation
        dataframe.to_csv(file, index=False)


        # Update the best hyperparameters if the current evaluation is better
        if current_evaluation < best_evaluation:
            best_evaluation = current_evaluation
            best_hyperparameters = current_hyperparameters

    print('=' * 100)
    print(f'| Best Parameters: {best_hyperparameters} |')
    print(f'| Best evaluation: {best_evaluation} |')
    print('=' * 100)

    return best_hyperparameters, best_evaluation

    
    



    


