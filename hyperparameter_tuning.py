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
import json
import pandas as pd



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

def train_lstm(parameters : dict, 
                      split_value1 = 1993, 
                      split_value2 = 2006,
                      gender = 'both',
                      raw_filename = 'Dataset/Mx_1x1_alt.txt',
                      country = "PT", 
                      seed = 0):
    
    random.seed(seed)
    np.random.seed(seed)
     
    
    # Control
    #split_value = 2000
    T_ = parameters['T']
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
            patience=50
        )
    else:
        best_model, history = trt.load_best_model(model, best_model_dir= best_model_dir)
        
    
    first_year, last_year = split_value1, split_value2 -1
    recursive_prediction = rf.recursive_forecast(data, first_year,last_year, (T_encoder, T_decoder), tau0, xmin, xmax, model, batch_size, xe_mask, tgt_mask)
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

    #return {'male_mse:': recursive_prediction_loss_male,'female_mse:': recursive_prediction_loss_female}


    if gender == 'both':
        return (recursive_prediction_loss_male + recursive_prediction_loss_female)
    
    elif gender == 'Male':
        return recursive_prediction_loss_male

    elif gender == 'Female':
        return recursive_prediction_loss_female
    

def gridSearch(parameters: dict, func_args: tuple, func: callable = train_transformer, model_name: str = 'model'):
    # Get hyperparameter names and values
    hyperparameter_names = list(parameters.keys())
    hyperparameter_values = list(parameters.values())

    # Generate all combinations of hyperparameter values
    hyperparameter_combinations = list(product(*hyperparameter_values))
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
        # Create a dictionary with current hyperparameter values
        current_hyperparameters = dict(zip(hyperparameter_names, combo))

        # Train the model with current hyperparameters and get the evaluation on the validation set
        print('-' * 100)
        print(f'| Training combo number: {i+1}/{len(hyperparameter_combinations)} |')
        print(f'| Hyperparameters:{combo} |')
        current_evaluation = func(current_hyperparameters, *func_args)
        print(f'| Current Avg. evaluation: {current_evaluation}')
        print('-' * 100)
        print('\n')
        
        results = {'hyperparameters': current_hyperparameters, 
                                        'evaluation': current_evaluation}
        
        pd.DataFrame(results).to_csv(f'results_hyperparameter_opt/hyperparameter_tuning_{model_name}_{list(current_hyperparameters.values())}.csv', index=False)


     
        # Update the best hyperparameters if the current evaluation is better
        if current_evaluation < best_evaluation:
            best_evaluation = current_evaluation
            best_hyperparameters = current_hyperparameters

    print('=' * 100)
    print(f'| Best Parameters: {best_hyperparameters} |')
    print(f'| Best evaluation: {best_evaluation} |')
    print('=' * 100)

    best_results = {'hyperparameters': best_hyperparameters, 
                                        'evaluation': best_evaluation}

    pd.DataFrame(best_results).to_csv(f'results_hyperparameter_opt/hyperparameter_tuning_{model_name}_best_parameters.csv', index=False)

    return best_hyperparameters, best_evaluation

    
    



    


