import pandas as pd
import numpy as np
import preprocessing_transformer as prt
import torch
import mortalityRateTransformer as mrt
import numpy as np
from sklearn.metrics import mean_squared_error


def get_data_in_time_range(data: pd.DataFrame, first_year: int = None, last_year: int = None ):
    assert first_year != None or last_year != None
    if first_year != None and last_year != None:
        new_data = data[(data['Year'] >= first_year) & (data['Year'] <= last_year)].copy()

    elif first_year == None: 
        new_data = data[(data['Year'] <= last_year)].copy()

    elif last_year == None: 
        new_data = data[(data['Year'] >= first_year)].copy()

    return new_data

import pandas as pd
import numpy as np

def get_mortality_dataframe_shell(year, gender_idx, model_pred_mx, model_pred_logmx, model_type='transformer', age_width=(0, 100), sort_age=True, gender=None):
    if model_type == 'lstm':
        assert gender is not None

    age_range = range(age_width[0], age_width[1])

    if model_type == 'transformer':

        if gender_idx is None:
            nb_genders = 1
        else:
            nb_genders = 2
            gender_idx = [int(gender_idx[i][0]) for i in range(len(gender_idx))]
            gender_idx = np.array(gender_idx)

    elif model_type == 'lstm':
        nb_genders = len(np.unique(gender_idx))

    age_list = sorted(list(age_range) * nb_genders) if sort_age else list(age_range) * nb_genders
    year_list = [year] * nb_genders * len(age_range)

    if gender == 'both':
        gender_list = pd.Categorical(gender_idx.tolist()).rename_categories({0: 'Female', 1: 'Male'})
    
    else:
        gender_list = [gender] * nb_genders  * len(age_range)

    return pd.DataFrame({'Year': year_list, 'Age': age_list, 'Gender': gender_list, 'mx': model_pred_mx.tolist(), 'logmx': model_pred_logmx.tolist()})

"""
def get_mortality_dataframe_shell( 
    year,
    gender_idx,
    model_pred_mx,
    model_pred_logmx,
    model_type = 'tranformer',
    age_width: tuple = (0,100),
    sort_age: bool = True,
    gender:str = None):
    if model_type == 'lstm':
        assert gender != None
    age_range = range(age_width[0], age_width[1])
    if model_type is 'transformer':
        nb_genders = len(gender_idx.unique())
    else:
        nb_genders = len(np.unique(gender_idx))


    age_list = list(age_range) * nb_genders
    if sort_age: age_list.sort()

    year_list = [year] * nb_genders * len(age_range)
    #gender_list = pd.Categorical(gender_idx.tolist()).rename_categories(['Female','Male'])
    gender_idx = [int(gender_idx[i][0]) for i in range(len(gender_idx))] if model_type == 'transformer' else gender_idx
    gender_idx = np.array(gender_idx )
    
    gender_list = pd.Categorical(gender_idx.tolist()).rename_categories({0:'Female',1:'Male'}) if gender == 'both' else gender
    model_pred_mx_list = model_pred_mx.tolist()
    model_pred_logmx_list = model_pred_logmx.tolist()

    return pd.DataFrame({ 'Year': year_list, 'Age': age_list, 'Gender': gender_list, 'mx': model_pred_mx_list, 'logmx': model_pred_logmx_list })"""

def recursive_forecast_both_genders(data,first_year,last_year,T,tau0, xmin, xmax, model, gender = 'Both'):
  with torch.no_grad():
    ObsYear = 1999 #last obs year

    # first window of T=10 years to predict the first year ( (ObsYear-T) to ObsYear => predicts ObsYear+1):
    mortality = data[(data['Year'] <= ObsYear)].copy()
    mortality = mortality[['Year', 'Age','mx', 'logmx', 'Gender']].copy()

    for  year in range(ObsYear+1, last_year+1): # The next year is recursively predicted 
        mort = mortality[( mortality['Year'] >= (year-T-1))].copy() #selection of only the last T years
        x_mort, gender_indicator, y_mort = prt.preprocessing_with_both_gendersLSTM(mort, T, tau0, xmin, xmax) 
        x_mort, gender_indicator,= prt.from_numpy_to_torch((x_mort, gender_indicator,))


        l=[]
        for i in range(0,100):
            l.extend([i]*2)

        predicted = pd.DataFrame({ 'Year': ([year]*200), 'Age': l, 
                                'Gender': pd.Categorical((gender_indicator.squeeze()).tolist()).rename_categories(['Female','Male']) }) 


        # Construction of prediction table for the test set:
        model_pred= (model(x_mort, gender_indicator)) #prediction of the model
        predicted['logmx'] = (-model_pred).squeeze().tolist() #substitution of real values for predicted ones
        predicted['mx'] = torch.exp((-model_pred).squeeze()).tolist()
        import warnings
        warnings.filterwarnings("ignore")

        # Construction of dataframe for the values that we are going to keep for the next iteration
        keep =  pd.DataFrame(mortality.copy())
        mortality= keep.append(predicted)

        prediction = (mortality[( mortality['Year'] >= (first_year)) ].copy())
        if gender in {'Male', 'Female'}:
            prediction = prediction[prediction['Gender'] == gender]
  
  return prediction



def recursive_forecast(
    raw_data: pd.DataFrame, 
    first_year: int,
    last_year: int, 
    T: tuple, 
    tau0: int, 
    xmin: float, 
    xmax: float, 
    model: mrt.MortalityRateTransformer, 
    batch_size:int,
    enc_out_mask: torch.Tensor = None, 
    dec_in_mask: torch.Tensor = None, 
    columns: list = ['Year', 'Age','mx', 'logmx', 'Gender'],
    model_type:str = 'transformer',
    gender:str = None) -> pd.DataFrame:

    last_obs_year = first_year-1

    

    if model_type == 'transformer':
        model.eval()
        timerange = T[0] + T[1]
    elif model_type == 'lstm':
        timerange = T

    # first window of T=10 years to predict the first year ( (last_obs_year-T) to last_obs_year => predicts last_obs_year+1):
    mortality = get_data_in_time_range(raw_data, last_year = last_obs_year)
    mortality = mortality[columns].copy()

    for  year in range(last_obs_year+1, last_year+1): # The next year is recursively predicted 
        mort = get_data_in_time_range(mortality, first_year = year-timerange-1) #selection of only the last timerange years
        if model_type == 'transformer':
            xe, xd, ind, yd = prt.preprocessing_with_both_genders(mort, T, tau0, xmin, xmax, 1, 1) if gender == 'both' else prt.preprocessed_data(mort, gender, T, tau0, xmin, xmax, 1, 1)
            xe, xd, ind = prt.from_numpy_to_torch((xe, xd, ind))
            xe, xd = xe.squeeze(), xd.squeeze(1)
            ind = ind.squeeze(1) if ind is not None else ind
            #ind_last_year = ind.squeeze(2)[:,-1]
        elif model_type == 'lstm': #REVER
            assert gender != None
            x, ind, y = prt.preprocessing_with_both_gendersLSTM(mort, T, tau0, xmin, xmax) if gender == 'both' else prt.preprocessed_dataLSTM(mort, gender, T, tau0, xmin, xmax)

        # Construction of prediction table for the test set:
        if model_type == 'transformer':
            model_forward = model(xe, xd, ind, enc_out_mask, dec_in_mask) #prediction of the model
            model_pred = model_forward.squeeze(2)[:,-1]
            log_mx = (-model_pred) #substitution of real values for predicted ones
            mx = torch.exp(-model_pred)
        elif model_type == 'lstm':
            inp = [x, ind] if gender == 'both' else x
            model_forward = np.array(model(inp)).flatten()   
            log_mx = (-model_forward) #substitution of real values for predicted ones
            mx = np.exp(-model_forward)

        #current year prediction on dictionary
        predicted = get_mortality_dataframe_shell(year, ind, mx, log_mx, model_type = model_type, gender = gender)

        # Construction of dataframe for the values that we are going to keep for the next iteration
        mortality= pd.concat([mortality, predicted])

    prediction = (mortality[( mortality['Year'] >= (first_year)) ].copy())

    return prediction


def loss_recursive_forecasting( raw_test_data: pd.DataFrame, pred_data: pd.DataFrame, mx_column:str = 'mx', gender_model = 'both'):
    if gender_model in ('both', 'Male'):
        real_test_male = (raw_test_data[raw_test_data['Gender'] == 'Male']).copy()[mx_column]
        recursive_prediction_male = (pred_data[pred_data['Gender'] == 'Male']).copy()[mx_column]
        recursive_prediction_loss_male = np.round(mean_squared_error(real_test_male.to_numpy(),recursive_prediction_male.to_numpy())*10**4,3)
        if gender_model == 'Male': recursive_prediction_loss_female = None
    
    if gender_model in ('both', 'Female'):
       real_test_female = (raw_test_data[raw_test_data['Gender'] == 'Female']).copy()[mx_column]
       recursive_prediction_female = (pred_data[pred_data['Gender'] == 'Female']).copy()[mx_column]
       recursive_prediction_loss_female = np.round(mean_squared_error(real_test_female.to_numpy(),recursive_prediction_female.to_numpy())*10**4,3)
       if gender_model == 'Female': recursive_prediction_loss_male = None

    return recursive_prediction_loss_male, recursive_prediction_loss_female


