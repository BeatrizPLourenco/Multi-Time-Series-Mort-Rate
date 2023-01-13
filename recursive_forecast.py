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

def get_mortality_dataframe_shell( 
    year,
    gender_idx,
    model_pred_mx,
    model_pred_logmx,
    age_width: tuple = (0,100),
    sort_age: bool = True):
    
    age_range = range(age_width[0], age_width[1])
    nb_genders = len(gender_idx.unique())

    age_list = list(age_range) * nb_genders
    if sort_age: age_list.sort()

    year_list = [year] * nb_genders * len(age_range)
    gender_list = pd.Categorical(gender_idx.tolist()).rename_categories(['Female','Male'])
    model_pred_mx_list = model_pred_mx.tolist()
    model_pred_logmx_list = model_pred_logmx.tolist()

    return pd.DataFrame({ 'Year': year_list, 'Age': age_list, 'Gender': gender_list, 'mx': model_pred_mx_list, 'logmx': model_pred_logmx_list })

def recursive_forecast(
    raw_data: pd.DataFrame, 
    first_year: int,
    last_year: int, 
    T: tuple, 
    tau0: int, 
    xmin: float, 
    xmax: float, 
    model: mrt.MortalityRateTransformer, 
    enc_out_mask: torch.Tensor, 
    dec_in_mask: torch.Tensor, 
    last_obs_year: int = 1999,
    columns: list = ['Year', 'Age','mx', 'logmx', 'Gender']) -> pd.DataFrame:

    model.eval()

    timerange = T[0] + T[1]

    # first window of T=10 years to predict the first year ( (last_obs_year-T) to last_obs_year => predicts last_obs_year+1):
    mortality = get_data_in_time_range(raw_data, last_year = last_obs_year)
    mortality = mortality[columns].copy()

    for  year in range(last_obs_year+1, last_year+1): # The next year is recursively predicted 
        mort = get_data_in_time_range(mortality, first_year = year-timerange-1) #selection of only the last timerange years
        xe, xd, ind, yd = prt.preprocessing_with_both_genders(mort, T, tau0, xmin, xmax, 1) 
        xe, xd, ind = prt.from_numpy_to_torch((xe, xd, ind))
        xe, xd, ind = xe.squeeze(), xd.squeeze(1), ind.squeeze(1)
        ind_last_year = ind.squeeze()[:,1]

        # Construction of prediction table for the test set:
        model_forward = model(xe, xd, ind, enc_out_mask, dec_in_mask) #prediction of the model
        model_pred = model_forward.squeeze()[:,-1]
        log_mx = (-model_pred) #substitution of real values for predicted ones
        mx = torch.exp(-model_pred)

        #current year prediction on dictionary
        predicted = get_mortality_dataframe_shell(year, ind_last_year, mx, log_mx)

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


