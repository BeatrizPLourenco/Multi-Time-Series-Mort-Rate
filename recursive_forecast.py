import pandas as pd
import numpy as np
import preprocessing_transformer as prt
import torch


def recursive_forecast_both_genders(raw_data, first_year,last_year, T, tau0, model, enc_out_mask, dec_in_mask, gender = 'Both'):
    ObsYear = 1999 #last obs year

    timerange = T[0] + T[1]
    # first window of T=10 years to predict the first year ( (ObsYear-T) to ObsYear => predicts ObsYear+1):
    mortality = raw_data[(raw_data['Year'] <= ObsYear)].copy()
    mortality = mortality[['Year', 'Age','mx', 'logmx', 'Gender']].copy()

    for  year in range(ObsYear+1, last_year+1): # The next year is recursively predicted 
        mort = mortality[( mortality['Year'] >= (year-timerange-1))].copy() #selection of only the last T years
        xe, xd, ind, yd = prt.preprocessing_with_both_genders(mort, T, tau0, 1) 

        l=[]
        for i in range(0,100):
            l.extend([i]*2)

        idx_shape = np.shape(ind)
        gender_idx_list = ind.reshape(idx_shape[0], idx_shape[2])[:,-1]
        predicted = pd.DataFrame({ 'Year': ([year]*200), 'Age': l, 
                                'Gender': pd.Categorical(gender_idx_list.tolist()).rename_categories(['Female','Male']) }) 


        # Construction of prediction table for the test set:
        xe, xd, ind = prt.from_numpy_to_torch((xe, xd, ind))
        xe, xd, ind = torch.squeeze(xe,1), torch.squeeze(xd,1), torch.squeeze(ind,1)
        model_foward= model(xe, xd, ind, enc_out_mask, dec_in_mask) #prediction of the model
        model_pred = model_foward[:,-1,:]
        predicted['logmx'] = (-model_pred).squeeze().tolist() #substitution of real values for predicted ones
        predicted['mx'] = torch.exp(-model_pred).squeeze().tolist()

        # Construction of dataframe for the values that we are going to keep for the next iteration
        keep =  pd.DataFrame(mortality.copy())
        mortality= keep.append(predicted)

    prediction = (mortality[( mortality['Year'] >= (first_year)) ].copy())
    if gender in {'Male', 'Female'}:
        prediction = prediction[prediction['Gender'] == gender]

    return prediction


