#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 09:30:54 2022

@author: beatrizlourenco
"""

import numpy as np
import pandas as pd
from torch import Tensor, device, cuda

device = device('cuda' if cuda.is_available() else 'cpu')


def transformer_input_shaping(padd_train,T_encoder, T_decoder,tau0, batch_size, num_out_features):
    map = {'female' : 0, 'male' : 1}

    t1 = padd_train.shape[0]-(T_encoder + T_decoder-1)-1
    a1 = padd_train.shape[1]-(tau0-1)
    n_train = t1 * a1 # number of training samples
    n_batches = n_train // batch_size
    delta0 = int(np.round((tau0-1)/2))

    x = np.empty((n_batches, batch_size, T_encoder,tau0)) #shaping
    x[:] = np.NaN 
    y_input = np.empty((n_batches, batch_size, T_decoder, num_out_features)) #shaping
    y_input[:] = np.NaN 
    y_expected = np.empty((n_batches, batch_size, T_decoder, num_out_features)) #shaping
    y_expected[:] = np.NaN 

    for t0 in range(0,t1):   # t=time, t0 = 0..39 => year 1950..1990      df_train_cbind.shape = 50,104
        for a0 in range(0,a1): # a=age,  a0=0-99 a0 => age 0..99
            pattern_idx = t0*a1 + a0
            pattern_per_batch_idx = pattern_idx % batch_size
            batch_idx = pattern_idx // batch_size 
            
            x[batch_idx, pattern_per_batch_idx,:,:] = padd_train.iloc[t0 : (t0 + T_encoder), a0 : (a0 + tau0)].copy()  # copy years from t0 to t9 and ages from a0 to a0+5 into xt_train [100*t0+a0, :, :]
            y_input[batch_idx, pattern_per_batch_idx, :, :] = padd_train.iloc[(t0 + T_encoder - 1) : (t0 + T_encoder + T_decoder - 1), a0 + int(delta0) : a0 + int(delta0) + num_out_features].copy()  # copy years from t9 to t13 and ages from a0 to a0+5 into xt_train [100*t0+a0, :, :]
            y_expected[batch_idx, pattern_per_batch_idx, :, :] = padd_train.iloc[(t0 + T_encoder) : (t0 + T_encoder + T_decoder), a0 + int(delta0)  : a0 + int(delta0) + num_out_features].copy()  # copy years from t9 to t13 and ages from a0 to a0+5 into xt_train [100*t0+a0, :, :]
                


    return x, y_input, y_expected



def data_to_logmat(raw_data, gender):    
    logmat_gender = raw_data.loc[raw_data['Gender'] == gender].copy()
    logmat_gender = logmat_gender[['Year', 'Age', 'logmx']].copy()
    logmat_gender = pd.crosstab(logmat_gender['Year'], logmat_gender['Age'], logmat_gender['logmx'],aggfunc='sum')
    return logmat_gender

def padding(raw_train,T,tau0):

    delta0 = int(np.round((tau0-1)/2))
    padding = [0]*delta0
    padding.extend(np.arange(100))
    padding.extend([99]*delta0)
    
    padd_train = raw_train.iloc[:,padding]
    
    #print("Dataset with padding: \n",padd_train)
    return padd_train

def preprocessed_data( data, gender, T , tau0, batch_size = 5, num_out_features = 1):
    T_encoder, T_decoder = T
    logmat = data_to_logmat(data, gender)
    padd_train= padding(logmat, T_encoder + T_decoder, tau0)
    xe,xd, yd = transformer_input_shaping(padd_train,T_encoder, T_decoder,tau0, batch_size, num_out_features)
    return xe, xd, None, yd


def preprocessing_with_both_genders(data, T, tau0, batch_size = 5, num_out_features = 1):

    data0 = (preprocessed_data(data, 'Female', T, tau0, batch_size, num_out_features)) # only training data
    data1 = (preprocessed_data(data, 'Male', T, tau0, batch_size, num_out_features))

    d = data0[0].shape[0]
    T_encoder = T[0]
    T_decoder = T[1]

    dimensions = (d * 2, batch_size, T[0], T[1])


    d = data0[0].shape[0]
    xe = np.empty ((2*d, batch_size, T_encoder, tau0)) #shaping
    xe[:] = np.NaN
    xd = np.empty ((2*d, batch_size, T_decoder, num_out_features)) #shaping
    xd[:] = np.NaN
    yd = np.empty((2*d, batch_size, T_decoder, num_out_features))
    yd[:] = np.NaN
    gender_indicator = np.array([([0]*T_decoder+[1]*T_decoder)] * d * batch_size)
    gender_indicator = np.reshape(gender_indicator, [2*d, batch_size, T_decoder, num_out_features])


    for i in range(d):
        for j in range(batch_size):
            xe[(i)*2] = data0[0][i][j] #even indexes corresponde to training pattern from the female dataset
            xd[(i)*2] = data0[1][i][j]
            yd[(i)*2] = data0[3][i][j]

            xe[(i)*2 + 1] = data1[0][i][j] #odd indexes corresponde to training pattern from the male dataset
            xd[(i)*2 + 1] = data1[1][i][j]
            yd[(i)*2 + 1] = data1[3][i][j]

    return xe, xd, gender_indicator, yd






    
