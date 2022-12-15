#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 09:30:54 2022

@author: beatrizlourenco
"""

import numpy as np
import pandas as pd


def transformer_input_shaping(padd_train,T_encoder, T_decoder,tau0):
    t1 = padd_train.shape[0]-(T_encoder + T_decoder-1)-1
    a1 = padd_train.shape[1]-(tau0-1)
    n_train = t1 * a1 # number of training samples
    delta0 = int(np.round((tau0-1)/2))
    
    xtrain_encoder = np.empty((n_train,T_encoder,tau0)) #shaping
    xtrain_encoder[:] = np.NaN 
    xtrain_decoder = np.empty((n_train,T_decoder ,tau0)) #shaping
    xtrain_decoder[:] = np.NaN 
    ytrain = np.empty((n_train,T_decoder ,tau0)) 
    ytrain[:] = np.NaN
    print(np.shape(xtrain_encoder), np.shape(xtrain_decoder), np.shape(ytrain))

    for t0 in range(0,t1):   # t=time, t0 = 0..39 => year 1950..1990      df_train_cbind.shape = 50,104
        for a0 in range(0,a1): # a=age,  a0=0-99 a0 => age 0..99
            pattern_idx = t0*a1 + a0
            xtrain_encoder[pattern_idx,:,:] = padd_train.iloc[t0 : (t0 + T_encoder), a0 : (a0 + tau0)].copy()  # copy years from t0 to t9 and ages from a0 to a0+5 into xt_train [100*t0+a0, :, :]
            xtrain_decoder[pattern_idx,:,:] = padd_train.iloc[(t0 + T_encoder - 1) : (t0 + T_encoder + T_decoder - 1), a0 : (a0 + tau0)].copy()  # copy years from t9 to t13 and ages from a0 to a0+5 into xt_train [100*t0+a0, :, :]
            ytrain[pattern_idx,:,:] = padd_train.iloc[(t0 + T_encoder) : (t0 + T_encoder + T_decoder), a0 : (a0 + tau0)].copy() # copy years from t10 to t14 and ages from a0 to a0+5 into xt_train [100*t0+a0, :, :]

    return xtrain_encoder, xtrain_decoder, ytrain



def data_to_logmat(raw_data, gender):    
    logmat_gender = raw_data.loc[raw_data['Gender'] == gender].copy()
    logmat_gender = logmat_gender[['Year', 'Age', 'logmx']].copy()
    logmat_gender = pd.crosstab(logmat_gender['Year'], logmat_gender['Age'], logmat_gender['logmx'],aggfunc='sum')

def preprocessed_data(raw_data_all, raw_data_train, gender, T , tau0, model = "LSTM"):
    all_data_mat = pd.crosstab(raw_data_all['Year'], raw_data_all['Age'], raw_data_all['logmx'],aggfunc='sum')
    
    train_gender = raw_data_train.loc[raw_data_train['Gender'] == gender].copy()
    train_gender = train_gender[['Year', 'Age', 'logmx']].copy()
    train_gender = pd.crosstab(train_gender['Year'], train_gender['Age'], train_gender['logmx'],aggfunc='sum')


    T_encoder, T_decoder = T
    padd_train= padding(train_gender, T_encoder + T_decoder, tau0)
    xe,xd,y = transformer_input_shaping(padd_train,T_encoder, T_decoder,tau0)
    xe = minMaxScale(xe, all_data_mat)
    xd = minMaxScale(xd, all_data_mat)
    y = minMaxScale(y, all_data_mat)
    return xe, xd, y





    
