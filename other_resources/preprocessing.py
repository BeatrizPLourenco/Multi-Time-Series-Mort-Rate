#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 09:30:54 2022

@author: beatrizlourenco
"""

import numpy as np
import pandas as pd

def LSTM_input_shaping(padd_train,T,tau0):
    t1 = padd_train.shape[0]-(T-1)-1
    a1 = padd_train.shape[1]-(tau0-1)
    
    n_train = t1 * a1 # number of training samples
    delta0 = int(np.round((tau0-1)/2))
    
    xtrain = np.empty ((n_train,T,tau0)) #shaping
    xtrain[:] = np.NaN 
    ytrain = np.empty((n_train))
    ytrain[:] = np.NaN
    
    for t0 in range(0,t1):   # t=time, t0 = 0..39 => year 1950..1990      df_train_cbind.shape = 50,104
      for a0 in range(0,a1): # a=age,  a0=0-99 a0 => age 0..99
        xtrain[(t0)*a1+a0,:,:] = padd_train.iloc[t0:(t0+T),a0:(a0+tau0)].copy()  # copy years from t0 to t0+10 and ages from a0 to a0+5 into xt_train [100*t0+a0, :, :]
        ytrain[(t0)*a1+a0] = padd_train.iloc[t0+T,a0+int(delta0)].copy()
    #print("shaped dataset: \n",xtrain)
    return xtrain,ytrain

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



"""Normalization using **MinMaxScale**:"""

def minMaxScale(data_to_normalize, reference_data):  # MinmaxScale with min and max of reference_data
    xmin = reference_data.to_numpy().min() #comparation between min from Male dataset and Female dataset
    xmax = reference_data.to_numpy().max()
    x_norm = 2*(data_to_normalize-xmin)/(xmin-xmax)-1
    
    return x_norm

def padding(raw_train,T,tau0):

    delta0 = int(np.round((tau0-1)/2))
    print(delta0)
    padding = [0]*delta0
    padding.extend(np.arange(100))
    padding.extend([99]*delta0)
    
    padd_train = raw_train.iloc[:,padding]
    
    #print("Dataset with padding: \n",padd_train)
    return padd_train

def preprocessed_data(raw_data_all, raw_data_train, gender, T , tau0, model = "LSTM"):
    all_data_mat = pd.crosstab(raw_data_all['Year'], raw_data_all['Age'], raw_data_all['logmx'],aggfunc='sum')
    
    train_gender = raw_data_train.loc[raw_data_train['Gender'] == gender].copy()
    train_gender = train_gender[['Year', 'Age', 'logmx']].copy()
    train_gender = pd.crosstab(train_gender['Year'], train_gender['Age'], train_gender['logmx'],aggfunc='sum')

  
  
    if model == "LSTM":
        padd_train= padding(train_gender,T, tau0)
        x,y = LSTM_input_shaping(padd_train,T,tau0)
        x = minMaxScale(x, all_data_mat)
        y = -y
      
        return x, y
  
    elif model == "transformer":
        T_encoder, T_decoder = T
        padd_train= padding(train_gender, T_encoder + T_decoder, tau0)
        xe,xd,y = transformer_input_shaping(padd_train,T_encoder, T_decoder,tau0)
        xe = minMaxScale(xe, all_data_mat)
        xd = minMaxScale(xd, all_data_mat)
        y = minMaxScale(y, all_data_mat)
        return xe, xd, y
    else:
        raise Exception("Invalid model!!")


def preprocessed_data_tranformer(raw_data_all, raw_data_train, gender, T, tau0):
  all_data_mat = pd.crosstab(raw_data_all['Year'], raw_data_all['Age'], raw_data_all['logmx'],aggfunc='sum')

  train_gender = raw_data_train.loc[raw_data_train['Gender'] == gender].copy()
  train_gender = train_gender[['Year', 'Age', 'logmx']].copy()
  train_gender = pd.crosstab(train_gender['Year'], train_gender['Age'], train_gender['logmx'],aggfunc='sum')
  
  padd_train= padding(train_gender,T, tau0)
  
  x, y = LSTM_input_shaping(padd_train,T,tau0)
  x = minMaxScale(x, all_data_mat)
  y = -y

  return x, y


def preprocessing_with_both_genders(raw_data_all, raw_data, T, tau0):

    data0 = (preprocessed_data(raw_data_all, raw_data, 'Female',T,tau0)) # only training data
    data1 = (preprocessed_data(raw_data_all, raw_data, 'Male',T,tau0))
    d = data0[0].shape[0]
    x = np.empty ((2*d,T,tau0)) #shaping
    x[:] = np.NaN
    y = np.empty((2*d))
    y[:] = np.NaN
    gender_indicator = np.array([0,1]*d)

    for i in range(0,d):
      x[(i)*2] = data0[0][i] #odd indexes corresponde to training pattern from the female dataset
      x[(i)*2+1] = data1[0][i] #even indexes corresponde to training pattern from the male dataset
      y[(i)*2] = data0[1][i] 
      y[(i)*2+1] = data1[1][i] 

    x = [x, gender_indicator]
    return x, y


    
