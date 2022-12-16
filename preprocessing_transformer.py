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

def batchify(data: Tensor, bsz: int) -> Tensor:
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


bptt = 35
def get_batch(source: Tensor, i: int):
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def transformer_input_shaping(padd_train,T_encoder, T_decoder,tau0, batch_size):
    t1 = padd_train.shape[0]-(T_encoder + T_decoder-1)-1
    a1 = padd_train.shape[1]-(tau0-1)
    n_train = t1 * a1 # number of training samples
    n_batches = n_train // batch_size
    delta0 = int(np.round((tau0-1)/2))
    xtrain_encoder = np.empty((n_batches, batch_size, T_encoder,tau0)) #shaping
    xtrain_encoder[:] = np.NaN 
    xtrain_decoder = np.empty((n_batches, batch_size, T_decoder ,tau0)) #shaping
    xtrain_decoder[:] = np.NaN 

    for t0 in range(0,t1):   # t=time, t0 = 0..39 => year 1950..1990      df_train_cbind.shape = 50,104
        for a0 in range(0,a1): # a=age,  a0=0-99 a0 => age 0..99
            pattern_idx = t0*a1 + a0
            pattern_per_batch_idx = pattern_idx % batch_size
            batch_idx = pattern_idx // batch_size 
            print(pattern_idx, batch_idx, (n_batches,n_train, T_encoder,tau0))
            xtrain_encoder[batch_idx, pattern_per_batch_idx,:,:] = padd_train.iloc[t0 : (t0 + T_encoder), a0 : (a0 + tau0)].copy()  # copy years from t0 to t9 and ages from a0 to a0+5 into xt_train [100*t0+a0, :, :]
            xtrain_decoder[batch_idx, pattern_per_batch_idx,:,:] = padd_train.iloc[(t0 + T_encoder - 1) : (t0 + T_encoder + T_decoder - 1), a0 : (a0 + tau0)].copy()  # copy years from t9 to t13 and ages from a0 to a0+5 into xt_train [100*t0+a0, :, :]

    return xtrain_encoder, xtrain_decoder



def data_to_logmat(raw_data, gender):    
    logmat_gender = raw_data.loc[raw_data['Gender'] == gender].copy()
    logmat_gender = logmat_gender[['Year', 'Age', 'logmx']].copy()
    logmat_gender = pd.crosstab(logmat_gender['Year'], logmat_gender['Age'], logmat_gender['logmx'],aggfunc='sum')
    return logmat_gender

def padding(raw_train,T,tau0):

    delta0 = int(np.round((tau0-1)/2))
    print(delta0)
    padding = [0]*delta0
    padding.extend(np.arange(100))
    padding.extend([99]*delta0)
    
    padd_train = raw_train.iloc[:,padding]
    
    #print("Dataset with padding: \n",padd_train)
    return padd_train

def preprocessed_data( logmat_train, gender, T , tau0, model = "LSTM", batch_size = 5):
    T_encoder, T_decoder = T
    padd_train= padding(logmat_train, T_encoder + T_decoder, tau0)
    xe,xd = transformer_input_shaping(padd_train,T_encoder, T_decoder,tau0, batch_size)
    #xe = minMaxScale(xe, all_data_mat)
    #xd = minMaxScale(xd, all_data_mat)
    return xe, xd





    
