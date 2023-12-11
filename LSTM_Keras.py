from numpy.random import seed
import tensorflow as tf
import pickle

import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
seed(1)
tf.random.set_seed(2) #89

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

#LSTM Configuration
import tensorflow as tf
from keras import Sequential
from keras.layers import Input
from keras.layers import Dense, LSTM, Flatten, Concatenate, Bidirectional, GRU
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras import backend
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils import plot_model


# Optimizer for comparing LSTM vs Lee-Carter
import cvxpy as cp

import pydot
import graphviz

def one_gender_model3(T,tau0,tau1,tau2,tau3,optimizer):
  y0 = 0.01
  #y0=np.mean(y_train_f) ####REVER ESTE VALOR
  input1 = Input(shape=[T,tau0],name="input")
  lstm1 = LSTM(units=tau1, activation='tanh', recurrent_activation='tanh', return_sequences= True, name ="lstm1")(input1) #20
  lstm2 = LSTM(units=tau2, activation='tanh', recurrent_activation='tanh', return_sequences=True, name ="lstm2")(lstm1) #15
  lstm3 = LSTM(units=tau3, activation='tanh', recurrent_activation='tanh', name ="lstm3")(lstm2) #10
  dense = Dense(units=1, name="Output", activation=backend.exp,
                weights=np.array([np.zeros((10,1)),np.array([np.log(y0)])], dtype='object'))(lstm3) #modificar para o ultimo
  # )

  model = Model(inputs=input1, outputs=dense)
  opt = optimizer # - if we want to improve results - lr=0.0003

  model.compile(optimizer=opt, loss='mean_squared_error')
  return model

#Baseline:
def lstm_model(T,tau0,tau1,tau2,tau3, gender = 'both'):
    #y0=np.mean(y_train_bg) ####REVER ESTE VALOR
    y0 = 0.1
    input1 = Input(shape=(T,tau0),name="input")
    #if gender == 'both':
    Gender = Input(shape=(1),name="Gender")

    lstm1 = LSTM(units=tau1, activation='tanh', recurrent_activation='sigmoid', return_sequences= True, name ="LSTM1")(input1) #20
    lstm2 = LSTM(units=tau2, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name ="LSTM2")(lstm1) #15
    lstm3 = LSTM(units=tau3, activation='tanh', recurrent_activation='sigmoid', name ="LSTM3")(lstm2) #10

    if gender == 'both':
      concat = Concatenate(axis=1,name='Concat')([lstm3, Gender])
      dense = Dense(units=1, name ="Output", activation = backend.exp,  weights=np.array([np.zeros((tau3+1,1)),np.array([np.log(y0)])], dtype='object')
                    )(concat) #modificar para o ultimo
      uni_model = Model(inputs=[input1, Gender], outputs=dense)

    if gender == 'Male' or gender == 'Female' :
       dense = Dense(units=1, name="Output", activation=backend.exp,
                weights=np.array([np.zeros((tau3,1)),np.array([np.log(y0)])], dtype='object'))(lstm3)
       uni_model = Model(inputs=input1, outputs=dense)

    opt = Adam(learning_rate=0.001, beta_1= 0.9, beta_2=0.999, )
    uni_model.compile(optimizer=opt, loss='mean_squared_error')
    return uni_model
