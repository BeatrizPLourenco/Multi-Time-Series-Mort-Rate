import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, LSTM, Flatten, Concatenate, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow_addons.optimizers import  RectifiedAdam, Lookahead
from tensorflow.keras import backend 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import numpy as np

class mortalityRateLSTM:

    def __init__(tau1: int,
                 tau2: int,
                 tau3: int,
                 y0: np.array):
        self.lstm1 = LSTM(units=tau1, activation='tanh', recurrent_activation='sigmoid', return_sequences= True, name ="lstm1")


def both_gender_model(tau1,tau2,tau3):

    y0=np.mean(y_train_bg) ####REVER ESTE VALOR
    input1 = Input(shape=(T,tau0),name="input")
    Gender = Input(shape=(1),name="Gender")

    lstm1 = LSTM(units=tau1, activation='tanh', recurrent_activation='sigmoid', return_sequences= True, name ="lstm1")(input1) #20
    lstm2 = LSTM(units=tau2, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name ="lstm2")(lstm1) #15
    lstm3 = LSTM(units=tau3, activation='tanh', recurrent_activation='sigmoid', name ="lstm3")(lstm2) #10
    concat = Concatenate(axis=1,name='Concat')([lstm3, Gender])
    dense = Dense(units=1, name ="Output", activation = backend.exp,  weights=np.array([np.zeros((11,1)),np.array([np.log(y0)])], dtype='object')
                  )(concat) #modificar para o ultimo


    uni_model = Model(inputs=[input1, Gender], outputs=dense)
    #radam = RectifiedAdam()
    #ranger = Lookahead(radam, sync_period=6, slow_step_size=0.5)
    #opt=ranger
    #opt = SGD(learning_rate=0.001, momentum=0.001) # - if we want to improve results - lr=0.0003
    opt = Adam(learning_rate=0.001, beta_1= 0.9, beta_2=0.999, )
    uni_model.compile(optimizer=opt, loss='mean_squared_error')
    return uni_model
