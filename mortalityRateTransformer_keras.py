import tensorflow as tf
from keras import layers, models
import data_cleaning as dtclean
import numpy as np
import preprocessing_transformer as prt
import train_transformer as trt
from scheduler import Scheduler
import mortalityRateTransformer as mrt
from torch import nn, optim, zeros
import recursive_forecast as rf
import explainability as xai

# Control
country = "PT"
#split_value = 2000
raw_filenamePT = 'Dataset/Mx_1x1_alt.txt'
raw_filenameSW = 'Dataset/CHE_mort.xlsx'
T = 10
T_encoder = 7
T_decoder = 3
tau0 = 5
split_value1 = 1993 # 1993 a 2005 corresponde a 13 anos (13/66 approx. 20%)
split_value2 = 2006 # 2006 a 2022 corresponde a 17 anos (17/83 approx. 20%)
gender = 'both'
both_gender_model = (gender == 'both')
checkpoint_dir = f'Saved_models/checkpoint_{gender}.pt'
best_model_dir = f'Saved_models/best_model_{gender}.pt'
batch_size = 5

# Preprocessing
data = dtclean.get_country_data(country)
data_logmat = prt.data_to_logmat(data, gender)
xmin, xmax = prt.min_max_from_dataframe(data_logmat)

# Split Data
training_data, validation_test_data  = dtclean.split_data(data, split_value1)
validation_data, testing_data  = dtclean.split_data(validation_test_data, split_value2)
training_val_data, testing_data  = dtclean.split_data(data, split_value2)    

# preprocessing for the transformer
if gender == 'both':
    train_data = prt.preprocessing_with_both_genders(training_data, (T_encoder, T_decoder), tau0, xmin, xmax, batch_size)
    val_data  = prt.preprocessing_with_both_genders(validation_data,  (T_encoder, T_decoder), tau0, xmin, xmax, batch_size)
    test_data = prt.preprocessing_with_both_genders(testing_data,  (T_encoder, T_decoder), tau0, xmin, xmax, batch_size)
    train_val_data = prt.preprocessing_with_both_genders(training_val_data,  (T_encoder, T_decoder), tau0, xmin, xmax, batch_size)


elif gender == 'Male' or gender == 'Female' :
    train_data = prt.preprocessed_data( training_data,  gender, (T_encoder, T_decoder),tau0, xmin, xmax, batch_size)
    val_data = prt.preprocessed_data( validation_data, gender, (T_encoder, T_decoder),tau0, xmin, xmax, batch_size)
    test_data = prt.preprocessed_data(testing_data, gender,  (T_encoder, T_decoder),tau0, xmin, xmax, batch_size)
    train_val_data = prt.preprocessed_data(training_val_data, gender,  (T_encoder, T_decoder),tau0, xmin, xmax, batch_size)



# Define the model
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

class PositionalEncoding(layers.Layer):
    def __init__(self, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        # apply sine to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cosine to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, seq_len, _ = input_shape
        pos_encoding = self.positional_encoding(seq_len, self.embed_dim)
        return inputs + pos_encoding[:, :seq_len, :]

# Rest of the code remains the same...


def build_model(embed_dim, num_heads, ff_dim, input_shape, output_shape, dropout=0.1):
    inputs = layers.Input(shape=input_shape)
    x = PositionalEncoding(input_shape[1], embed_dim)(inputs)
    x = TransformerEncoder(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(output_shape, activation="linear")(x)

    model = models.Model(inputs=inputs, outputs=x)
    return model

# Define hyperparameters
embed_dim = 32  # Embedding dimension
num_heads = 2   # Number of attention heads
ff_dim = 32     # Feedforward dimension

# Build and compile the model
model = build_model(embed_dim, num_heads, ff_dim, input_shape=(10, 5), output_shape=1)
model.compile(optimizer="adam", loss="mean_squared_error")

# Display the model summary
model.summary()

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

# Function to create the model
def create_model(embed_dim, num_heads, ff_dim, input_shape, output_shape, dropout=0.1):
    model = build_model(embed_dim, num_heads, ff_dim, input_shape, output_shape, dropout)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Define hyperparameter grid for grid search
param_grid = {
    'embed_dim': [16, 32, 64],
    'num_heads': [2, 4],
    'ff_dim': [16, 32, 64],
    'dropout': [0.1, 0.2, 0.3],
}

# Create KerasRegressor
model = KerasRegressor(build_fn=create_model, input_shape=(10, 5), output_shape=1, epochs=10, batch_size=32, verbose=0)

X_train = train_data[:-1]
y_train = train_data[-1]
X_val = val_data[:-1]
y_val = val_data[-1]

# Use GridSearchCV with specified training and validation sets
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=[(list(range(len(X_train))), list(range(len(X_train), len(X_train) + len(X_val))))])
grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val))

# Print the best parameters and corresponding mean squared error
print(f"Best parameters: {grid_result.best_params_}")
print(f"Best mean squared error: {-grid_result.best_score_}")