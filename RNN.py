# -*- coding: utf-8 -*-
# Created on Mon Jun 15 17:18:52 2020
# @author: Upquark00

import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Activation, Dense, Dropout
from time import strptime
from math import ceil

# Load data 
file = r'DJIA_historical' + '.csv'
dirname = os.path.dirname(__file__)
filename = os.path.join((dirname), 'data/' +file)
df = pd.read_csv(filename)

# Visualize the total dataset
data = df['Close']
dates = df['Date']
adj_dates = mdates.datestr2num(dates)
plt.plot_date(adj_dates, data, '-')
    
plt.title('DJIA Close vs. Date, 1985-2020')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close value', fontsize=16)
plt.show()

# Check for, and handle, missing values 
if data.isna().sum():
    print('Alert! Missing data')
    # do something to handle empty data rows 

# Normalization is not ideal since closing values trend upwards. 
# Standardization is not appropriate since data does not approximate a Gaussian

# Convert data into format acceptable to Keras
# Input must be 3D ndarray of shape (samples x time steps x features)
train_fraction = 0.80
sequence_length = 30

data = data.to_numpy()
num_train_samples = ceil(len(data) * train_fraction)
train_data = data[:num_train_samples]
test_data = data[num_train_samples:]

# Reshape data 
# x_train.shape = (7105,30)
# y_train.shape = (7105,)
x_train = np.array([train_data[i - sequence_length:i] for i in range(sequence_length, len(train_data))])
y_train = np.array([train_data[i + 1 - sequence_length: (i + 1)] for i in range(sequence_length, len(train_data))])

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))

# x_train.shape should now be (7105,30, 1)

x_test = np.array([test_data[i - sequence_length:i] for i in range(sequence_length, len(test_data))])
y_test = np.array([test_data[i + 1 - sequence_length: (i + 1)] for i in range(sequence_length, len(test_data))])

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))


# Build model 
units = 15
epochs = 25
batch_size = 350

# Build model 
layer_1 = LSTM(
    units = units, 
    input_shape = (sequence_length, 1),
    return_sequences = True,
    )


model = Sequential()
model.add(layer_1)
model.add(Dense(units = 1))
# model.add(Activation('linear'))
model.compile(optimizer = 'adam',
             loss = 'mean_squared_error',
             )

model.fit(x_train,
          y_train,
          epochs = epochs, 
          batch_size = batch_size,
          # validation_split = 0.05
          )

# Evaluate performance 
model.summary()
loss = model.evaluate(x_test, y_test)
print('Test Loss: %f' % (loss))
# print('Test Accuracy: %f' % (accuracy * 100))

history = model.history.history

# =============================================================================
# plt.plot(history['accuracy'])
# plt.title('Model Accuracy vs. Epoch')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
# =============================================================================

plt.plot(history['loss'])
plt.title('Model Loss vs. Epoch')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()















