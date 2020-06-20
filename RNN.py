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
from tensorflow.keras.metrics import Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from time import strptime
from math import ceil
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

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
scaler = MinMaxScaler(feature_range = (0, 1))
data2 = data.reshape(-1, 1)
data = scaler.fit_transform(data2)

num_train_samples = ceil(len(data) * train_fraction)
train_data = data[:num_train_samples]
test_data = data[num_train_samples:]

# Reshape data 
# x_train.shape = (7105,30)
# y_train.shape = (7105,)

x_train_0 = np.array([train_data[i - sequence_length:i] for i in range(sequence_length, len(train_data))])
y_train_0 = np.array([train_data[i + 1 - sequence_length: (i + 1)] for i in range(sequence_length, len(train_data))])
# x_train_0.shape = (7105,30)
# y_train_0.shape = (7105, 30)

x_train = np.reshape(x_train_0, (x_train_0.shape[0], x_train_0.shape[1], 1))
y_train = np.reshape(y_train_0, (y_train_0.shape[0], y_train_0.shape[1], 1))

# x_train.shape should now be (7105,30, 1)

x_test = np.array([test_data[i - sequence_length:i] for i in range(sequence_length, len(test_data))])
y_test = np.array([test_data[i + 1 - sequence_length: (i + 1)] for i in range(sequence_length, len(test_data))])

# Test shape is (1753, 30, 1)
# Trim the test samples
x_test, y_test = x_test[:1750, :], y_test[:1750, :] 

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))


# Build model 
units = 1
epochs = 50
batch_size = 35

# Build model 
layer_1 = LSTM(
    units = 25, 
    batch_input_shape = (batch_size, sequence_length, 1),
    return_sequences = True,
    stateful = True, 
    kernel_initializer = 'random_uniform'
    )
# Output shape is (batch_size, steps, units)

layer_2 = LSTM(
    units = units,
    return_sequences = True,
    stateful = True, 
    kernel_initializer = 'random_uniform'    
    )

model = Sequential()
model.add(layer_1)
model.add(layer_2)
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'mean_squared_error', 
             optimizer = 'adam', 
             metrics = [
                 'accuracy', 
                 Precision(name='precision'), 
                 Recall(name='recall'),
                 TruePositives(name='TP'),
                 TrueNegatives(name='TN'),
                 FalsePositives(name='FP'),
                 FalseNegatives(name='FN')])


# =============================================================================
# early_stopping = EarlyStopping(monitor='val_loss', 
#                                mode='min', verbose=1, 
#                                patience = 9,
#                                restore_best_weights = False
#                                )
# =============================================================================

history = model.fit(x_train,
          y_train,
          epochs = epochs, 
          batch_size = batch_size,
          verbose = 2, 
          validation_data = (x_test, y_test),
          # callbacks = [early_stopping],
          )

# Evaluate performance 
model.summary()
loss, accuracy, precision, recall, TP, TN, FP, FN = model.evaluate(x_test, y_test, batch_size = batch_size)
print('Test Loss: %f' % (loss))
print('Test Accuracy: %f' % (accuracy * 100))
# print('Training stopped after',early_stopping.stopped_epoch,'epochs.')

plt.plot(history.history['accuracy'])
plt.title('Model Accuracy vs. Epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model Loss vs. Epoch')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['precision'])
plt.title('Model Precision vs. Epoch')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['recall'])
plt.title('Model Recall vs. Epoch')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

loss = model.evaluate(x_test, y_test)
# print('Test Loss: %f' % (loss))
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















