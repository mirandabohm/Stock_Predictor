# -*- coding: utf-8 -*-
# Created on Mon Jun 15 17:18:52 2020
# @author: Upquark00

import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from time import strptime

# Load data 
file = r'all_stocks_2006-01-01_to_2018-01-01' + '.csv'
dirname = os.path.dirname(__file__)
filename = os.path.join((dirname), 'data/' +file)

df = pd.read_csv(filename)
tickers = df.Name.unique().tolist()

# Visualize the total dataset
for ticker in tickers: 
    data = df[df.Name.eq(ticker)]
    dates = df['Date'][:len(data)]
    adj_dates = mdates.datestr2num(dates)
    plt.plot_date(adj_dates, data['Close'], '-')
    
plt.title('Closing Value vs. Date for 30 Top DJIA Equities')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close value', fontsize=16)
plt.show()

# Start with one stock
stock = 'AMZN'
metric = 'Close'
data = df[df['Name'] == stock][metric]

# Choose metric to predict
# input('Please enter the metric you want to predict: ')
# metric = metric.capitalize()

# Check for, and handle, missing values 
if data.isna().sum():
    pass
    # do something to handle empty data rows 

# Normalization is not ideal since closing values trend upwards. 
# Standardization is not appropriate since data does not approximate a Gaussian

# Convert data into format acceptable to Keras
# Input must be 3D ndarray of shape (samples x time steps x features)
test_fraction = 0.80
data = data.to_numpy()

# Define data
x_train = 0;
y_train = 0;

# Build model 
units = 32
epochs = 25
batch_size = 15

model = Sequential()
model.add(LSTM(
    units = units, 
    return_sequences = True,
    ))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam',
              loss = 'mean_squared_error')

model.fit(x_train,
          y_train,
          epochs = epochs, 
          batch_size = batch_size
          )

# Reshape data 

# Build model 

# Evaluate performance 
