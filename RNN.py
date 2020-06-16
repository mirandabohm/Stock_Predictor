# -*- coding: utf-8 -*-
# Created on Mon Jun 15 17:18:52 2020
# @author: Upquark00

import os
import tensorflow as tf
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from time import strptime

# Load data 
file = r'all_stocks_2006-01-01_to_2018-01-01' + '.csv'
dirname = os.path.dirname(__file__)
filename = os.path.join((dirname), 'data/' +file)

df = pd.read_csv(filename)
tickers = df.Name.unique().tolist()

# Start with one stock
apple = tickers[2]

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

# Process data 

# Reshape data 

# Build model 

# Evaluate performance 
