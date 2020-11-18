# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:27:29 2020
@author: upquark00
"""

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import date

yf.pdr_override()

ticker_list = ['DJIA']
today = date.today()
start_date = '1985-01-28'
files = []

def save_data(df, filename):
    df.to_csv('./data/' + filename + '.csv')

def get_data(ticker_symbol):
    print('Ticker:', ticker_symbol)
    data = pdr.get_data_yahoo(ticker_symbol, start = start_date, end = today)
    dataname = ticker_symbol + '_all_time'
    files.append(dataname)
    save_data(data, dataname)

def fetch_data(ticker_list):
    for ticker in ticker_list:
        get_data(ticker)
        
    for i in range(len(files)):
        dataframe = pd.read_csv('./data/' + str(files[i]) + '.csv')
    return dataframe

df = fetch_data(ticker_list)
print (df.head())