#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 01:09:04 2019

@author: tawehbeysolow
"""

import pandas as pan, matplotlib.pyplot as plt
from pandas_datareader import data
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
    

def model_iris_data(n_neighbors):
    '''
    This function loads the iris data from the sklearn datasets class, models this data set using KNN, 
    and then displays performance data on the algorithm (Classification)
    
    Arguments:
    
        n_neighbors - int - the number of neighbors to an observation KNN uses to output a discrete or continuous label
        
    '''

    raw_iris_data = datasets.load_iris()
    print('Data Set Shape: %s,%s'%(pan.DataFrame(raw_iris_data.data).shape))
    print('\nDescription:\n %s'%pan.DataFrame(raw_iris_data.data).describe())        
    print('\nCorrelation Coefficient Matrix:\n %s'%(pan.DataFrame(raw_iris_data.data).corr()))

    x = pan.DataFrame(raw_iris_data.data[:, :4], columns=raw_iris_data.feature_names)
    y = pan.DataFrame(raw_iris_data.target)

    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors).fit(x, y)
    predicted_labels = knn_model.predict(x)
    in_sample_accuracy = accuracy_score(y, predicted_labels)    
    print('KNN Accuracy w/ %s Neighbors: %s'%(n_neighbors, in_sample_accuracy))
    print('Confusion Matrix:\n %s'%(confusion_matrix(y, predicted_labels)))
    
    
def model_stock_data(n_neighbors, ticker_symbol, start_date, end_date, data_source='yahoo'):
   '''
    This function loads the stock data from yahoo finance, models this data set using KNN, 
    and then displays performance data on the algorithm (Regression). Time series is sampled daily
    
    Arguments:
    
        n_neighbors - int - the number of neighbors to an observation KNN uses to output a discrete or continuous label
        ticker_symbol - str - the SPY ticker of the stock that is being modeled 
        start_date - str - initial sampling date of the time series (Format - YYYY-MM-DD)
        end_date - str - final sampling date of the time series (Format - YYYY-MM-DD)
        data_source - str - the data source from which the data is pulled (default: 'yahoo')
    '''
     
   stock_data = data.DataReader(name=ticker_symbol, 
                                 start=start_date, 
                                 end=end_date,
                                 data_source=data_source)

   x = stock_data.shift(1).dropna().reset_index(drop='index')
   y = stock_data.Close.shift(-1).dropna().reset_index(drop='index')

   knn_model = KNeighborsRegressor(n_neighbors=5).fit(x, y)
   predicted_labels = knn_model.predict(x)

   plt.plot(predicted_labels, label='predicted price')
   plt.plot(y, label='actual price')
   plt.xlabel('N Days')
   plt.ylabel('Price')
   plt.legend(loc='upper right')
   plt.show()
    
   print('KNN Mean Squared Error: %s'%mean_squared_error(y, predicted_labels))

        
if __name__ == '__main__':
    
    
    model_iris_data(n_neighbors=3)
    
    model_stock_data(n_neighbors=4,
                     name='AAPL', 
                     start='2017-01-01', 
                     end='2019-01-01',
                     data_source='yahoo')
