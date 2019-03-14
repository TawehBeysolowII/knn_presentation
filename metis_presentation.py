#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 01:09:04 2019

@author: tawehbeysolow
"""

import pandas as pan, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pandas_datareader import data
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
    
np.random.seed(2018)
    
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
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.20)

    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors).fit(train_X, train_y)
    predicted_labels = knn_model.predict(test_X)
  
    print('KNN Accuracy w/ %s Neighbors: %s'%(n_neighbors, accuracy_score(test_y, predicted_labels)))
    print('Confusion Matrix:\n %s'%(confusion_matrix(test_y, predicted_labels)))
    
    
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
    _mean_squared_error = []
    
    for i in range(1000):
        train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.20)
        test_y = test_y.reset_index(drop='index')
        knn_model = KNeighborsRegressor(n_neighbors=n_neighbors).fit(train_X, train_y)
        predicted_labels = knn_model.predict(test_X)
        _mean_squared_error.append(mean_squared_error(test_y, predicted_labels))
        
    prediction_difference = pan.DataFrame([test_y[i] - predicted_labels[i] for i in range(0, len(predicted_labels))])
    price_percentage_changes = [test_y[i]/float(test_y[i-1]) - 1 for i in range(1, len(test_y))]
    
    plt.plot(predicted_labels, label='predicted price')
    plt.plot(test_y, label='actual price')
    plt.title('Predicted vs Actual Price')
    plt.xlabel('N Days')
    plt.ylabel('Price')
    plt.legend(loc='upper right')
    plt.show()
        
    print('\nSummary Statistics on Average Mean Squared Error: %s'%pan.DataFrame(_mean_squared_error).describe())  
    print('\nSummary Statistics on AAPL Close Price:\n %s'%pan.DataFrame(test_y).describe())
    print('\nSummary Statistics on AAPL Close Price Percentage Changes:\n %s'%pan.DataFrame(price_percentage_changes).describe())
    print('\nSummary Statistics on Prediction Difference: %s'%prediction_difference.describe())
    sns.distplot(prediction_difference, bins=10).set_title('Predicted Minus Actual Stock Price Distribution')
    plt.show()
    
def visualize_iris_data():
    '''
    Visualize iris data    
    
    :return: None
    '''
    raw_iris_data = datasets.load_iris()
    iris_data = pan.DataFrame(raw_iris_data.data[:, :4], columns=raw_iris_data.feature_names)
    
    for start, end in zip([0, 2], [2, 4]):
        
        X = iris_data.ix[:, start:end]  
        y = raw_iris_data.target
        
        plt.scatter(X.ix[:, 0], X.ix[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
        plt.title('Scatter Plot of Iris Data Features')
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        plt.show()
        
def visualize_stock_data(ticker_symbol, start_date, end_date, data_source='yahoo'):
    '''
    Visualize closing price of stock data    
    
    :return None
    '''
    stock_data = data.DataReader(name=ticker_symbol,
                                 start=start_date,
                                 end=end_date,
                                 data_source='yahoo')
        
    plt.plot(stock_data.Close)
    plt.title('Closing Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.show()
    
if __name__ == '__main__':
    
    visualize_iris_data()
    
    visualize_stock_data(ticker_symbol='AAPL', 
                       start_date='2017-01-01', 
                       end_date='2019-01-01')
    
    model_iris_data(n_neighbors=3)
    
    model_stock_data(n_neighbors=4,
                     ticker_symbol='AAPL', 
                     start_date='2017-01-01', 
                     end_date='2019-01-01')
