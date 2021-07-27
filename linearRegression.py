# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:35:20 2021

@author: klio_ks
"""

from sklearn.linear_model import LinearRegression

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
def callLinearRegression (seriesAvg, data_df_hour, data_df_min):

   # data_df=data_df.map(dt.datetime.toordinal)
   # data_df=data_df.apply(lambda x: time.mktime(x.timetuple()))
   # print(data_df)
   XTimes =np.array(data_df_min).reshape((-1, 1))
   # XTimes = np.arange(numWindowSize).reshape((-1, 1))  
   yAvgPrices = seriesAvg  

   modelRegression = LinearRegression()
   modelRegression.fit(XTimes, yAvgPrices)

  
   X_predict=XTimes
   y_predict = modelRegression.predict(X_predict)
   
  
   plt.plot(yAvgPrices.to_numpy())
   plt.plot(y_predict)
   plt.show()
   
   
   train_size = int(0.85 * yAvgPrices.shape[0])
   X_train = X_predict[:train_size]
   y_train = yAvgPrices[:train_size]
   X_test = X_predict[train_size:]
   y_test = yAvgPrices[train_size:]
   
   modelRegressionRandomForest = RandomForestRegressor(n_estimators=5000, oob_score=True, random_state=100)
   modelRegressionRandomForest.fit(X_train, y_train)

  
   # X_predict=XTimes
   # X_predict = np.array([25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]).reshape((-1, 1)) 
   
   y_predict = modelRegressionRandomForest.predict(X_test)
  
   
  

   plt.plot(y_test.to_numpy(), 'b')
   plt.plot(y_predict, 'r')
   plt.show()