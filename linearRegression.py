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
from talib import EMA, RSI, OBV
from sklearn.ensemble import RandomForestRegressor



from sklearn.model_selection import train_test_split # for the initial split to a train set and a untouched test set 
from sklearn.model_selection import TimeSeriesSplit # for roll forward cross vallidation
from sklearn import metrics
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def callLinearRegression (seriesAvg, dataframe,dataTimestamp):

   # df = dataframe.copy()
   # XTimes =np.array(dataTimestamp).reshape((-1, 1))
   # yAvgPrices = seriesAvg  

   # train_size = int(0.85 * yAvgPrices.shape[0])
   # X_train = XTimes[:train_size]
   # y_train = yAvgPrices[:train_size]
   # X_test = XTimes[train_size:]
   # y_test = yAvgPrices[train_size:]


   ema_short_period=10
   ema_long_period=30
   rsi_period=14

   df=pd.read_csv("C:/Users/klio_ks/Downloads/Christos-Gklinavos-EPAT-Project\Chrsitos 1/forex.csv", index_col=0, parse_dates=True)
   df.head()
   df.columns=["open","high","low", "close", "volume"]
   df=df.loc['2016-12-31':'2018-12-31']
   df=df.dropna()
   df.head()
   df.shape
   
   
   df["ema_short"]=pd.DataFrame(df["close"]).apply(lambda row: EMA(row,timeperiod=ema_short_period))
   df["ema_long"]=pd.DataFrame(df["close"]).apply(lambda row: EMA(row,timeperiod=ema_long_period))
   df["rsi"]=pd.DataFrame(df["close"]).apply(lambda row: RSI(row,timeperiod=rsi_period))
   # df['obv'] =pd.DataFrame(df.apply(lambda row: OBV(row["close"], row["volume"])), axis=1)
   df.tail()
   

   df["close_nextday"]=df.close.shift(-1) # used as target prediction
   
   df.dropna(inplace=True)
   df.head()
   
   
   df1=df
   
   predictors=['open', 'high', 'low', 'close', 'volume', 'ema_short', 'ema_long', 'rsi']
   
   prediction=['close_nextday']
   x_train_df, x_test_df, y_train_df, y_test_df=train_test_split(df1[predictors], df1[prediction], test_size=0.3)
   x_train_df.head()
   y_train_df.head()
   x_test_df.head()
   y_test_df.head()
   x=x_train_df.values # selected features
   y=y_train_df.values.ravel()# labels
   y=np.round(y, 4)
   sc=StandardScaler(copy=True, with_mean=True, with_std=True)
   x_normal=np.round(sc.fit_transform(x), 4)
   tss = TimeSeriesSplit(n_splits=5)
   
   print(tss)
   model_score = []
   model=RandomForestRegressor(n_estimators=20, max_features="sqrt", max_depth=10,random_state=1)
   for train_index, test_index in tss.split(x):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model=model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
        model_score.append(metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average' ))   
        print('Model score= ', model_score)
        print('Importance of predictors: ', model.feature_importances_)
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
       
   np.isinf(y_test).sum(),np.isnan(y_test).sum()
   y_test=pd.Series(y_test)
   (y_test[np.isnan(y_test)])    
   len(y_test)
   points=range(y_test.shape[0])
   print(points)
   print(y_test.shape[0])
   
   plt.plot(points, y_test, color='blue', label='y_test')
   plt.plot(points, y_pred, color='red', label='y_pred')
   plt.ylabel('close prices')
   plt.xlabel('days')
   plt.title('Predicted Prices vs Actual Prices on test set')
   plt.legend(loc='upper left')
       
   plt.show()
   x_test_df['price_prediction'] = model.predict(x_test_df.loc[:,predictors])
       
   x_test_df['model_signal'] = np.where(x_test_df.price_prediction>x_test_df.close, 1, -1)
   def compute_ret(df): 
        # ret is daily returns
        df['ret'] = df.close.shift(-1).pct_change()
        df['cum_ret'] = np.cumsum(df.ret)
        df['strategy_ret'] = df.ret * df.model_signal.shift(1)
        df['cum_strategy_ret'] = np.cumsum(df.strategy_ret)
        return df
   df2=compute_ret(x_test_df)
   df2.head()
   df2['cum_ret'].plot(figsize=(8,4))
   df2['cum_strategy_ret'].plot(figsize=(8,4))
   plt.legend(loc='upper right')
   plt.show()
   def sharpe(df):    
        sharpe_ratio = df.strategy_ret.mean()/df.strategy_ret.std()*np.sqrt(252)
        return sharpe_ratio
   print(sharpe(df2))
   plt.plot(range(len(y_test_df['close_nextday'])), y_test_df['close_nextday'], color='blue', label='Actual Price')
   plt.plot(range(len(y_test_df['close_nextday'])), x_test_df['price_prediction'], color='red', label='Predicted Price')
   plt.ylabel('close prices')
   plt.xlabel('days')
   plt.title('Predicted Prices vs Actual Prices on untouched set')
   plt.legend(loc='upper left')
   plt.show()
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   # df['ema_short']=pd.DataFrame(df['EndPrice']).apply(lambda row: EMA(row,timeperiod=ema_short_period))
   # df["ema_long"]=pd.DataFrame(df['EndPrice']).apply(lambda row: EMA(row,timeperiod=ema_long_period))
   # df["rsi"]=pd.DataFrame(df['EndPrice']).apply(lambda row: RSI(row,timeperiod=rsi_period))
   # # df['obv']=df.apply(lambda row: OBV(row['EndPrice'], row['addedVolume']), axis=1)

   # df.dropna(inplace=True)

   # modelRegression = LinearRegression()
   # modelRegression.fit(XTimes, yAvgPrices)

  
   # y_predict = modelRegression.predict(X_test)
   
  
   # plt.plot(yAvgPrices.to_numpy())
   # plt.plot(y_predict)
   # plt.show()
   
   
   # train_size = int(0.85 * yAvgPrices.shape[0])
   # X_train = XTimes[:train_size]
   # y_train = yAvgPrices[:train_size]
   # X_test = XTimes[train_size:]
   # y_test = yAvgPrices[train_size:]
   
   # modelRegressionRandomForest = RandomForestRegressor(n_estimators=20, max_features="sqrt", max_depth=10,random_state=1)
   # modelRegressionRandomForest.fit(X_train, y_train)

  
   
   # y_predict = modelRegressionRandomForest.predict(X_test)
  
   

   # plt.plot(y_test.to_numpy(), 'b')
   # plt.plot(y_predict, 'r')
   # plt.show()