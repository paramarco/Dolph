# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:20:45 2020

@author: mvereda
"""

import os
import logging
import pandas as pd
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
log = logging.getLogger("TradingPlatform")

class NARemover:
    def __init__(self, name):
        self.name = name
    def transform(self, single_stock):
        before = single_stock.shape[0]
        single_stock = single_stock.dropna()
        after = single_stock.shape[0]
        print("{}: Dropped {:2.2f} % of records due to NA".format(self.name, 100.0*(before - after)/(0.0001 + before)))
        return single_stock

class Featurizer:
    
    def __init__(self, target):
        # target := "training" | "prediction"
        self.target = target
        
    def transform(self, single_stock, max_offset = 30):

        predictionWindow = 5
        
        single_stock['x(DOW)'] = single_stock['CalcDateTime'].dt.dayofweek
        single_stock['x(Hour)'] = single_stock['CalcDateTime'].dt.hour
        # single_stock['x(DOY)'] = single_stock['CalcDateTime'].dt.dayofyear
       
        
        for offset in range(1, max_offset ):
            
            single_stock['x(open(t-{})'.format(str(offset))] = single_stock['StartPrice'] - single_stock['StartPrice'].shift(offset) 
            single_stock['x(high(t-{})'.format(str(offset))] = single_stock['MaxPrice'] - single_stock['MaxPrice'].shift(offset)   
            single_stock['x(close(t-{})'.format(str(offset))] = single_stock['EndPrice'] - single_stock['EndPrice'].shift(offset)   
            single_stock['x(low(t-{})'.format(str(offset))] = single_stock['MinPrice'] - single_stock['MinPrice'].shift(offset)   
            single_stock['x(vol(t-{})'.format(str(offset))] = single_stock['addedVolume'] - single_stock['addedVolume'].shift(offset)   

        if (self.target  == "training"):      
            for offset in range(1, predictionWindow ):
                single_stock['y(open(t+{})'.format(str(offset))] =  single_stock['StartPrice'].shift(-offset) - single_stock['StartPrice']
                single_stock['y(high(t+{})'.format(str(offset))] = single_stock['MaxPrice'].shift(-offset)    - single_stock['MaxPrice'] 
                single_stock['y(close(t+{})'.format(str(offset))] = single_stock['EndPrice'].shift(-offset)   - single_stock['EndPrice'] 
                single_stock['y(low(t+{})'.format(str(offset))] =   single_stock['MinPrice'].shift(-offset)   - single_stock['MinPrice']
            
        return single_stock

class TrainingSet:
    def __init__(self, X, y, orig_df):
        self.X = X
        self.y = y
        self.original_df = orig_df
        
class TrainingSetBuilder:
    def transform(self, single_stock):
        x_features = filter(lambda name: name.startswith('x('), list(single_stock.dtypes.index))
        X = single_stock[x_features].values
        
        y_features = filter(lambda name: name.startswith('y('), list(single_stock.dtypes.index))
        y = single_stock[y_features].values        
        
        return TrainingSet(X, y, single_stock)
    
    
class Predictions:
    def __init__(self, predictions, training_set):
        self.predictions = predictions
        self.training_set = training_set       
    
class MLModel:
    def __init__(self, df, params, period):
        self.model = None
        self.period = period
        # month = datetime.now().strftime('%h') 
        self.fileName =  __name__ + "_" + self.period 
        
        if ( os.path.isdir( self.fileName ) ):
            log.info('pre-trainned model found! loading it ...')
            self.model = load_model( self.fileName )            
        else:        
            self.loadData(df)
        
    def loadData(self, df):        
     
        all_mnemonics = df.Mnemonic.unique()
        
        def date_part(dt):
            return str(dt).split(' ')[0]
        unique_days = sorted(list(set(map(date_part , list(df.index.unique())))))

        percent_train = 80.0
        percent_valid = 20.0
        # percent_test = 100.0 - percent_train - percent_valid
        
        offset_train = int(len(unique_days)*percent_train/100.0)
        offset_test = offset_train + int(len(unique_days)*percent_valid/100.0)
        
        train_valid_days = list(set(unique_days[0:offset_test]))
        
        np.random.seed(484811945)
        # np.random.shuffle(train_valid_days)
        
        train_days = train_valid_days[0:offset_train]
        valid_days = train_valid_days[offset_train:]
        
        
        df['CalcDateTime'] = df.index
        df['Date'] = df['CalcDateTime'].dt.strftime("%Y-%m-%d")
        
        df_train = df[ df.Date.isin(list(train_days)) ]
        df_valid = df[ df.Date.isin(list(valid_days)) ]
        
        combined_training_set = []
        combined_valid_set = []
        
        for mnemonic in all_mnemonics: 
        
            single_stock = df_train[df_train.Mnemonic == mnemonic].copy()
            single_stock = single_stock[single_stock.HasTrade == 1.0]
            single_stock = Featurizer("training").transform(single_stock)
            single_stock = NARemover(mnemonic).transform(single_stock)
            combined_training_set.append(single_stock)
            log.info(single_stock.shape)
        
            single_stock = df_valid[df_valid.Mnemonic == mnemonic].copy()
            single_stock = single_stock[single_stock.HasTrade == 1.0] 
            single_stock = Featurizer("training").transform(single_stock)
            single_stock = NARemover(mnemonic).transform(single_stock)
            combined_valid_set.append(single_stock)            
            
            
        combined_training_set_df = pd.concat(combined_training_set, axis=0)
        training_set = TrainingSetBuilder().transform(combined_training_set_df)
            
        combined_valid_set_df = pd.concat(combined_valid_set, axis=0)
        valid_set = TrainingSetBuilder().transform(combined_valid_set_df) 
                
        log.info('Trainning Machine....' )        
        self.fit(training_set, valid_set)        

        
    def fit(self, training_set, valid_set = None):
        train_X, train_y = training_set.X, training_set.y
        
        if valid_set is None:
            valid_X, valid_y = train_X, train_y
        else:
            valid_X, valid_y = valid_set.X, valid_set.y

        model = Sequential()

        model.add(  
            Dense(1024, 
                  activation='tanh', 
                  input_shape =(train_X.shape[1],) , 
                  kernel_regularizer=regularizers.l2(0.001)
                  )
        ) 
        model.add(
            Dense(512, 
                  activation='tanh',
                  input_shape =(train_X.shape[1],) ,
                  kernel_regularizer=regularizers.l2(0.001)
                  )
        )
        model.add(
            Dense(256, 
                  activation='tanh',
                  input_shape =(train_X.shape[1],) ,
                  kernel_regularizer=regularizers.l2(0.001)
                  )
        ) 
        model.add(
            Dense(128, 
                  activation='tanh',
                  input_shape =(train_X.shape[1],) ,
                  kernel_regularizer=regularizers.l2(0.001)
                  )
        )
        model.add(Dense(train_y.shape[1]))
        
        loss_fn = tf.keras.losses.MeanSquaredError(reduction='sum')
        
        # model.compile(loss='mean_squared_error', optimizer='adam')
        opt = optimizers.Adam(learning_rate=0.000001)
        model.compile(loss=loss_fn, optimizer=opt)
        # model.compile(loss=loss_fn,  optimizer='adam')
        self.model = model            

        # fit network
        history = model.fit(train_X, train_y, epochs=5, batch_size=5, validation_data=(valid_X, valid_y), verbose=2, shuffle=False)
       
        # save network 
        model.save(self.fileName)
        
        # plot history
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='valid')
        pyplot.legend()
        pyplot.show()
        
    def transform(self, input_set):
        predictions = self.model.predict(input_set.X)
        return Predictions(predictions, input_set)
    
    def fit_transform(self, training_set, valid_set):
        self.fit(training_set, valid_set)
        return self.transform(training_set), self.transform(valid_set)

    def predict(self, df ):
        
        all_mnemonics = df.Mnemonic.unique()
        
        df['CalcDateTime'] = df.index
        df['Date'] = df['CalcDateTime'].dt.strftime("%Y-%m-%d")
        
        combined_data_set = []
        
        for mnemonic in all_mnemonics: 
        
            single_stock = df[df.Mnemonic == mnemonic].copy()
            single_stock = single_stock[single_stock.HasTrade == 1.0]
            single_stock = Featurizer("prediction").transform(single_stock)
            single_stock = NARemover(mnemonic).transform(single_stock)
            single_stock = single_stock.tail(1)
            combined_data_set.append(single_stock)
            
        combined_data_set_df = pd.concat(combined_data_set, axis=0)
        data_set = TrainingSetBuilder().transform(combined_data_set_df)
        
        predictions = self.model.predict(data_set.X)
        myNewPred = Predictions(predictions, data_set)
        return myNewPred
