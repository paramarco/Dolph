# -*- coding: utf-8 -*-
"""
Created on Wed Aug 05 12:31:22 2020

@author: mvereda
"""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv1D, MaxPooling1D, BatchNormalization, LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
# tf.compat.v1.disable_eager_execution()
import gc; gc.collect()
import pandas as pd
from pandas import concat
from datetime import timedelta
from matplotlib import pyplot
    
numInputWith = 10
numEpochs = 25
batch_size = 50
units = 15
numOffset = 1
learningRate = 0.001
activationFunc = 'relu'

class Model:
    def __init__( self,  df , params, period):
        
        self.model = None
        self.period = period
        self.fileName =  __name__ + "_" + self.period 
        
        self.scaler_X = None 
        self.scaler_y = None
        
        self.df = None
        self.df_scaled = None
        self.df_Window = None
        
        if ( os.path.isdir( self.fileName ) ):
            print ('pre-trainned model found! loading it ...')
            self.model = load_model( self.fileName )                        
            self.scaler_X = joblib.load(self.fileName + "/scaler_X.dmp" ) 
            self.scaler_y = joblib.load(self.fileName + "/scaler_y.dmp" ) 
        else:        
            self.loadData(df)
        
        
        
    # convert series to LSTMNN input format
    # pre-condicition: the 1st coloumn ist the target for forcasting 
    # that means the first security in the list of securities will be
    # the target for predictions
    def dataWindowing( self, data, n_in=10, n_out=0, dropnan=True):
    	
        df = data.iloc[:,1:]
        n_vars = df.shape[1]
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
        df = data
        n_vars = df.shape[1]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))           
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        X = concat(cols, axis=1)
        X.columns = names  
        
        X.iloc[:, -n_vars  ] = X.iloc[:, -n_vars  ].diff()
        # drop rows with NaN values
        if dropnan:
            X.dropna(inplace=True)
            
        Y =  X.iloc[:, -n_vars  ] 
        X =  X.drop( X.columns[-n_vars], axis='columns'  )        
        
        return X, Y
    
    def x_inputWindowing( self, data, n_in=10, n_out=0, dropnan=True):
    	
        n_in = n_in - n_out
        
        df = data
        n_vars = df.shape[1]
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]        

        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))           
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        X = concat(cols, axis=1)
        X.columns = names  
        # drop rows with NaN values
        if dropnan:
            X.dropna(inplace=True)
            
        return X    
    
    def createFeatures(self,df):
        self.dates = df.index.tolist()
         
        # close_fft = np.fft.fft(np.asarray(df.iloc[:,1].tolist()))
        # df["fft_ang_CLOSE"] = np.angle(close_fft)
        # df["fft_abs_CLOSE"] = np.abs(close_fft)
        # df["dif_sqr_CLOSE"] = df.iloc[:,0].diff().apply(lambda x: np.power(x,2))
        # df["dif_sqr_OPEN"] = df.iloc[:,1].diff().apply(lambda x: np.power(x,2))
        # df["dif_sqr_HIGH"] = df.iloc[:,2].diff().apply(lambda x: np.power(x,2))
        # df["dif_sqr_LOW"] = df.iloc[:,3].diff().apply(lambda x: np.power(x,2))
        # df = df.drop(
        #     [df.columns[1], df.columns[2], df.columns[3] ] ,
        #     axis='columns'
        # ) 
        
        # minus 1 cus 1st is the CLOSE of the target to predict 
        numFeatures = df.shape[1] - 1
        
        return numFeatures, df
    
    def splitTrainTestData(self, X_values, y_values , numFeatures):
        
        # split into train and test sets
        train_end = int(np.floor(0.8*X_values.shape[0]))
        
        train_X = X_values[ : train_end, : -numFeatures]
        test_X =  X_values[ train_end :, : -numFeatures]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], numInputWith, numFeatures))
        test_X = test_X.reshape((test_X.shape[0], numInputWith, numFeatures))
        
        train_y = y_values[: train_end ]
        test_y =  y_values[ train_end :]        
        
        return train_X, test_X, train_y, test_y
        
        
    def loadData(self, df):
        
        numFeatures, df = self.createFeatures(df) 
 
        df_X_window, df_y_window = self.dataWindowing( 
            df, 
            numInputWith, 
            numOffset 
        )
        
        self.scaler_X = MinMaxScaler( feature_range=(0, 1) ).fit( df_X_window.values.astype('float64') )
        X_scaled = self.scaler_X.transform( df_X_window.values.astype('float64') )
       
        self.scaler_y = MinMaxScaler( feature_range=(0, 1) ).fit( df_y_window.values.astype('float64').reshape(-1, 1) )
        y_scaled = self.scaler_y.transform( df_y_window.values.astype('float64').reshape(-1, 1))
        
        
        train_X, test_X, train_y, test_y = self.splitTrainTestData(
            X_scaled, y_scaled , numFeatures
        )
        
        # build a LSTM neuronal net
        self.model = keras.Sequential()
        self.model.add(keras.layers.LSTM(
          activation=activationFunc ,
          units=units,
          input_shape=(train_X.shape[1], train_X.shape[2])
          # ,return_sequences=True
        ))
        # self.model.add(keras.layers.LSTM(
        #   activation=activationFunc ,
        #   units=units,
        #   input_shape=(train_X.shape[1], train_X.shape[2]),
        #   return_sequences=True
        # ))
        # self.model.add(keras.layers.LSTM(
        #   activation=activationFunc ,
        #   units=units,
        #   input_shape=(train_X.shape[1], train_X.shape[2])
          
        # ))
        self.model.add(keras.layers.Dense(units=1))
        self.model.compile(
          loss='mean_squared_error',
          optimizer=keras.optimizers.Adam(learningRate)
        )
        self.model.summary()
        
        # train the LSTM neuronal net
        history = self.model.fit(
            train_X, train_y,
            epochs=numEpochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
            shuffle=False
        )
        
        # save network
        self.model.save(self.fileName)
        # save Scaler
        joblib.dump(self.scaler_X, self.fileName + "/scaler_X.dmp" ) 
        joblib.dump(self.scaler_y, self.fileName + "/scaler_y.dmp" )         


        # plot history
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

    def predict( _ , df ):       
       
        numFeatures, df = _.createFeatures(df) 
        
        lastClosePrice = df.iloc[-1,0]
        df_meta = df.iloc[:,1:]
       
        df_X = _.x_inputWindowing( df_meta, numInputWith, numOffset)
        
        values = df_X.values.astype('float64')        
        scaler_X_test = MinMaxScaler( feature_range=(0, 1) ).fit( values )
        X_scaled = scaler_X_test.transform( values )        
        
        
        X = X_scaled.reshape((X_scaled.shape[0], numInputWith, numFeatures))
        print( X.shape )
        
        X = X[-1,:,:].reshape((1, numInputWith, numFeatures)) 
        
        print( X.shape )
        
        pred_y = _.model.predict(X)   
        
        # invert scaling for forecast                    
        data2inverse = np.full( (1, 1), 0.)
        data2inverse[0,0] = pred_y   
        predInOriginalScale =  _.scaler_y.inverse_transform( data2inverse )
        
        # data2inverse = np.full( (1, numFeatures), 0.)
        # data2inverse[0,0] = X[0,-1,0]      
        # lastClosePrice =  _.scaler.inverse_transform( data2inverse )        
        
        prediction = {}
        prediction['predLastData'] = predInOriginalScale[0,0]
        prediction['lastClosePrice'] = lastClosePrice
        prediction['predWindow'] = _.period
        prediction['algorithm'] = __name__ 
        
        return prediction
    