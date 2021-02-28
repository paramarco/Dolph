# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:20:45 2020

@author: mvereda
"""

import os
import math
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
from keras.callbacks import LearningRateScheduler
class NARemover:
    def __init__(self, name):
        self.name = name
    def transform(self, single_stock):
        before = single_stock.shape[0]
        single_stock = single_stock.dropna()
        after = single_stock.shape[0]
        m = "{}: Dropped {:2.2f} % of records due to NA"
        print(m.format(self.name, 100.0*(before - after)/(0.0001 + before)))
        return single_stock

class Featurizer:
    
    def __init__(self, target, numPastSamples):
        # target := "training" | "prediction"
        self.target = target
        self.numPastSamples = numPastSamples
        
    def transform(self, sec):

        predictionWindow = 2
        
        # sec['x(DOW)'] = sec['CalcDateTime'].dt.dayofweek
        sec['x(Hour)'] = sec['CalcDateTime'].dt.hour
        end_price_fixed = sec['EndPrice'].shift(1)
        steps = 5
        stds = 0.000001 + 0.9 * end_price_fixed.rolling(steps).std() + 0.1*end_price_fixed.rolling(10).std()
        delta=1
        

        # close_fft = np.fft.fft(np.asarray(sec['EndPrice'].tolist()))
        # fft_df = pd.DataFrame({'fft':close_fft})
        # fft_list = np.asarray(fft_df['fft'].tolist())
        
        # numRows = sec.shape[0]
        
        # for f in [0.1, 0.2, 0.4]:
        #     c = math.ceil( numRows * f )
        #     fft_list_component = np.copy(fft_list)
        #     fft_list_component[ c : -c ] = 0
        #     ifft = np.fft.ifft(fft_list_component)
            
        #     j = 'x(fft_{}_mag)'.format(str(c))
        #     sec[j] = np.abs(ifft)
            
            # j = 'x(fft_{}_ang)'.format(str(c))
            # sec[j] = np.angle(ifft)        

        
        for offset in range(1, self.numPastSamples ):
            
            # j = 'x(mult(t-{})'.format(str(offset))
            # sec[j] =    (sec['MaxPrice'] - sec['MaxPrice'].shift(offset)) * \
            #             (sec['MinPrice'] - sec['MinPrice'].shift(offset))    
 
            # j = 'x(multEnd(t-{})'.format(str(offset))
            # sec[j] =    (sec['EndPrice'] - sec['StartPrice']) * \
            #             (sec['EndPrice'] - sec['StartPrice'].shift(offset) )  #            ) 
            # isLong =    (sec['EndPrice'].shift(offset-1) - sec['StartPrice'].shift(offset-1) >= 0)
            # isShort =   (sec['EndPrice'].shift(offset-1) - sec['StartPrice'].shift(offset-1) < 0)
            # sec.loc[ isLong, 'coef'] = sec['MaxPrice'].shift(offset-1) - sec['EndPrice'].shift(offset-1)  
            # sec.loc[ isShort,'coef'] = sec['EndPrice'].shift(offset-1) - sec['MinPrice'].shift(offset-1)  
            
            # j = 'x(dir(t-{})'.format(str(offset-1))
            # sec[j] = (sec['EndPrice'].shift(offset-1) - sec['StartPrice'].shift(offset-1)) * sec['addedVolume'].shift(offset-1) * sec['coef']
            
            # j = 'diffVol'.format(str(offset))
            # sec[j] =    sec['addedVolume'].shift(offset-1) - sec['addedVolume'].shift(offset)
            j = 'x(high(t-{})'.format(str(offset))
            sec[j] =  (sec['MaxPrice'].shift(offset-1) - sec['MaxPrice'].shift(offset))            
            j = 'x(low(t-{})'.format(str(offset))
            sec[j] =   ( sec['MinPrice'].shift(offset-1) - sec['MinPrice'].shift(offset)) 
            
            j = 'x(open(t-{})'.format(str(offset))
            sec[j] =    (sec['StartPrice'].shift(offset-1) - sec['StartPrice'].shift(offset))
            
            j = 'x(close(t-{})'.format(str(offset))
            sec[j] =    (sec['EndPrice'].shift(offset-1) - sec['EndPrice'].shift(offset))
            
            j = 'x(Volatility(t-{})'.format(str(offset))
            sec[j] = 100 * ( sec['MaxPrice'].shift(offset) - sec['MinPrice'].shift(offset) ) / sec['MinPrice'].shift(offset)
            
            j = 'x(PercentageChange(t-{})'.format(str(offset))
            sec[j] = 100 * ( sec['EndPrice'].shift(offset) - sec['StartPrice'].shift(offset) ) / sec['EndPrice'].shift(offset)
            
          
            
           

        if (self.target  == "training"):
            for offset in range(1, predictionWindow ):

                
                # j = 'y(high(t+{})'.format(str(offset))
                # sec[j] = sec['MaxPrice'].shift(-offset) - sec['EndPrice']
                
                # j = 'y(low(t+{})'.format(str(offset))
                # sec[j] = sec['MinPrice'].shift(-offset) - sec['EndPrice']
                
                j = 'y(close(t+{})'.format(str(offset))
                sec[j] = sec['EndPrice'].shift(-offset) - sec['EndPrice']
                
                            
        return sec

class TrainingSet:
    def __init__(self, X, y, orig_df):
        self.X = X
        self.y = y
        self.original_df = orig_df
        
class TrainingSetBuilder:
    def transform(self, single_stock):
        x_features = filter(
            lambda name: name.startswith('x('), list(single_stock.dtypes.index)
        )
        X = single_stock[x_features].values
        
        y_features = filter(
            lambda name: name.startswith('y('), list(single_stock.dtypes.index)
        )
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
        self.params = params        
        samples = str(params['minNumPastSamples'])  +'Samples'
        nameParts=[ __name__, self.period, samples]
        self.fileName = '_'.join(nameParts)
        
        if ( os.path.isdir( self.fileName ) ):
            log.info('pre-trainned model found! loading it ...')
            self.model = load_model( self.fileName )            
        else:        
            self.loadData(df)
        
    def loadData(self, df):        
     
        all_mnemonics = df.Mnemonic.unique()
        
        def date_part(dt):
            return str(dt).split(' ')[0]
        unique_days = sorted(
            list(
                set(
                    map(date_part , list( df.index.unique() ) )
                )
            )
        )

        percent_train = 80.0
        percent_valid = 20.0
        
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
            single_stock = Featurizer(
                "training", self.params['minNumPastSamples']
            ).transform(single_stock)
            single_stock = NARemover(mnemonic).transform(single_stock)
            combined_training_set.append(single_stock)
            log.info(single_stock.shape)
        
            single_stock = df_valid[df_valid.Mnemonic == mnemonic].copy()
            single_stock = single_stock[single_stock.HasTrade == 1.0] 
            single_stock = Featurizer(
                "training", self.params['minNumPastSamples']
            ).transform(single_stock)
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
            Dense(
                64, activation='selu', input_shape =(train_X.shape[1],)
                , bias_regularizer=regularizers.l2(1e-2)
                , kernel_initializer='lecun_normal'
                
            )
        ) 
        # model.add(  
        #     tf.keras.layers.Dropout(0.05)
        # ) 
        model.add(
            Dense(
                32, activation='selu', input_shape =(train_X.shape[1],)
                , bias_regularizer=regularizers.l2(1e-1)
                , kernel_initializer='lecun_normal'
                # , activity_regularizer=regularizers.l2(1e-5)
            )
        )
        # model.add(  
        #     tf.keras.layers.Dropout(0.05)
        # ) 
        model.add(
            Dense(
                16, activation='selu', input_shape =(train_X.shape[1],)
                , bias_regularizer=regularizers.l2(1e-2)
                , kernel_initializer='lecun_normal'
                # , activity_regularizer=regularizers.l2(1e-5)
            )
        ) 
        # model.add(
        #     Dense(
        #         32, activation='selu', input_shape =(train_X.shape[1],)
        #         , bias_regularizer=regularizers.l2(1e-2)
        #         , kernel_initializer='lecun_normal'
        #         # , activity_regularizer=regularizers.l2(1e-5)
        #     )
        # ) 

        model.add(Dense(train_y.shape[1]))
        
        # loss_fn = tf.keras.losses.MeanSquaredError(reduction='sum')
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=0.001,
        #     decay_steps=0,
        #     decay_rate=0)
        
        learningRate=0.000005
        opt = optimizers.Adam(learning_rate=learningRate)
        

        # model.compile(loss='mean_squared_error',  optimizer=opt)
        
        # opt = optimizers.Adam(learning_rate=0.000005)
        model.compile(loss='mean_squared_error', optimizer=opt)

 
            
            
        self.model = model            

        # fit network
        history = model.fit(
            train_X, train_y, epochs=100, batch_size=32, 
            validation_data=(valid_X, valid_y), verbose=2, shuffle=False
        )
       
        # save network 
        model.save(self.fileName)
        
        # plot history
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='valid')
        # pyplot.plot(history.history['acc'], label='train')
        # pyplot.plot(history.history['val_acc'], label='valid')
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
            single_stock = Featurizer(
                "prediction", self.params['minNumPastSamples']
            ).transform(single_stock)
            single_stock = NARemover(mnemonic).transform(single_stock)
            single_stock = single_stock.tail(1)
            combined_data_set.append(single_stock)
            
        combined_data_set_df = pd.concat(combined_data_set, axis=0)
        data_set = TrainingSetBuilder().transform(combined_data_set_df)
        
        predictions = self.model.predict(data_set.X)
        myNewPred = Predictions(predictions, data_set)
        return myNewPred
