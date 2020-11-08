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
from datetime import datetime
from matplotlib import pyplot
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers, activations
from tensorflow.keras.models import load_model

log = logging.getLogger("TradingPlatform")

def pred_baseline(d):
    single_feature = 'x((MaxP-EndP)-(EndP-MinP))(t - 1)'
    preds = d.training_set.original_df[single_feature].values
    preds = preds.reshape((preds.shape[0], 1))
    return Predictions(preds, d.training_set).evaluate()

def readable_summary(which_set, p):
    achieved = p.evaluate()['strategies']['achieved']
    achieved_baseline = pred_baseline(p)['strategies']['achieved']
    per_change = np.mean(np.absolute(p.training_set.original_df['pseudo_y(pctChange)']))
    n = p.training_set.original_df.shape[0]
    print ("""
           So if you play {} times on the {} with 1 EUR and you always guess 
           the movement,ignoring all transactions cost, you will make {}. 
           Instead you make {} or {} percent of the ideally achievable. 
           If you use the baseline you will make {} or {} percent of ideal
           """.format( 
               n, which_set, n * per_change, 
               achieved, 100.0*achieved/(n*per_change), achieved_baseline,
               100.0*achieved_baseline/(n*per_change)
            )
    )

def plot_accuracy_by(grouping_feature, predictions):
    df = predictions.training_set.original_df
    s = df[['Date', 'Mnemonic', 'pseudo_y(SignReturn)']].copy()
    s['Predictions'] = predictions.predictions
    s['Baseline'] = df['x((MaxP-EndP)-(EndP-MinP))(t - 1)']

    def agg(group):
        pred = group['Predictions']
        baseline = group['Baseline']
        rets = group['pseudo_y(SignReturn)']
        c = pred.corr(rets)
        c = np.where(np.sign(pred)*np.sign(rets) == 1.0, 1.0, 0.0).sum()
        e = np.where(np.sign(pred)*np.sign(rets) == -1.0, 1.0, 0.0).sum()
        acc = c/(c + e)

        c_baseline = np.where(np.sign(baseline)*np.sign(rets) == 1.0, 1.0, 0.0).sum()
        e_baseline = np.where(np.sign(baseline)*np.sign(rets) == -1.0, 1.0, 0.0).sum()
        acc_baseline = c_baseline/(c_baseline + e_baseline)

        l = group.shape[0]
        return {"corr": c, 'size': l, 'accuracy': acc, 'acc_baseline': acc_baseline}
    f = s.groupby(grouping_feature).apply(agg).to_frame("agg")

    f['AccuracyPred'] = f['agg'].map(lambda i: i['accuracy'])
    f['AccuracyBaseline'] = f['agg'].map(lambda i: i['acc_baseline'])
    f['AccPred - AccBaseline'] = f['AccuracyPred'] - f['AccuracyBaseline']
    f = f.drop(columns=['agg'])

    f = f[f.index != '2017-10-14'] # remove this date which has one data point
    f[['AccuracyPred', 'AccuracyBaseline']].plot()
    return f

def clipped(ind, limit):
    return np.where(ind < -limit, -limit, np.where(ind > limit, limit, ind ))

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
    def transform(self, single_stock, max_offset = 30):

        predictionWindow = 5
        # stds = single_stock['EndPrice'].rolling(max_offset).std()       
        
        # # take the last EndPrice at [ t ]
        # end_price = single_stock['EndPrice'].tail(1)
        # # take the max price in the past at [  t - offset , t ]        
        # min_price = single_stock['MinPrice'].rolling(max_offset).min()
        # # take the min price in the past at [  t - offset , t ]   
        # max_price = single_stock['MaxPrice'].rolling(max_offset).max()       
        # # compute an indicator for time interval [  t - offset , t ] 
        # ind = ((max_price - end_price) - (end_price - min_price)) / stds
        # single_stock['x((MaxP-EndP)-(EndP-MinP))(t - {})'.format(str(max_offset))] = ind
        
        single_stock['x(DOW)'] = single_stock['CalcDateTime'].dt.dayofweek
        single_stock['x(Hour)'] = single_stock['CalcDateTime'].dt.hour
        single_stock['x(DOY)'] = single_stock['CalcDateTime'].dt.dayofyear
        
        for offset in range(0, max_offset ):
            
            single_stock['x(open(t-{})'.format(str(offset))] = single_stock['StartPrice'].shift(offset)   
            single_stock['x(high(t-{})'.format(str(offset))] = single_stock['MaxPrice'].shift(offset)   
            single_stock['x(close(t-{})'.format(str(offset))] = single_stock['EndPrice'].shift(offset)   
            single_stock['x(low(t-{})'.format(str(offset))] = single_stock['MinPrice'].shift(offset)   
            single_stock['x(vol(t-{})'.format(str(offset))] = single_stock['addedVolume'].shift(offset)   

             
        for offset in range(1, predictionWindow ):
            single_stock['y(open(t+{})'.format(str(offset))] = single_stock['StartPrice'].shift(-offset)   
            single_stock['y(high(t+{})'.format(str(offset))] = single_stock['MaxPrice'].shift(-offset)   
            single_stock['y(close(t+{})'.format(str(offset))] = single_stock['EndPrice'].shift(-offset)   
            single_stock['y(low(t+{})'.format(str(offset))] = single_stock['MinPrice'].shift(-offset)   
            

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
        
    def evaluate(self):
        single_feature = 'x((MaxP-EndP)-(EndP-MinP))(t - 1)'
        stats_df = pd.DataFrame({
                      'predictions': self.predictions[:,0],
                      'single_feature_pred': self.training_set.original_df[single_feature].values,
                      'pseudo_y(SignReturn)': self.training_set.y[:,0],
                      'pseudo_y(pctChange)': self.training_set.original_df['pseudo_y(pctChange)'].values,
                       'pseudo_y(ClippedReturn)': self.training_set.original_df['pseudo_y(ClippedReturn)'].values,
                      'y(Return)': self.training_set.original_df['y(Return)'].values})
        
        corr = stats_df. \
            corr()[['predictions', 'single_feature_pred']]. \
            iloc[1:]
            
        pred_signs = np.sign(stats_df['predictions'])
        y_signs = np.sign(stats_df['y(Return)'])
        has_answer = np.absolute(pred_signs * y_signs).sum()
        correct = np.where(pred_signs * y_signs == 1.0, 1.0, 0.0).sum()
        
        thresholds = []
        accuracy = []
        correct_lst = []
        errors = []
        percent_has_answer = []
        abs_has_answer = []
        achieved_returns = []

        preds = stats_df['predictions']
        
        for d in range(5, 46, 5):
            low = np.percentile(preds, d) 
            high = np.percentile(preds, 100 - d)
            thresholded = np.where(preds > high, 1.0, np.where(preds < low, -1.0, 0.0))
            c = np.where(np.sign(thresholded)*np.sign(y_signs) == 1.0, 1.0, 0.0).sum()
            e = np.where(np.sign(thresholded)*np.sign(y_signs) == -1.0, 1.0, 0.0).sum()
            achieved_ret = (stats_df['pseudo_y(pctChange)']*thresholded).sum()
            correct_lst.append(c)
            errors.append(e)
            accuracy.append(c/(c + e))
            percent_has_answer.append(100.0*(c + e)/pred_signs.shape[0])
            abs_has_answer.append((c + e))
            achieved_returns.append(achieved_ret)
            thresholds.append(d)
            
        at_cutoff = DataFrame({
                    'thresholds': thresholds,
                    'accuracy': accuracy,
                    'percent_with_answer': percent_has_answer,
                    'absolute_has_answer': abs_has_answer,
                    'achieved_returns': achieved_returns,
                    'correct': correct_lst,
                    'errors': errors
        })
        at_cutoff['achieved_norm_returns'] = at_cutoff['achieved_returns']/at_cutoff['absolute_has_answer']
        
        ret = stats_df['pseudo_y(pctChange)']
        rand_feature = np.where(np.random.rand(ret.shape[0]) > 0.5, 1.0, -1.0)    
        random_returns = (ret * rand_feature).sum()
        always_up_returns = (ret*1.0).sum()
        always_down_returns = (ret*-1.0).sum()
        omnicient_returns = (np.absolute(ret)).sum()
        achieved = (ret * pred_signs).sum()
        return {
            'corr': corr,
            'accuracy_at_cutoff': at_cutoff,
            'matches': {
                'percent_correct': 100*correct/has_answer,
                'percent_has_answer': has_answer/pred_signs.shape[0],
                'absolute_with_answer': has_answer,
                'size': pred_signs.shape[0]
            },
            'strategies': {
                'omniscient': omnicient_returns,
                'random': random_returns,
                'always_up': always_up_returns,
                'always_down': always_down_returns,
                'achieved': achieved,
                'num_trials': np.absolute(pred_signs).sum()
            }
        }
    
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
        np.random.shuffle(train_valid_days)
        
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
            single_stock = Featurizer().transform(single_stock)
            single_stock = NARemover(mnemonic).transform(single_stock)
            combined_training_set.append(single_stock)
            log.info(single_stock.shape)
        
            single_stock = df_valid[df_valid.Mnemonic == mnemonic].copy()
            single_stock = single_stock[single_stock.HasTrade == 1.0] 
            single_stock = Featurizer().transform(single_stock)
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

        model.add(Dense(100, activation='tanh', input_shape =(train_X.shape[1],),
                        kernel_regularizer=regularizers.l2(0.001))) 
        model.add(Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))        

        model.add(Dense(train_y.shape[1]))
        
        loss_fn = tf.keras.losses.MeanSquaredError(reduction='sum')
        
        # model.compile(loss='mean_squared_error', optimizer='adam')
        model.compile(loss=loss_fn,  optimizer='adam')
        self.model = model            

        # fit network
        history = model.fit(train_X, train_y, epochs=50, batch_size=15, validation_data=(valid_X, valid_y), verbose=2, shuffle=True)
       
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
            single_stock = Featurizer().transform(single_stock)
            single_stock = NARemover(mnemonic).transform(single_stock)
            single_stock = single_stock.tail(1)
            combined_data_set.append(single_stock)
            
        combined_data_set_df = pd.concat(combined_data_set, axis=0)
        data_set = TrainingSetBuilder().transform(combined_data_set_df)
        
        predictions = self.model.predict(data_set.X)
        myNewPred = Predictions(predictions, data_set)
        return myNewPred






















