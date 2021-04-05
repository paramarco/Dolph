# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:59:41 2020

@author: mvereda
"""

import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15, 10) # use bigger graphs

import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import numpy as np
import pandas as pd
import gc; gc.collect()
import math
import time 
import numbers
import datetime
import datetime as dt

# import NeuronalNet_v6 as nn_v6
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def findPeaksValleys (data_test):
    
    data_test = data_test[['timeDate','StartPrice','MaxPrice','EndPrice'] ].values
    
    series = data_test[:,3]
    numWindowSize = 25
    series = series[-numWindowSize:]
    print (series)    
   
    # Find indices of peaks
    peak_idx, _ = find_peaks(series, distance=3)
    
    # Find indices of valleys (from inverting the signal)
    valley_idx, _ = find_peaks(-series, distance=3)
    
    # Plot 
    t = np.arange(start=0, stop=len(series), step=1, dtype=int)
    plt.plot(t, series)   
    
    # Plot peaks (red) and valleys (blue)
    plt.plot(t[peak_idx], series[peak_idx], 'g^')
    plt.plot(t[valley_idx], series[valley_idx], 'rv')
    
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
    
    plt.show() 


class TrendViewer:
    
    def __init__(self, periods, func):
        self.preds = []
        self.predsplot = []
        self.predsplot1 = []
        self.index_plot = 0
        self.array_X_index = []
        self.array_X_pred = []
        self.N_all = 19
        self.index = 1
        self.data_test = None
        self.data_train = None
        self.listPredictions = []
        self.periods = periods
        self.previousPrice =0
        self.numTotalPrices = 0
        self.numPositivePrices=0
        self.numNegativePrices=0
        self.previousPrediction=[]
        self.totalcounter= 0
        self.df_four=[]
        self.entrancePrice =0
        self.outPrice=0
        self.printPrices = False
        self.prediction_sign = 0
        self.evaluatePositionTest = func
        self.arrayPredictionsSigns = []
        self.arrayTruecandlesSigns = []
        self.moscowHourarray = []
        self.positiveHour = []
        self.negativeHour =[]
    def setDataTest(self, inputData):
        self.data_test = inputData
        
        

                
                
    def alignCruves (self, inputPredicion ):
        
        predLastDataRescaled = inputPredicion['predLastData']
        predTestDataRescaled = inputPredicion['predTestingData']
     
        # we take the 3rd coloumn because it's a close price in the origin!
        prediction_price_CLOSE = predLastDataRescaled[-1:][0][3]
        
        # here we take the 3.coloumn for predicted array
        array_pred_coef = predTestDataRescaled[-19:, 3]     
        
        
        # take 20 points for the plotting at the end the results
        Num_points = 20   

    
        data_test_plot = self.data_test [::, 3]
        data_test_plot = data_test_plot[:-1]
        data_close_for_graf=data_test_plot[-19:]
    
    
        array_real_coef = data_close_for_graf # real data 
        
    
        
        diff_coeff_scaled = sum(abs(array_pred_coef-array_real_coef))/Num_points
        # diff_coeff_scaled =0
       
        #tae last value from real data
        last_value_real = array_real_coef[-1:]
        #take last value from predicted data
        last_value_prediction = array_pred_coef[-1:]
     
    
        if(math.isnan(last_value_prediction)):
            prediction_number_scaled=0
        #condictions to align both lines: predicted and real
        if  (last_value_real>=last_value_prediction):
             array_prediction_new_scaled = array_pred_coef + diff_coeff_scaled
             prediction_number_scaled= prediction_price_CLOSE+diff_coeff_scaled
        if   (last_value_real<last_value_prediction):
             array_prediction_new_scaled = array_pred_coef-diff_coeff_scaled
             prediction_number_scaled=prediction_price_CLOSE-diff_coeff_scaled
        
       
        print_prediction_number_scaled = prediction_number_scaled
     
        print ('prediction: ' + str( print_prediction_number_scaled) )
        
        triple = ( print_prediction_number_scaled,
                  data_close_for_graf,
                  array_prediction_new_scaled )
        
        return triple
        
    def displayPrediction (self, inputPrediction, period ): 
               
        period = "check this out...."           
        prediction_p, array_real_p, array_prediction_p = \
            self.alignCruves(inputPrediction)
            
        result = 'prediction ' + period + ':    ' + str(prediction_p)
        self.preds.append(result )
        if (len(self.predsplot1)==19):
            self.predsplot1 = []
            self.index_plot = 0
            self.indx = 0
        self.predsplot.append(prediction_p)
        self.predsplot1.append((prediction_p))  
        array_plot_prediction=array_prediction_p
        array_plot_real = array_real_p
        timeframe= [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 ]
        fig1= plt.figure(period)
        ax = fig1.gca()
        ax.set_xticks(np.arange(0, self.N_all+1, 1))
        plt.plot(timeframe,array_plot_real) 
        plt.plot(timeframe,array_plot_prediction) 
        plt.plot(self.N_all+1, prediction_p, 'ro')
        start_history=0
        stop_history=len(self.predsplot1)-1
        step_history=1
        indx=stop_history
        logging.info( self.predsplot1 )
        if (self.index_plot >0):
            for j in range(start_history, stop_history, step_history):
                plt.plot((self.N_all+1)-indx, self.predsplot1[j], 'go')
                self.array_X_index.append((self.N_all+1)-indx)
                self.array_X_pred.append(self.predsplot1[j])
                indx=indx-1
    
        self.index_plot=self.index_plot+1            
        red_patch = mpatches.Patch(color='red', label='prediction')
        blue_patch = mpatches.Patch(color='blue', label='real data')
        plt.legend(handles=[red_patch, blue_patch])
        plt.grid()
        plt.title('Prediction for ' + period)
        plt.show()
                 
        self.predsplot = []
        logging.info( self.preds)
             
        
    def displayPrediction_v2 (self, prediction, period  ):
 
        lastClosePrice = self.data_test [-1, 3]
        logging.info( 'when close price is:' + str(lastClosePrice) )
        numWindowSize = 20
        numPredWindow = 5
        self.listPredictions.append(prediction)
        lastIndexes = numWindowSize - numPredWindow
        
        predictions = self.listPredictions[-lastIndexes:]
        listClosePrice = [i['lastClosePrice'] for i in  predictions]
        listPredLastData = [i['predLastData'] for i in  predictions]
        list2MinPred = [i[0,0] for i in listPredLastData]
        # list5MinPred = [i[1,0] for i in listPredLastData]

        
        timeframeClosePrice = range( 0, len(predictions) )       
        plt.plot(timeframeClosePrice, listClosePrice) 
        
        timeframe2MinPred = range( 4, len(predictions)+4 )       
        plt.plot(timeframe2MinPred, list2MinPred) 
        
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))        
        plt.show() 
        
    def displayPrediction_v3 (self, prediction, period  ):
        
        lastClosePrice = self.data_test [-1, 3]
        logging.info( 'when close price is:' + str(lastClosePrice) )
        numWindowSize = 20
        numPredWindow = 5
        self.listPredictions.append(prediction)
        lastIndexes = numWindowSize - numPredWindow
        
        predictions = self.listPredictions[-lastIndexes:]
        listClosePrice = [i['lastClosePrice'] for i in  predictions]
        listPredLastData = [i['predLastData'] for i in  predictions]
        list2MinPred = [i[0,0] for i in listPredLastData]
        # list5MinPred = [i[1,0] for i in listPredLastData]

        
        timeframeClosePrice = range( 0, len(predictions) )       
        plt.plot(timeframeClosePrice, listClosePrice) 
        
        timeframe2MinPred = range( 5, len(predictions)+5 )       
        plt.plot(timeframe2MinPred, list2MinPred) 
        
        # timeframe5MinPred = range( 5, len(predictions)+5 )       
        # plt.plot(timeframe5MinPred, list5MinPred) 
        
        
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))        
        plt.show()     
        
        
    def displayPrediction_v4 (self, prediction, period ):  

        # signal = np.copy(df['Close_Price'].values)
        plt.plot(
            np.arange(len(prediction['date_ori'])),
            prediction['df'][:, -1],
            label = 'prediction',
        )
        # plt.plot(np.arange(len(signal)), signal, label = 'real close price')
        plt.legend()
        plt.show()
        
    def displayPrediction_v5 (self, lisrPreds, period ):
        
        self.listPredictions = lisrPreds
        
        prediction = self.listPredictions[-1]
        time.sleep(0.3)
        lastClosePrice = prediction['lastClosePrice']
        logging.info( 'when close price is:' + str(lastClosePrice) )
        numWindowSize = 20
        numPredWindow = int(prediction['predWindow'][0])          
        
        # self.listPredictions.append(prediction)
        lastIndexes = numWindowSize - numPredWindow
        
        predictions = self.listPredictions[-lastIndexes:]
        listClosePrice = [i['lastClosePrice'] for i in  predictions]
        listPredLastData = [i['predLastData'] for i in  predictions]
        listXMinPred = [i for i in listPredLastData]
        # list5MinPred = [i[1,0] for i in listPredLastData]
        
        listXMinPred = [a + b for a, b in zip(listClosePrice, listPredLastData)]
        
        timeframeClosePrice = range( 0, len(predictions) )       
        plt.plot(timeframeClosePrice, listClosePrice) 
        
        timeframeXMinPred = range( 1, len(predictions)+1 )       
        plt.plot(timeframeXMinPred, listXMinPred) 
        
        # timeframe5MinPred = range( 5, len(predictions)+5 )       
        # plt.plot(timeframe5MinPred, list5MinPred) 
        
        
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))        
        plt.show()
        
    def displayPrediction_v6 (self, predictions, period):
        
        single_feature = 'x((MaxP-EndP)-(EndP-MinP))(t - 1)'
        df_in = pd.DataFrame(columns=['predictions',
                                      'time',
                                      'Mnemonic',
                                      'EndPrice',
                                      'single_feature_pred',
                                      'period' ])
        
        for p in predictions:
            row = pd.DataFrame({
                'predictions':          p.predictions[:,0],
                'time':                 p.training_set.original_df['CalcDateTime'],
                'Mnemonic':             p.training_set.original_df['Mnemonic'],
                'EndPrice':             p.training_set.original_df['EndPrice'],
                'single_feature_pred':  p.training_set.original_df[single_feature].values,
                'period':               period
                }
            )
            df_in = df_in.append(row)
        df_in.set_index('time')
        
        time.sleep(0.3)
        numWindowSize = 20
       
            
        numWindowSize = int(numWindowSize / int(period[0]) )
                            
        df = df_in[df_in.period == period]
        
        times = sorted(list(df.index.unique()))
        times = times[-numWindowSize:]
        prices = [] 
        for t in times:
            close =  df.loc[t].EndPrice
            if isinstance(close,  float):
                prices.append(close)
            else:
                prices.append(close[0])
        
        def setLabel(sign):
            label = ""
            if ( sign > 0.05 ):
                label = "\u2197"
            elif ( sign < -0.05 ):
                label = "\u2198" 
            else:
                label = "\u003D"
            return label
            
        for t,price in zip(times,prices):            
            label = u""
            signs =  df.loc[t].predictions.tolist()
            if isinstance(signs,  float):
                label += setLabel(signs)
            else:  
                for s in signs:
                    label += setLabel(s)
            
            plt.annotate(
                label, # this is the text
                (t,price), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center',  # horizontal alignment 
                size=20
            )
        
        etiquete= 'close price, prediction for:' + period
        plt.title(label = etiquete )
        plt.plot(times, prices)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        xlocator = mdates.MinuteLocator(byminute=range(60))
        plt.gca().xaxis.set_major_locator(xlocator)  
        plt.show() 
        
        
        
    def displayPrediction_v8 (self, predictions, period):
            
       
        df_in = pd.DataFrame(columns=['predictions',
                  'time',
                  'Mnemonic',
                  'EndPrice',
                  'single_feature_pred',
                  'period' ])
        
        for p in predictions:
            row = pd.DataFrame({
            'predictions':          p.predictions[:,0],
            'time':                 p.training_set.original_df['CalcDateTime'],
            'Mnemonic':             p.training_set.original_df['Mnemonic'],
            'EndPrice':             p.training_set.original_df['LastEndPrice'],
            # 'single_feature_pred':  p.training_set.original_df[single_feature].values,
            'period':               period
            }
            )
            df_in = df_in.append(row)
        df_in = df_in.set_index('time')
        
        time.sleep(0.3)
        numWindowSize = 20
           
        numWindowSize = int(numWindowSize / int(period[0]) )
        
        df = df_in[df_in.period == period]
        
        times = sorted(list(df.index.unique()))
        times = times[-numWindowSize:]
        prices = [] 
        for t in times:
            close =  df.loc[t].EndPrice
            if isinstance(close,  float):
                prices.append(close)
            else:
                prices.append(close[0])
            
              
        def setLabel(sign):
            label = ""
            if ( sign > 0.0 ):
                label = "\u2197"
            elif ( sign < 0.0 ):
                label = "\u2198" 
            else:
                label = "\u003D"
            return label
            
        prediction_sign = (df.loc[t].predictions.tolist())
        print('prediction is:'+ str(prediction_sign))
        currentPrice=close
        if (self.numTotalPrices > 0):
            self.numTotalPrices+=1         
            print('current:' + str(currentPrice))
            print('previous:' + str(self.previousPrice))
            signPriceDiff=np.sign(currentPrice-self.previousPrice)
            signPrediction=np.sign( self.previousPrediction)
            self.previousPrediction = prediction_sign #sign of prediction -1 or +1
            
            if(signPriceDiff==signPrediction):
                    self.numPositivePrices+=1
            else:
                    self.numNegativePrices+=1
            probaility=self.numPositivePrices/self.numTotalPrices
            self.previousPrice=currentPrice
            print('probability:' + str(probaility))
            print('numTotal:' + str(self.numTotalPrices))
            print('numPositiv:' + str(self.numPositivePrices))
        else:
            self.numTotalPrices=1
            self.previousPrice = close
            self.previousPrediction = prediction_sign #sign of prediction -1 or +1
           
        
        for t,price in zip(times,prices):            
            label = u""
            signs = df.loc[t].predictions.tolist()
            if isinstance(signs,  float):
                label += setLabel(signs)
            else:  
                for s in signs:
                    label += setLabel(s)
        
            plt.annotate(
            label, # this is the text
            (t,price), # this is the point to label
            textcoords="offset points", # how to position the text
            xytext=(0,10), # distance from text to points (x,y)
            ha='center',  # horizontal alignment 
            size=20
            )
        
        etiquete= 'close price, prediction for:' + period
        plt.title(label = etiquete )
        plt.plot(times, prices)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        xlocator = mdates.MinuteLocator(byminute=range(60))
        plt.gca().xaxis.set_major_locator(xlocator)  
        plt.show() 
        
    def displayPrediction_v9 (self, predictions, period):
        
        single_feature = 'x((MaxP-EndP)-(EndP-MinP))(t - 1)'
        df_in = pd.DataFrame(columns=['predictions',
                                      'time',
                                      'Mnemonic',
                                      'EndPrice',
                                      'single_feature_pred',
                                      'period' ])
        
        for p in predictions:
            row = pd.DataFrame({
                'predictions':          p.predictions[:,0],
                'time':                 p.training_set.original_df['CalcDateTime'],
                'Mnemonic':             p.training_set.original_df['Mnemonic'],
                'StartPrice':           p.training_set.original_df['StartPrice'],
                'EndPrice':             p.training_set.original_df['EndPrice'],
                'MinPrice':             p.training_set.original_df['MinPrice'],
                'MaxPrice':             p.training_set.original_df['MaxPrice'],                
                'single_feature_pred':  p.training_set.original_df[single_feature].values,
                'period':               period
                }
            )
            df_in = df_in.append(row)
        df_in['timeDate'] = df_in['time']  
        df_in = df_in.set_index('time')
        
        time.sleep(0.3)
        numWindowSize = 10    
            
        # numWindowSize = int(numWindowSize / int(period[0]) )
                            
        df = df_in[df_in.period == period]
        
        times = sorted(list(df.index.unique()))
        times = times[-numWindowSize:]
        prices = [] 
        for t in times:
            high =  df.loc[t].MaxPrice
            if isinstance(high,  float):
                prices.append(high)
            else:
                prices.append(high[0])
        
  
        def setLabel(sign):
            label = ""
            if ( sign > 0.0 ):
                label = "\u2197"
            elif ( sign < 0.0 ):
                label = "\u2198" 
            else:
                label = "\u003D"
            return label
            
        prediction_sign = df.loc[t].predictions.tolist()
        currentPrice=df.iloc[-1].EndPrice
        if (self.numTotalPrices > 0):
                self.numTotalPrices+=1 
                print('current:' + str(currentPrice))
                print('previous:' + str(self.previousPrice))
                signPriceDiff=np.sign(currentPrice-self.previousPrice)
                signPrediction=np.sign( self.previousPrediction)
                self.previousPrediction = prediction_sign #sign of prediction -1 or +1

                if(signPriceDiff==signPrediction):
                    self.numPositivePrices+=1
                else:
                    self.numNegativePrices+=1
                probaility=self.numPositivePrices/(self.numTotalPrices-1)
                self.previousPrice=currentPrice
                print('probability:' + str(probaility))
                print('numTotal:' + str(self.numTotalPrices-1))
                print('numPositiv:' + str(self.numPositivePrices))
        else:
             self.numTotalPrices=1
             self.previousPrice = currentPrice
             self.previousPrediction = prediction_sign #sign of prediction -1 or +1


        for t,price in zip(times,prices):            
            label = u""
            signs = df.loc[t].predictions.tolist()
            if isinstance(signs,  float):
                label += setLabel(signs)
            else:  
                for s in signs:
                    label += setLabel(s)
            
            plt.annotate(
                label, # this is the text
                (t,price), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center',  # horizontal alignment 
                size=20
            )
        
        etiquete= 'close price, prediction for:' + period
        plt.title(label = etiquete )
        
        ohlc = df[['timeDate','StartPrice','MaxPrice','MinPrice','EndPrice'] ]
        ohlc = ohlc[-numWindowSize:]
        ohlc.columns = ['Date', 'Open', 'High', 'Low', 'Close'] 
        ohlc['Date'] = pd.to_datetime(ohlc['Date'])
        ohlc['Date'] = ohlc['Date'].apply(mdates.date2num)
        ohlc = ohlc.astype(float)        
        candlestick_ohlc(plt.gca(), ohlc.values, width=0.0001, colorup='green', colordown='red', alpha=0.9)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%M'))
        xlocator = mdates.MinuteLocator(byminute=range(60))
        plt.gca().xaxis.set_major_locator(xlocator)  
        plt.show() 


        
        
        
    def displayPrediction_v10 (self, predictions, period):
        
        numPeriod = int(period[0])
        df_in = pd.DataFrame(columns=['predictions',
                                      'time',
                                      'Mnemonic',
                                      'EndPrice' ])
        
        for p in predictions:
            row = pd.DataFrame({
                'time':                 p.training_set.original_df['CalcDateTime'],
                'Mnemonic':             p.training_set.original_df['Mnemonic'],
                'StartPrice':           p.training_set.original_df['StartPrice'],
                'EndPrice':             p.training_set.original_df['EndPrice'],
                'MinPrice':             p.training_set.original_df['MinPrice'],
                'MaxPrice':             p.training_set.original_df['MaxPrice'] ,
                'close_t+1':            p.predictions[0][0] 
                }
            )
            df_in = df_in.append(row)
        df_in['timeDate'] = df_in['time']  
        df_in = df_in.set_index('time')
        
        time.sleep(0.3)
        numWindowSize = 13    
            
        # numWindowSize = int(numWindowSize / int(period[0]) )
                            
        df = df_in
        
        times = sorted(list(df.index.unique()))
        times = times[-numWindowSize:]
        prices = [] 
        lastTime = None
        for t in times:
            high =  df.loc[t].MaxPrice
            if isinstance(high,  float):
                prices.append(high)
            else:
                prices.append(high[0])
            lastTime = t
        
          
        etiquete= 'close price, prediction for:' + period
        plt.title(label = etiquete )
        
        ohlc = df[['timeDate','StartPrice','MaxPrice','MinPrice','EndPrice'] ]
        ohlc = ohlc[-numWindowSize:]
        ohlc.columns = ['Date', 'Open', 'High', 'Low', 'Close'] 
        ohlc['Date'] = pd.to_datetime(ohlc['Date'])
        ohlc['Date'] = ohlc['Date'].apply(mdates.date2num)
        ohlc = ohlc.astype(float)        
        candlestick_ohlc(plt.gca(), ohlc.values, width=0.6/(24*60), colorup='green', colordown='black', alpha=0.9)
        
        t1 = lastTime + datetime.timedelta(minutes = (1 * numPeriod) )
        moscowHour = t1.hour

        
        
        currentOpen = df.iloc[-1].StartPrice
        currentHigh = df.iloc[-1].MaxPrice
        currentLow = df.iloc[-1].MinPrice
        currentClose = df.iloc[-1].EndPrice

        p = df.loc[lastTime]
        
        candlePredList = [
            {'Date': t1, 'Open': currentClose, 'High': currentClose  + 1, 'Low': currentClose  - 1, 'Close': currentClose + p['close_t+1']}   #,            # {'Date': t2, 'Open': currentClose + p['close_t+1'], 'High': currentHigh + p['high_t+2'], 'Low': currentLow + p['low_t+2'], 'Close': currentClose + p['close_t+1'] + p['close_t+2']},
        ]         
            
        ohlc2 = pd.DataFrame(candlePredList)  
        ohlc2['Date'] = pd.to_datetime(ohlc2['Date'])
        ohlc2['Date'] = ohlc2['Date'].apply(mdates.date2num)
        ohlc2 = ohlc2.astype(float)
        candlestick_ohlc(plt.gca(), ohlc2.values, width=0.6/(24*60), colorup='green', colordown='black', alpha=0.15)
        
        # plt.yticks(np.arange(min(ohlc['Close']), max(ohlc['Close'])+1, 10.0))
        
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%M'))
        xlocator = mdates.MinuteLocator(byminute=range(60))
        plt.gca().xaxis.set_major_locator(xlocator)  
                            
        lastCandle = { 
            'currentOpen' : currentOpen,
            'currentHigh' : currentHigh,
            'currentLow' : currentLow,
            'currentClose' : currentClose                      
        }        
        entryPrice, exitPrice, takePosition, printPrices = \
            self.evaluatePositionTest(candlePredList, lastCandle)
            
        predictionCandle = { 
            'currentOpen' : entryPrice,
            'currentClose' : exitPrice                      
        }        
        def checkCandleBlue(firstcandle):
            blue = False 
            if (firstcandle['currentClose']>firstcandle['currentOpen']):
                blue= True
            return blue
        def checkCandleBlack(firstcandle):
            black = False 
            if ((firstcandle['currentClose']<firstcandle['currentOpen'])):
                black= True
            return black   
        
        blue_prediction = checkCandleBlue(predictionCandle)
        black_prediction=checkCandleBlack(predictionCandle)
        if (blue_prediction == True):
            self.prediction_sign  = 1
        if (black_prediction == True):
            self.prediction_sign  = -1
            
        blue_current = checkCandleBlue(lastCandle)
        black_current=checkCandleBlack(lastCandle)
        if (blue_current == True):
            self.current_sign  = 1
        if (black_current == True):
            self.current_sign  = -1  
            
        self.arrayPredictionsSigns.append(self.prediction_sign)
        self.arrayTruecandlesSigns.append(self.current_sign )
       
        if (self.numTotalPrices > 0):
            if (moscowHour == self.moscowHourarray[-1]):
                print('previos:' + str(self.moscowHourarray[-1]))
                print('current:' + str(moscowHour))
                self.numTotalPrices+=1 
                if(self.arrayPredictionsSigns[-2]==self.arrayTruecandlesSigns[-1]):
                    self.numPositivePrices+=1
                else:
                    self.numNegativePrices+=1
                    

                print('numTotal:' + str(self.numTotalPrices-1))
                print('numPositiv:' + str(self.numPositivePrices))
                print('numNegative:' + str(self.numNegativePrices))
            else:   

                 self.positiveHour.append(self.numPositivePrices)
                 self.negativeHour.append(self.numNegativePrices)
                 print('array positive:' + str(self.positiveHour))
                 print('array negative:' + str(self.negativeHour))
                 self.numTotalPrices=1
                 self.numPositivePrices=0
                 self.numNegativePrices =0
        else:
             self.numTotalPrices=1
             
        self.moscowHourarray.append(moscowHour)    
             
             
             
        if (printPrices == True):
            plt.annotate(
                    "entrance="+ str(entryPrice), # this is the text
                    (t1,currentLow), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center',  # horizontal alignment 
                    size=20)
            plt.annotate(
                "exit="+ str(exitPrice), # this is the text
                (t1,currentHigh), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center',  # horizontal alignment 
                size=20)
        plt.show() 
        
        
        findPeaksValleys(df)
        
        
    def displayNothing (self, predictions, period):
        pass

