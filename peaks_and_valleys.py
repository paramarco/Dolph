# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 13:31:56 2021

@author: klio_ks
"""


import os
import math
import logging
import pandas as pd
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
log = logging.getLogger("TradingPlatform")
from keras.callbacks import LearningRateScheduler
from keras import backend as K


# import NeuronalNet_v6 as nn_v6
from scipy.signal import find_peaks

  

class Predictions:
    def __init__(self, predictions, training_set):
        self.predictions = predictions
        self.training_set = training_set  
    
class Model:
    def __init__(self, df, params, period, mode, currentHour):
        self.bestDistanceValley = 0
        self.bestDistancePeak = 0
        self.bestPositionMargin = 10
        
        self.id = 'peaks_and_valleys'
        
        self.model = None
        self.period = period
        self.params = params        
        self.mode = mode                
        samples = str(params['minNumPastSamples'])  +'Samples'  
        
        self.TargetHours = None
        if currentHour in range(9,14):
            self.TargetHours = range(9,14)
        else:
            self.TargetHours = range(14,20)
            
        tb = self.TargetHours[0]
        te = self.TargetHours[-1]
        
        interval = str(tb) + '_' + str(te) + 'Hour'        
        nameParts = [ __name__, self.period, samples, interval]        
        self.fileName = '_'.join(nameParts)
    
        self.findBestfluctiation(df, params, period, mode, currentHour)        
        
    def findBestfluctiation(self, df, params, period, mode, currentHour):
        self.bestDistanceValley = 8
        self.bestDistancePeak = 8
        self.bestPositionMargin = 10       
        
        
    
    def isSuitableForThisTime(self, hour):
        answer = True
        # answer = False
        # if hour in self.TargetHours:
        #     answer = True
        # else:
        #     answer = False
        return answer
    
    def findPeaksValleys (self, df):
        
        numWindowSize = 25
        df = df.tail(numWindowSize)
        fluctuation = {}
        fluctuation['samplingWindow'] = df[['CalcDateTime','StartPrice','MaxPrice','MinPrice','EndPrice',] ]

        seriesEnd = df['EndPrice']
        seriesMax = df['MaxPrice']
        seriesMin = df['MinPrice']
        times =     df['CalcDateTime']
        
        log.info('from ' + str(times[0]) + ' to ' + str(times[-1]) )
        # Find indices of peaks
        peak_idx, _ = find_peaks(seriesMax, distance=self.bestDistancePeak)        
        # Find indices of valleys (from inverting the signal)
        valley_idx, _ = find_peaks(-seriesMin, distance=self.bestDistanceValley)
        
        fluctuation['peak_idx'] = peak_idx
        fluctuation['valley_idx'] = valley_idx
        
        self.plotPeaksAndValleys (seriesMax,seriesEnd, seriesMin,peak_idx,valley_idx, fluctuation )  
              
        return fluctuation
    
    
    def plotPeaksAndValleys (self, seriesMax,seriesEnd, seriesMin,peak_idx,valley_idx,fluctuation ):  
        
        # Plot curves
        t = np.arange(start=0, stop=len(seriesEnd), step=1, dtype=int)
        plt.plot(t, seriesMax)
        plt.plot(t, seriesEnd)
        plt.plot(t, seriesMin)
        
        # Plot peaks (red) and valleys (blue)
        plt.plot(t[peak_idx], seriesMax[peak_idx], 'g^')
        plt.plot(t[valley_idx], seriesMin[valley_idx], 'rv')
        
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))        

        
        #PLOT CLOSE PRICE WHICH IS ALMOST ENRTY+-
        dataInSamplinWindow = fluctuation['samplingWindow']
        currentClose = dataInSamplinWindow.iloc[-1].EndPrice
        currentLow = dataInSamplinWindow.iloc[-1].MinPrice
        currentHigh=dataInSamplinWindow.iloc[-1].MaxPrice
        entryPrice=currentClose
        exitPrice=entryPrice+20
        
        new_peak_ind=np.array(peak_idx)+1
        plt.plot(t[new_peak_ind],seriesEnd[new_peak_ind], 'm^') 
        new_valley_ind=np.array(valley_idx)+1
        plt.plot(t[new_valley_ind],seriesEnd[new_valley_ind], 'm^') 
        
        plt.show()
        
    def predict(self, df ):        
      
        df['CalcDateTime'] = df.index
        fluctuation = self.findPeaksValleys(df)
        
               
        return fluctuation