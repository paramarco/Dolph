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
        self.bestDistanceValley = 5
        self.bestDistancePeak = 5
        self.bestPositionMargin = 10       
        
        
    
    def isSuitableForThisTime(self, hour):
        answer = False
        if hour in self.TargetHours:
            answer = True
        else:
            answer = False
        return answer
    
    def findPeaksValleys (self, data_test):
        
        data_test = data_test[['CalcDateTime','StartPrice','MaxPrice','EndPrice','MinPrice'] ].values
        
        series = data_test[:,3]
        seriesMax = data_test[:,2]
        seriesMin = data_test[:,4]

        numWindowSize = 25
        series = series[-numWindowSize:]
        seriesMax = seriesMax[-numWindowSize:]
        seriesMin = seriesMin[-numWindowSize:]

       
        # Find indices of peaks
        # peak_idx, _ = find_peaks(series, distance=self.bestDistancePeak)
        peak_idx, _ = find_peaks(seriesMax, distance=self.bestDistancePeak)
        
        # Find indices of valleys (from inverting the signal)
        # valley_idx, _ = find_peaks(-series, distance=self.bestDistanceValley)
        valley_idx, _ = find_peaks(-seriesMin, distance=self.bestDistanceValley)
        
        # Plot 
        t = np.arange(start=0, stop=len(series), step=1, dtype=int)
        plt.plot(t, seriesMax)
        plt.plot(t, series)
        plt.plot(t, seriesMin)   

        
        # Plot peaks (red) and valleys (blue)
        plt.plot(t[peak_idx], seriesMax[peak_idx], 'g^')
        plt.plot(t[valley_idx], seriesMin[valley_idx], 'rv')
        
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        
        plt.show()
        
        status = 0
        
        indexLastPeak = peak_idx[-1]
        indexLastValley = valley_idx[-1]
        
        if indexLastPeak == (numWindowSize - 2) :
            status = 1
        elif indexLastValley == (numWindowSize - 2) :
            status = -1
        else:
            status = 0
       
        return status
        
    def predict(self, df ):        
      
        df['CalcDateTime'] = df.index
        status = self.findPeaksValleys(df)
        log.info('last change was = ' + str(status) )
        
       
        return status   