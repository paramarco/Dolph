# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 13:31:56 2021

@author: klio_ks
"""



import logging
import pandas as pd
from pandas import DataFrame
import numpy as np

import matplotlib.pyplot as plt

log = logging.getLogger("TradingPlatform")

import scipy.signal as signal

# import NeuronalNet_v6 as nn_v6
from scipy.signal import find_peaks

import pyampd.ampd as ampd 

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
        
        # self.TargetHours = None
        # if currentHour in range(7,23):
        self.TargetHours = range(7,23)
        # else:
        #     self.TargetHours = range(14,20)
            
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
    
    def findPeaksValleys (self, dataframe, sec, p):
        
        numWindowSize = 20
        if (p=='1Min'):
            numWindowSize=60
        elif (p == '30Min'):
            numWindowSize=60
        else:
            log.info('we careful')
            numWindowSize=100
        
        
        
        dataframe = dataframe.tail(numWindowSize)
        fluctuation = {}
        fluctuation_filtered = {}
        df = dataframe.copy()
        df['CalcDateTime'] = dataframe.index
        fluctuation['samplingWindow'] = df[['CalcDateTime','StartPrice','MaxPrice','MinPrice','EndPrice',] ]
        fluctuation_filtered['samplingWindow'] = df[['CalcDateTime','StartPrice','MaxPrice','MinPrice','EndPrice',] ]

        seriesEnd = df['EndPrice']
        seriesMax = df['MaxPrice']
        seriesMin = df['MinPrice']
        seriesStart = df['StartPrice']
        seriesAvg = (seriesEnd + seriesMax + seriesMin + seriesStart)/4
        times =     df['CalcDateTime']
        
        length = len(seriesAvg)
        log.debug('we careful: size for period: ' + p + ': '  + str(length))
        
        
        b, a = signal.butter(2, 0.2)
        zi = signal.lfilter_zi(b, a)

        filtered, _ = signal.lfilter(b, a, seriesAvg, zi=zi*seriesAvg[0])
        filtered2, _ = signal.lfilter(b, a, filtered, zi=zi*filtered[0])
        y = signal.filtfilt(b, a, seriesAvg)
        # plt.plot(y ,'g')
        # plt.plot(seriesAvg.to_numpy(), 'r')
        # plt.show()

        log.info('from ' + str(times[0]) + ' to ' + str(times[-1]) )
        
        # Find indices of peaks
        peak_idx_filtered, _ = find_peaks(y, distance=self.bestDistancePeak)        
        # Find indices of valleys (from inverting the signal)
        valley_idx_filtered, _ = find_peaks(-y, distance=self.bestDistanceValley)
        
        
        fluctuation_filtered['peak_idx'] = peak_idx_filtered
        fluctuation_filtered['valley_idx'] = valley_idx_filtered
        
        self.plotPeaksAndValleys (
            y,y, y,y, peak_idx_filtered,valley_idx_filtered, 
            fluctuation_filtered, sec, times,p
        )  
        
        
        # Find indices of peaks
        peak_idx, _ = find_peaks(seriesAvg, distance=self.bestDistancePeak )        
        
        # Find indices of valleys (from inverting the signal)
        # seriesAvgInv= -seriesAvg
        
        
        valley_idx, _ = find_peaks(seriesAvgInv, distance=self.bestDistanceValley)
        
        fluctuation['peak_idx'] = peak_idx
        fluctuation['valley_idx'] = valley_idx
        
        # self.plotPeaksAndValleys (
        #     seriesMax,seriesEnd, seriesMin, y, peak_idx,valley_idx, 
        #     fluctuation, sec, times,p
        # )  


        
        # peaksAMPD= ampd.find_peaks(seriesAvg, numWindowSize)
        # valleysAMPD= ampd.find_peaks(-seriesAvg, numWindowSize)
        # plt.plot( seriesAvg, 'b')
        # plt.plot(seriesAvg[peaksAMPD], 'k^', markersize=5)
        # plt.plot(seriesAvg[valleysAMPD], 'rv', markersize=5)
        # plt.title('Prediction for ' + p)
        # plt.show()
        return fluctuation
    
    
    def plotPeaksAndValleys (self, seriesMax,seriesEnd, seriesMin, filteredDataLine,
                             peak_idx,valley_idx,fluctuation,sec, times,p ):  
        
        seseccode = sec['seccode']
        lable = sec['label']
        if (seseccode == 'GZM1'):
            lable= 'GAZPROM'
        elif(seseccode == 'SRZ1'):
             lable= 'SBERBANK'
        elif(seseccode == 'SiM1'):
             lable= 'SIH'
        
        
        # Plot curves
        # t = np.arange(start=0, stop=len(seriesEnd), step=1, dtype=int)
        t = times
        plt.plot(t, seriesMax)
        plt.plot(t, seriesEnd)
        plt.plot(t, seriesMin)
        plt.plot(t , filteredDataLine, 'b')

        # Plot peaks (red) and valleys (blue)
        plt.plot(t[peak_idx], seriesMax[peak_idx], 'k^', markersize=15)
        plt.plot(t[valley_idx], seriesMin[valley_idx], 'rv', markersize=15)
        
 
        
        new_peak_ind=np.array(peak_idx)+1
        plt.plot(t[new_peak_ind],seriesEnd[new_peak_ind], 'm^') 
        new_valley_ind=np.array(valley_idx)+1
        plt.plot(t[new_valley_ind],seriesEnd[new_valley_ind], 'm^') 
        plt.title('Prediction for ' + lable + '   ' + p)
        

        
        plt.show()
        
    def predict(self, df, sec, p ):        
      
        fluctuation = self.findPeaksValleys(df, sec, p)
        
               
        return fluctuation