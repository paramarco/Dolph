# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 20:24:28 2023

@author: klio_ks
"""
import logging
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger("TradingPlatform")

class Predictions:
    def __init__(self, predictions, training_set):
        self.predictions = predictions
        self.training_set = training_set  
    
class Model:
    def __init__(self, df, params, period, mode, currentHour):
        
        self.id = 'fractal_dimension'
        
        self.df = df
        self.model = None
        self.params = params        
        self.period = period
        self.mode = mode 
                 
        self.TargetHours = range(7,23)     
        tb = self.TargetHours[0]
        te = self.TargetHours[-1]
        interval = str(tb) + '_' + str(te) + 'Hour'
        samples = str(params['minNumPastSamples'])  +'Samples'
        nameParts = [ __name__, self.period, samples, interval]        
        self.fileName = '_'.join(nameParts)
        
        self.numSamples = self.getNumSamples()
        self.minPrice = self.getMinPrice()
        self.maxPrice = self.getMaxPrice()
        self.ΔPrice = self.getΔPrice()
        
        log.info(" loading Model ...")
        msg = 'Number of Samples to compute: ' + str(self.numSamples) 
        msg += ', MinPrice: ' + str(self.minPrice) 
        msg += ', MaxPrice: ' + str(self.maxPrice)
        msg += ', delta Price: ' + str(self.ΔPrice)
        msg += ', fractal dimension: ' + str(self.findCurveDimension())
        
        log.info(msg)
    
    
    def findNumberOfCells (self):
        numOfCells = 0
        condition_1 = False
        condition_2 = False        
        maxMinSeriesDelta = self.df['MaxPrice'] - self.df['MinPrice']
        minPrice = self.minPrice
        ΔPrice = self.ΔPrice 
        
        for i in range( 1, self.numSamples) :
            for j in range( 0, self.numSamples):
                
                maxPriceItem = self.df['MaxPrice'][i]
                minPriceItem = self.df['MinPrice'][i]
   
                if ( j == 0):
                    condition_1 = maxPriceItem > ( minPrice ) 
                    condition_2 = maxPriceItem < ( minPrice + (ΔPrice) )
                    condition_3 = minPriceItem > ( minPrice  ) 
                    condition_4 = minPriceItem < ( minPrice + (ΔPrice) )
                else:
                    condition_1 = maxPriceItem > ( minPrice + (ΔPrice * (j  ) )) 
                    condition_2 = maxPriceItem < ( minPrice + (ΔPrice * (j+1) ))
                    condition_3 = minPriceItem > ( minPrice + (ΔPrice * (j  ) )) 
                    condition_4 = minPriceItem < ( minPrice + (ΔPrice * (j+1) ))
               
                if ( (condition_1 and condition_2) or (condition_3 and condition_4)):
                    numOfCells = numOfCells + 1
                                    
        log.info( "number of Cells: " + str(numOfCells) )
        
        return numOfCells

    def predict(self, df, sec, p ):        
        
        return None
    
    def getNumSamples(self):
        return len(self.df.index)
    
    def getMinPrice(self):
        return self.df['MinPrice'].min()
    
    def getMaxPrice(self):
        return self.df['MaxPrice'].max()
    
    def getΔPrice(self):
        Δ = (self.getMaxPrice() - self.getMinPrice())/self.getNumSamples()
        return Δ
    
    def findCurveDimension(self):
        
        numerator = np.log(self.findNumberOfCells())
        denominator =  np.log( (self.numSamples**2) )
        dimension = numerator / denominator
        
        return dimension
        
    
