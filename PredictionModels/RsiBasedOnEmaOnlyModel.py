# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:07:13 2025

@author: klio_ks
"""


import pandas as pd
import logging

log = logging.getLogger("PredictionModel")


class RsiBasedOnEmaOnlyModel:
    
    def __init__(self, data, params, dolph):
        
        df = data['1Min']
       
        # Exclude non-price columns ('mnemonic', 'hastrade', 'addedvolume', 'numberoftrades')
        df = df.drop(columns=['hastrade', 'addedvolume', 'numberoftrades'], errors='ignore')
        
        # Ensure df has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a datetime index.")
        
        # Now proceed with renaming the columns as before
        self.df = df.rename(columns={
            'startprice': 'open',
            'maxprice': 'high',
            'minprice': 'low',
            'endprice': 'close'
        })
        self.params = params
        self.dolph = dolph
        # Calculate indicators for the dataframe
        self.df['RSI'] = self._calculate_rsi(self.df['close'], 14)
        self.df.dropna(inplace=True)

    def build_model(self):
        pass

    def train(self):
        pass

    def load_trained_model(self, security):
        return True


    def predict(self, df, sec, period):
        """
        Predict whether to go long, short, or no-go based on RSI and Stochastic indicators.
        """
        seccode = sec['seccode']
        entryPrice = exitPrice = 0.0

        lastClosePrice = self.dolph.getLastClosePrice( seccode)
        openPosition = self.dolph.tp.isPositionOpen( seccode )
        
        if openPosition:            
            positions = self.dolph.tp.get_PositionsByCode(seccode)
            for p in positions :
                entryPrice = p.entryPrice
                exitPrice = p.exitPrice 
                
        m = f"{seccode} last: {lastClosePrice}, entry: {entryPrice}, exit: {exitPrice}"                
        log.info(m)

        # Ensure df has a mnemonic column, and filter by seccode
        if 'mnemonic' in df.columns:
            df = df[df['mnemonic'] == seccode]
        else:
            log.error(f"DataFrame does not have a 'mnemonic' column.")
            raise KeyError("DataFrame is missing 'mnemonic' column.")            

        # Ensure the necessary columns are renamed to the correct price columns
        self.df = self.df.rename(columns={
            'startprice': 'open',
            'maxprice': 'high',
            'minprice': 'low',
            'endprice': 'close'
        })
    
        # Exclude non-price columns ('mnemonic', 'hastrade', 'addedvolume', 'numberoftrades')
        df = df.drop(columns=['hastrade', 'addedvolume', 'numberoftrades'], errors='ignore')
        
        # Ensure df has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a datetime index.")
        
        # Now proceed with renaming the columns as before
        self.df = df.rename(columns={
            'startprice': 'open',
            'maxprice': 'high',
            'minprice': 'low',
            'endprice': 'close'
        })
        # Calculate indicators for the dataframe

        self.df['RSI'] = self._calculate_rsi(self.df['close'], 14)
        self.df.dropna(inplace=True)
        
        # Print column names to ensure correct dataframe structure
        #log.debug(f"Columns in DataFrame before prediction: {self.df.columns}")
        
        # Ensure you're only working with the renamed price columns ('close', 'open', 'high', 'low')
        rsi= 0
        try:
        
            #log.debug(f"Total number of rows in the DataFrame: {len(self.df)}")
            rsi = self.df['RSI'].iloc[-1]
          
        
        except KeyError as e:
            log.debug(f"KeyError: {e}. Check if the dataframe has the correct price columns.")
            raise

        log.info(f"RSI: {rsi}")

        # Buy conditions: RSI < 30 (oversold), 
        if rsi < 30:
            log.info("predictor says long")
            return 'long'  # Buy signal
    
        # Sell conditions: RSI > 70 (overbought)
        elif rsi > 70 :
            log.info("predictor says short")
            return 'short'  # Sell signal
        log.info("predictor says nogo")
        # No clear signal to buy or sell
        return 'no-go'

    def _calculate_rsi(self, series, period=14):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
    
        # Compute EMA for average gain and average loss
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
    
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))



