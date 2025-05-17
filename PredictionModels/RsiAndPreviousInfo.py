# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:07:13 2025

@author: klio_ks
"""

import pandas as pd
import logging

log = logging.getLogger("PredictionModel")

def plot_candles_with_indicators(df, seccode, filename="chart.png", share_name="Stock"):
    sub_df = df.tail(200).copy()
    sub_df.index.name = 'Date'

    apds = [
        mpf.make_addplot(sub_df['EMA50'], color='blue', linestyle='dashed'),
        mpf.make_addplot(sub_df['EMA200'], color='purple', linestyle='dotted'),
        mpf.make_addplot(sub_df['RSI'], panel=1, color='orange', ylabel='RSI'),
        mpf.make_addplot([30] * len(sub_df), panel=1, color='gray', linestyle='--'),
        mpf.make_addplot([70] * len(sub_df), panel=1, color='gray', linestyle='--')
    ]

    fig, axes = mpf.plot(
        sub_df,
        type='candle',
        style='charles',
        volume=False,
        addplot=apds,
        panel_ratios=(2, 1),
        figscale=1.2,
        figratio=(10, 6),
        returnfig=True
    )

    axes[0].set_title(f"{share_name} ({seccode})", fontsize=14)
    axes[0].legend(
        handles=[
            plt.Line2D([0], [0], color='blue', linestyle='dashed', label='EMA50'),
            plt.Line2D([0], [0], color='purple', linestyle='dotted', label='EMA200'),
            plt.Line2D([0], [0], color='orange', label='RSI'),
            plt.Line2D([0], [0], color='gray', linestyle='--', label='RSI Limits (30/70)')
        ],
        loc='upper left',
        frameon=True
    )

    fig.savefig(filename)
    plt.close(fig)
    return filename
class RsiAndPreviousInfo:
    
    def __init__(self, data, params, dolph):
        df = data['1Min']
       
        # Exclude non-price columns ('mnemonic', 'hastrade', 'addedvolume', 'numberoftrades')
        df = df.drop(columns=['hastrade', 'addedvolume', 'numberoftrades'], errors='ignore')
        
        # Ensure df has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a datetime index.")
        
        # Rename the columns to standard format
        self.df = df.rename(columns={
            'startprice': 'open',
            'maxprice': 'high',
            'minprice': 'low',
            'endprice': 'close'
        })
        self.params = params
        self.dolph = dolph
        self.rsiCoeff = self.params['rsiCoeff']
        # Calculate RSI and additional indicators
        self.df['RSI'] = self._calculate_rsi(self.df['close'], self.rsiCoeff)
        self.df['PrevRSI'] = self.df['RSI'].shift(1)
        self.df['PrevClose'] = self.df['close'].shift(1)
        self.df.dropna(inplace=True)

    def build_model(self):
        pass

    def train(self):
        pass

    def load_trained_model(self, security):
        return True

    def predict(self, df, sec, period):
        """
        Predict whether to go long, short, or no-go based on RSI and price action trends.
        """
        try:
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
            
            # Define lookback period
            lookback_period = 2  # Number of previous RSI and close values to check
            
            self.df['RSI'] = self._calculate_rsi(self.df['close'], self.rsiCoeff)
            self.df.dropna(inplace=True)
    
            # Get the last N RSI and close values
            prev_rsi_values = self.df['RSI'].iloc[-lookback_period-1:-1].values
            prev_close_values = self.df['close'].iloc[-lookback_period-1:-1].values
    
            # Ensure sufficient data exists
            if len(prev_rsi_values) < lookback_period or len(prev_close_values) < lookback_period:
                log.warning("Insufficient data for prediction")
                return 'no-go'
    
            # Log the last three RSI and close prices
            log.info(f"Last three RSI values: {prev_rsi_values}")
    
            # Get the latest RSI and close price
            rsi = self.df['RSI'].iloc[-1]
            image_filename = f"{seccode}_decision_chart.png"
            plot_candles_with_indicators(df, seccode, filename=image_filename, share_name=seccode)
            log.info(f"Making plot for {seccode} ...")
            # Buy conditions: RSI < 30, consistent oversold RSI, and decreasing close prices
            if all(rsi < 30 for rsi in prev_rsi_values) and rsi < 30 :
                log.info("predictor says long")
                return 'long'  # Buy signal
    
            # Sell conditions: RSI > 70, consistent overbought RSI, and increasing close prices
            if all(rsi > 70 for rsi in prev_rsi_values) and rsi > 70 :
                log.info("predictor says short")
                return 'short'  # Sell signal
            
            log.info("predictor says nogo")
            # No clear signal to buy or sell
            
            return  'no-go'
       
        except Exception as e:        
            log.error(f"Failed : {e}",e)            
            return 'no-go'

    def _calculate_rsi(self, series, period):
        
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
    
        # Compute EMA for average gain and average loss
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
    
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

