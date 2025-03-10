import pandas as pd
import logging

log = logging.getLogger("PredictionModel")

class RsiAndAtr:
    def __init__(self, data, params, dolph):
        self.df = data['1Min'].copy()
       
        # Exclude non-price columns ('mnemonic', 'hastrade', 'addedvolume', 'numberoftrades')
        self.df = self.df.drop(columns=['hastrade', 'addedvolume', 'numberoftrades'], errors='ignore')
        
        # Ensure df has a datetime index
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a datetime index.")
        
        # Rename the columns to standard format
        self.df = self.df.rename(columns={
            'startprice': 'open',
            'maxprice': 'high',
            'minprice': 'low',
            'endprice': 'close'
        })
        self.params = params
        self.dolph = dolph

    def build_model(self):
        pass

    def train(self):
        pass

    def load_trained_model(self, security):
        return True

    def predict(self, df, sec, period):
        """
        Predict whether to go long, short, or no-go based on RSI, ATR, EMA, and price trends.
        """
        try:
            seccode = sec['seccode']
            entryPrice = exitPrice = 0.0
    
            lastClosePrice = self.dolph.getLastClosePrice(seccode)
            openPosition = self.dolph.tp.isPositionOpen(seccode)
            
            if openPosition:            
                positions = self.dolph.tp.get_PositionsByCode(seccode)
                for p in positions:
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
            
    
            # Calculate indicators for this specific seccode
            self.df['RSI'] = self._calculate_rsi(self.df['close'], 14)
            self.df.dropna(inplace=True)

            self.df['PrevRSI'] = self.df['RSI'].shift(1)
            self.df['PrevPrevRSI'] = self.df['RSI'].shift(2)  # Two steps before
            self.df['PrevClose'] = self.df['close'].shift(1)
            self.df['ATR'] = self._calculate_atr(self.df, 14)
            self.df['EMA50'] = self.df['close'].ewm(span=50, adjust=False).mean()
            self.df['EMA200'] = self.df['close'].ewm(span=200, adjust=False).mean()
            self.df.dropna(inplace=True)

            # Get latest ATR value
            atr = self.df['ATR'].iloc[-1]
            price = self.df['close'].iloc[-1]
            atr_threshold = 0.002 * price  # 0.2% of price
            
            tpmultiplier = 2
            coef = ( atr * tpmultiplier ) / price

            log.info(f"setting margin for {seccode}: {coef} ")
            # updating new calculated params
            coef = 0.002
            params = {'longPositionMargin': coef, 'stopLossCoefficient': 2 }
            self.dolph.setSecurityParams( seccode, **params )   

           
    
            # Get the latest and previous RSI values

            rsi = self.df['RSI'].iloc[-1]
            prev_rsi = self.df['RSI'].iloc[-2]
            prev_prev_rsi = self.df['RSI'].iloc[-3]
    
            # Get EMA values
            ema50 = self.df['EMA50'].iloc[-1]
            ema200 = self.df['EMA200'].iloc[-1]

            log.info(f"Values: {rsi} {prev_rsi} {prev_prev_rsi} {ema50} {ema200}")
    
            # Buy conditions: RSI was below 30, remained there, now increasing; EMA50 > EMA200 (bullish trend); Price > EMA50
            if prev_rsi < 30 and rsi > prev_rsi and ema50 > ema200:
                log.info(f"{seccode}: predictor says long")
                return 'long'  # Buy signal
    
            # Sell conditions: RSI was above 70, remained there, now decreasing; EMA50 < EMA200 (bearish trend); Price < EMA50
            if prev_rsi > 70 and rsi < prev_rsi and ema50 < ema200:
                log.info(f"{seccode}: predictor says short")
                return 'short'  # Sell signal
            
            log.info(f"{seccode}: predictor says nogo")
            return 'no-go'
       
        except Exception as e:        
            log.error(f"{seccode}: Failed : {e}", e)            
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

    def _calculate_atr(self, df, period=14):

        self.df['high-low'] = self.df['high'] - self.df['low']
        self.df['high-prevclose'] = abs(self.df['high'] - self.df['close'].shift(1))
        self.df['low-prevclose'] = abs(self.df['low'] - self.df['close'].shift(1))
        self.df['true_range'] = self.df[['high-low', 'high-prevclose', 'low-prevclose']].max(axis=1)
        return self.df['true_range'].rolling(period).mean()

