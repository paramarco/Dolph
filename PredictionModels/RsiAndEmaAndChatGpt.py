import pandas as pd
import logging

log = logging.getLogger("PredictionModel")

class RsiAndEmaAndChatGpt:
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
            self.df['RSI'] = self._calculate_rsi(self.df['close'], 7)
            self.df.dropna(inplace=True)

            self.df['PrevClose'] = self.df['close'].shift(1)
            self.df['EMA50'] = self.df['close'].ewm(span=50, adjust=False).mean()
            self.df['EMA200'] = self.df['close'].ewm(span=200, adjust=False).mean()
            self.df.dropna(inplace=True)

            # Get the latest RSI value
            rsi = self.df['RSI'].iloc[-1]
    
            # Get EMA values
            ema50 = self.df['EMA50'].iloc[-1]
            ema200 = self.df['EMA200'].iloc[-1]

            log.info(f"Values rsi ema50 ema200 : {rsi} {ema50} {ema200}")
    
            # Buy conditions
            if rsi < 30 and ema50 > ema200:
                log.info(f"{seccode}: predictor says long")
                return 'long'
    
            # Sell conditions
            if rsi > 70 and ema50 < ema200:
                log.info(f"{seccode}: predictor says short")
                return 'short'
            
            log.info(f"{seccode}: predictor says nogo")

            # -------- 5-minute candle and indicator calculation (non-intrusive) --------
            df_5min = self.df.resample('5T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()

            df_5min['RSI_5min'] = self._calculate_rsi(df_5min['close'], 7)
            df_5min['EMA50_5min'] = df_5min['close'].ewm(span=50, adjust=False).mean()
            df_5min['EMA200_5min'] = df_5min['close'].ewm(span=200, adjust=False).mean()
            df_5min.dropna(inplace=True)

            if not df_5min.empty:
                rsi_5 = df_5min['RSI_5min'].iloc[-1]
                ema50_5 = df_5min['EMA50_5min'].iloc[-1]
                ema200_5 = df_5min['EMA200_5min'].iloc[-1]
                log.info(f"{seccode}: 5-min RSI={rsi_5:.2f}, EMA50={ema50_5:.2f}, EMA200={ema200_5:.2f}")

            return 'no-go'
       
        except Exception as e:        
            log.error(f"{seccode}: Failed : {e}", e)            
            return 'no-go'

    def _calculate_rsi(self, series, period):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
    
        # Initialize average gain/loss using SMA
        avg_gain = gain[:period].mean()
        avg_loss = loss[:period].mean()
    
        if avg_loss == 0:
            rs = float('inf')
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
    
        # If not enough data, return None
        if len(series) < period:
            return None
    
        # Smooth over remaining data, if available
        for i in range(period, len(series) - 1):
            current_gain = gain.iloc[i + 1]
            current_loss = loss.iloc[i + 1]
    
            avg_gain = (avg_gain * (period - 1) + current_gain) / period
            avg_loss = (avg_loss * (period - 1) + current_loss) / period
    
            if avg_loss == 0:
                rs = float('inf')
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
    
        return rsi

    def _calculate_atr(self, df, period):
        self.df['high-low'] = self.df['high'] - self.df['low']
        self.df['high-prevclose'] = abs(self.df['high'] - self.df['close'].shift(1))
        self.df['low-prevclose'] = abs(self.df['low'] - self.df['close'].shift(1))
        self.df['true_range'] = self.df[['high-low', 'high-prevclose', 'low-prevclose']].max(axis=1)
        return self.df['true_range'].rolling(period).mean()
