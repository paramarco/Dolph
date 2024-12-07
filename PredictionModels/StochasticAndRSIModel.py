import pandas as pd
import logging

log = logging.getLogger("PredictionModel")


class StochasticAndRSIModel:
    
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
        self.df['SMA50'] = self.df['close'].rolling(window=50).mean()
        self.df['SMA200'] = self.df['close'].rolling(window=200).mean()
        self.df['RSI'] = self._calculate_rsi(self.df['close'], 14)
        self.df['Stochastic_K'], self.df['Stochastic_D'] = self._calculate_stochastic(self.df['close'], self.df['low'], self.df['high'])
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
        self.df['SMA50'] = self.df['close'].rolling(window=50).mean()
        self.df['SMA200'] = self.df['close'].rolling(window=200).mean()
        self.df['RSI'] = self._calculate_rsi(self.df['close'], 14)
        self.df['Stochastic_K'], self.df['Stochastic_D'] = self._calculate_stochastic(self.df['close'], self.df['low'], self.df['high'])
        self.df.dropna(inplace=True)
        
        # Print column names to ensure correct dataframe structure
        #log.debug(f"Columns in DataFrame before prediction: {self.df.columns}")
        
        # Ensure you're only working with the renamed price columns ('close', 'open', 'high', 'low')
        rsi = sma50 = sma200 = 0
        try:
        
            #log.debug(f"Total number of rows in the DataFrame: {len(self.df)}")
            rsi = self.df['RSI'].iloc[-1]
            sma50 = self.df['SMA50'].iloc[-1]
            sma200 = self.df['SMA200'].iloc[-1]
        
        except KeyError as e:
            log.debug(f"KeyError: {e}. Check if the dataframe has the correct price columns.")
            raise

        log.info(f"RSI: {rsi} sma50: {sma50}, sma200: {sma200}")

        # Buy conditions: RSI < 30 (oversold), Stochastic %K > %D (bullish momentum)
        if rsi < 30 and sma50 > sma200:
            log.info("predictor says long")
            return 'long'  # Buy signal
    
        # Sell conditions: RSI > 70 (overbought), Stochastic %K < %D (bearish momentum)
        elif rsi > 70 and sma50 < sma200 :
            log.info("predictor says short")
            return 'short'  # Sell signal
        log.info("predictor says nogo")
        # No clear signal to buy or sell
        return 'no-go'


    def _calculate_rsi(self, series, period=14):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_stochastic(self, close, low, high, k_period=14, d_period=3):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_k, stoch_d