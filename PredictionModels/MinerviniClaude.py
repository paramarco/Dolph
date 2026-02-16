# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
import datetime as dt
from Configuration import Conf as cm

log = logging.getLogger("PredictionModel")

class MinerviniClaude:

    _calibration_cache = {}

    def __init__(self, data, security, dolph):
        self.df = data['1Min'].copy()
       
        # Exclude non-price columns ('mnemonic', 'hastrade', 'addedvolume', 'numberoftrades')
        self.df = self.df.drop(columns=['hastrade', 'numberoftrades'], errors='ignore')
        
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
        self.security = security
        self.seccode = security['seccode']      
        self.security_id = security['id']       
        self.params = security['params']  
        self.dolph = dolph

        if self.seccode not in MinerviniClaude._calibration_cache:
            log.info(f"seccode={self.seccode} calibrating margins from historical DB...")
            self._calibrate_margins_from_db()

    # =====================================================
    # PUBLIC
    # =====================================================

    def build_model(self):
        pass

    def train(self):
        pass

    def load_trained_model(self, security):
        return True

    def predict(self, df, sec, period):

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
                    m = f"seccode={seccode} already in a position, last close: {lastClosePrice}, entry: {entryPrice}, exit: {exitPrice}"                
                    log.info(m)
                           
            self.df = self._prepare_ohlcv(df)
            self.df = self._compute_indicators(self.df)

            phase = self._detect_phase(self.df)

            signal = self._generate_signal(self.df, phase)

            context = self._volume_context(self.df)

            # Trust strengthens breakout
            if signal == 'long' and context['trust']:
                log.info(f"{self.seccode}: TRUST breakout confirmed")

            # Divergence cancels longs
            if signal == 'long' and context['divergence']:
                log.info(f"{self.seccode}: volume divergence detected → cancelling long")
                signal = 'no-go'

            if signal == 'long' and context['buying_climax']:
                signal = 'no-go'

            # Strong reversal (only if real confluence)
            if (phase == 'expansion' and context['buying_climax'] and context['absorption']):
                signal = 'short'

            if (phase == 'trend' and signal == 'no-go'and context['no_supply']):
                latest = self.df.iloc[-1]
                # Only if the trend is upward
                if latest['EMA_FAST'] > latest['EMA_MID'] > latest['EMA_SLOW']:
                    signal = 'long'

            self._adapt_margin(sec, phase, self.df)

            utc_now = dt.datetime.now(dt.timezone.utc)
            factorMargin_Position = sec['params']['positionMargin']
            margin = lastClosePrice * factorMargin_Position
            if signal == 'long':
                exitPrice = lastClosePrice  + margin                     
            elif signal == 'short':
                exitPrice = lastClosePrice  - margin            


            exitPrice = "{0:0.{prec}f}".format(exitPrice, prec=sec['decimals'])
            margin = "{0:0.{prec}f}".format(sec['params']['positionMargin'], prec=5)

            log.info(
                f"seccode={self.seccode} phase={phase}, signal={signal},"
                f" margin={margin}, entryPrice={lastClosePrice},"
                f" exitPrice={exitPrice}, UTC-time={utc_now.isoformat()}"
            )

            return signal

        except Exception as e:
            log.error(f"{self.seccode}: MinerviniClaude failed: {e}")
            return 'no-go'

    # =====================================================
    # DATA PREP
    # =====================================================

    def _prepare_ohlcv(self, df):

        # Filter by mnemonic
        if 'mnemonic' in df.columns:
            df = df[df['mnemonic'] == self.seccode].copy()
        else:
            raise KeyError("DataFrame missing 'mnemonic' column")
        # Rename OHLC
        df = df.rename(columns={
            'startprice': 'open',
            'maxprice': 'high',
            'minprice': 'low',
            'endprice': 'close',
            'addedvolume': 'volume'
        })

        #log.debug(f"{self.seccode} columns AFTER rename : {df.columns.tolist()}")
        # Keep only required columns
        df = df[['open', 'high', 'low', 'close', 'volume']]

        return df

    # =====================================================
    # INDICATORS
    # =====================================================

    def _compute_indicators(self, df):
        """ This method computes all technical indicators required for:
            Phase detection (Contraction / Expansion / Trend)
            Signal generation (long / short / no-go)
            Volatility-based margin adaptation
        """
        # ---------------------------------------------------------
        # Exponential moving averages used for trend alignment. Used for:
        #   EMA alignment (trend detection)
        #   Short-term vs mid-term structure
        #   Pullback detection    
        # ---------------------------------------------------------
        df['EMA_FAST'] = df['close'].ewm(span=cm.EMA_FAST).mean()   # cm.EMA_FAST = 9
        df['EMA_MID'] = df['close'].ewm(span=cm.EMA_MID).mean()     # cm.EMA_MID = 21
        df['EMA_SLOW'] = df['close'].ewm(span=cm.EMA_SLOW).mean()   # cm.EMA_SLOW = 50
        # ---------------------------------------------------------
        # RSI (Momentum Filter). Used for:
        #   Avoid overbought/oversold extremes
        #   Confirm directional entries
        #   Filter false breakouts
        # ---------------------------------------------------------
        df['RSI'] = self._rsi_series(df['close'], cm.RSI_PERIOD) # cm.RSI_PERIOD = 14
        # ---------------------------------------------------------
        # ATR + ATR Slope (Volatility Regime). Measures volatility level, Used to:
        #   Detect expansion (volatility rising)
        #   Detect contraction (volatility compressing)
        # ---------------------------------------------------------
        df['ATR'] = self._atr(df, cm.ATR_PERIOD)                # cm.ATR_PERIOD = 14 
        df['ATR_slope'] = df['ATR'].diff(cm.ATR_SLOPE_WINDOW)   # cm.ATR_SLOPE_WINDOW = 5
        # ---------------------------------------------------------
        # ADX + DI (Trend Strength). Measures trend strength and direction, Used to:
        #   Confirm directional strength
        #   Filter ranging markets
        # ---------------------------------------------------------
        df['ADX'], df['+DI'], df['-DI'] = self._adx(df, cm.ADX_PERIOD) # cm.ADX_PERIOD = 14
        # ---------------------------------------------------------
        # BOLLINGER BAND + Percentile. Measures compression vs expansion of volatility, Used to:
        #   Detect volatility compression
        #   Detect breakout expansion
        # ---------------------------------------------------------        
        ma = df['close'].rolling(cm.BB_WINDOW).mean()           # cm.BB_WINDOW = 20
        std = df['close'].rolling(cm.BB_WINDOW).std()           # cm.BB_WINDOW = 20
        # Normalized width (volatility relative to price level)
        df['BB_width'] = (cm.BB_STD * std) / ma                 # cm.BB_STD = 2
        # Percentile of BB width over rolling window            # cm.BB_PERCENTILE_WINDOW = 100
        df['BB_width_pctile'] = df['BB_width'].rolling(cm.BB_PERCENTILE_WINDOW).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        # ---------------------------------------------------------
        # FAIR VALUE PRICE (FVP). Rolling mean of close. Used for:
        # Statistical center of recent price movement
        # Used for mean-reversion during expansion
        # ---------------------------------------------------------
        df['FVP'] = df['close'].rolling(cm.FVP_WINDOW).mean()   # cm.FVP_WINDOW = 30

        df.dropna(inplace=True)

        return df

    def _rsi_series(self, series, period):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period).mean()
        avg_loss = loss.ewm(alpha=1/period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _atr(self, df, period):
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _adx(self, df, period):

        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = self._atr(df, period)

        plus_di = 100 * (plus_dm.rolling(period).sum() / tr)
        minus_di = 100 * (minus_dm.rolling(period).sum() / tr)

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()

        return adx, plus_di, minus_di

    # =====================================================
    # PHASE DETECTION
    # =====================================================

    def _detect_phase(self, df):
        """ 
        This method determines the current market regime using volatility and trend metrics.
        It classifies the security into one of three VCP phases:
        Expansion
            Volatility is increasing strongly.
            Bollinger width percentile indicates widening range.
            Early breakout regime.
        Trend
            Strong directional movement.
            ADX confirms trend strength.
            EMA alignment confirms direction.
        Contraction
            Default fallback.
            Low volatility compression phase.
        """
        latest = df.iloc[-1]  # Get the latest row of indicators
        # ---------------------------------------------------------
        # Expansion is detected when volatility increases rapidly
        # ATR_slope > threshold AND Bollinger width percentile high
        # ---------------------------------------------------------
        if (
            latest['ATR_slope'] > cm.VCP_ATR_SLOPE_EXPANSION  and  # VCP_ATR_SLOPE_EXPANSION = 0.15
            latest['BB_width_pctile'] > cm.VCP_BB_WIDTH_PERCENTILE_EXPANSION  # VCP_BB_WIDTH_PERCENTILE_EXPANSION = 0.5
        ):
            return 'expansion'
        # ---------------------------------------------------------
        # Trend is detected when both:
        #   1) ADX above threshold (strong directional movement)
        #   2) EMA alignment either bullish or bearish
        # ---------------------------------------------------------
        if (
            latest['ADX'] > cm.VCP_ADX_TREND_THRESHOLD and  # VCP_ADX_TREND_THRESHOLD = 25
            (
                latest['EMA_FAST'] > latest['EMA_MID'] > latest['EMA_SLOW']                
                or                
                latest['EMA_FAST'] < latest['EMA_MID'] < latest['EMA_SLOW']
            )
        ):
            return 'trend'
        # ---------------------------------------------------------
        # Default = Contraction Phase, Low volatility compression regime
        # ---------------------------------------------------------
        return 'contraction'

    # =====================================================
    # SIGNAL GENERATION
    # =====================================================

    def _generate_signal(self, df, phase):

        latest = df.iloc[-1]        # Get latest indicator values
        # ---------------------------------------------------------
        # CONTRACTION PHASE No trades allowed in volatility compression regime
        # ---------------------------------------------------------
        if phase == 'contraction':
            return 'no-go'
        # ---------------------------------------------------------
        # EXPANSION PHASE
        #   Mean-reversion relative to Fair Value Price (FVP)
        #   Entry when price deviates significantly from center
        # ---------------------------------------------------------
        if phase == 'expansion':
            # Relative deviation from statistical center
            deviation = (latest['close'] - latest['FVP']) / latest['FVP'] 
            # Short signal:  Price above fair value AND RSI not weak
            if (
                deviation > cm.EXPANSION_DEVIATION_THRESHOLD and    # cm.EXPANSION_DEVIATION_THRESHOLD = 0.0005
                latest['RSI'] > cm.EXPANSION_RSI_SHORT_MIN          # cm.EXPANSION_RSI_SHORT_MIN = 40
            ):                
                return 'short'
            # Long signal: Price below fair value AND RSI not overheated
            if (
                deviation < -cm.EXPANSION_DEVIATION_THRESHOLD and   # cm.EXPANSION_DEVIATION_THRESHOLD = 0.0005
                latest['RSI'] < cm.EXPANSION_RSI_LONG_MAX           # cm.EXPANSION_RSI_LONG_MAX = 60
            ):
                return 'long'
            return 'no-go'

        # ---------------------------------------------------------
        # TREND PHASE Trend-following continuation entries
        #   Requires EMA alignment + DI confirmation + RSI filter
        # ---------------------------------------------------------
        if phase == 'trend':
            # Bullish Trend Continuation
            #   cm.TREND_RSI_LONG_MIN = 40
            #   cm.TREND_RSI_LONG_MAX = 70
            if (
                latest['EMA_FAST'] > latest['EMA_MID'] > latest['EMA_SLOW']
                and latest['+DI'] > latest['-DI']
                and cm.TREND_RSI_LONG_MIN < latest['RSI'] < cm.TREND_RSI_LONG_MAX
            ):
                return 'long'
            # Bearish Trend Continuation
            #   cm.TREND_RSI_SHORT_MIN = 30
            #   cm.TREND_RSI_SHORT_MAX = 60
            if (
                latest['EMA_FAST'] < latest['EMA_MID'] < latest['EMA_SLOW']
                and latest['-DI'] > latest['+DI']
                and cm.TREND_RSI_SHORT_MIN < latest['RSI'] < cm.TREND_RSI_SHORT_MAX                
            ):
                return 'short'
        
        # Default Fallback
        return 'no-go'

    # =====================================================
    # MARGIN ADAPTATION
    # =====================================================

    def _adapt_margin(self, sec, phase, df):

        latest = df.iloc[-1]        # Get latest indicator values
        close = latest['close']     # Current price (used to normalize ATR)

        # If calibration exists, use pre-optimized margins
        if self.seccode in MinerviniClaude._calibration_cache:          
            margins = MinerviniClaude._calibration_cache[self.seccode]
            if phase in margins:
                m = margins[phase]
                msg = f"seccode {self.seccode} calibration exists, using pre-optimized margins {m}"
                log.debug(msg)
                sec['params']['positionMargin'] = m                
                return
        # Contraction Phase. Very tight margin because volatility is low
        if phase == 'contraction':
            m = cm.MARGIN_CONTRACTION_FIXED                 # cm.MARGIN_CONTRACTION_FIXED = 0.0015
        
        # Expansion Phase. Reflects volatility breakout amplitude
        elif phase == 'expansion':
           # cm.MARGIN_EXPANSION_MULTIPLIER = 1.5
            raw_margin = latest['BB_width'] * cm.MARGIN_EXPANSION_MULTIPLIER 
            # Clamp margin inside configured bounds
            m = min(
                max(raw_margin, cm.MARGIN_EXPANSION_MIN),   # cm.MARGIN_EXPANSION_MIN = 0.002
                cm.MARGIN_EXPANSION_MAX                     # cm.MARGIN_EXPANSION_MAX = 0.008
            )
        # Trend Phase. Reflects sustained directional volatility. Margin proportional to ATR normalized
        elif phase == 'trend':
            m = min(max(2.0 * latest['ATR'] / close, 0.002), 0.006)
            # cm.MARGIN_TREND_ATR_MULTIPLIER = 2.0
            raw_margin = (
                cm.MARGIN_TREND_ATR_MULTIPLIER *
                latest['ATR'] / close
            )
            # Clamp margin inside configured bounds
            m = min(
                max(raw_margin, cm.MARGIN_TREND_MIN),       # cm.MARGIN_TREND_MIN = 0.002
                cm.MARGIN_TREND_MAX                         # cm.MARGIN_TREND_MAX = 0.006
            )

        # Fallback (safety)    
        else:
            m = cm.MARGIN_CONTRACTION_FIXED                 # cm.MARGIN_CONTRACTION_FIXED = 0.0015
        
        # Apply computed margin to security parameters
        sec['params']['positionMargin'] = float(m)
        

    # =====================================================
    # DB CALIBRATION
    # =====================================================

    def _calibrate_margins_from_db(self):

        try:
            # Define historical lookback period,  cm.CALIBRATION_LOOKBACK_DAYS = 90
            since = dt.datetime.now() - dt.timedelta(days=cm.CALIBRATION_LOOKBACK_DAYS)
            # Fetch historical 1-minute data from database, cm.CALIBRATION_LIMIT_RESULTS = 5000
            df = self.dolph.ds.searchData(since, limitResult=5000)
            df = df['1Min'].copy()
            # Prepare OHLCV format
            hist = self._prepare_ohlcv(df)
            # Ensure sufficient data for statistical validity, cm.CALIBRATION_MIN_ROWS = 1000
            if hist is None or len(hist) < cm.CALIBRATION_MIN_ROWS:
                log.info(f"{self.seccode}: calibration failed: less than 1000 items")
                return
            # Compute ATR for volatility regime split, cm.ATR_PERIOD = 14
            hist['ATR'] = self._atr(hist, cm.ATR_PERIOD)
            median_atr = hist['ATR'].median()
            # Split dataset into low and high volatility halves
            # It’s a statistical proxy for phase without running full phase detection historically (faster).
            # This approximates:
            #   Regime	            Proxy
            #   Low volatility	    Expansion
            #   High volatility	    Trend
            low_vol = hist[hist['ATR'] < median_atr]
            high_vol = hist[hist['ATR'] >= median_atr]
            # Generate candidate margin values, (MIN=0.001, MAX=0.006, STEPS=10)
            margins = np.linspace(
                cm.CALIBRATION_MARGIN_MIN,
                cm.CALIBRATION_MARGIN_MAX,
                cm.CALIBRATION_MARGIN_STEPS
            )
            # Run calibration simulation for each regime
            best_low = self._run_calibration(low_vol, margins)
            best_high = self._run_calibration(high_vol, margins)
            # Store optimized results
            MinerviniClaude._calibration_cache[self.seccode] = {
                'expansion': best_low,
                'trend': best_high
            }

            log.info(
                f"calibration complete for seccode={self.seccode}, results:"
                f"{MinerviniClaude._calibration_cache[self.seccode]}"
            )

        except Exception as e:
            log.error(f"{self.seccode}: calibration failed {e}")


    def _run_calibration(self, df, margins):
        """This is a simplified Monte Carlo forward simulation
            • Iterates over candidate margin values.
            • Simulates a trade at every historical bar:
                • Entry at close
                • TP = entry x (1 + margin)
                • SL = entry x (1 - k x margin)
            • Looks forward N bars.
            • Checks which is hit first: TP or SL.
            • Computes win rate, score = wins / total That maximizes probability of TP hit.
            • Selects margin with best score.
        """
        # Initialize best score and margin
        best_score = 0  
        best_margin = 0.003
        # Loop over candidate margin values
        for m in margins:
            wins = 0
            total = 0
            # Slide through historical dataset cm.CALIBRATION_LOOKAHEAD_BARS = 60
            for i in range(len(df) - cm.CALIBRATION_LOOKAHEAD_BARS):    
                entry = df['close'].iloc[i]
                # Takeprofit level
                tp = entry * (1 + m)        
                # Stoploss level cm.CALIBRATION_STOPLOSS_MULTIPLIER = 3.0
                sl = entry * ( 1 - cm.CALIBRATION_STOPLOSS_MULTIPLIER * m ) 
                # Lookahead window, cm.CALIBRATION_LOOKAHEAD_BARS = = 60
                window = df.iloc[i:i+cm.CALIBRATION_LOOKAHEAD_BARS]
                # Check if TP is reached within window
                hit_tp = (window['high'] >= tp).any()
                # Check if SL is reached within window
                hit_sl = (window['low'] <= sl).any()
                # Count win only if TP hit and SL not hit first
                if hit_tp and not hit_sl:
                    wins += 1

                total += 1
            # Compute score (TP hit probability)
            if total > 0:
                score = wins / total
                # Update best margin if better score found
                if score > best_score:
                    best_score = score
                    best_margin = m

        return float(best_margin)


    def _volume_context(self, df):

        latest = df.iloc[-1]

        # Compute relative metrics
        volume_avg = df['volume'].rolling(cm.VOLUME_AVG_WINDOW).mean().iloc[-1]
        relative_volume = latest['volume'] / volume_avg

        candle_body = abs(latest['close'] - latest['open'])
        candle_range = latest['high'] - latest['low']
        relative_body = candle_body / latest['ATR']

        price_slope = df['close'].diff(cm.VOLUME_SLOPE_WINDOW).iloc[-1]
        volume_slope = df['volume'].diff(cm.VOLUME_SLOPE_WINDOW).iloc[-1]

        context = {}

        context['healthy'] = (
            (price_slope > 0 and volume_slope > 0)
            or
            (price_slope < 0 and volume_slope < 0)
        )

        context['absorption'] = (
            relative_volume > cm.BIG_VOLUME_THRESHOLD
            and relative_body < 0.5
        )

        context['trust'] = (
            relative_body > cm.BIG_BODY_ATR_THRESHOLD
            and relative_volume > cm.BIG_VOLUME_THRESHOLD
        )

        context['divergence'] = (
            df['close'].iloc[-1] > df['close'].iloc[-cm.DIVERGENCE_LOOKBACK]
            and df['volume'].iloc[-1] < df['volume'].iloc[-cm.DIVERGENCE_LOOKBACK]
        )

        context['stopping_volume'] = (
            relative_volume > cm.EXTREME_VOLUME_THRESHOLD
            and latest['close'] < latest['open']
            and latest['close'] > latest['low'] + (candle_range * 0.3)
        )

        context['buying_climax'] = (
            relative_volume > cm.EXTREME_VOLUME_THRESHOLD
            and relative_body > cm.EXTREME_BODY_ATR_THRESHOLD
            and latest['close'] > latest['open']
        )

        context['no_supply'] = (
            price_slope < 0
            and relative_volume < 0.7
        )

        return context   
