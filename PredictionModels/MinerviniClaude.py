# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
import datetime as dt

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
            log.info(f"{self.seccode}: calibrating margins from historical DB...")
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
                    
            m = f"{seccode} last: {lastClosePrice}, entry: {entryPrice}, exit: {exitPrice}"                
            log.info(m)
       
            self.df = self._prepare_ohlcv(df)
            self.df = self._compute_indicators(self.df)

            phase = self._detect_phase(self.df)
            signal = self._generate_signal(self.df, phase)

            self._adapt_margin(sec, phase, self.df)

            log.info(
                f"{self.seccode}: phase={phase}, "
                f"signal={signal}, "
                f"longMargin={sec['params']['positionMargin']} "
            )

            return signal

        except Exception as e:
            log.error(f"{self.seccode}: MinerviniClaude failed: {e}")
            return 'no-go'

    # =====================================================
    # DATA PREP
    # =====================================================

    def _prepare_ohlcv(self, df):

        # Ensure df has a mnemonic column, and filter by seccode
        if 'mnemonic' in df.columns:
            df = df[df['mnemonic'] == self.seccode]
        else:
            log.error(f"DataFrame does not have a 'mnemonic' column.")
            raise KeyError("DataFrame is missing 'mnemonic' column.")

        df = df[df['mnemonic'] == self.seccode].copy()

        # Exclude non-price columns ('mnemonic', 'hastrade', 'numberoftrades')
        df = df.drop(columns=['hastrade', 'numberoftrades'], errors='ignore')
            
        # Ensure df has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a datetime index.")

        # Now proceed with renaming the columns as before
        df = df.rename(columns={
            'startprice': 'open',
            'maxprice': 'high',
            'minprice': 'low',
            'endprice': 'close',
            'addedvolume' : 'volume'
        })

        df = df[['open', 'high', 'low', 'close','volume']]

        return df

    # =====================================================
    # INDICATORS
    # =====================================================

    def _compute_indicators(self, df):

        df['EMA9'] = df['close'].ewm(span=9).mean()
        df['EMA21'] = df['close'].ewm(span=21).mean()
        df['EMA50'] = df['close'].ewm(span=50).mean()

        df['RSI'] = self._rsi_series(df['close'], 14)

        df['ATR'] = self._atr(df, 14)
        df['ATR_slope'] = df['ATR'].diff(5)

        df['ADX'], df['+DI'], df['-DI'] = self._adx(df, 14)

        # Bollinger width
        ma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        df['BB_width'] = (2 * std) / ma
        df['BB_width_pctile'] = df['BB_width'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )

        # Fair Value Price
        df['FVP'] = df['close'].rolling(30).mean()

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

        latest = df.iloc[-1]

        if (
            latest['ATR_slope'] > 0.15 and
            (latest['BB_width_pctile'] > 0.5)
        ):
            return 'expansion'

        if (
            latest['ADX'] > 25 and
            (
                latest['EMA9'] > latest['EMA21'] > latest['EMA50']
                or
                latest['EMA9'] < latest['EMA21'] < latest['EMA50']
            )
        ):
            return 'trend'

        return 'contraction'

    # =====================================================
    # SIGNAL GENERATION
    # =====================================================

    def _generate_signal(self, phase, df):

        latest = df.iloc[-1]

        if phase == 'contraction':
            return 'no-go'

        if phase == 'expansion':
            deviation = (latest['close'] - latest['FVP']) / latest['FVP']

            if deviation > 0.0005 and latest['RSI'] > 40:
                return 'short'
            if deviation < -0.0005 and latest['RSI'] < 60:
                return 'long'
            return 'no-go'

        if phase == 'trend':

            if (
                latest['EMA9'] > latest['EMA21'] > latest['EMA50']
                and latest['+DI'] > latest['-DI']
                and 40 < latest['RSI'] < 70
            ):
                return 'long'

            if (
                latest['EMA9'] < latest['EMA21'] < latest['EMA50']
                and latest['-DI'] > latest['+DI']
                and 30 < latest['RSI'] < 60
            ):
                return 'short'

        return 'no-go'

    # =====================================================
    # MARGIN ADAPTATION
    # =====================================================

    def _adapt_margin(self, sec, phase, df):

        latest = df.iloc[-1]
        close = latest['close']

        if self.seccode in MinerviniClaude._calibration_cache:
            margins = MinerviniClaude._calibration_cache[self.seccode]
            if phase in margins:
                m = margins[phase]
                sec['params']['positionMargin'] = m                
                return

        if phase == 'contraction':
            m = 0.0015

        elif phase == 'expansion':
            m = min(max(latest['BB_width'] * 1.5, 0.002), 0.008)

        elif phase == 'trend':
            m = min(max(2.0 * latest['ATR'] / close, 0.002), 0.006)

        sec['params']['positionMargin'] = float(m)
        

    # =====================================================
    # DB CALIBRATION
    # =====================================================

    def _calibrate_margins_from_db(self):

        try:

            since = dt.datetime.now() - dt.timedelta(days=90)
            df = self.dolph.ds.searchData(since)
            df = df['1Min'].copy()

            hist = self._prepare_ohlcv(df)

            if hist is None or len(hist) < 1000:
                log.info(f"{self.seccode}: calibration failed: less than 1000 items")
                return

            hist['ATR'] = self._atr(hist, 14)
            median_atr = hist['ATR'].median()

            low_vol = hist[hist['ATR'] < median_atr]
            high_vol = hist[hist['ATR'] >= median_atr]

            margins = np.linspace(0.001, 0.006, 10)

            best_low = self._run_calibration(low_vol, margins)
            best_high = self._run_calibration(high_vol, margins)

            MinerviniClaude._calibration_cache[self.seccode] = {
                'expansion': best_low,
                'trend': best_high
            }

            log.info(
                f"{self.seccode}: calibration complete: "
                f"{MinerviniClaude._calibration_cache[self.seccode]}"
            )

        except Exception as e:
            log.error(f"{self.seccode}: calibration failed {e}")

    def _run_calibration(self, df, margins):

        best_score = 0
        best_margin = 0.003

        for m in margins:
            wins = 0
            total = 0

            for i in range(len(df) - 60):
                entry = df['close'].iloc[i]
                tp = entry * (1 + m)
                sl = entry * (1 - 3*m)

                window = df.iloc[i:i+60]

                hit_tp = (window['high'] >= tp).any()
                hit_sl = (window['low'] <= sl).any()

                if hit_tp and not hit_sl:
                    wins += 1

                total += 1

            if total > 0:
                score = wins / total
                if score > best_score:
                    best_score = score
                    best_margin = m

        return float(best_margin)
