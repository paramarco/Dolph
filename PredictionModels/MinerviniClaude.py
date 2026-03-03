# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
import datetime as dt
import pytz
from Configuration import Conf as cm

log = logging.getLogger("PredictionModel")

class MinerviniClaude:

    _calibration_cache = {}

    # Params requiring full indicator recomputation when changed
    _INDICATOR_PARAMS = frozenset({
        'EMA_FAST', 'EMA_MID', 'EMA_SLOW', 'RSI_PERIOD',
        'ATR_PERIOD', 'ATR_SLOPE_WINDOW', 'ADX_PERIOD',
        'BB_WINDOW', 'BB_STD', 'BB_PERCENTILE_WINDOW', 'FVP_WINDOW'
    })

    # Params excluded from optimization (non-numeric or meta-calibration)
    _EXCLUDE_PARAMS = frozenset({
        'algorithm', 'entryByMarket', 'period',
        'exitTimeSeconds', 'minNumPastSamples',
        'CALIBRATION_LOOKBACK_DAYS', 'CALIBRATION_LIMIT_RESULTS',
        'CALIBRATION_MIN_ROWS', 'CALIBRATION_MARGIN_MIN',
        'CALIBRATION_MARGIN_MAX', 'CALIBRATION_MARGIN_STEPS',
        'CALIBRATION_LOOKAHEAD_BARS', 'BUYING_CLIMAX_COOLDOWN_SECONDS',
        'POSITION_COOLDOWN_SECONDS',
        'MAX_CALIBRATION_PASSES', 'MIN_CALIBRATION_IMPROVEMENT',
        'TRAILING_TP_ENABLED',
        'MIN_CONFIDENCE'
    })

    def __init__(self, data, security, dolph):
        """Initialize the prediction model for a single security.

        Loads 1-minute OHLCV data, sets up security params, and runs
        calibration (TEST_OFFLINE) or loads cached params (OPERATIONAL).
        """
        if '1Min' in data:
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
        else:
            self.df = pd.DataFrame()
        self.security = security
        self.seccode = security['seccode']
        self.security_id = security['id']
        self.params = security['params']
        self.dolph = dolph

        if cm.MODE == 'OPERATIONAL':
            # Skip calibration, use pre-loaded params from DB (loaded by DolphRobot)
            if self.seccode in MinerviniClaude._calibration_cache:
                cached = MinerviniClaude._calibration_cache[self.seccode]
                for k, v in cached.items():
                    self.security['params'][k] = v
                self.params = self.security['params']
            log.info(f"seccode={self.seccode} OPERATIONAL mode, using pre-loaded params")
        else:
            # TEST_OFFLINE or other modes: run calibration
            if self.seccode not in MinerviniClaude._calibration_cache:
                log.info(f"seccode={self.seccode} calibrating params from historical DB...")
                self._calibrate_params_from_db()
            else:
                cached = MinerviniClaude._calibration_cache[self.seccode]
                for k, v in cached.items():
                    self.security['params'][k] = v
                self.params = self.security['params']
                log.info(f"seccode={self.seccode} loaded calibrated params from cache")

    # =====================================================
    # PUBLIC
    # =====================================================

    def build_model(self):
        """No-op. MinerviniClaude uses rule-based signals, no ML model to build."""
        pass

    def train(self):
        """No-op. Calibration replaces traditional training (see _calibrate_params_from_db)."""
        pass

    def load_trained_model(self, security):
        """No-op. Returns True for interface compatibility with DolphRobot."""
        return True

    def predict(self, df, sec, period):
        """Generate a trading signal (long/short/no-go) for the current bar.

        Pipeline: prepare OHLCV → compute indicators → detect VCP phase →
        generate signal with confidence score → adapt margin to volatility.
        Returns dict with 'signal' and 'confidence'.
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
                    m = f"seccode={seccode} already in a position, last close: {lastClosePrice}, entry: {entryPrice}, exit: {exitPrice}"
                    log.info(m)

            self.df = self._prepare_ohlcv(df)
            self.df = self._compute_indicators(self.df)

            phase = self._detect_phase(self.df)

            signal_result = self._generate_signal(self.df, phase)
            signal = signal_result['signal']
            confidence = signal_result['confidence']
            volume_contexts = signal_result.get('volume_contexts', [])

            if signal not in ['long', 'short', 'no-go']:
                log.error(f"{self.seccode}: invalid signal {signal}, forcing no-go")
                signal = 'no-go'
                confidence = 0.0

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

            vol_ctx_str = ','.join(volume_contexts) if volume_contexts else 'none'
            log.info(
                f"seccode={self.seccode} phase={phase}, signal={signal},"
                f" confidence={confidence:.4f}, volume_contexts=[{vol_ctx_str}],"
                f" margin={margin}, entryPrice={lastClosePrice},"
                f" exitPrice={exitPrice}, UTC-time={utc_now.isoformat()}"
            )

            return {'signal': signal, 'confidence': confidence}

        except Exception as e:
            log.error(f"{self.seccode}: MinerviniClaude failed: {e}")
            return {'signal': 'no-go', 'confidence': 0.0}

    # =====================================================
    # DATA PREP
    # =====================================================

    def _prepare_ohlcv(self, df):
        """Filter raw DataFrame by seccode and rename columns to standard OHLCV format.

        Input columns: mnemonic, startprice, maxprice, minprice, endprice, addedvolume.
        Output columns: open, high, low, close, volume (indexed by datetime).
        """
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
        p = self.params
        # ---------------------------------------------------------
        # Exponential moving averages used for trend alignment. Used for:
        #   EMA alignment (trend detection)
        #   Short-term vs mid-term structure
        #   Pullback detection
        # ---------------------------------------------------------
        df['EMA_FAST'] = df['close'].ewm(span=p['EMA_FAST']).mean()     # cm.EMA_FAST = 9
        df['EMA_MID']  = df['close'].ewm(span=p['EMA_MID']).mean()      # cm.EMA_MID = 21
        df['EMA_SLOW'] = df['close'].ewm(span=p['EMA_SLOW']).mean()     # cm.EMA_SLOW = 50
        # ---------------------------------------------------------
        # RSI (Momentum Filter). Used for:
        #   Avoid overbought/oversold extremes
        #   Confirm directional entries
        #   Filter false breakouts
        # ---------------------------------------------------------
        df['RSI'] = self._rsi_series(df['close'], p['RSI_PERIOD'])      # cm.RSI_PERIOD = 14
        # ---------------------------------------------------------
        # ATR + ATR Slope (Volatility Regime). Measures volatility level, Used to:
        #   Detect expansion (volatility rising)
        #   Detect contraction (volatility compressing)
        # ---------------------------------------------------------
        df['ATR']       = self._atr(df, p['ATR_PERIOD'])                # cm.ATR_PERIOD = 14
        df['ATR_slope'] = df['ATR'].diff(p['ATR_SLOPE_WINDOW'])         # cm.ATR_SLOPE_WINDOW = 5
        # ---------------------------------------------------------
        # ADX + DI (Trend Strength). Measures trend strength and direction, Used to:
        #   Confirm directional strength
        #   Filter ranging markets
        # ---------------------------------------------------------
        df['ADX'], df['+DI'], df['-DI'] = self._adx(df, p['ADX_PERIOD']) # cm.ADX_PERIOD = 14
        # ---------------------------------------------------------
        # BOLLINGER BAND + Percentile. Measures compression vs expansion of volatility, Used to:
        #   Detect volatility compression
        #   Detect breakout expansion
        # ---------------------------------------------------------
        ma  = df['close'].rolling(p['BB_WINDOW']).mean()                # cm.BB_WINDOW = 20
        std = df['close'].rolling(p['BB_WINDOW']).std()                 # cm.BB_WINDOW = 20
        # Normalized width (volatility relative to price level)
        df['BB_width'] = (p['BB_STD'] * std) / ma.replace(0, np.nan)      # cm.BB_STD = 2
        # Percentile of BB width over rolling window                    # cm.BB_PERCENTILE_WINDOW = 100
        df['BB_width_pctile'] = df['BB_width'].rolling(p['BB_PERCENTILE_WINDOW']).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        # ---------------------------------------------------------
        # FAIR VALUE PRICE (FVP). Rolling mean of close. Used for:
        # Statistical center of recent price movement
        # Used for mean-reversion during expansion
        # ---------------------------------------------------------
        df['FVP'] = df['close'].rolling(p['FVP_WINDOW']).mean()         # cm.FVP_WINDOW = 30

        df.dropna(inplace=True)

        return df

    def _rsi_series(self, series, period):
        """Compute Relative Strength Index (RSI) using exponential moving average.

        RSI measures momentum on a 0-100 scale. Used to confirm directional
        entries and filter overbought/oversold extremes.
        """
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period).mean()
        avg_loss = loss.ewm(alpha=1/period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _atr(self, df, period):
        """Compute Average True Range (ATR) — measures volatility level.

        True Range = max(high-low, |high-prev_close|, |low-prev_close|).
        ATR = rolling mean of TR over `period` bars.
        """
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _adx(self, df, period):
        """Compute ADX (Average Directional Index) with +DI and -DI.

        ADX measures trend strength (0-100). +DI/-DI indicate bullish/bearish
        directional movement. Used to confirm trend phase and filter ranging markets.
        """
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = self._atr(df, period)

        plus_di = 100 * (plus_dm.rolling(period).sum() / tr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(period).sum() / tr.replace(0, np.nan))

        di_sum = plus_di + minus_di
        dx = (abs(plus_di - minus_di) / di_sum.replace(0, np.nan)) * 100
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
        p = self.params
        latest = df.iloc[-1]  # Get the latest row of indicators
        # ---------------------------------------------------------
        # Expansion is detected when volatility increases rapidly
        # ATR_slope > threshold AND Bollinger width percentile high
        # ---------------------------------------------------------
        if (
            latest['ATR_slope'] > p['VCP_ATR_SLOPE_EXPANSION'] and              # cm.VCP_ATR_SLOPE_EXPANSION = 0.15
            latest['BB_width_pctile'] > p['VCP_BB_WIDTH_PERCENTILE_EXPANSION']  # cm.VCP_BB_WIDTH_PERCENTILE_EXPANSION = 0.5
        ):
            return 'expansion'
        # ---------------------------------------------------------
        # Trend is detected when both:
        #   1) ADX above threshold (strong directional movement)
        #   2) EMA alignment either bullish or bearish
        # ---------------------------------------------------------
        if (
            latest['ADX'] > p['VCP_ADX_TREND_THRESHOLD'] and                       # cm.VCP_ADX_TREND_THRESHOLD = 25
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

    def _detect_phase_vectorized(self, df, params):
        """Vectorized phase detection for all bars (calibration version).

        Same logic as _detect_phase but applied to the entire DataFrame at once.
        Returns (expansion_mask, trend_mask, bullish, bearish) boolean Series.
        """
        bullish = (df['EMA_FAST'] > df['EMA_MID']) & (df['EMA_MID'] > df['EMA_SLOW'])
        bearish = (df['EMA_FAST'] < df['EMA_MID']) & (df['EMA_MID'] < df['EMA_SLOW'])

        expansion_mask = (
            (df['ATR_slope'] > params['VCP_ATR_SLOPE_EXPANSION']) &
            (df['BB_width_pctile'] > params['VCP_BB_WIDTH_PERCENTILE_EXPANSION'])
        )
        trend_mask = (
            (df['ADX'] > params['VCP_ADX_TREND_THRESHOLD']) &
            (bullish | bearish) &
            ~expansion_mask
        )
        return expansion_mask, trend_mask, bullish, bearish

    # =====================================================
    # SIGNAL GENERATION
    # =====================================================

    def _generate_signal(self, df, phase):
        """Generate a single trading signal for the latest bar (real-time prediction).

        Scores long and short independently using phase-specific rules and volume
        context. Returns dict with 'signal' (long/short/no-go), 'confidence'
        (0-1 directional conviction), and 'volume_contexts' (active patterns).
        """
        p = self.params
        long_score = 0.0
        short_score = 0.0
        latest = df.iloc[-1]        # Get latest indicator values
        bullish = latest['EMA_FAST'] > latest['EMA_MID'] > latest['EMA_SLOW']
        bearish = latest['EMA_FAST'] < latest['EMA_MID'] < latest['EMA_SLOW']

        # ---------------------------------------------------------
        # CONTRACTION PHASE: low-volatility compression, wait for breakout
        # ---------------------------------------------------------
        if phase == 'contraction':
            return {'signal': 'no-go', 'confidence': 0.0, 'volume_contexts': []}

        # Volume confirmation gate
        volume_avg = df['volume'].rolling(p.get('VOLUME_AVG_WINDOW', 4)).mean().iloc[-1]
        relative_volume = latest['volume'] / volume_avg if volume_avg > 0 else 0
        MIN_REL_VOL = p.get('MIN_RELATIVE_VOLUME', getattr(cm, 'MIN_RELATIVE_VOLUME', 0.8))
        if relative_volume < MIN_REL_VOL:
            return {'signal': 'no-go', 'confidence': 0.0, 'volume_contexts': []}

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
                deviation > p['EXPANSION_DEVIATION_THRESHOLD'] and      # cm.EXPANSION_DEVIATION_THRESHOLD = 0.0005
                latest['RSI'] > p['EXPANSION_RSI_SHORT_MIN']            # cm.EXPANSION_RSI_SHORT_MIN = 40
            ):
                short_score += 1.0
            # Long signal: Price below fair value AND RSI not overheated
            if (
                deviation < -p['EXPANSION_DEVIATION_THRESHOLD'] and     # cm.EXPANSION_DEVIATION_THRESHOLD = 0.0005
                latest['RSI'] < p['EXPANSION_RSI_LONG_MAX']             # cm.EXPANSION_RSI_LONG_MAX = 60
            ):
                long_score += 1.0

        # ---------------------------------------------------------
        # TREND PHASE Trend-following continuation entries
        #   Requires EMA alignment + DI confirmation + RSI filter
        # ---------------------------------------------------------
        if phase == 'trend':
            # Base score: confirmed trend phase (ADX + EMA alignment) deserves
            # a baseline contribution so that trend structure + this base
            # can reach MIN_TOTAL_SCORE even when RSI is outside the window.
            if bullish:
                long_score += 0.5
            elif bearish:
                short_score += 0.5

            # Bullish Trend Continuation (RSI-confirmed bonus)
            #   cm.TREND_RSI_LONG_MIN = 40
            #   cm.TREND_RSI_LONG_MAX = 70
            if (
                bullish and latest['+DI'] > latest['-DI']
                and p['TREND_RSI_LONG_MIN'] < latest['RSI'] < p['TREND_RSI_LONG_MAX']
            ):
                long_score += 1.5

            # Bearish Trend Continuation (RSI-confirmed bonus)
            #   cm.TREND_RSI_SHORT_MIN = 30
            #   cm.TREND_RSI_SHORT_MAX = 60
            if (
                bearish and latest['-DI'] > latest['+DI']
                and p['TREND_RSI_SHORT_MIN'] < latest['RSI'] < p['TREND_RSI_SHORT_MAX']
            ):
                short_score += 1.5

        # =============================
        # TREND STRUCTURE SCORE
        # =============================
        if latest['ADX'] > p['VCP_ADX_TREND_THRESHOLD']:
            if latest['+DI'] > latest['-DI']:
                long_score += 0.5
            else:
                short_score += 0.5

        # =============================
        # VOLUME SCORE
        # =============================
        context = self._volume_context(df)

        if context['buying_climax']:
            short_score += 1.0

        if context['no_supply'] and bullish :
            long_score += 0.8

        if context['divergence']:
            short_score += 0.7
            long_score -= 0.3  # penalize long: price-volume divergence warns against buying
            long_score = max(0.0, long_score)

        if context['absorption']:
            if latest['close'] < latest['open']:  # bearish candle = distribution
                short_score += 0.6
            # bullish candle absorption = ambiguous (possible accumulation), no score

        if context['trust']:
            if latest['close'] > latest['open']:
                long_score += 0.5
            else:
                short_score += 0.5

        if context['stopping_volume']:
            long_score += 0.6

        if context['healthy']:
            if latest['close'] > latest['open']:   # bullish confirmed by volume
                long_score += 0.3
            else:                                    # bearish confirmed by volume
                short_score += 0.3

        # =============================
        # FINAL DECISION
        # =============================
        score_diff = long_score - short_score
        total_score = long_score + short_score

        # Active volume contexts (only those that are True)
        active_contexts = [k for k, v in context.items() if v]

        if total_score == 0:
            return {'signal': 'no-go', 'confidence': 0.0, 'volume_contexts': active_contexts}

        confidence = abs(score_diff) / total_score

        # Scale confidence by signal strength - prevent conf=1.0 on minimal evidence
        MIN_QUALITY_SCORE = 1.5
        quality_factor = min(1.0, total_score / MIN_QUALITY_SCORE)
        confidence *= quality_factor

        # ---- Low energy filter with dynamic phase threshold (Idea #5) ----
        base_min_score = max(p['MIN_TOTAL_SCORE'], 0.50)
        EXPANSION_SCORE_MULT = getattr(cm, 'EXPANSION_SCORE_MULT', 1.2)
        TREND_SCORE_MULT = getattr(cm, 'TREND_SCORE_MULT', 0.9)

        if phase == 'expansion':
            min_score = base_min_score * EXPANSION_SCORE_MULT
        elif phase == 'trend':
            min_score = base_min_score * TREND_SCORE_MULT
        else:
            min_score = base_min_score
        if total_score < min_score:
            return {'signal': 'no-go', 'confidence': 0.0, 'volume_contexts': active_contexts}

        # ---- Volume support penalty ----
        if score_diff > 0:  # long signal
            has_vol_support = (context.get('no_supply') or context.get('stopping_volume')
                               or (context.get('trust') and latest['close'] > latest['open']))
        else:  # short signal
            has_vol_support = (context.get('absorption') or context.get('divergence')
                               or context.get('buying_climax')
                               or (context.get('trust') and latest['close'] <= latest['open']))

        if not has_vol_support:
            NO_VOL_PENALTY = p.get('NO_VOLUME_CONFIDENCE_PENALTY', 0.40)
            confidence *= (1.0 - NO_VOL_PENALTY)

        # ---- Counter-trend price momentum penalty ----
        MOMENTUM_LOOKBACK = int(p.get('MOMENTUM_LOOKBACK', 5))
        COUNTER_TREND_THRESHOLD = p.get('COUNTER_TREND_THRESHOLD', 0.003)
        COUNTER_TREND_FACTOR = p.get('COUNTER_TREND_FACTOR', 10.0)

        if len(df) > MOMENTUM_LOOKBACK + 1:
            price_change = latest['close'] / df['close'].iloc[-(MOMENTUM_LOOKBACK + 1)] - 1.0
            is_counter = ((score_diff > 0 and price_change < -COUNTER_TREND_THRESHOLD)
                       or (score_diff < 0 and price_change > COUNTER_TREND_THRESHOLD))
            if is_counter:
                penalty = min(0.7, abs(price_change) * COUNTER_TREND_FACTOR)
                confidence *= (1.0 - penalty)

        # ---- Conflict filter ----  cm.MIN_CONFIDENCE = 0.6
        # Minimum conviction filter. Avoid trades when there is internal conflict.
        # Floor minimum prevents calibration from over-relaxing this threshold.
        min_conf = max(p['MIN_CONFIDENCE'], 0.80)
        if confidence < min_conf:
            return {'signal': 'no-go', 'confidence': 0.0, 'volume_contexts': active_contexts}

        # ---- Direction ----
        if score_diff > 0:
            return {'signal': 'long', 'confidence': confidence, 'volume_contexts': active_contexts}
        else:
            return {'signal': 'short', 'confidence': confidence, 'volume_contexts': active_contexts}

    def _generate_signals_vectorized(self, df, params, expansion_mask, trend_mask, bullish, bearish):
        """Vectorized signal generation for all bars (calibration version).

        Same scoring logic as _generate_signal but applied to the full DataFrame
        at once for performance. Returns int8 numpy array: 0=no-go, 1=long, -1=short.
        """
        long_score  = pd.Series(0.0, index=df.index)
        short_score = pd.Series(0.0, index=df.index)

        # -- Expansion signals --
        deviation = (df['close'] - df['FVP']) / df['FVP'].replace(0, np.nan)
        exp_short = (
            expansion_mask &
            (deviation > params['EXPANSION_DEVIATION_THRESHOLD']) &
            (df['RSI'] > params['EXPANSION_RSI_SHORT_MIN'])
        )
        exp_long = (
            expansion_mask &
            (deviation < -params['EXPANSION_DEVIATION_THRESHOLD']) &
            (df['RSI'] < params['EXPANSION_RSI_LONG_MAX'])
        )
        short_score[exp_short] += 1.0
        long_score[exp_long]   += 1.0

        # -- Trend phase base score (EMA alignment confirmation) --
        long_score[trend_mask & bullish]   += 0.5
        short_score[trend_mask & bearish]  += 0.5

        # -- Trend signals (RSI-confirmed bonus) --
        trend_long = (
            trend_mask & bullish &
            (df['+DI'] > df['-DI']) &
            (df['RSI'] > params['TREND_RSI_LONG_MIN']) &
            (df['RSI'] < params['TREND_RSI_LONG_MAX'])
        )
        trend_short = (
            trend_mask & bearish &
            (df['-DI'] > df['+DI']) &
            (df['RSI'] > params['TREND_RSI_SHORT_MIN']) &
            (df['RSI'] < params['TREND_RSI_SHORT_MAX'])
        )
        long_score[trend_long]   += 1.5
        short_score[trend_short] += 1.5

        # -- ADX trend structure --
        adx_high = df['ADX'] > params['VCP_ADX_TREND_THRESHOLD']
        long_score[adx_high & (df['+DI'] > df['-DI'])]  += 0.5
        short_score[adx_high & (df['-DI'] > df['+DI'])] += 0.5

        # -- Volume context (vectorized, no cooldown) --
        vol_avg_win   = int(params['VOLUME_AVG_WINDOW'])
        vol_slope_win = int(params['VOLUME_SLOPE_WINDOW'])
        vol_avg     = df['volume'].rolling(vol_avg_win).mean()
        rel_volume  = df['volume'] / vol_avg.replace(0, np.nan)
        candle_body = abs(df['close'] - df['open'])
        rel_body    = candle_body / df['ATR'].replace(0, np.nan)
        price_slope = df['close'].diff(vol_slope_win)

        absorption = (rel_volume > params['BIG_VOLUME_THRESHOLD']) & (rel_body < 0.5)
        bearish_candle_vol = df['close'] < df['open']
        short_score[absorption & bearish_candle_vol] += 0.6
        # bullish absorption = ambiguous, no score contribution

        div_lb = int(params['DIVERGENCE_LOOKBACK'])
        divergence = (
            (df['close'] > df['close'].shift(div_lb)) &
            (df['volume'] < df['volume'].shift(div_lb))
        )
        short_score[divergence] += 0.7
        long_score[divergence] -= 0.3
        long_score = long_score.clip(lower=0.0)

        no_supply = (price_slope < 0) & (rel_volume < 0.7)
        long_score[no_supply & bullish] += 0.8

        # -- Trust: high volume + big body = convicted move, reinforce direction --
        trust = (rel_body > params['BIG_BODY_ATR_THRESHOLD']) & (rel_volume > params['BIG_VOLUME_THRESHOLD'])
        bullish_candle = df['close'] > df['open']
        long_score[trust & bullish_candle]   += 0.5
        short_score[trust & ~bullish_candle] += 0.5

        # -- Stopping volume: extreme volume + bearish candle closing near high = exhaustion --
        candle_range = df['high'] - df['low']
        stopping_vol = (
            (rel_volume > params['EXTREME_VOLUME_THRESHOLD']) &
            (df['close'] < df['open']) &
            (df['close'] > df['low'] + candle_range * 0.3)
        )
        long_score[stopping_vol] += 0.6

        healthy = (
            ((price_slope > 0) & (df['volume'].diff(vol_slope_win) > 0)) |
            ((price_slope < 0) & (df['volume'].diff(vol_slope_win) < 0))
        )
        long_score[healthy & bullish_candle] += 0.3
        short_score[healthy & ~bullish_candle] += 0.3

        bc_lb       = int(params['BUYING_CLIMAX_LOOKBACK'])
        bc_trend_lb = int(params['BUYING_CLIMAX_TREND_LOOKBACK'])
        recent_high = df['high'].rolling(bc_lb).max().shift(1)
        is_breakout = df['high'] >= recent_high
        recent_mean = df['close'].rolling(bc_trend_lb).mean()
        prior_mean  = df['close'].rolling(bc_trend_lb).mean().shift(bc_trend_lb)
        trend_up    = recent_mean > prior_mean
        ext         = (df['close'] - df['FVP']) / df['FVP'].replace(0, np.nan)
        is_overext  = ext > params['BUYING_CLIMAX_EXTENSION']
        buying_climax = (
            (rel_volume > params['EXTREME_VOLUME_THRESHOLD']) &
            (rel_body > params['EXTREME_BODY_ATR_THRESHOLD']) &
            is_breakout & trend_up & is_overext
        )
        short_score[buying_climax] += 1.0

        # -- Final signal decision --
        score_diff  = long_score - short_score
        total_score = long_score + short_score
        confidence  = pd.Series(0.0, index=df.index)
        nonzero     = total_score > 0
        confidence[nonzero] = abs(score_diff[nonzero]) / total_score[nonzero]

        # Scale by signal strength
        MIN_QUALITY_SCORE = 1.5
        quality_factor = (total_score / MIN_QUALITY_SCORE).clip(upper=1.0)
        confidence *= quality_factor

        # ---- Volume support penalty ----
        long_vol_support = no_supply | stopping_vol | (trust & bullish_candle)
        short_vol_support = absorption | divergence | buying_climax | (trust & ~bullish_candle)

        no_vol_support = pd.Series(False, index=df.index)
        no_vol_support[(score_diff > 0) & ~long_vol_support] = True
        no_vol_support[(score_diff < 0) & ~short_vol_support] = True

        NO_VOL_PENALTY = params.get('NO_VOLUME_CONFIDENCE_PENALTY', 0.40)
        confidence[no_vol_support] *= (1.0 - NO_VOL_PENALTY)

        # ---- Counter-trend price momentum penalty ----
        MOMENTUM_LOOKBACK = int(params.get('MOMENTUM_LOOKBACK', 5))
        COUNTER_TREND_THRESHOLD = params.get('COUNTER_TREND_THRESHOLD', 0.003)
        COUNTER_TREND_FACTOR = params.get('COUNTER_TREND_FACTOR', 10.0)

        price_change = df['close'] / df['close'].shift(MOMENTUM_LOOKBACK) - 1.0
        price_change.iloc[:MOMENTUM_LOOKBACK] = 0  # avoid artifacts

        counter_long = (score_diff > 0) & (price_change < -COUNTER_TREND_THRESHOLD)
        counter_short = (score_diff < 0) & (price_change > COUNTER_TREND_THRESHOLD)
        counter_mask = counter_long | counter_short

        penalty = (price_change.abs() * COUNTER_TREND_FACTOR).clip(upper=0.7)
        confidence[counter_mask] *= (1.0 - penalty[counter_mask])

        # Contraction phase = no-go (wait for breakout)
        contraction_mask = ~expansion_mask & ~trend_mask
        min_conf  = max(params['MIN_CONFIDENCE'], 0.80)

        # Dynamic signal threshold by phase (Idea #5)
        base_min_score = max(params['MIN_TOTAL_SCORE'], 0.50)
        EXPANSION_SCORE_MULT = getattr(cm, 'EXPANSION_SCORE_MULT', 1.2)
        TREND_SCORE_MULT = getattr(cm, 'TREND_SCORE_MULT', 0.9)

        min_score_arr = np.full(len(df), base_min_score)
        min_score_arr[expansion_mask.values] = base_min_score * EXPANSION_SCORE_MULT
        min_score_arr[trend_mask.values] = base_min_score * TREND_SCORE_MULT

        # 0 = no-go, 1 = long, -1 = short
        signals = np.zeros(len(df), dtype=np.int8)
        valid = (~contraction_mask) & (total_score.values >= min_score_arr) & (confidence >= min_conf)
        signals[(valid & (score_diff > 0)).values]  =  1
        signals[(valid & (score_diff < 0)).values]  = -1

        # Volume confirmation gate (Idea #3)
        volume_avg = df['volume'].rolling(params.get('VOLUME_AVG_WINDOW', 4)).mean()
        relative_volume = df['volume'] / volume_avg.replace(0, np.nan)
        MIN_REL_VOL = params.get('MIN_RELATIVE_VOLUME', getattr(cm, 'MIN_RELATIVE_VOLUME', 0.8))
        signals[relative_volume.values < MIN_REL_VOL] = 0

        return signals

    # =====================================================
    # MARGIN ADAPTATION
    # =====================================================

    def _adapt_margin(self, sec, phase, df):
        """Adapt positionMargin based on the current VCP phase (real-time prediction).

        Contraction: fixed tight margin (low volatility).
        Expansion: BB_width proportional (breakout amplitude).
        Trend: ATR-normalized (sustained directional volatility).
        Writes the computed margin into sec['params']['positionMargin'].
        """
        p = self.params
        latest = df.iloc[-1]        # Get latest indicator values
        close = latest['close']     # Current price (used to normalize ATR)

        # Contraction Phase. Very tight margin because volatility is low
        if phase == 'contraction':
            m = p['MARGIN_CONTRACTION_FIXED']               # cm.MARGIN_CONTRACTION_FIXED = 0.0015

        # Expansion Phase. Reflects volatility breakout amplitude
        elif phase == 'expansion':
           # cm.MARGIN_EXPANSION_MULTIPLIER = 1.5
            raw_margin = latest['BB_width'] * p['MARGIN_EXPANSION_MULTIPLIER']
            # Clamp margin inside configured bounds
            m = min(
                max(raw_margin, p['MARGIN_EXPANSION_MIN']),
                p['MARGIN_EXPANSION_MAX']
            )

        # Trend Phase. Reflects sustained directional volatility. Margin proportional to ATR normalized
        elif phase == 'trend':
            # cm.MARGIN_TREND_ATR_MULTIPLIER = 2.0
            raw_margin = (
                p['MARGIN_TREND_ATR_MULTIPLIER'] *
                latest['ATR'] / close
            )
            # Clamp margin inside configured bounds
            m = min(
                max(raw_margin, p['MARGIN_TREND_MIN']),
                p['MARGIN_TREND_MAX']
            )

        # Fallback (safety)
        else:
            m = p['MARGIN_CONTRACTION_FIXED']           # cm.MARGIN_CONTRACTION_FIXED = 0.0015

        # Dynamic cost floor (Idea #4): ensure margin covers transaction costs
        _margin_mult = getattr(cm, 'MIN_ABS_MARGIN_MULTIPLIER', 1.5)
        cost_floor = _margin_mult * max(0.02, close * 0.0001) / close
        m = max(m, cost_floor)

        # Apply computed margin to security parameters
        sec['params']['positionMargin'] = float(m)

    def _compute_margin_vectorized(self, df, params, expansion_mask, trend_mask):
        """Vectorized margin computation for all bars (calibration version).

        Same logic as _adapt_margin but applied to the full DataFrame at once.
        Returns numpy array of margin factors (one per bar).
        """
        margin_factor = np.full(len(df), params['MARGIN_CONTRACTION_FIXED'])

        exp_raw     = (df['BB_width'] * params['MARGIN_EXPANSION_MULTIPLIER']).values
        exp_clamped = np.clip(exp_raw, params['MARGIN_EXPANSION_MIN'], params['MARGIN_EXPANSION_MAX'])
        exp_idx     = expansion_mask.values
        margin_factor[exp_idx] = exp_clamped[exp_idx]

        trend_raw     = (params['MARGIN_TREND_ATR_MULTIPLIER'] * df['ATR'] / df['close'].replace(0, np.nan)).values
        trend_clamped = np.clip(trend_raw, params['MARGIN_TREND_MIN'], params['MARGIN_TREND_MAX'])
        trend_idx     = trend_mask.values
        margin_factor[trend_idx] = trend_clamped[trend_idx]

        # Dynamic cost floor: ensure margin covers transaction costs (Idea #4)
        closes = df['close'].values
        _margin_mult = getattr(cm, 'MIN_ABS_MARGIN_MULTIPLIER', 1.5)
        cost_floor = _margin_mult * np.maximum(0.02, closes * 0.0001) / closes
        margin_factor = np.maximum(margin_factor, cost_floor)

        return margin_factor

    # =====================================================
    # DB CALIBRATION (Coordinate Descent Param Optimization)
    # =====================================================

    def _calibrate_params_from_db(self):
        """Coordinate descent optimization of all numeric sec['params'].
        For each parameter, iterates 8 candidates in [-30%, +30%]
        of its initial value and picks the one that maximizes cumulative
        takeProfit benefit in a historical backtest simulation.

        Entry timing model:
          - Signal at bar i uses indicator data from bar i (= candle n-1)
          - Prediction occurs during candle n (= bar i+1)
          - Entry de facto at candle n+1 (= bar i+2)
          - TP/SL checked from bar i+3 onwards

        Optimization is split in two phases for performance:
          Phase 1: indicator params (require full indicator recomputation)
          Phase 2: non-indicator params (reuse cached indicator DataFrame)
        """
        try:
            p = self.params
            since = dt.datetime.now() - dt.timedelta(days=p['CALIBRATION_LOOKBACK_DAYS'])
            df_raw = self.dolph.ds.searchData(since, limitResult=p['CALIBRATION_LIMIT_RESULTS'])
            df_raw = df_raw['1Min'].copy()
            hist = self._prepare_ohlcv(df_raw)

            if hist is None or len(hist) < p['CALIBRATION_MIN_ROWS']:
                rows = len(hist) if hist is not None else 0
                log.warning(
                    f"{self.seccode}: calibration skipped, insufficient data "
                    f"({rows} rows, need {p['CALIBRATION_MIN_ROWS']})"
                )
                return

            # ---- Walk-forward split: train on first portion, validate on unseen data ----
            # Score = weighted blend of train + test. Test weight forces optimizer to
            # choose params that generalize, not just memorize the train period.
            TRAIN_RATIO = getattr(cm, 'CALIBRATION_TRAIN_RATIO', 0.67)
            TEST_WEIGHT = getattr(cm, 'CALIBRATION_TEST_WEIGHT', 0.40)
            split_idx = int(len(hist) * TRAIN_RATIO)
            train_df = hist.iloc[:split_idx].copy()
            test_df  = hist.iloc[split_idx:].copy()
            log.info(f"{self.seccode}: walk-forward split: train={len(train_df)} bars, test={len(test_df)} bars, test_weight={TEST_WEIGHT}")

            def _blended_score(params, train_indicator_df=None, test_indicator_df=None):
                """Score = (1-w)*train + w*test. Forces generalization."""
                s_train = self._simulate_profit(train_df, params, indicator_df=train_indicator_df)
                s_test  = self._simulate_profit(test_df, params, indicator_df=test_indicator_df)
                return (1.0 - TEST_WEIGHT) * s_train + TEST_WEIGHT * s_test

            # Working copy of params to optimize
            best_params = dict(p)

            # Collect optimizable params (numeric, non-zero, non-excluded)
            optimizable = [
                (k, v) for k, v in best_params.items()
                if k not in self._EXCLUDE_PARAMS
                and isinstance(v, (int, float)) and v != 0
            ]

            indicator_group = [(k, v) for k, v in optimizable if k in self._INDICATOR_PARAMS]
            other_group     = [(k, v) for k, v in optimizable if k not in self._INDICATOR_PARAMS]

            # Baseline score (blended train+test)
            baseline = _blended_score(best_params)
            log.info(f"{self.seccode}: calibration baseline score={baseline:.6f} (blended)")

            # ---- Phase 1: Indicator params (each step recomputes indicators) ----
            for param_name, base_value in indicator_group:
                candidates = self._make_candidates(base_value, 8)
                best_score = -np.inf
                best_value = base_value
                for c in candidates:
                    best_params[param_name] = c
                    score = _blended_score(best_params)
                    if score > best_score:
                        best_score = score
                        best_value = c
                best_params[param_name] = best_value
                if best_value != base_value:
                    log.info(f"{self.seccode}: {param_name}: {base_value} -> {best_value} (score={best_score:.6f})")

            # ---- Phase 2: Iterative refinement of non-indicator params ----
            cached_train_df = self._compute_indicators_for_calibration(train_df.copy(), best_params)
            cached_test_df  = self._compute_indicators_for_calibration(test_df.copy(), best_params)
            MAX_CALIBRATION_PASSES = getattr(cm, 'MAX_CALIBRATION_PASSES', 2)
            MIN_CALIBRATION_IMPROVEMENT = getattr(cm, 'MIN_CALIBRATION_IMPROVEMENT', 0.01)
            prev_score = _blended_score(best_params, cached_train_df, cached_test_df)

            for pass_num in range(MAX_CALIBRATION_PASSES):
                for param_name, _ in other_group:
                    base_value = best_params[param_name]
                    candidates = self._make_candidates(base_value, 8)
                    best_score = -np.inf
                    best_value = base_value
                    for c in candidates:
                        best_params[param_name] = c
                        score = _blended_score(best_params, cached_train_df, cached_test_df)
                        if score > best_score:
                            best_score = score
                            best_value = c
                    best_params[param_name] = best_value
                    if best_value != base_value:
                        log.info(f"{self.seccode}: pass {pass_num+1} {param_name}: {base_value} -> {best_value} (score={best_score:.6f})")

                current_score = _blended_score(best_params, cached_train_df, cached_test_df)
                improvement = (current_score - prev_score) / max(abs(prev_score), 1.0)
                log.info(f"{self.seccode}: calibration pass {pass_num+1} complete, score={current_score:.2f}, improvement={improvement:.2%} (blended)")
                if improvement < MIN_CALIBRATION_IMPROVEMENT:
                    break
                prev_score = current_score

            # ---- Final diagnostic: train vs test breakdown ----
            train_score = self._simulate_profit(train_df, best_params)
            test_score  = self._simulate_profit(test_df, best_params)
            log.info(f"{self.seccode}: walk-forward result: train={train_score:.2f}, test={test_score:.2f}, ratio={test_score/max(abs(train_score),1.0):.2f}")

            # ---- Save optimized params ----
            # Note: stopLossCoefficient is NOT clamped here — stored value (e.g. 8) is used
            # by OPERATIONAL as-is. The [1.5, 3.0] clamp only applies inside _simulate_profit()
            # during calibration to enforce realistic RR ratios for parameter search.
            MinerviniClaude._calibration_cache[self.seccode] = dict(best_params)

            for k, v in best_params.items():
                self.security['params'][k] = v
            self.params = self.security['params']

            final = self._simulate_profit(hist, best_params)
            log.info(
                f"seccode={self.seccode} calibration complete. "
                f"profit score: {baseline:.6f} -> {final:.6f}"
            )
            log.info(f"seccode={self.seccode} sec['params']={self.security['params']}")

        except Exception as e:
            log.error(f"{self.seccode}: calibration failed: {e}")


    @staticmethod
    def _tp_reward_gaussian(tp_profit, optimal_min, optimal_max):
        """Gaussian reward multiplier for TP profit.

        Smooth bell curve centered on the optimal TP range, replacing the old
        binary step function (2.5x inside / 0.3x outside) that created an
        8.33:1 discontinuity at the range boundaries.

        Returns:
            float multiplier:
              - ~2.5 at center of [optimal_min, optimal_max]
              - ~1.6 at range edges (±1 sigma)
              - ~0.6 at ±2 sigma
              - 0.3 floor far outside the range
        """
        center = (optimal_min + optimal_max) / 2.0
        # sigma = half the range width → edges of range sit at ±1 sigma
        sigma = (optimal_max - optimal_min) / 2.0
        if sigma <= 0:
            return 1.0
        z = (tp_profit - center) / sigma
        # peak=2.2 + floor=0.3 → 2.5 at center, decays smoothly outside
        return 0.3 + 2.2 * np.exp(-0.5 * z * z)

    def _make_candidates(self, base_value, steps=8):
        """Generate candidate values in [-30%, +30%] of base for coordinate descent.

        For int params: rounds to int, deduplicates, enforces minimum of 1.
        For float params: returns `steps` linearly spaced candidates.
        """
        candidates = [base_value * (1 + f) for f in np.linspace(-0.3, 0.3, steps)]
        if isinstance(base_value, int):
            candidates = [max(1, int(round(c))) for c in candidates]
            # Deduplicate preserving order
            seen = set()
            unique = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)
            candidates = unique
        return candidates


    def _compute_indicators_for_calibration(self, df, params):
        """Compute all technical indicators for calibration backtesting.

        Same indicators as _compute_indicators but takes an explicit params dict
        (instead of self.params) and uses a fast numpy-based BB_width percentile
        (avoids the slow pandas lambda). Called once per indicator-param candidate.
        """
        # EMA (Exponential Moving Averages) — trend alignment and pullback detection
        df['EMA_FAST'] = df['close'].ewm(span=int(params['EMA_FAST'])).mean()
        df['EMA_MID']  = df['close'].ewm(span=int(params['EMA_MID'])).mean()
        df['EMA_SLOW'] = df['close'].ewm(span=int(params['EMA_SLOW'])).mean()
        # RSI (Relative Strength Index) — momentum filter, overbought/oversold
        df['RSI']      = self._rsi_series(df['close'], int(params['RSI_PERIOD']))
        # ATR (Average True Range) — volatility level measurement
        df['ATR']      = self._atr(df, int(params['ATR_PERIOD']))
        # ATR slope — volatility expansion/contraction speed
        df['ATR_slope'] = df['ATR'].diff(int(params['ATR_SLOPE_WINDOW']))
        # ADX + DI (Average Directional Index) — trend strength and direction
        df['ADX'], df['+DI'], df['-DI'] = self._adx(df, int(params['ADX_PERIOD']))

        # BB (Bollinger Band) width — volatility compression vs expansion
        bb_win = int(params['BB_WINDOW'])
        ma  = df['close'].rolling(bb_win).mean()
        std = df['close'].rolling(bb_win).std()
        df['BB_width'] = (params['BB_STD'] * std) / ma.replace(0, np.nan)

        # BB width percentile — relative position within rolling volatility history (fast numpy version)
        bb_vals = df['BB_width'].values
        pctile_win = int(params['BB_PERCENTILE_WINDOW'])
        pctile = np.full(len(bb_vals), np.nan)
        for i in range(pctile_win - 1, len(bb_vals)):
            w = bb_vals[i - pctile_win + 1:i + 1]
            valid = w[~np.isnan(w)]
            if len(valid) > 0:
                pctile[i] = np.sum(valid <= valid[-1]) / len(valid)
        df['BB_width_pctile'] = pctile

        # FVP (Fair Value Price) — statistical center for mean-reversion in expansion phase
        df['FVP'] = df['close'].rolling(int(params['FVP_WINDOW'])).mean()
        df.dropna(inplace=True)
        return df


    def _simulate_profit(self, hist_ohlcv, params, indicator_df=None):
        """Simulate the full pipeline and return cumulative takeProfit benefit.

        Timing model (1-min candles):
          bar i   = last completed candle (data available = n-1)
          bar i+1 = current candle during which prediction runs (n)
          bar i+2 = entry candle (n+1)
          bar i+3..i+2+lookahead = TP/SL evaluation window

        For each bar that generates a long/short signal, a simulated trade
        is opened at the close of bar i+2.  Profit = +margin on TP hit,
        loss = -stopLossCoefficient*margin on SL hit (whichever comes first).

        Args:
            hist_ohlcv: raw OHLCV DataFrame (before indicators)
            params: dict of params to use for the simulation
            indicator_df: pre-computed indicator DataFrame (skip recomputation)
        Returns:
            float: cumulative profit (positive = net TP benefit)
        """
        try:
            # Step 1: Compute or reuse indicators
            if indicator_df is not None:
                df = indicator_df
            else:
                df = self._compute_indicators_for_calibration(hist_ohlcv.copy(), params)

            if len(df) < 50:
                return -np.inf

            # Step 2: Vectorized phase detection
            expansion_mask, trend_mask, bullish, bearish = self._detect_phase_vectorized(df, params)

            # Step 3: Vectorized signal scoring
            signals = self._generate_signals_vectorized(df, params, expansion_mask, trend_mask, bullish, bearish)

            # Step 4: Vectorized adaptive margin
            margin_factor = self._compute_margin_vectorized(df, params, expansion_mask, trend_mask)

            # Step 5: Trade simulation
            # Compute position size: quantity = (net_balance * factorPosition_Balance) / entry_price
            # This matches DolphRobot.positionAssessment() logic
            if cm.MODE == 'TEST_OFFLINE':
                net_balance = getattr(cm, 'simulation_net_balance', 29000)
            else:
                try:
                    net_balance = self.dolph.tp.get_net_balance()
                    if net_balance is None or net_balance <= 0:
                        net_balance = 20000.0
                except Exception:
                    net_balance = 20000.0
            cash_4_position = net_balance * cm.factorPosition_Balance

            closes  = df['close'].values
            highs   = df['high'].values
            lows    = df['low'].values
            volumes = df['volume'].values
            # Clamp SL coefficient to [1.5, 3.0] during calibration for realistic RR ratios.
            # With coef=5 → RR=1:5 → need >83% win rate (unstable intraday).
            # With coef=3 → RR=1:3 → need >75%. With coef=1.5 → RR=1:1.5 → need >60%.
            _MIN_SL_COEFF = getattr(cm, 'MIN_STOP_LOSS_COEFFICIENT', 1.5)
            _MAX_SL_COEFF = getattr(cm, 'MAX_STOP_LOSS_COEFFICIENT', 3.0)
            sl_coeff  = max(_MIN_SL_COEFF, min(_MAX_SL_COEFF, params['stopLossCoefficient']))
            lookahead = int(params['CALIBRATION_LOOKAHEAD_BARS'])
            n = len(df)

            # Trading hours filter setup - per-security timezone
            sec_tz = pytz.timezone(self.security.get('timezone', 'America/New_York'))
            trading_start, trading_end = self.security.get('tradingTimes',
                getattr(cm, 'tradingTimes', (dt.time(9, 30), dt.time(16, 0))))
            time2close = self.security.get('time2close',
                getattr(cm, 'time2close', dt.time(16, 0)))
            use_trading_hours = (cm.MODE == 'TEST_OFFLINE')

            # exitTimeSeconds -> bars (1 bar = 1 min)
            exit_time_seconds = self.security.get('params', {}).get(
                'exitTimeSeconds', getattr(cm, 'exitTimeSeconds', 11400))
            exit_timeout_bars = int(exit_time_seconds / 60)

            # entryTimeSeconds -> bars for fill simulation
            entry_time_seconds = int(params.get(
                'entryTimeSeconds', getattr(cm, 'entryTimeSeconds', 360)))
            entry_timeout_bars = max(1, entry_time_seconds // 60)

            # Exposure and position tracking for TEST_OFFLINE
            track_constraints = (cm.MODE == 'TEST_OFFLINE')
            active_positions = []  # each: {'dir': 1/-1, 'exposure': float, 'close_bar': int}
            long_exposure = 0.0
            short_exposure = 0.0
            has_open_position = False  # only 1 position at a time per security (mirrors isPositionOpen)

            # Simulation counters for DEBUG logging
            stats_signals = 0
            stats_skip_hours = 0
            stats_skip_margin = 0
            stats_skip_position_open = 0
            stats_skip_exposure = 0
            stats_entry_expired = 0  # entry LMT order didn't fill within entryTimeSeconds
            stats_blocked_pending = 0  # signal skipped because capital reserved for pending LMT
            stats_tp = 0
            stats_tp_fast = 0  # TP hit in < half exitTimeSeconds
            stats_tp_time_sum = 0  # sum of bars to TP (for avg tracking)
            stats_sl = 0
            stats_sl_loss = 0.0  # cumulative real SL loss (for proportional penalty)
            stats_expired = 0
            stats_forced_close = 0
            stats_exit_timeout = 0  # closed by exitTimeSeconds timeout
            stats_max_concurrent = 0

            total_profit = 0.0
            peak_profit = 0.0   # high-water mark for max drawdown tracking
            max_drawdown = 0.0  # worst peak-to-trough decline
            pending_until_bar = -1  # bar until which capital is reserved for a pending LMT order

            # Mejora 1: Optimal TP range as % of cash_4_position (currency-independent)
            # Replaces the old absolute USD range that ignored EUR/GBP/JPY differences
            optimal_tp_min = cash_4_position * getattr(cm, 'OPTIMAL_TP_RATIO_MIN', 0.0035)
            optimal_tp_max = cash_4_position * getattr(cm, 'OPTIMAL_TP_RATIO_MAX', 0.0046)

            for i in range(n - lookahead - 2):
                sig = signals[i]
                if sig == 0:
                    continue

                stats_signals += 1

                # Block signals while a pending LMT order is waiting for fill.
                # Capital is reserved → cannot enter new trades. This is the real
                # opportunity cost of unfilled orders (no artificial penalty needed).
                if i < pending_until_bar:
                    stats_blocked_pending += 1
                    continue

                # Trading hours filter
                if use_trading_hours:
                    bar_time = df.index[i]
                    bar_ny = bar_time.tz_convert(sec_tz) if bar_time.tzinfo else bar_time
                    if not (trading_start <= bar_ny.time() <= trading_end):
                        stats_skip_hours += 1
                        continue

                    # Filter entries with insufficient bars before time2close
                    bars_to_close = 0
                    for j in range(i + 2, min(i + 2 + lookahead, n)):
                        bar_t = df.index[j]
                        bar_local = bar_t.tz_convert(sec_tz) if bar_t.tzinfo else bar_t
                        if bar_local.time() >= time2close:
                            break
                        bars_to_close += 1
                    min_bars_for_trade = 10  # minimum 10 bars (10 min) for a trade to make sense
                    if bars_to_close < min_bars_for_trade:
                        stats_skip_hours += 1
                        continue

                # Entry fill simulation: LMT order at closes[i+2], search
                # entry_timeout_bars for the price to cross the limit.
                # Conservative fill: require price to cross FROM the correct direction
                # (prev bar close must be on opposite side of limit) to avoid fills
                # that assume favorable intra-candle order with 1-min OHLC data.
                limit_bar = i + 2
                if limit_bar >= n:
                    continue
                limit_price = closes[limit_bar]
                fill_end = min(limit_bar + entry_timeout_bars, n)

                # Reserve capital: block new signals while this LMT order is pending
                pending_until_bar = fill_end

                # Volume participation: order can only fill if candle volume can absorb it.
                # MAX_PARTICIPATION=0.10 means order cannot exceed 10% of candle volume.
                # Deterministic (no randomness) so optimizer produces stable scores.
                MAX_PARTICIPATION = getattr(cm, 'CALIBRATION_MAX_VOLUME_PARTICIPATION', 0.10)
                est_quantity = round(cash_4_position / limit_price) if limit_price > 0 else 0

                entry_idx = -1
                for fb in range(limit_bar, fill_end):
                    prev_close = closes[fb - 1] if fb > 0 else closes[fb]
                    if sig == 1 and prev_close > limit_price and lows[fb] <= limit_price:
                        # BUY LMT: price crossed down to limit — check volume can absorb
                        bar_vol = volumes[fb]
                        if bar_vol > 0 and est_quantity > bar_vol * MAX_PARTICIPATION:
                            continue  # insufficient liquidity, try next bar
                        entry_idx = fb
                        break
                    elif sig == -1 and prev_close < limit_price and highs[fb] >= limit_price:
                        # SELL LMT: price crossed up to limit — check volume can absorb
                        bar_vol = volumes[fb]
                        if bar_vol > 0 and est_quantity > bar_vol * MAX_PARTICIPATION:
                            continue  # insufficient liquidity, try next bar
                        entry_idx = fb
                        break
                if entry_idx < 0:
                    # Entry order expired — didn't fill within entryTimeSeconds
                    # No artificial penalty: opportunity cost is organic (blocked signals above)
                    stats_entry_expired += 1
                    continue

                # Apply slippage: fill slightly worse than limit price.
                # Prevents backtest from assuming perfect fills at exact limit.
                # Default 0.01% ≈ $0.01 on $100 stock (half spread for liquid names).
                FILL_SLIPPAGE = getattr(cm, 'CALIBRATION_FILL_SLIPPAGE', 0.0001)
                if sig == 1:
                    entry_price = limit_price * (1.0 + FILL_SLIPPAGE)   # buy slightly higher
                else:
                    entry_price = limit_price * (1.0 - FILL_SLIPPAGE)   # sell slightly lower
                m_abs       = entry_price * margin_factor[i]
                quantity    = round(cash_4_position / entry_price)
                if quantity <= 0:
                    continue

                # Mirror DolphRobot.evaluatePosition() min_margin check
                # IB: ~$0.02/share round-trip, proportional floor 0.01% for expensive stocks
                _margin_mult = getattr(cm, 'MIN_ABS_MARGIN_MULTIPLIER', 1.5)
                min_abs_margin = _margin_mult * max(0.02, entry_price * 0.0001)
                if m_abs < min_abs_margin:
                    stats_skip_margin += 1
                    continue

                # Realistic IB transaction cost: max($1.00, qty × $0.005) per side
                round_trip_cost = max(1.0, quantity * 0.005) * 2

                # Expire resolved positions and update exposure
                if track_constraints:
                    still_active = []
                    for pos in active_positions:
                        if pos['close_bar'] <= i:
                            if pos['dir'] == 1:
                                long_exposure -= pos['exposure']
                            else:
                                short_exposure -= pos['exposure']
                        else:
                            still_active.append(pos)
                    active_positions = still_active
                    has_open_position = len(active_positions) > 0

                    # Block if there is already an open position on this security
                    # Mirrors TradingPlatform.isPositionOpen() / processingCheck()
                    if has_open_position:
                        stats_skip_position_open += 1
                        continue

                    # Check exposure limits before opening
                    new_exposure = quantity * entry_price
                    if sig == 1 and (long_exposure + new_exposure) > net_balance:
                        stats_skip_exposure += 1
                        continue
                    if sig == -1 and (short_exposure + new_exposure) > net_balance:
                        stats_skip_exposure += 1
                        continue

                if sig == 1:   # long
                    tp_price = entry_price + m_abs
                    sl_price = entry_price - sl_coeff * m_abs
                else:          # short
                    tp_price = entry_price - m_abs
                    sl_price = entry_price + sl_coeff * m_abs

                # Check TP/SL from bar i+3 onwards
                start = entry_idx + 1
                end   = min(start + lookahead, n)
                window_highs = highs[start:end]
                window_lows  = lows[start:end]

                if sig == 1:   # long
                    tp_hits = np.where(window_highs >= tp_price)[0]
                    sl_hits = np.where(window_lows  <= sl_price)[0]
                else:          # short
                    tp_hits = np.where(window_lows  <= tp_price)[0]
                    sl_hits = np.where(window_highs >= sl_price)[0]

                tp_first = tp_hits[0] if len(tp_hits) > 0 else lookahead + 1
                sl_first = sl_hits[0] if len(sl_hits) > 0 else lookahead + 1

                # Conservative same-bar conflict resolution: with 1-min OHLC we cannot
                # know if TP or SL was hit first within a candle. Assume worst case (SL).
                if tp_first == sl_first and tp_first <= lookahead:
                    tp_first = lookahead + 1  # suppress TP → SL branch will handle it

                # Calculate forced close bar: first bar where time >= time2close
                forced_close_bar = lookahead + 1  # default: no forced close
                if use_trading_hours:
                    for j_offset in range(end - start):
                        bar_idx = start + j_offset
                        bar_t = df.index[bar_idx]
                        bar_local = bar_t.tz_convert(sec_tz) if bar_t.tzinfo else bar_t
                        if bar_local.time() >= time2close:
                            forced_close_bar = j_offset
                            break

                # exitTimeSeconds timeout bar (position open too long)
                exit_timeout_bar = min(exit_timeout_bars, lookahead + 1)
                # effective deadline = earliest of exit_timeout and forced_close (time2close)
                deadline_bar = min(exit_timeout_bar, forced_close_bar)
                half_exit_timeout = exit_timeout_bars // 2

                # Determine outcome and close bar
                if tp_first <= sl_first and tp_first <= lookahead and tp_first < deadline_bar:
                    # TP hit before any deadline
                    standard_tp_profit = quantity * m_abs

                    # Trailing TP (Idea #6): after TP hit, scan forward for excess via trailing stop
                    # Disabled during calibration (TEST_OFFLINE) — optimize base strategy on
                    # deterministic TP profit; trailing adds variance that destabilises the objective.
                    TRAILING_TP_ENABLED = getattr(cm, 'TRAILING_TP_ENABLED', True) and cm.MODE == 'OPERATIONAL'
                    TRAILING_TP_RETRACE = getattr(cm, 'TRAILING_TP_RETRACE', 0.50)

                    if TRAILING_TP_ENABLED and (start + tp_first + 1) < min(start + deadline_bar, n):
                        trail_start = start + tp_first
                        trail_end = min(start + deadline_bar, n)

                        if sig == 1:  # long
                            best_high = tp_price
                            trail_exit_price = None
                            for tb in range(trail_start, trail_end):
                                if highs[tb] > best_high:
                                    best_high = highs[tb]
                                trail_stop = tp_price + (best_high - tp_price) * (1 - TRAILING_TP_RETRACE)
                                if lows[tb] <= trail_stop and best_high > tp_price:
                                    trail_exit_price = trail_stop
                                    break
                            if trail_exit_price is None:
                                trail_exit_price = closes[trail_end - 1]
                            trailing_profit = (trail_exit_price - entry_price) * quantity
                        else:  # short
                            best_low = tp_price
                            trail_exit_price = None
                            for tb in range(trail_start, trail_end):
                                if lows[tb] < best_low:
                                    best_low = lows[tb]
                                trail_stop = tp_price - (tp_price - best_low) * (1 - TRAILING_TP_RETRACE)
                                if highs[tb] >= trail_stop and best_low < tp_price:
                                    trail_exit_price = trail_stop
                                    break
                            if trail_exit_price is None:
                                trail_exit_price = closes[trail_end - 1]
                            trailing_profit = (entry_price - trail_exit_price) * quantity

                        tp_profit = max(standard_tp_profit, trailing_profit) - round_trip_cost
                    else:
                        tp_profit = standard_tp_profit - round_trip_cost

                    # Stats only (no reward shaping — score = net_profit)
                    stats_tp_time_sum += tp_first
                    if tp_first < half_exit_timeout:
                        stats_tp_fast += 1

                    total_profit += tp_profit
                    close_bar = entry_idx + 1 + tp_first
                    stats_tp += 1
                elif sl_first < tp_first and sl_first <= lookahead and sl_first < deadline_bar:
                    # SL hit before any deadline
                    sl_loss = quantity * sl_coeff * m_abs + round_trip_cost
                    total_profit -= sl_loss
                    stats_sl_loss += sl_loss
                    close_bar = entry_idx + 1 + sl_first
                    stats_sl += 1
                elif exit_timeout_bar <= forced_close_bar and exit_timeout_bar <= lookahead:
                    # exitTimeSeconds timeout: position open too long, penalty
                    close_idx = min(start + exit_timeout_bar, n - 1)
                    close_price = closes[close_idx]
                    if sig == 1:
                        pnl = (close_price - entry_price) * quantity - round_trip_cost
                    else:
                        pnl = (entry_price - close_price) * quantity - round_trip_cost
                    total_profit += pnl
                    close_bar = close_idx
                    stats_exit_timeout += 1
                elif forced_close_bar <= lookahead:
                    # Forced close at time2close
                    close_idx = start + forced_close_bar
                    close_price = closes[close_idx]
                    if sig == 1:
                        pnl = (close_price - entry_price) * quantity - round_trip_cost
                    else:
                        pnl = (entry_price - close_price) * quantity - round_trip_cost
                    total_profit += pnl
                    close_bar = close_idx
                    stats_forced_close += 1
                else:
                    # Expired (no TP, no SL, no timeout, no forced close in window)
                    # No artificial penalty — score = net_profit only
                    close_bar = end
                    stats_expired += 1

                # Update max drawdown after each trade outcome
                if total_profit > peak_profit:
                    peak_profit = total_profit
                drawdown = peak_profit - total_profit
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

                # Track exposure and position
                if track_constraints:
                    new_exp = quantity * entry_price
                    active_positions.append({'dir': sig, 'exposure': new_exp, 'close_bar': close_bar})
                    if sig == 1:
                        long_exposure += new_exp
                    else:
                        short_exposure += new_exp
                    if len(active_positions) > stats_max_concurrent:
                        stats_max_concurrent = len(active_positions)

            # Mejora 4: Dynamic trades/day target adapted per security
            # Stocks with higher signal density get a higher acceptable frequency,
            # so high-signal securities (e.g. MARA 9.5/day) are not penalized unfairly
            trades_opened = stats_tp + stats_sl + stats_expired + stats_forced_close + stats_exit_timeout
            total_signals_attempted = trades_opened + stats_entry_expired
            num_trading_days = 1
            trades_per_day = 0.0
            dynamic_target = 0.0
            if use_trading_hours and trades_opened > 0:
                # Count unique trading days in the simulation window
                trading_days = set()
                for idx in range(n):
                    bar_t = df.index[idx]
                    bar_local = bar_t.tz_convert(sec_tz) if bar_t.tzinfo else bar_t
                    if trading_start <= bar_local.time() <= trading_end:
                        trading_days.add(bar_local.date())
                num_trading_days = max(len(trading_days), 1)
                trades_per_day = trades_opened / num_trading_days

                # Dynamic target: derive from this security's signal density
                # eligible = signals that passed trading-hours filter (before margin/position checks)
                eligible_per_day = max((stats_signals - stats_skip_hours) / num_trading_days, 1.0)
                # FREQ_SIGNAL_CONVERSION = expected % of eligible signals → actual trades
                # (limited by position_open blocking, margin filter, exposure limits)
                freq_conv = getattr(cm, 'FREQ_SIGNAL_CONVERSION', 0.04)
                freq_min = getattr(cm, 'FREQ_TARGET_MIN', 3.0)
                freq_max = getattr(cm, 'FREQ_TARGET_MAX', 12.0)
                dynamic_target = min(max(eligible_per_day * freq_conv, freq_min), freq_max)

                # Stats only — no frequency/efficiency multipliers (score = net_profit)

            # DEBUG summary for this simulation run
            if track_constraints:
                log.debug(
                    f"seccode={self.seccode} simulation: "
                    f"signals={stats_signals} skip_hours={stats_skip_hours} "
                    f"skip_margin={stats_skip_margin} "
                    f"skip_position_open={stats_skip_position_open} skip_exposure={stats_skip_exposure} "
                    f"entry_expired={stats_entry_expired} blocked_pending={stats_blocked_pending} | "
                    f"trades={trades_opened} TP={stats_tp} TP_fast={stats_tp_fast} SL={stats_sl} "
                    f"avg_tp_bars={stats_tp_time_sum/max(stats_tp,1):.1f} "
                    f"exit_timeout={stats_exit_timeout} forced_close={stats_forced_close} expired={stats_expired} "
                    f"max_concurrent={stats_max_concurrent} "
                    f"trades/day={trades_per_day:.1f} freq_target={dynamic_target:.1f} "
                    f"optimal_tp=[{optimal_tp_min:.1f},{optimal_tp_max:.1f}] "
                    f"profit={total_profit:.2f}"
                )

            # SL aversion: penalize proportional to real SL losses so optimizer prefers
            # fewer, higher-quality entries. Scales automatically with position size/capital.
            # alpha=0.15 means each $100 of SL loss costs an extra $15 in the objective.
            SL_AVERSION_ALPHA = getattr(cm, 'CALIBRATION_SL_AVERSION', 0.15)
            total_profit -= SL_AVERSION_ALPHA * stats_sl_loss

            # Max drawdown penalty: penalize SL clustering / deep equity curves.
            # 5 SL in 10 min is worse than 5 SL spread over 3 months even if total loss is equal.
            # beta=0.10 means $500 max drawdown costs extra $50 in the objective.
            DD_AVERSION_BETA = getattr(cm, 'CALIBRATION_DD_AVERSION', 0.10)
            total_profit -= DD_AVERSION_BETA * max_drawdown

            # Min activity: prevent ultra-selective strategies (2 trades in 90 days = luck, not edge).
            # Scales score linearly when below threshold so optimizer can't game it with 0 trades.
            MIN_TRADES_PER_DAY = getattr(cm, 'CALIBRATION_MIN_TRADES_PER_DAY', 0.5)
            if num_trading_days > 0 and trades_per_day < MIN_TRADES_PER_DAY:
                total_profit *= trades_per_day / MIN_TRADES_PER_DAY

            return total_profit

        except Exception as e:
            import traceback
            log.error(f"{self.seccode}: simulation error: {e}\n{traceback.format_exc()}")
            return -np.inf


    def _volume_context(self, df):
        """Analyze volume patterns on the latest bar for signal scoring.

        Detects six Wyckoff-inspired contexts: healthy (price-volume alignment),
        absorption (high volume, small body), trust (big body + big volume),
        divergence (price up, volume down), stopping_volume (exhaustion),
        buying_climax (overextended breakout), and no_supply (pullback on low volume).
        Returns dict of bool flags.
        """
        p = self.params
        latest = df.iloc[-1]

        # Compute relative metrics
        volume_avg      = df['volume'].rolling(p['VOLUME_AVG_WINDOW']).mean().iloc[-1]
        relative_volume = latest['volume'] / volume_avg

        candle_body = abs(latest['close'] - latest['open'])
        candle_range = latest['high'] - latest['low']
        relative_body = candle_body / latest['ATR']

        price_slope  = df['close'].diff(p['VOLUME_SLOPE_WINDOW']).iloc[-1]
        volume_slope = df['volume'].diff(p['VOLUME_SLOPE_WINDOW']).iloc[-1]

        context = {}

        context['healthy'] = (
            (price_slope > 0 and volume_slope > 0)
            or
            (price_slope < 0 and volume_slope < 0)
        )

        context['absorption'] = (
            relative_volume > p['BIG_VOLUME_THRESHOLD']
            and relative_body < 0.5
        )

        context['trust'] = (
            relative_body  > p['BIG_BODY_ATR_THRESHOLD']
            and relative_volume > p['BIG_VOLUME_THRESHOLD']
        )

        context['divergence'] = (
            df['close'].iloc[-1]  > df['close'].iloc[-p['DIVERGENCE_LOOKBACK']]
            and df['volume'].iloc[-1] < df['volume'].iloc[-p['DIVERGENCE_LOOKBACK']]
        )

        context['stopping_volume'] = (
            relative_volume > p['EXTREME_VOLUME_THRESHOLD']
            and latest['close'] < latest['open']
            and latest['close'] > latest['low'] + (candle_range * 0.3)
        )
        # ==========================================
        # context['buying_climax']
        # ==========================================

        # cm.BUYING_CLIMAX_LOOKBACK = 20
        recent_high    = df['high'].rolling(p['BUYING_CLIMAX_LOOKBACK']).max().iloc[-2]
        is_breakout_high = latest['high'] >= recent_high

        # cm.BUYING_CLIMAX_TREND_LOOKBACK = 15
        trend_up = (
            df['close'].iloc[-p['BUYING_CLIMAX_TREND_LOOKBACK']:].mean()
            >
            df['close'].iloc[-2*p['BUYING_CLIMAX_TREND_LOOKBACK']:-p['BUYING_CLIMAX_TREND_LOOKBACK']].mean()
        )
        # cm.BUYING_CLIMAX_EXTENSION = 0.004   # 0.4%
        extension = (latest['close'] - latest['FVP']) / latest['FVP']
        is_overextended = extension > p['BUYING_CLIMAX_EXTENSION']

        # cm.BUYING_CLIMAX_COOLDOWN_SECONDS = 900  # 15 minutos
        cooldown_ok = (
            not hasattr(self, "_last_climax_time")
            or self._last_climax_time is None
            or (df.index[-1] - self._last_climax_time).seconds > p['BUYING_CLIMAX_COOLDOWN_SECONDS']
        )

        context['buying_climax'] = (
            relative_volume  > p['EXTREME_VOLUME_THRESHOLD']
            and relative_body    > p['EXTREME_BODY_ATR_THRESHOLD']
            and is_breakout_high
            and trend_up
            and is_overextended
            and cooldown_ok
        )

        if context['buying_climax']:
            self._last_climax_time = df.index[-1]

        context['no_supply'] = (
            price_slope < 0
            and relative_volume < 0.7
        )

        return context
