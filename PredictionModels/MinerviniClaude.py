# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import collections
import logging
import datetime as dt
import pytz
from Configuration import Conf as cm

log = logging.getLogger("PredictionModel")

class MinerviniClaude:

    _calibration_cache = {}

    # Meta params whose change triggers full indicator recomputation
    # (via _derive_params -> EMA_FAST/MID/SLOW, ATR_PERIOD, BB_WINDOW, etc.)
    _INDICATOR_PARAMS = frozenset({'EMA_BASE', 'VOL_WINDOW', 'TREND_WINDOW'})

    # The 10 params the optimizer is allowed to touch (whitelist)
    _OPTIMIZABLE_PARAMS = frozenset({
        'EMA_BASE', 'VOL_WINDOW', 'TREND_WINDOW',
        'VCP_ATR_SLOPE_EXPANSION', 'VCP_BB_WIDTH_PERCENTILE_EXPANSION',
        'VCP_ADX_TREND_THRESHOLD', 'EXPANSION_DEVIATION_THRESHOLD',
        'MIN_RELATIVE_VOLUME',
        'TP_MULT', 'SL_RR',
    })

    # Minimum allowed values for the 10 optimizable params.
    _PARAM_FLOORS = {
        'EMA_BASE': 5,
        'VOL_WINDOW': 5,
        'TREND_WINDOW': 5,
        'VCP_ATR_SLOPE_EXPANSION': 0.005,
        'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.05,
        'VCP_ADX_TREND_THRESHOLD': 8,
        'EXPANSION_DEVIATION_THRESHOLD': 0.00005,
        'MIN_RELATIVE_VOLUME': 0.2,
        'TP_MULT': 0.5,
        'SL_RR': 1.0,
    }

    # Maximum allowed values — prevents the optimizer from pushing thresholds
    # so high that no signals fire (zero trades → -inf degeneration).
    _PARAM_CEILINGS = {
        'EMA_BASE': 60,
        'VOL_WINDOW': 60,
        'TREND_WINDOW': 60,
        'VCP_ATR_SLOPE_EXPANSION': 0.15,
        'VCP_BB_WIDTH_PERCENTILE_EXPANSION': 0.80,
        'VCP_ADX_TREND_THRESHOLD': 40,
        'EXPANSION_DEVIATION_THRESHOLD': 0.01,
        'MIN_RELATIVE_VOLUME': 3.0,
        'TP_MULT': 5.0,
        'SL_RR': 5.0,
    }

    @staticmethod
    def _derive_params(params):
        """Derive dependent params from the 5 meta-params.

        Must be called whenever EMA_BASE, VOL_WINDOW, TREND_WINDOW, TP_MULT,
        or SL_RR change. Downstream code reads the derived keys unchanged.
        """
        if 'EMA_BASE' not in params:
            return params  # legacy params without meta-keys
        ema_base = int(params['EMA_BASE'])
        params['EMA_FAST'] = ema_base
        params['EMA_MID']  = round(1.5 * ema_base)
        params['EMA_SLOW'] = round(2.5 * ema_base)

        vol_win = int(params['VOL_WINDOW'])
        params['ATR_PERIOD']           = vol_win
        params['BB_WINDOW']            = vol_win
        params['BB_PERCENTILE_WINDOW'] = 3 * vol_win
        params['ATR_SLOPE_WINDOW']     = max(3, vol_win // 3)
        params['FVP_WINDOW']           = 2 * vol_win

        params['ADX_PERIOD'] = int(params['TREND_WINDOW'])

        params['stopLossCoefficient'] = params['SL_RR']
        return params

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
        self._signal_history = collections.deque(maxlen=10)

        # Ensure derived params are consistent with meta params
        MinerviniClaude._derive_params(self.params)

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

            # Signal stability: require N consecutive identical signals before acting.
            # Liquidity (SMC) signals are exempt from this filter.
            p = self.params
            stability_required = int(p.get('SIGNAL_STABILITY_REQUIRED', 3))
            self._signal_history.append(signal)
            has_liquidity = ('liquidity_long' in volume_contexts
                            or 'liquidity_short' in volume_contexts)

            if signal in ('long', 'short') and not has_liquidity:
                if len(self._signal_history) >= stability_required:
                    recent = list(self._signal_history)[-stability_required:]
                    if not all(s == signal for s in recent):
                        log.info(
                            f"{self.seccode} signal={signal} suppressed: stability check failed "
                            f"(need {stability_required} consecutive, history={recent})"
                        )
                        signal = 'no-go'
                        confidence = 0.0
                elif len(self._signal_history) < stability_required:
                    log.info(
                        f"{self.seccode} signal={signal} suppressed: insufficient history "
                        f"({len(self._signal_history)}/{stability_required})"
                    )
                    signal = 'no-go'
                    confidence = 0.0

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

            entry_type = 'pullback' if has_liquidity else 'breakout'
            return {'signal': signal, 'confidence': confidence, 'entry_type': entry_type}

        except Exception as e:
            log.error(f"{self.seccode}: MinerviniClaude failed: {e}")
            return {'signal': 'no-go', 'confidence': 0.0, 'entry_type': 'breakout'}

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
        """Generate a trading signal using mandatory/optional architecture.

        Mandatory conditions determine direction (expansion deviation or trend EMA).
        Optional confirmations (ADX, RSI, volume, liquidity) validate the signal.
        Entry requires: all mandatory + >= 1 optional confirmation.
        """
        p = self.params
        latest = df.iloc[-1]
        bullish = latest['EMA_FAST'] > latest['EMA_MID'] > latest['EMA_SLOW']
        bearish = latest['EMA_FAST'] < latest['EMA_MID'] < latest['EMA_SLOW']

        # Gate 1: contraction → no-go
        if phase == 'contraction':
            return {'signal': 'no-go', 'confidence': 0.0, 'volume_contexts': []}

        # Gate 2: minimum volume
        volume_avg = df['volume'].rolling(p.get('VOLUME_AVG_WINDOW', 4)).mean().iloc[-1]
        relative_volume = latest['volume'] / volume_avg if volume_avg > 0 else 0
        if relative_volume < p.get('MIN_RELATIVE_VOLUME', 0.8):
            return {'signal': 'no-go', 'confidence': 0.0, 'volume_contexts': []}

        # === MANDATORY CONDITIONS (determine direction) ===
        long_mandatory = False
        short_mandatory = False

        if phase == 'expansion':
            deviation = (latest['close'] - latest['FVP']) / latest['FVP']
            if (deviation > p['EXPANSION_DEVIATION_THRESHOLD']
                    and latest['RSI'] > p['EXPANSION_RSI_SHORT_MIN']):
                short_mandatory = True
            if (deviation < -p['EXPANSION_DEVIATION_THRESHOLD']
                    and latest['RSI'] < p['EXPANSION_RSI_LONG_MAX']):
                long_mandatory = True
        elif phase == 'trend':
            if bullish:
                long_mandatory = True
            elif bearish:
                short_mandatory = True

        if not long_mandatory and not short_mandatory:
            return {'signal': 'no-go', 'confidence': 0.0, 'volume_contexts': []}

        # === OPTIONAL CONFIRMATIONS (direction-specific, need >= 1) ===
        long_opt = 0
        short_opt = 0

        # Opt 1: ADX direction
        if latest['ADX'] > p['VCP_ADX_TREND_THRESHOLD']:
            if latest['+DI'] > latest['-DI']:
                long_opt += 1
            else:
                short_opt += 1

        # Opt 2: Trend RSI (trend phase only)
        if phase == 'trend':
            if (bullish and latest['+DI'] > latest['-DI']
                    and p['TREND_RSI_LONG_MIN'] < latest['RSI'] < p['TREND_RSI_LONG_MAX']):
                long_opt += 1
            if (bearish and latest['-DI'] > latest['+DI']
                    and p['TREND_RSI_SHORT_MIN'] < latest['RSI'] < p['TREND_RSI_SHORT_MAX']):
                short_opt += 1

        # Opt 3: Volume support (any supportive context for the direction)
        context = self._volume_context(df)
        active_contexts = [k for k, v in context.items() if v]

        if ((context['no_supply'] and bullish) or context['stopping_volume']
                or (context['trust'] and latest['close'] > latest['open'])
                or (context['healthy'] and latest['close'] > latest['open'])):
            long_opt += 1
        if (context['buying_climax'] or context['divergence']
                or (context['absorption'] and latest['close'] < latest['open'])
                or (context['trust'] and latest['close'] <= latest['open'])
                or (context['healthy'] and latest['close'] <= latest['open'])):
            short_opt += 1

        # Opt 4: Liquidity / SMC
        liq = self._detect_liquidity_pattern(df)
        if liq['liquidity_long']:
            long_opt += 1
            active_contexts.append('liquidity_long')
        if liq['liquidity_short']:
            short_opt += 1
            active_contexts.append('liquidity_short')

        # === SIGNAL DECISION ===
        if long_mandatory and short_mandatory:
            signal_dir = 'long' if long_opt >= short_opt else 'short'
            n_opt = long_opt if signal_dir == 'long' else short_opt
        elif long_mandatory:
            signal_dir = 'long'
            n_opt = long_opt
        else:
            signal_dir = 'short'
            n_opt = short_opt

        if n_opt < 1:
            return {'signal': 'no-go', 'confidence': 0.0, 'volume_contexts': active_contexts}

        # === CONFIDENCE ===
        confidence = min(1.0, 0.5 + 0.15 * n_opt)

        # Counter-trend price momentum penalty (safety)
        MOMENTUM_LOOKBACK = int(p.get('MOMENTUM_LOOKBACK', 5))
        COUNTER_TREND_THRESHOLD = p.get('COUNTER_TREND_THRESHOLD', 0.003)
        COUNTER_TREND_FACTOR = p.get('COUNTER_TREND_FACTOR', 10.0)

        if len(df) > MOMENTUM_LOOKBACK + 1:
            price_change = latest['close'] / df['close'].iloc[-(MOMENTUM_LOOKBACK + 1)] - 1.0
            is_counter = ((signal_dir == 'long' and price_change < -COUNTER_TREND_THRESHOLD)
                       or (signal_dir == 'short' and price_change > COUNTER_TREND_THRESHOLD))
            if is_counter:
                penalty = min(0.7, abs(price_change) * COUNTER_TREND_FACTOR)
                confidence *= (1.0 - penalty)

        return {'signal': signal_dir, 'confidence': confidence, 'volume_contexts': active_contexts}

    def _generate_signals_vectorized(self, df, params, expansion_mask, trend_mask, bullish, bearish):
        """Vectorized signal generation using mandatory/optional architecture.

        Same logic as _generate_signal but applied to the full DataFrame at once.
        Mandatory conditions determine direction; optional confirmations validate.
        Returns int8 numpy array: 0=no-go, 1=long, -1=short.
        """
        n = len(df)

        # === MANDATORY CONDITIONS ===
        deviation = (df['close'] - df['FVP']) / df['FVP'].replace(0, np.nan)

        exp_long_mandatory = (
            expansion_mask
            & (deviation < -params['EXPANSION_DEVIATION_THRESHOLD'])
            & (df['RSI'] < params['EXPANSION_RSI_LONG_MAX'])
        )
        exp_short_mandatory = (
            expansion_mask
            & (deviation > params['EXPANSION_DEVIATION_THRESHOLD'])
            & (df['RSI'] > params['EXPANSION_RSI_SHORT_MIN'])
        )

        long_mandatory = exp_long_mandatory | (trend_mask & bullish)
        short_mandatory = exp_short_mandatory | (trend_mask & bearish)

        # === OPTIONAL CONFIRMATIONS ===
        long_opt = pd.Series(0, index=df.index, dtype=int)
        short_opt = pd.Series(0, index=df.index, dtype=int)

        # Opt 1: ADX direction
        adx_high = df['ADX'] > params['VCP_ADX_TREND_THRESHOLD']
        long_opt[adx_high & (df['+DI'] > df['-DI'])] += 1
        short_opt[adx_high & (df['-DI'] > df['+DI'])] += 1

        # Opt 2: Trend RSI (trend phase only)
        trend_rsi_long = (
            trend_mask & bullish
            & (df['+DI'] > df['-DI'])
            & (df['RSI'] > params['TREND_RSI_LONG_MIN'])
            & (df['RSI'] < params['TREND_RSI_LONG_MAX'])
        )
        trend_rsi_short = (
            trend_mask & bearish
            & (df['-DI'] > df['+DI'])
            & (df['RSI'] > params['TREND_RSI_SHORT_MIN'])
            & (df['RSI'] < params['TREND_RSI_SHORT_MAX'])
        )
        long_opt[trend_rsi_long] += 1
        short_opt[trend_rsi_short] += 1

        # Opt 3: Volume support (vectorized computation)
        vol_avg_win   = int(params['VOLUME_AVG_WINDOW'])
        vol_slope_win = int(params['VOLUME_SLOPE_WINDOW'])
        vol_avg     = df['volume'].rolling(vol_avg_win).mean()
        rel_volume  = df['volume'] / vol_avg.replace(0, np.nan)
        candle_body = abs(df['close'] - df['open'])
        rel_body    = candle_body / df['ATR'].replace(0, np.nan)
        price_slope = df['close'].diff(vol_slope_win)
        bullish_candle = df['close'] > df['open']
        bearish_candle = df['close'] < df['open']
        candle_range = df['high'] - df['low']

        no_supply = (price_slope < 0) & (rel_volume < 0.7)
        stopping_vol = (
            (rel_volume > params['EXTREME_VOLUME_THRESHOLD'])
            & bearish_candle
            & (df['close'] > df['low'] + candle_range * 0.3)
        )
        trust = (
            (rel_body > params['BIG_BODY_ATR_THRESHOLD'])
            & (rel_volume > params['BIG_VOLUME_THRESHOLD'])
        )
        healthy = (
            ((price_slope > 0) & (df['volume'].diff(vol_slope_win) > 0))
            | ((price_slope < 0) & (df['volume'].diff(vol_slope_win) < 0))
        )
        absorption = (rel_volume > params['BIG_VOLUME_THRESHOLD']) & (rel_body < 0.5)

        div_lb = int(params['DIVERGENCE_LOOKBACK'])
        divergence = (
            (df['close'] > df['close'].shift(div_lb))
            & (df['volume'] < df['volume'].shift(div_lb))
        )

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
            (rel_volume > params['EXTREME_VOLUME_THRESHOLD'])
            & (rel_body > params['EXTREME_BODY_ATR_THRESHOLD'])
            & is_breakout & trend_up & is_overext
        )
        if buying_climax.any():
            cooldown_secs = int(params.get('BUYING_CLIMAX_COOLDOWN_SECONDS', 900))
            bc_indices = np.where(buying_climax.values)[0]
            bc_times = df.index[bc_indices]
            cooldown_td = pd.Timedelta(seconds=cooldown_secs)
            last_climax_time = None
            for idx, t in zip(bc_indices, bc_times):
                if last_climax_time is not None and (t - last_climax_time) < cooldown_td:
                    buying_climax.iloc[idx] = False
                else:
                    last_climax_time = t

        # Aggregate volume support per direction (any supportive context = +1)
        long_vol_support = (
            (no_supply & bullish) | stopping_vol
            | (trust & bullish_candle) | (healthy & bullish_candle)
        )
        short_vol_support = (
            buying_climax | divergence
            | (absorption & bearish_candle)
            | (trust & ~bullish_candle) | (healthy & ~bullish_candle)
        )
        long_opt[long_vol_support] += 1
        short_opt[short_vol_support] += 1

        # Opt 4: Liquidity / SMC pattern
        liq_adx_min = params.get('LIQ_ADX_MIN', 15)
        bos_lb   = int(params.get('LIQ_BOS_LOOKBACK', 20))
        sweep_lb = int(params.get('LIQ_SWEEP_LOOKBACK', 10))
        tol_liq  = params.get('LIQ_ZONE_TOLERANCE', 0.0015)
        adx_ok   = df['ADX'] > liq_adx_min

        df5 = df.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()

        liq_long  = pd.Series(False, index=df.index)
        liq_short = pd.Series(False, index=df.index)

        if len(df5) > bos_lb:
            recent_high_5m = df5['high'].rolling(bos_lb).max().shift(1)
            recent_low_5m  = df5['low'].rolling(bos_lb).min().shift(1)
            bos_up_5m   = df5['close'] > recent_high_5m
            bos_down_5m = df5['close'] < recent_low_5m

            bos_up_1m   = bos_up_5m.reindex(df.index, method='ffill').fillna(False)
            bos_down_1m = bos_down_5m.reindex(df.index, method='ffill').fillna(False)

            zone_body_low  = df5[['open','close']].min(axis=1)
            zone_body_high = df5[['open','close']].max(axis=1)
            z_range = zone_body_high - zone_body_low
            zone_lo = (zone_body_low  - z_range * tol_liq).reindex(df.index, method='ffill')
            zone_hi = (zone_body_high + z_range * tol_liq).reindex(df.index, method='ffill')

            prev_low_1m  = df['low'].rolling(sweep_lb).min().shift(1)
            prev_high_1m = df['high'].rolling(sweep_lb).max().shift(1)
            sweep_long_v  = (df['low'] < prev_low_1m) & (df['close'] > prev_low_1m)
            sweep_short_v = (df['high'] > prev_high_1m) & (df['close'] < prev_high_1m)

            in_zone_v = (df['close'] >= zone_lo) & (df['close'] <= zone_hi)
            bull_candle = df['close'] > df['open']
            bear_candle = df['close'] < df['open']

            liq_long  = adx_ok & bos_up_1m & sweep_long_v & in_zone_v & bull_candle
            liq_short = adx_ok & bos_down_1m & sweep_short_v & in_zone_v & bear_candle

            long_opt[liq_long] += 1
            short_opt[liq_short] += 1

        # === SIGNAL DECISION ===
        contraction_mask = ~expansion_mask & ~trend_mask
        volume_avg_gate = df['volume'].rolling(params.get('VOLUME_AVG_WINDOW', 4)).mean()
        relative_volume_gate = df['volume'] / volume_avg_gate.replace(0, np.nan)
        MIN_REL_VOL = params.get('MIN_RELATIVE_VOLUME', 0.8)
        vol_ok = relative_volume_gate >= MIN_REL_VOL

        valid_long = long_mandatory & vol_ok & ~contraction_mask & (long_opt >= 1)
        valid_short = short_mandatory & vol_ok & ~contraction_mask & (short_opt >= 1)

        signals = np.zeros(n, dtype=np.int8)
        signals[valid_long.values] = 1
        signals[valid_short.values] = -1

        # If both valid, take direction with more optionals
        both = valid_long & valid_short
        signals[(both & (short_opt > long_opt)).values] = -1

        # Build liquidity exemption mask for stability filter
        liq_exempt = np.zeros(n, dtype=bool)
        if len(df5) > bos_lb:
            liq_exempt = (liq_long | liq_short).values

        return signals, liq_exempt

    # =====================================================
    # MARGIN ADAPTATION
    # =====================================================

    def _adapt_margin(self, sec, phase, df):
        """Unified margin: TP_MULT * max(ATR/close, BB_width).

        No phase dependency — naturally adapts to volatility regime.
        During contraction both ATR and BB_width are small → small margin.
        During expansion/trend the dominant measure grows → larger margin.
        """
        p = self.params
        latest = df.iloc[-1]
        close = latest['close']
        atr_norm = latest['ATR'] / close if close > 0 else 0
        bb_w = latest['BB_width'] if not np.isnan(latest['BB_width']) else 0
        m = p['TP_MULT'] * max(atr_norm, bb_w)

        # Dynamic cost floor: ensure margin covers transaction costs
        _margin_mult = getattr(cm, 'MIN_ABS_MARGIN_MULTIPLIER', 1.5)
        cost_floor = _margin_mult * max(0.02, close * 0.0001) / close
        m = max(m, cost_floor)

        sec['params']['positionMargin'] = float(m)

    def _compute_margin_vectorized(self, df, params):
        """Vectorized unified margin for calibration.

        Same formula as _adapt_margin: TP_MULT * max(ATR/close, BB_width).
        Returns numpy array of margin factors (one per bar).
        """
        closes = df['close'].values
        atr_norm = np.nan_to_num(
            (df['ATR'] / df['close'].replace(0, np.nan)).values, nan=0.0)
        bb_w = np.nan_to_num(df['BB_width'].values, nan=0.0)
        margin_factor = params['TP_MULT'] * np.maximum(atr_norm, bb_w)

        # Dynamic cost floor: ensure margin covers transaction costs
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

            # ---- Data fingerprint for convergence detection ----
            data_fingerprint = f"{len(hist)}:{hist.index[-1].isoformat()}"
            if (p.get('_calibration_converged', False)
                    and p.get('_calibration_fingerprint', '') == data_fingerprint
                    and p.get('_calibration_perturb_count', 0) >= getattr(cm, 'CALIBRATION_MAX_PERTURBS', 3)):
                log.info(f"{self.seccode}: calibration SKIPPED (converged, data unchanged, "
                         f"perturbs exhausted={p.get('_calibration_perturb_count', 0)})")
                MinerviniClaude._calibration_cache[self.seccode] = dict(p)
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

            # Collect optimizable params (whitelist of 12 core params)
            MinerviniClaude._derive_params(best_params)
            optimizable = [
                (k, v) for k, v in best_params.items()
                if k in self._OPTIMIZABLE_PARAMS
                and isinstance(v, (int, float))
            ]

            indicator_group = [(k, v) for k, v in optimizable if k in self._INDICATOR_PARAMS]
            other_group     = [(k, v) for k, v in optimizable if k not in self._INDICATOR_PARAMS]

            # Baseline score (blended train+test)
            baseline = _blended_score(best_params)
            log.info(f"{self.seccode}: calibration baseline score={baseline:.6f} (blended)")

            # ---- Multi-resolution coordinate descent ----
            STEP_SIZES = getattr(cm, 'CALIBRATION_STEP_SIZES', [0.30, 0.15, 0.08])
            MAX_CALIBRATION_PASSES = getattr(cm, 'MAX_CALIBRATION_PASSES', 2)
            MIN_CALIBRATION_IMPROVEMENT = getattr(cm, 'MIN_CALIBRATION_IMPROVEMENT', 0.01)

            def _run_multi_resolution(bp, ind_group, oth_group):
                """Run full multi-resolution coordinate descent on bp (mutated in place).
                Returns (any_improved, cached_train_df, cached_test_df)."""
                any_improved = False
                c_train_df = None
                c_test_df = None

                for step_size in STEP_SIZES:
                    indicator_improved = False

                    # Phase 1: indicator meta-params at this resolution
                    for param_name, _ in ind_group:
                        base_value = bp[param_name]
                        candidates = self._make_candidates(base_value, 8, param_name, step_size)
                        best_score = -np.inf
                        best_value = base_value
                        for c in candidates:
                            bp[param_name] = c
                            MinerviniClaude._derive_params(bp)
                            score = _blended_score(bp)
                            if score > best_score:
                                best_score = score
                                best_value = c
                        bp[param_name] = best_value
                        MinerviniClaude._derive_params(bp)
                        if best_value != base_value:
                            indicator_improved = True
                            any_improved = True
                            log.info(f"{self.seccode}: step={step_size} {param_name}: {base_value} -> {best_value} (score={best_score:.6f})")

                    # Recompute cached indicators if changed or first resolution
                    if indicator_improved or c_train_df is None:
                        c_train_df = self._compute_indicators_for_calibration(train_df.copy(), bp)
                        c_test_df  = self._compute_indicators_for_calibration(test_df.copy(), bp)

                    # Phase 2: non-indicator params at this resolution
                    prev_score = _blended_score(bp, c_train_df, c_test_df)
                    for pass_num in range(MAX_CALIBRATION_PASSES):
                        for param_name, _ in oth_group:
                            base_value = bp[param_name]
                            candidates = self._make_candidates(base_value, 8, param_name, step_size)
                            best_score = -np.inf
                            best_value = base_value
                            for c in candidates:
                                bp[param_name] = c
                                MinerviniClaude._derive_params(bp)
                                score = _blended_score(bp, c_train_df, c_test_df)
                                if score > best_score:
                                    best_score = score
                                    best_value = c
                            bp[param_name] = best_value
                            MinerviniClaude._derive_params(bp)
                            if best_value != base_value:
                                any_improved = True
                                log.info(f"{self.seccode}: step={step_size} pass {pass_num+1} {param_name}: {base_value} -> {best_value} (score={best_score:.6f})")

                        current_score = _blended_score(bp, c_train_df, c_test_df)
                        improvement = (current_score - prev_score) / max(abs(prev_score), 1.0)
                        log.info(f"{self.seccode}: step={step_size} pass {pass_num+1} score={current_score:.2f}, improvement={improvement:.2%}")
                        if improvement < MIN_CALIBRATION_IMPROVEMENT:
                            break
                        prev_score = current_score

                return any_improved, c_train_df, c_test_df

            any_improved_overall, cached_train_df, cached_test_df = _run_multi_resolution(
                best_params, indicator_group, other_group)

            # ---- Stochastic perturbation if no improvement ----
            # Reset perturb_count when data changes (new fingerprint)
            if p.get('_calibration_fingerprint', '') != data_fingerprint:
                perturb_count = 0
            else:
                perturb_count = p.get('_calibration_perturb_count', 0)

            if not any_improved_overall:
                import random
                max_perturbs = getattr(cm, 'CALIBRATION_MAX_PERTURBS', 3)

                if perturb_count < max_perturbs:
                    perturb_range = getattr(cm, 'CALIBRATION_PERTURB_RANGE', 0.15)
                    saved_params = dict(best_params)
                    saved_score = _blended_score(best_params, cached_train_df, cached_test_df)

                    # Perturb all optimizable params randomly
                    for k, _ in optimizable:
                        v = best_params[k]
                        factor = 1.0 + random.uniform(-perturb_range, perturb_range)
                        new_val = v * factor
                        if isinstance(v, int):
                            new_val = max(1, int(round(new_val)))
                        floor = self._PARAM_FLOORS.get(k)
                        ceil = self._PARAM_CEILINGS.get(k)
                        if floor is not None:
                            new_val = max(floor, new_val) if not isinstance(v, int) else max(int(floor), new_val)
                        if ceil is not None:
                            new_val = min(ceil, new_val) if not isinstance(v, int) else min(int(ceil), new_val)
                        best_params[k] = new_val

                    # Propagate meta -> derived after perturbation
                    MinerviniClaude._derive_params(best_params)

                    # Re-optimize from perturbed state
                    _, cached_train_df, cached_test_df = _run_multi_resolution(
                        best_params, indicator_group, other_group)

                    perturbed_score = _blended_score(best_params, cached_train_df, cached_test_df)
                    if perturbed_score > saved_score:
                        log.info(f"{self.seccode}: perturbation {perturb_count+1} improved: "
                                 f"{saved_score:.2f} -> {perturbed_score:.2f}")
                        perturb_count = 0
                        any_improved_overall = True
                    else:
                        log.info(f"{self.seccode}: perturbation {perturb_count+1} failed: "
                                 f"{perturbed_score:.2f} <= {saved_score:.2f}, reverting")
                        best_params = saved_params
                        perturb_count += 1

            # ---- Final diagnostic: train vs test breakdown ----
            train_score = self._simulate_profit(train_df, best_params)
            test_score  = self._simulate_profit(test_df, best_params)
            log.info(f"{self.seccode}: walk-forward result: train={train_score:.2f}, test={test_score:.2f}, ratio={test_score/max(abs(train_score),1.0):.2f}")

            # ---- Store convergence metadata ----
            self.security['params']['_calibration_converged'] = not any_improved_overall
            self.security['params']['_calibration_fingerprint'] = data_fingerprint
            self.security['params']['_calibration_perturb_count'] = perturb_count if not any_improved_overall else 0

            # ---- Save optimized params ----
            # Note: stopLossCoefficient is NOT clamped here — stored value (e.g. 8) is used
            # by OPERATIONAL as-is. The [1.5, 3.0] clamp only applies inside _simulate_profit()
            # during calibration to enforce realistic RR ratios for parameter search.
            # Ensure derived params are consistent before saving
            MinerviniClaude._derive_params(best_params)
            MinerviniClaude._calibration_cache[self.seccode] = dict(best_params)

            for k, v in best_params.items():
                self.security['params'][k] = v
            self.params = self.security['params']

            final = self._simulate_profit(hist, best_params)

            # Persist calibration score so OPERATIONAL can filter low-score securities
            self.security['params']['calibration_score'] = round(float(final), 2)

            # Anti-degeneration: reset to _BASE_PARAMS when calibration yields
            # score=0, so the next cycle starts from sane values instead of
            # continuing to shrink towards zero.
            if final <= 0.0:
                base_params = getattr(cm, '_BASE_PARAMS', None)
                if base_params:
                    for k, v in base_params.items():
                        self.security['params'][k] = v
                    MinerviniClaude._derive_params(self.security['params'])
                    # Reset convergence flags so next cycle re-calibrates from base
                    self.security['params']['_calibration_converged'] = False
                    self.security['params']['_calibration_perturb_count'] = 0
                    self.params = self.security['params']
                    log.warning(
                        f"seccode={self.seccode} score=0, reset params to "
                        f"_BASE_PARAMS to prevent degeneration"
                    )

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

    def _make_candidates(self, base_value, steps=8, param_name=None, step_size=0.30):
        """Generate candidate values in [-step_size, +step_size] of base for coordinate descent.

        For int params: rounds to int, deduplicates, enforces minimum of 1.
        For float params: returns `steps` linearly spaced candidates.
        base_value is always the first candidate so ties preserve the current
        value instead of drifting toward -30% each cycle.
        If param_name is in _PARAM_FLOORS/_PARAM_CEILINGS, candidates are clamped.
        """
        candidates = [base_value * (1 + f) for f in np.linspace(-step_size, step_size, steps)]
        # Apply floor and ceiling if defined for this parameter
        floor = self._PARAM_FLOORS.get(param_name) if param_name else None
        ceil = self._PARAM_CEILINGS.get(param_name) if param_name else None
        if floor is not None:
            if isinstance(base_value, int):
                candidates = [max(int(floor), c) for c in candidates]
            else:
                candidates = [max(floor, c) for c in candidates]
        if ceil is not None:
            if isinstance(base_value, int):
                candidates = [min(int(ceil), c) for c in candidates]
            else:
                candidates = [min(ceil, c) for c in candidates]
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
        # Anchor: base_value first prevents degeneration when all candidates tie
        candidates = [base_value] + [c for c in candidates if c != base_value]
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
            signals, liq_exempt = self._generate_signals_vectorized(df, params, expansion_mask, trend_mask, bullish, bearish)

            # Step 3b: Signal stability filter — require N consecutive identical
            # signals before acting (mirrors real-time _signal_history check).
            # Liquidity (SMC) signals are exempt from this filter.
            stability_required = int(params.get('SIGNAL_STABILITY_REQUIRED', 3))
            if stability_required > 1:
                filtered = np.zeros_like(signals)
                for i in range(stability_required - 1, len(signals)):
                    if signals[i] != 0:
                        if liq_exempt[i]:
                            filtered[i] = signals[i]
                        else:
                            window = signals[i - stability_required + 1:i + 1]
                            if np.all(window == signals[i]):
                                filtered[i] = signals[i]
                signals = filtered

            # Step 4: Vectorized adaptive margin
            margin_factor = self._compute_margin_vectorized(df, params)

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
            # Position sizing via shared DolphRobot method (single source of truth)
            cash_4_position, fx_rate, board_lot, _ = self.dolph.compute_position_size(
                net_balance, 1.0, self.security)
            exposure_limit = net_balance * fx_rate  # exposure limit in local currency

            closes  = df['close'].values
            highs   = df['high'].values
            lows    = df['low'].values
            volumes = df['volume'].values
            # Clamp SL coefficient to [1.5, 3.0] during calibration for realistic RR ratios.
            # With coef=5 → RR=1:5 → need >83% win rate (unstable intraday).
            # With coef=3 → RR=1:3 → need >75%. With coef=1.5 → RR=1:1.5 → need >60%.
            _MIN_SL_COEFF = getattr(cm, 'MIN_STOP_LOSS_COEFFICIENT', 1.0)
            _MAX_SL_COEFF = getattr(cm, 'MAX_STOP_LOSS_COEFFICIENT', 4.0)
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

                # Entry fill depends on signal type:
                # Type A (breakout): aggressive fill at close[i+2] + slippage (quasi-market)
                # Type B (liquidity/pullback): conservative LMT fill (price must retrace)
                limit_bar = i + 2
                if limit_bar >= n:
                    continue
                limit_price = closes[limit_bar]
                fill_end = min(limit_bar + entry_timeout_bars, n)
                pending_until_bar = fill_end

                MAX_PARTICIPATION = getattr(cm, 'CALIBRATION_MAX_VOLUME_PARTICIPATION', 0.10)
                _, _, _, est_quantity = self.dolph.compute_position_size(
                    net_balance, limit_price, self.security)

                if liq_exempt[i]:
                    # === TYPE B: Pullback/Retest — conservative LMT fill ===
                    # Price must retrace to limit from the opposite direction.
                    entry_idx = -1
                    for fb in range(limit_bar, fill_end):
                        prev_close = closes[fb - 1] if fb > 0 else closes[fb]
                        if sig == 1 and prev_close > limit_price and lows[fb] <= limit_price:
                            bar_vol = volumes[fb]
                            if bar_vol > 0 and est_quantity > bar_vol * MAX_PARTICIPATION:
                                continue
                            entry_idx = fb
                            break
                        elif sig == -1 and prev_close < limit_price and highs[fb] >= limit_price:
                            bar_vol = volumes[fb]
                            if bar_vol > 0 and est_quantity > bar_vol * MAX_PARTICIPATION:
                                continue
                            entry_idx = fb
                            break
                    if entry_idx < 0:
                        stats_entry_expired += 1
                        continue
                else:
                    # === TYPE A: Breakout — aggressive fill at limit_bar ===
                    # Momentum signal: fill immediately at close[i+2].
                    # Mirrors OPERATIONAL quasi-market (close + 0.1%).
                    bar_vol = volumes[limit_bar]
                    if bar_vol > 0 and est_quantity > bar_vol * MAX_PARTICIPATION:
                        stats_entry_expired += 1
                        continue
                    entry_idx = limit_bar
                    pending_until_bar = limit_bar + 1  # capital freed immediately

                # Slippage: breakout (market) gets higher slippage than pullback (LMT)
                if liq_exempt[i]:
                    FILL_SLIPPAGE = getattr(cm, 'CALIBRATION_FILL_SLIPPAGE', 0.0001)
                else:
                    FILL_SLIPPAGE = getattr(cm, 'CALIBRATION_BREAKOUT_SLIPPAGE', 0.0005)
                if sig == 1:
                    entry_price = limit_price * (1.0 + FILL_SLIPPAGE)
                else:
                    entry_price = limit_price * (1.0 - FILL_SLIPPAGE)
                m_abs       = entry_price * margin_factor[i]
                _, _, _, quantity = self.dolph.compute_position_size(
                    net_balance, entry_price, self.security)
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
                # Convert from USD to local currency so it matches profit units
                round_trip_cost = max(1.0, quantity * 0.005) * 2 * fx_rate

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

                    # Check exposure limits before opening (in local currency)
                    new_exposure = quantity * entry_price
                    if sig == 1 and (long_exposure + new_exposure) > exposure_limit:
                        stats_skip_exposure += 1
                        continue
                    if sig == -1 and (short_exposure + new_exposure) > exposure_limit:
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
            # Zero trades returns -inf so "do nothing" is NEVER preferred over "trade and lose".
            # This prevents the optimizer from killing all signals (e.g. EMA_FAST=EMA_MID).
            MIN_TRADES_PER_DAY = getattr(cm, 'CALIBRATION_MIN_TRADES_PER_DAY', 0.5)
            if trades_opened == 0:
                sample_price = closes[len(closes) // 2]
                _, _, bl, sample_qty = self.dolph.compute_position_size(
                    net_balance, sample_price, self.security)
                if bl > 1 and sample_qty == 0:
                    log.warning(
                        f"{self.seccode}: 0 trades, board_lot={bl} too large for "
                        f"net_balance={net_balance} at price={sample_price:.2f} "
                        f"(need {bl * sample_price / fx_rate:.0f} USD min)")
                return -np.inf
            if num_trading_days > 0 and trades_per_day < MIN_TRADES_PER_DAY:
                total_profit *= trades_per_day / MIN_TRADES_PER_DAY

            # Normalize profit to USD for cross-security comparability.
            # Without this, GBX stocks (fx_rate=79) appear ~79× more profitable
            # than equivalent USD positions, inflating calibration_score.
            if fx_rate > 0 and fx_rate != 1.0:
                total_profit /= fx_rate

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

    def _detect_liquidity_pattern(self, df):
        """Detect SMC liquidity pattern on latest bar.
        BOS on 5-min resampled data, sweep + zone entry on 1-min.
        Requires ADX > LIQ_ADX_MIN.
        Returns dict with 'liquidity_long' and 'liquidity_short' bools.
        """
        p = self.params
        result = {'liquidity_long': False, 'liquidity_short': False}

        # Gate: only in trending conditions
        if df.iloc[-1]['ADX'] < p.get('LIQ_ADX_MIN', 15):
            return result

        bos_lb   = int(p.get('LIQ_BOS_LOOKBACK', 20))
        sweep_lb = int(p.get('LIQ_SWEEP_LOOKBACK', 10))
        tol      = p.get('LIQ_ZONE_TOLERANCE', 0.0015)

        # Resample to 5-min for BOS
        df5 = df.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
        if len(df5) < bos_lb + 1:
            return result

        latest   = df.iloc[-1]
        latest5  = df5.iloc[-1]

        # BOS on 5-min
        recent_high_5m = df5['high'].iloc[-(bos_lb+1):-1].max()
        recent_low_5m  = df5['low'].iloc[-(bos_lb+1):-1].min()
        bos_up   = latest5['close'] > recent_high_5m
        bos_down = latest5['close'] < recent_low_5m
        if not bos_up and not bos_down:
            return result

        # Demand/Supply zone = body of the 5-min impulse candle
        zone_low  = min(latest5['open'], latest5['close'])
        zone_high = max(latest5['open'], latest5['close'])
        zone_range = zone_high - zone_low
        zone_low  -= zone_range * tol
        zone_high += zone_range * tol

        # Liquidity sweep on 1-min
        prev_low_1m  = df['low'].iloc[-(sweep_lb+1):-1].min()
        prev_high_1m = df['high'].iloc[-(sweep_lb+1):-1].max()
        sweep_long  = (latest['low'] < prev_low_1m) and (latest['close'] > prev_low_1m)
        sweep_short = (latest['high'] > prev_high_1m) and (latest['close'] < prev_high_1m)

        # Entry: price in zone + bounce candle
        in_zone = zone_low <= latest['close'] <= zone_high

        if bos_up and sweep_long and in_zone and latest['close'] > latest['open']:
            result['liquidity_long'] = True
        if bos_down and sweep_short and in_zone and latest['close'] < latest['open']:
            result['liquidity_short'] = True

        return result
