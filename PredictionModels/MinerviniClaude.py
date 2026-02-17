# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
import datetime as dt
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
        'exitTimeSeconds', 'entryTimeSeconds', 'minNumPastSamples',
        'CALIBRATION_LOOKBACK_DAYS', 'CALIBRATION_LIMIT_RESULTS',
        'CALIBRATION_MIN_ROWS', 'CALIBRATION_MARGIN_MIN',
        'CALIBRATION_MARGIN_MAX', 'CALIBRATION_MARGIN_STEPS',
        'CALIBRATION_LOOKAHEAD_BARS', 'CALIBRATION_STOPLOSS_MULTIPLIER',
        'CALIBRATION_DEFAULT_MARGIN', 'BUYING_CLIMAX_COOLDOWN_SECONDS'
    })

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

            if signal not in ['long', 'short', 'no-go']:
                log.error(f"{self.seccode}: invalid signal {signal}, forcing no-go")
                signal = 'no-go'

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
        df['BB_width'] = (p['BB_STD'] * std) / ma                       # cm.BB_STD = 2
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

    # =====================================================
    # SIGNAL GENERATION
    # =====================================================

    def _generate_signal(self, df, phase):

        p = self.params
        long_score = 0.0
        short_score = 0.0
        latest = df.iloc[-1]        # Get latest indicator values
        bullish = latest['EMA_FAST'] > latest['EMA_MID'] > latest['EMA_SLOW']
        bearish = latest['EMA_FAST'] < latest['EMA_MID'] < latest['EMA_SLOW']

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
            # Bullish Trend Continuation
            #   cm.TREND_RSI_LONG_MIN = 40
            #   cm.TREND_RSI_LONG_MAX = 70
            if (
                bullish and latest['+DI'] > latest['-DI']
                and p['TREND_RSI_LONG_MIN'] < latest['RSI'] < p['TREND_RSI_LONG_MAX']
            ):
                long_score += 1.5

            # Bearish Trend Continuation
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
            long_score -= 0.7

        if context['absorption']:
            short_score += 0.6

        # =============================
        # FINAL DECISION
        # =============================
        score_diff = long_score - short_score
        total_score = long_score + short_score

        if total_score == 0:
            return 'no-go'

        confidence = abs(score_diff) / total_score

        # ---- Low energy filter ---- cm.MIN_TOTAL_SCORE = 1.5
        # Minimum intensity filter. Avoid trades when there is very little total energy.
        if total_score < p['MIN_TOTAL_SCORE']:
            return 'no-go'

        # ---- Conflict filter ----  cm.MIN_CONFIDENCE = 0.6
        # Minimum conviction filter. Avoid trades when there is internal conflict.
        if confidence < p['MIN_CONFIDENCE']:
            return 'no-go'

        # ---- Direction ----
        if score_diff > 0:
            return 'long'
        else:
            return 'short'

        log.error("returning no-go out of logic")
        return 'no-go'


    # =====================================================
    # MARGIN ADAPTATION
    # =====================================================

    def _adapt_margin(self, sec, phase, df):

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

        # Apply computed margin to security parameters
        sec['params']['positionMargin'] = float(m)


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

            # Baseline score
            baseline = self._simulate_profit(hist, best_params)
            log.info(f"{self.seccode}: calibration baseline profit score={baseline:.6f}")

            # ---- Phase 1: Indicator params (each step recomputes indicators) ----
            for param_name, base_value in indicator_group:
                candidates = self._make_candidates(base_value, 8)
                best_score = -np.inf
                best_value = base_value
                for c in candidates:
                    best_params[param_name] = c
                    score = self._simulate_profit(hist, best_params)
                    if score > best_score:
                        best_score = score
                        best_value = c
                best_params[param_name] = best_value
                if best_value != base_value:
                    log.info(f"{self.seccode}: {param_name}: {base_value} -> {best_value} (score={best_score:.6f})")

            # ---- Phase 2: Non-indicator params (cache indicators, cheaper) ----
            cached_df = self._compute_indicators_for_calibration(hist.copy(), best_params)

            for param_name, base_value in other_group:
                candidates = self._make_candidates(base_value, 8)
                best_score = -np.inf
                best_value = base_value
                for c in candidates:
                    best_params[param_name] = c
                    score = self._simulate_profit(hist, best_params, indicator_df=cached_df)
                    if score > best_score:
                        best_score = score
                        best_value = c
                best_params[param_name] = best_value
                if best_value != base_value:
                    log.info(f"{self.seccode}: {param_name}: {base_value} -> {best_value} (score={best_score:.6f})")

            # ---- Save optimized params ----
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


    def _make_candidates(self, base_value, steps=8):
        """Generate candidate values: -30% to +30% of base in `steps` steps."""
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
        """Like _compute_indicators but takes explicit params dict and uses
        a fast numpy-based BB_width_pctile (avoids slow pandas lambda)."""
        df['EMA_FAST'] = df['close'].ewm(span=int(params['EMA_FAST'])).mean()
        df['EMA_MID']  = df['close'].ewm(span=int(params['EMA_MID'])).mean()
        df['EMA_SLOW'] = df['close'].ewm(span=int(params['EMA_SLOW'])).mean()
        df['RSI']      = self._rsi_series(df['close'], int(params['RSI_PERIOD']))
        df['ATR']      = self._atr(df, int(params['ATR_PERIOD']))
        df['ATR_slope'] = df['ATR'].diff(int(params['ATR_SLOPE_WINDOW']))
        df['ADX'], df['+DI'], df['-DI'] = self._adx(df, int(params['ADX_PERIOD']))

        bb_win = int(params['BB_WINDOW'])
        ma  = df['close'].rolling(bb_win).mean()
        std = df['close'].rolling(bb_win).std()
        df['BB_width'] = (params['BB_STD'] * std) / ma

        # Fast BB_width percentile using numpy
        bb_vals = df['BB_width'].values
        pctile_win = int(params['BB_PERCENTILE_WINDOW'])
        pctile = np.full(len(bb_vals), np.nan)
        for i in range(pctile_win - 1, len(bb_vals)):
            w = bb_vals[i - pctile_win + 1:i + 1]
            valid = w[~np.isnan(w)]
            if len(valid) > 0:
                pctile[i] = np.sum(valid <= valid[-1]) / len(valid)
        df['BB_width_pctile'] = pctile

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

            # Step 3: Vectorized signal scoring
            long_score  = pd.Series(0.0, index=df.index)
            short_score = pd.Series(0.0, index=df.index)

            # -- Expansion signals --
            deviation = (df['close'] - df['FVP']) / df['FVP']
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

            # -- Trend signals --
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
            rel_volume  = df['volume'] / vol_avg
            candle_body = abs(df['close'] - df['open'])
            rel_body    = candle_body / df['ATR']
            price_slope = df['close'].diff(vol_slope_win)

            absorption = (rel_volume > params['BIG_VOLUME_THRESHOLD']) & (rel_body < 0.5)
            short_score[absorption] += 0.6

            div_lb = int(params['DIVERGENCE_LOOKBACK'])
            divergence = (
                (df['close'] > df['close'].shift(div_lb)) &
                (df['volume'] < df['volume'].shift(div_lb))
            )
            long_score[divergence] -= 0.7

            no_supply = (price_slope < 0) & (rel_volume < 0.7)
            long_score[no_supply & bullish] += 0.8

            bc_lb       = int(params['BUYING_CLIMAX_LOOKBACK'])
            bc_trend_lb = int(params['BUYING_CLIMAX_TREND_LOOKBACK'])
            recent_high = df['high'].rolling(bc_lb).max().shift(1)
            is_breakout = df['high'] >= recent_high
            recent_mean = df['close'].rolling(bc_trend_lb).mean()
            prior_mean  = df['close'].rolling(bc_trend_lb).mean().shift(bc_trend_lb)
            trend_up    = recent_mean > prior_mean
            ext         = (df['close'] - df['FVP']) / df['FVP']
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

            # 0 = no-go, 1 = long, -1 = short
            signals = np.zeros(len(df), dtype=np.int8)
            valid = (total_score >= params['MIN_TOTAL_SCORE']) & (confidence >= params['MIN_CONFIDENCE'])
            signals[(valid & (score_diff > 0)).values]  =  1
            signals[(valid & (score_diff < 0)).values]  = -1

            # Step 4: Vectorized adaptive margin per bar
            margin_factor = np.full(len(df), params['MARGIN_CONTRACTION_FIXED'])

            exp_raw     = (df['BB_width'] * params['MARGIN_EXPANSION_MULTIPLIER']).values
            exp_clamped = np.clip(exp_raw, params['MARGIN_EXPANSION_MIN'], params['MARGIN_EXPANSION_MAX'])
            exp_idx     = expansion_mask.values
            margin_factor[exp_idx] = exp_clamped[exp_idx]

            trend_raw     = (params['MARGIN_TREND_ATR_MULTIPLIER'] * df['ATR'] / df['close']).values
            trend_clamped = np.clip(trend_raw, params['MARGIN_TREND_MIN'], params['MARGIN_TREND_MAX'])
            trend_idx     = trend_mask.values
            margin_factor[trend_idx] = trend_clamped[trend_idx]

            # Step 5: Trade simulation
            closes = df['close'].values
            highs  = df['high'].values
            lows   = df['low'].values
            sl_coeff  = params['stopLossCoefficient']
            lookahead = int(params['CALIBRATION_LOOKAHEAD_BARS'])
            n = len(df)

            total_profit = 0.0

            for i in range(n - lookahead - 2):
                sig = signals[i]
                if sig == 0:
                    continue

                # Entry at bar i+2 (n+1)
                entry_idx   = i + 2
                entry_price = closes[entry_idx]
                m_abs       = entry_price * margin_factor[i]

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

                if tp_first <= sl_first and tp_first <= lookahead:
                    total_profit += m_abs
                elif sl_first < tp_first and sl_first <= lookahead:
                    total_profit -= sl_coeff * m_abs

            return total_profit

        except Exception as e:
            log.error(f"{self.seccode}: simulation error: {e}")
            return -np.inf


    def _volume_context(self, df):

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
