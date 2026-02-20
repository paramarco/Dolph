import pandas as pd
import logging

log = logging.getLogger("PredictionModel")


class RsiAndEmaAndChatGpt:
    def __init__(self, data, params, dolph):
        self.df = data['1Min'].copy()

        # Remove non-price columns
        self.df = self.df.drop(
            columns=['hastrade', 'addedvolume', 'numberoftrades'],
            errors='ignore'
        )

        # Ensure datetime index
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a datetime index.")

        # Standardize column names
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
        Predicts 'long', 'short', or 'no-go' based solely on 5-minute RSI/EMA values.
        """
        try:
            seccode = sec['seccode']
            entryPrice = exitPrice = 0.0

            # Log current position info
            lastClosePrice = self.dolph.getLastClosePrice(seccode)
            if self.dolph.tp.isPositionOpen(seccode):
                for p in self.dolph.tp.get_PositionsByCode(seccode):
                    entryPrice = p.entryPrice
                    exitPrice = p.exitPrice
            log.info(f"{seccode} last: {lastClosePrice}, entry: {entryPrice}, exit: {exitPrice}")

            # Filter historical 1-minute data for this symbol
            df_sec = self.df[self.df['mnemonic'] == seccode].copy()
            if df_sec.empty:
                log.info(f"{seccode}: no historical 1-minute data available.")
                return {'signal': 'no-go', 'confidence': 0.0}

            # Resample to 5-minute candles
            df_5min = df_sec.resample('5T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()

            # Require at least 200 rows for EMA200
            if len(df_5min) < 200:
                log.info(f"{seccode}: not enough 5-minute candles for EMA200 (have {len(df_5min)}).")
                return {'signal': 'no-go', 'confidence': 0.0}

            # Compute indicators
            df_5min['RSI'] = self._calculate_rsi(df_5min['close'], 7)
            df_5min['EMA50'] = df_5min['close'].ewm(span=50, adjust=False).mean()
            df_5min['EMA200'] = df_5min['close'].ewm(span=200, adjust=False).mean()
            df_5min.dropna(inplace=True)

            if df_5min.empty:
                log.info(f"{seccode}: indicators not available.")
                return {'signal': 'no-go', 'confidence': 0.0}

            # Use latest values
            rsi = df_5min['RSI'].iloc[-1]
            ema50 = df_5min['EMA50'].iloc[-1]
            ema200 = df_5min['EMA200'].iloc[-1]
            log.info(f"{seccode}: [5-min] RSI={rsi:.2f}, EMA50={ema50:.2f}, EMA200={ema200:.2f}")

            # Decision logic
            if rsi < 30 and ema50 > ema200:
                confidence = (30.0 - rsi) / 30.0
                log.info(f"{seccode}: predictor says long (5-min indicators, confidence={confidence:.4f})")
                return {'signal': 'long', 'confidence': round(confidence, 4)}

            if rsi > 70 and ema50 < ema200:
                confidence = (rsi - 70.0) / 30.0
                log.info(f"{seccode}: predictor says short (5-min indicators, confidence={confidence:.4f})")
                return {'signal': 'short', 'confidence': round(confidence, 4)}

            log.info(f"{seccode}: predictor says no-go (5-min indicators)")
            return {'signal': 'no-go', 'confidence': 0.0}

        except Exception as e:
            log.error(f"{seccode}: Failed: {e}", e)
            return {'signal': 'no-go', 'confidence': 0.0}

    def _calculate_rsi(self, series, period):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain[:period].mean()
        avg_loss = loss[:period].mean()

        if avg_loss == 0:
            rs = float('inf')
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        if len(series) < period:
            return None

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
