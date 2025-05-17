import pandas as pd
import logging
import os
import base64

from openai import OpenAI
import matplotlib.pyplot as plt
import mplfinance as mpf

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


class RsiAndEmaAndChatGpt:
    def __init__(self, data, params, dolph):
        self.params = params
        self.dolph = dolph
        self.df = self._prepare_df(data['1Min'].copy())
    
    def _prepare_df(self, df):
        df = df.drop(columns=['hastrade', 'addedvolume', 'numberoftrades'], errors='ignore')
        df = df.rename(columns={
            'startprice': 'open',
            'maxprice': 'high',
            'minprice': 'low',
            'endprice': 'close'
        })
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a datetime index.")
        return df

    def build_model(self):
        pass

    def train(self):
        pass

    def load_trained_model(self, security):
        return True

    def predict(self, df, sec, period):
        try:
            log.info("i am in predictiong")

            seccode = sec['seccode']
            entryPrice = exitPrice = 0.0
            lastClosePrice = self.dolph.getLastClosePrice(seccode)
            openPosition = self.dolph.tp.isPositionOpen(seccode)

            if openPosition:
                positions = self.dolph.tp.get_PositionsByCode(seccode)
                for p in positions:
                    entryPrice = p.entryPrice
                    exitPrice = p.exitPrice

            log.info(f"{seccode} last: {lastClosePrice}, entry: {entryPrice}, exit: {exitPrice}")

            if 'mnemonic' not in df.columns:
                raise KeyError("DataFrame is missing 'mnemonic' column.")

            df = df[df['mnemonic'] == seccode]
            df = self._prepare_df(df)

            # ✅ Only make prediction on 5-minute boundary
            now = df.index[-1]
            if now.minute % 5 != 0 or now.second != 0:
                log.info(f"{seccode}: Skipping prediction — not a 5-min boundary (now={now})")
                return 'no-go'

            # === 5-minute candles ===
            df_5min = df.resample('5min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            df_5min['RSI'] = self._calculate_rsi(df_5min['close'], 14)
            df_5min['EMA50'] = df_5min['close'].ewm(span=50, adjust=False).mean()
            df_5min['EMA200'] = df_5min['close'].ewm(span=200, adjust=False).mean()
            df_5min.dropna(inplace=True)

            rsi = df_5min['RSI'].iloc[-1]
            ema50 = df_5min['EMA50'].iloc[-1]
            ema200 = df_5min['EMA200'].iloc[-1]

            log.info(f"5-minute Values RSI={rsi:.2f}, EMA50={ema50:.2f}, EMA200={ema200:.2f}")

            # ✅ Plot only on new 5-min candle
            image_filename = f"{seccode}_5min_chart_{now.strftime('%Y%m%d_%H%M')}.png"
            plot_candles_with_indicators(df_5min, seccode, filename=image_filename, share_name=f"{seccode} (5min)")
            log.info(f"5-min plot saved: {image_filename}")

            if rsi < 30 and ema50 > ema200:
                return 'long'
            elif rsi > 70 and ema50 < ema200:
                return 'short'

            return 'no-go'

        except Exception:
            log.exception(f"{sec.get('seccode', 'UNKNOWN')}: Failed during prediction")
            return 'no-go'

    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rsi = pd.Series(dtype=float, index=series.index)

        if avg_loss.iloc[period] == 0:
            rsi.iloc[period] = 100
        else:
            rs = avg_gain.iloc[period] / avg_loss.iloc[period]
            rsi.iloc[period] = 100 - (100 / (1 + rs))

        for i in range(period + 1, len(series)):
            current_gain = gain.iloc[i]
            current_loss = loss.iloc[i]
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + current_gain) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + current_loss) / period

            if avg_loss.iloc[i] == 0:
                rsi.iloc[i] = 100
            else:
                rs = avg_gain.iloc[i] / avg_loss.iloc[i]
                rsi.iloc[i] = 100 - (100 / (1 + rs))

        return rsi
