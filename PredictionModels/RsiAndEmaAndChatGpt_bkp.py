import pandas as pd
import logging
import os
import base64

from openai import OpenAI
import matplotlib.pyplot as plt
import mplfinance as mpf

log = logging.getLogger("PredictionModel")
for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(proxy_var, None)


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


def ask_chatgpt_image_decision(image_path, action_type="long", client=None):
    prompt = f"Should I open a {action_type} position?"

    with open(image_path, "rb") as image_file:
        image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "auto"
                            }
                        }
                    ]
                }
            ],
            max_tokens=10
        )
        log.info(f"GPT raw reply: {response}")
        return response.choices[0].message.content.strip()
    except Exception:
        log.exception("Failed to get ChatGPT decision.")
        return "no"


class RsiAndEmaAndChatGpt:
    def __init__(self, data, params, dolph):
        self.params = params
        self.dolph = dolph
        self.df = self._prepare_df(data['1Min'].copy())
        log.info("i am initilizing next is chat gpt")
        log.info(f"OpenAI key being used: {self.dolph.open_ai_key}")
     
        client = OpenAI(api_key=self.dolph.open_ai_key)
        response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": "Hello, who won the last world cup?"}],
        max_tokens=10,
        timeout=10  # seconds
          )
      
        if response and response.choices and response.choices[0].message:
            log.info(f"GPT says: {response.choices[0].message.content}")
        else:
            log.warning("Received empty or incomplete response from OpenAI.")
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",                
                messages=[
                    {"role": "system", "content": "You are a test bot."},
                    {"role": "user", "content": "Say hello."}
                ],
                max_tokens=10
            )
            log.info(f"OpenAI key is valid. GPT says: {response.choices[0].message.content}")
        except Exception:
            log.exception("OpenAI API key check failed.")

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
            log.info(f"i am in predictiong")

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

            df['RSI'] = self._calculate_rsi(df['close'], 14)
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
            df.dropna(inplace=True)

            rsi = df['RSI'].iloc[-1]
            ema50 = df['EMA50'].iloc[-1]
            ema200 = df['EMA200'].iloc[-1]

            log.info(f"Values rsi ema50 ema200 : {rsi} {ema50} {ema200}")

            image_filename = f"{seccode}_decision_chart.png"
            plot_candles_with_indicators(df, seccode, filename=image_filename, share_name=seccode)
            log.info(f"Making plot for {seccode} ...")

            if rsi < 30 and ema50 > ema200:
                log.info(f"{seccode}: RSI={rsi:.2f}, EMA50={ema50:.2f}, EMA200={ema200:.2f} → asking GPT...")
                gpt_reply = ask_chatgpt_image_decision(image_filename, action_type="long", client=self.client)
                log.info(f"{seccode} GPT reply: {gpt_reply}")
                if gpt_reply.lower().startswith("yes"):
                    return 'long'

            elif rsi > 70 and ema50 < ema200:
                log.info(f"{seccode}: RSI={rsi:.2f}, EMA50={ema50:.2f}, EMA200={ema200:.2f} → asking GPT...")
                gpt_reply = ask_chatgpt_image_decision(image_filename, action_type="short", client=self.client)
                log.info(f"{seccode} GPT reply: {gpt_reply}")
                if gpt_reply.lower().startswith("yes"):
                    return 'short'

            log.info(f"{seccode}: predictor says no-go")
            return 'no-go'

        except Exception:
            log.exception(f"{seccode}: Failed during prediction")
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
