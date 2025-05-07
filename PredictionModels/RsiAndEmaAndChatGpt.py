import pandas as pd
import logging
import os
import base64
import openai
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

def ask_chatgpt_image_decision(image_path, action_type="long"):
    prompt = f"Should I open a {action_type} position?"
    with open(image_path, "rb") as image_file:
        image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful trading advisor. Reply only with 'Yes' or 'No'."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "auto"}}
            ]}
        ],
        max_tokens=10
    )
    return response.choices[0].message.content.strip()



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
        openai.api_key =self.dolph.open_ai_key
        
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a test bot."},
                    {"role": "user", "content": "Say hello."}
                ],
                max_tokens=10
            )
            log.info(f"OpenAI key is valid. GPT says: {response.choices[0].message.content}")
        except Exception as e:
            log.exception("OpenAI API key check failed. GPT call is not working.")

        
    def build_model(self):
        pass

    def train(self):
        pass

    def load_trained_model(self, security):
        return True

    def predict(self, df, sec, period):
        """
        Predict whether to go long, short, or no-go based on RSI, EMA, and price trends.
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
            self.df['RSI'] = self._calculate_rsi(self.df['close'], 14)
            self.df.dropna(inplace=True)

        # Two steps before
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
                # File name per seccode
            image_filename = f"{seccode}_decision_chart.png"
            plot_candles_with_indicators(self.df, seccode, filename=image_filename, share_name=seccode)
            log.info(f"Making plot for {seccode} ...")

            # Decide based on GPT
            if rsi < 30 and ema50 > ema200:
                log.info(f"{seccode}: RSI={rsi:.2f}, EMA50={ema50:.2f}, EMA200={ema200:.2f} next then asking GPT ...")
                gpt_reply = ask_chatgpt_image_decision(image_filename, action_type="long")
                log.info(f"{seccode}: RSI={rsi:.2f}, EMA50={ema50:.2f}, EMA200={ema200:.2f} → GPT: {gpt_reply}")
                if gpt_reply.lower().startswith("yes"):
                    return 'long'
            
            elif rsi > 70 and ema50 < ema200:
                log.info(f"{seccode}: RSI={rsi:.2f}, EMA50={ema50:.2f}, EMA200={ema200:.2f} next then asking GPT ...")
                gpt_reply = ask_chatgpt_image_decision(image_filename, action_type="short")
                log.info(f"{seccode}: RSI={rsi:.2f}, EMA50={ema50:.2f}, EMA200={ema200:.2f} → GPT: {gpt_reply}")
                if gpt_reply.lower().startswith("yes"):
                    return 'short'
            
            log.info(f"{seccode}: predictor says nogo")
            return 'no-go'
       
        except Exception as e:        
            log.error(f"{seccode}: Failed : {e}", e)            
            return 'no-go'


    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
    
        # Step 1: calculate first average gain and loss
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
        # Step 2: create a series to hold RSI values
        rsi = pd.Series(dtype=float, index=series.index)
    
        # Step 3: first RSI value after initial SMA
        rs = avg_gain.iloc[period] / avg_loss.iloc[period]
        rsi.iloc[period] = 100 - (100 / (1 + rs))
    
        # Step 4: smoothing
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



    
