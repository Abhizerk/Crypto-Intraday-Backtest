import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime

st.set_page_config(page_title="Crypto Intraday Strategy", layout="wide")

# --- Indicator Functions ---

def download_data(symbol, interval, lookback_minutes=1440):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(minutes=lookback_minutes)

    df = yf.download(tickers=symbol, start=start, end=end, interval=interval, progress=False)

    if df.empty:
        st.error("⚠️ No data found. Please check symbol, internet connection, or try a longer timeframe.")
        return pd.DataFrame()

    df = df.reset_index()
    df.dropna(inplace=True)
    df.columns = [c.lower() for c in df.columns]
    return df

def calculate_ema(df, period=9):
    df['ema'] = df['close'].ewm(span=period, adjust=False).mean()

def calculate_fibonacci_levels(df):
    max_price = df['high'].max()
    min_price = df['low'].min()
    diff = max_price - min_price
    fib_levels = {
        '0.0': max_price,
        '0.236': max_price - 0.236 * diff,
        '0.382': max_price - 0.382 * diff,
        '0.5': max_price - 0.5 * diff,
        '0.618': max_price - 0.618 * diff,
        '1.0': min_price
    }
    return fib_levels

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

def calculate_macd(df):
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

def detect_bullish_engulfing(df):
    df['bullish_engulfing'] = (
        (df['close'].shift(1) < df['open'].shift(1)) &
        (df['close'] > df['open']) &
        (df['open'] < df['close'].shift(1)) &
        (df['close'] > df['open'].shift(1))
    )

def detect_volume_spike(df, multiplier=2):
    avg_volume = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = df['volume'] > (multiplier * avg_volume)

def detect_breakouts(df):
    df['breakout_high'] = df['close'] > df['high'].rolling(window=20).max().shift(1)
    df['breakout_low'] = df['close'] < df['low'].rolling(window=20).min().shift(1)

# --- Strategy Filter ---

def apply_strategy(df, risk_reward=2):
    df['buy_signal'] = (
        (df['close'] > df['ema']) &
        (df['rsi'] > 50) &
        (df['macd'] > df['signal']) &
        df['bullish_engulfing'] &
        df['volume_spike'] &
        df['breakout_high']
    )
    df['target'] = df['close'] + (df['close'] - df['low']) * risk_reward
    df['stop_loss'] = df['low']
    return df[df['buy_signal']]

# --- Streamlit UI ---

st.title("🚀 Solana Intraday Strategy Dashboard")
st.markdown("Track potential **buy signals** using technical indicators on 1m, 5m, 15m charts.")

symbol = st.selectbox("Select Symbol", ["SOL-USD"])
interval = st.selectbox("Select Timeframe", ["1m", "5m", "15m"])
lookback_minutes = st.slider("Lookback Period (in minutes)", min_value=60, max_value=2880, step=60, value=720)
risk_reward = st.slider("Risk to Reward Ratio", 1, 5, 2)

df = download_data(symbol, interval, lookback_minutes)

if not df.empty:
    calculate_ema(df)
    calculate_rsi(df)
    calculate_macd(df)
    detect_bullish_engulfing(df)
    detect_volume_spike(df)
    detect_breakouts(df)
    fib_levels = calculate_fibonacci_levels(df)

    signals = apply_strategy(df, risk_reward)

    st.subheader("📈 Latest Signals")
    st.dataframe(signals[['datetime', 'close', 'ema', 'rsi', 'macd', 'signal', 'volume', 'target', 'stop_loss']].tail(10), use_container_width=True)

    st.subheader("📊 Fibonacci Levels")
    fib_df = pd.DataFrame.from_dict(fib_levels, orient='index', columns=['Price Level'])
    st.dataframe(fib_df)

    st.line_chart(df.set_index('datetime')[['close', 'ema']].tail(200))
