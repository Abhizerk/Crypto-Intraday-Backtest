import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
import matplotlib.pyplot as plt

# --------------------------------------------
# 🔧 Custom Functions
# --------------------------------------------

def download_data(symbol, interval, lookback_minutes=1440):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(minutes=lookback_minutes)
    df = yf.download(tickers=symbol, start=start, end=end, interval=interval)
    df = df.reset_index()
    df.dropna(inplace=True)
    df.columns = [c.lower() for c in df.columns]
    return df

def apply_indicators(df):
    df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    # Bullish engulfing
    df['bullish_engulfing'] = (df['close'].shift(1) < df['open'].shift(1)) & \
                              (df['close'] > df['open']) & \
                              (df['close'] > df['open'].shift(1)) & \
                              (df['open'] < df['close'].shift(1))

    # Volume spike
    df['vol_spike'] = df['volume'] > df['volume'].rolling(20).mean() * 1.5

    # Breakout (50-bar high)
    df['recent_high'] = df['high'].rolling(50).max()
    df['breakout'] = df['close'] > df['recent_high'].shift(1)

    return df

def get_fibonacci_levels(price):
    fib_38 = price * 0.382
    fib_61 = price * 0.618
    return fib_38, fib_61

def backtest_strategy(df, target_pct=1.0, stop_loss_pct=0.5):
    entries = []
    position = False
    buy_price = 0
    df['position'] = 0

    for i in range(51, len(df)):
        row = df.iloc[i]
        fib_38, fib_61 = get_fibonacci_levels(row['close'])

        if not position:
            condition = (
                row['close'] > row['ema9'] and
                row['low'] <= fib_61 and row['high'] >= fib_38 and
                row['rsi'] > 50 and
                row['macd'] > row['macd_signal'] and
                row['bullish_engulfing'] and
                row['vol_spike'] and
                row['breakout']
            )
            if condition:
                position = True
                buy_price = row['close']
                entries.append((row['datetime'], row['close'], 'BUY'))
                df.at[i, 'position'] = 1
            else:
                df.at[i, 'position'] = 0
        else:
            if row['close'] >= buy_price * (1 + target_pct / 100):
                entries.append((row['datetime'], row['close'], 'TARGET'))
                position = False
                df.at[i, 'position'] = 0
            elif row['close'] <= buy_price * (1 - stop_loss_pct / 100):
                entries.append((row['datetime'], row['close'], 'STOP'))
                position = False
                df.at[i, 'position'] = 0
            else:
                df.at[i, 'position'] = 1

    return entries, df

# --------------------------------------------
# 🚀 Streamlit App
# --------------------------------------------

st.set_page_config(page_title="Solana Intraday Strategy", layout="wide")
st.title("🚀 Solana Intraday Strategy Backtest Dashboard")

# Sidebar inputs
st.sidebar.header("📊 Backtest Settings")
target_pct = st.sidebar.slider("🎯 Target %", 0.5, 5.0, 1.0)
stop_loss_pct = st.sidebar.slider("🛑 Stop Loss %", 0.5, 5.0, 0.5)
lookback_minutes = st.sidebar.slider("🔁 Lookback Minutes", 60, 1440, 720)

# Get data
st.subheader("📈 Market Data (SOL-USD)")
df = download_data("SOL-USD", "1m", lookback_minutes)
df = apply_indicators(df)

# Run strategy
entries, df = backtest_strategy(df, target_pct, stop_loss_pct)

# Show latest indicators
with st.expander("🔢 Fibonacci & Indicators"):
    latest = df.iloc[-1]
    fib_38, fib_61 = get_fibonacci_levels(latest['close'])
    st.write(f"🔹 Current Price: {latest['close']:.2f}")
    st.write(f"🔹 EMA9: {latest['ema9']:.2f}")
    st.write(f"🔹 RSI: {latest['rsi']:.2f}")
    st.write(f"🔹 MACD: {latest['macd']:.4f}, Signal: {latest['macd_signal']:.4f}")
    st.write(f"🔹 Fib 38.2%: {fib_38:.2f}, Fib 61.8%: {fib_61:.2f}")
    st.write(f"🔹 Volume Spike: {latest['vol_spike']}")
    st.write(f"🔹 Breakout: {latest['breakout']}")
    st.write(f"🔹 Bullish Engulfing: {latest['bullish_engulfing']}")

# Trade log
st.subheader("📋 Trade Log")
if entries:
    trade_log = pd.DataFrame(entries, columns=["Time", "Price", "Action"])
    st.dataframe(trade_log)
else:
    st.warning("No trades found with current filters.")

# Plot trades
st.subheader("📊 Price Chart with Trades")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['datetime'], df['close'], label='Price', color='gray')
buy_signals = df[df['position'] == 1]
ax.scatter(buy_signals['datetime'], buy_signals['close'], marker='^', color='green', label='Buy')
ax.set_title("Solana - 1 Min Chart")
ax.legend()
st.pyplot(fig)
