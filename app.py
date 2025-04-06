import ccxt
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import streamlit as st

# 1️⃣ Page Setup
st.set_page_config(layout="wide")
st.title("📈 Advanced Crypto Backtester – EMA + Fib + RSI + MACD + Candles")

# 2️⃣ User Inputs
col1, col2, col3 = st.columns(3)
with col1:
    selected_symbol = st.selectbox("Select Coin", ['SOL/USDT', 'BTC/USDT', 'ETH/USDT'])
with col2:
    timeframe = st.selectbox("Timeframe", ['1m', '5m', '15m'])
with col3:
    risk_reward = st.slider("Risk-Reward Ratio", 1.0, 5.0, 2.0, step=0.5)

stop_loss_pct = 1.0
target_pct = stop_loss_pct * risk_reward

# 3️⃣ Fetch Live Data
@st.cache_data
def fetch_data(symbol, tf):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=500)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

df = fetch_data(selected_symbol, timeframe)
df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
macd = ta.trend.MACD(df['close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()

# 4️⃣ Candlestick Pattern: Bullish Engulfing
df['bullish_engulfing'] = (df['open'].shift(1) > df['close'].shift(1)) & (df['open'] < df['close']) & (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))

# 5️⃣ Fibonacci
swing_low = df['low'][-100:].min()
swing_high = df['high'][-100:].max()
fib_38 = swing_high - (swing_high - swing_low) * 0.382
fib_50 = swing_high - (swing_high - swing_low) * 0.5
fib_61 = swing_high - (swing_high - swing_low) * 0.618

# 6️⃣ Entry Strategy
entries = []
position = False
buy_price = 0

for i in range(1, len(df)):
    row = df.iloc[i]
    
    if not position:
        condition = (
            row['close'] > row['ema9'] and
            row['low'] <= fib_61 and row['high'] >= fib_38 and
            row['rsi'] > 50 and
            row['macd'] > row['macd_signal'] and
            row['bullish_engulfing']
        )
        if condition:
            position = True
            buy_price = row['close']
            entries.append((row['timestamp'], row['close'], 'BUY'))
            df.at[i, 'position'] = 1
        else:
            df.at[i, 'position'] = 0
    else:
        if row['close'] >= buy_price * (1 + target_pct / 100):
            entries.append((row['timestamp'], row['close'], 'TARGET'))
            position = False
            df.at[i, 'position'] = 0
        elif row['close'] <= buy_price * (1 - stop_loss_pct / 100):
            entries.append((row['timestamp'], row['close'], 'STOP'))
            position = False
            df.at[i, 'position'] = 0
        else:
            df.at[i, 'position'] = 1

df['position'] = df['position'].fillna(0)
df['returns'] = df['close'].pct_change()
df['strategy_returns'] = df['returns'] * df['position']
cumulative = (1 + df['strategy_returns'].fillna(0)).cumprod()

# 7️⃣ Chart
st.subheader("📊 Strategy vs Buy & Hold")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['timestamp'], cumulative, label='Strategy', color='green')
ax.plot(df['timestamp'], (1 + df['returns'].fillna(0)).cumprod(), label='Buy & Hold', color='gray')

for t, price, tag in entries:
    color = 'blue' if tag == 'BUY' else ('green' if tag == 'TARGET' else 'red')
    ax.scatter(t, price, color=color, label=tag if tag not in [e[2] for e in entries[:entries.index((t, price, tag))]] else "", marker='o')

ax.legend()
st.pyplot(fig)

# 8️⃣ Fibonacci & Indicators
with st.expander("🔢 Fibonacci & Indicators"):
    st.write(f"**Swing High**: {swing_high:.2f}, **Swing Low**: {swing_low:.2f}")
    st.write(f"Fib 38.2%: {fib_38:.2f}, Fib 50%: {fib_50:.2f}, Fib 61.8%: {fib_61:.2f}")

# 9️⃣ Trades Table
st.subheader("📋 Trades Log")
trades_df = pd.DataFrame(entries, columns=['Time', 'Price', 'Action'])
st.dataframe(trades_df.tail(10), use_container_width=True)

csv = trades_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Trade Log", csv, "trades.csv", "text/csv")

# 🔟 Summary
st.subheader("📈 Performance Summary")
st.metric("Strategy Return", f"{(cumulative.iloc[-1] - 1)*100:.2f}%")
