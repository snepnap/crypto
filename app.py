import streamlit as st
import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Crypto AI Oracle", page_icon="⚡", layout="wide")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Market Settings")

# 1. COIN LIST (Crypto + Gaming Tokens)
coin_options = ["Bitcoin", "Ethereum", "Solana", "XRP", "Dogecoin", "The Sandbox", "Decentraland"]
coin = st.sidebar.selectbox("Select Asset", coin_options)

# 2. ASSET MAPPING
coin_map = {
    "Bitcoin":  {"ticker": "BTC-USD",  "keyword": "bitcoin"},
    "Ethereum": {"ticker": "ETH-USD",  "keyword": "ethereum"},
    "Solana":   {"ticker": "SOL-USD",  "keyword": "solana"},
    "XRP":      {"ticker": "XRP-USD",  "keyword": "xrp"},
    "Dogecoin": {"ticker": "DOGE-USD", "keyword": "dogecoin"},
    "The Sandbox": {"ticker": "SAND-USD", "keyword": "sandbox crypto"},
    "Decentraland": {"ticker": "MANA-USD", "keyword": "decentraland"}
}

ticker_symbol = coin_map[coin]["ticker"]
keyword = coin_map[coin]["keyword"]
rss_url = f"https://www.coindesk.com/arc/outboundfeeds/rss/?keywords={keyword}"

# --- HELPER FUNCTIONS ---
def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- SETUP NLP ENGINE ---
nltk.download('vader_lexicon', quiet=True)
analyzer = SentimentIntensityAnalyzer()
new_words = {
    'sinks': -4.0, 'slide': -4.0, 'plunge': -4.0, 'crash': -4.0,
    'soar': 2.0, 'surge': 3.5, 'jump': 3.0, 'rebound': 2.5,
    'fades': -2.0, 'slips': -2.0, 'loss': -3.0, 'gain': 3.0,
    'bullish': 3.5, 'bearish': -3.5, 'rally': 3.0, 'dump': -4.0,
    'explode': 3.5, 'skyrocket': 4.0, 'rekt': -4.0, 'moon': 3.5
}
analyzer.lexicon.update(new_words)

# --- MAIN PAGE HEADER ---
st.title(f"⚡ {coin} AI Oracle & Financial Advisor")
st.markdown(f"**The Ultimate Dashboard:** Fundamentals + Technicals + Sentiment + **ML Prediction** + Risk Management.")

# --- FETCH DATA ONCE (EFFICIENCY) ---
try:
    with st.spinner(f'Fetching all data for {ticker_symbol}...'):
        # Get 14 days of data for analysis
        ticker_data = yf.Ticker(ticker_symbol)
        price_df = ticker_data.history(period="14d", interval="1h")
        
        # Get Fundamental Info
        info = ticker_data.info

        if price_df.empty:
            st.error("⚠️ Error: No price data found. API might be down.")
            st.stop()
            
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()


# --- PART 0: FUNDAMENTAL SNAPSHOT (Context) ---
st.subheader(f"🏦 {coin} Fundamental Health")

try:
    # 1. Get Key Metrics
    market_cap = info.get('marketCap', 0)
    year_high = info.get('fiftyTwoWeekHigh', 0)
    year_low = info.get('fiftyTwoWeekLow', 0)
    current_price_raw = price_df['Close'].iloc[-1]
    
    # 2. Calculate "Discount" (How far from the top?)
    if year_high > 0:
        drawdown = ((year_high - current_price_raw) / year_high) * 100
    else:
        drawdown = 0
    
    # 3. Determine "Safety Score"
    if market_cap > 100_000_000_000:
        safety = "🛡️ Safe Haven (Large Cap)"
    elif market_cap > 10_000_000_000:
        safety = "🏢 Established (Mid Cap)"
    else:
        safety = "🎰 High Risk (Speculative)"

    # 4. Display Metrics
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("💰 Market Cap", f"${market_cap:,.0f}")
    f2.metric("🛡️ Asset Class", safety)
    f3.metric("📉 Down from High", f"-{drawdown:.1f}%")
    f4.metric("📊 52-Week Range", f"${year_low:,.2f} - ${year_high:,.2f}")

    # 5. The "FOMO Meter" Progress Bar
    if year_high > year_low:
        range_percent = ((current_price_raw - year_low) / (year_high - year_low))
        st.write("**Price Position (Yearly Range):**")
        st.progress(float(range_percent))
        st.caption(f"Left = Yearly Low (Cheap) | Right = Yearly High (Expensive). Current: {range_percent*100:.0f}%")

except Exception as e:
    st.warning("⚠️ Could not load fundamental data (Yahoo API limit). Proceeding with technicals.")


# --- PART 1: TECHNICAL ANALYSIS (The Pro Chart) ---
st.divider()
st.subheader(f"📉 Professional Technical Analysis")

# --- CALCULATIONS ---
# 1. RSI
price_df['RSI'] = calculate_rsi(price_df)

# 2. Moving Average (50 Hours)
price_df['MA50'] = price_df['Close'].rolling(window=50).mean()

# 3. Bollinger Bands
price_df['std_dev'] = price_df['Close'].rolling(window=50).std()
price_df['BB_Upper'] = price_df['MA50'] + (price_df['std_dev'] * 2)
price_df['BB_Lower'] = price_df['MA50'] - (price_df['std_dev'] * 2)

# Get current metrics
current_price = price_df['Close'].iloc[-1]
rsi_value = price_df['RSI'].iloc[-1]
start_price = price_df['Close'].iloc[-24]
change_pct = ((current_price - start_price) / start_price) * 100

# Determine Trend
if change_pct > 0.5: price_trend = "Uptrend 📈"
elif change_pct < -0.5: price_trend = "Downtrend 📉"
else: price_trend = "Sideways ➡️"

# --- DISPLAY METRICS ---
c1, c2, c3 = st.columns(3)
c1.metric("Current Price", f"${current_price:,.4f}", f"{change_pct:.2f}%")

rsi_label = "Neutral"
if rsi_value > 70: rsi_label = "OVERBOUGHT ⚠️"
elif rsi_value < 30: rsi_label = "OVERSOLD 🟢"

c2.metric("RSI (14-Hour)", f"{rsi_value:.1f}", rsi_label)
c3.metric("Trend (24h)", price_trend)

# --- DOWNLOAD BUTTON ---
csv = price_df.to_csv().encode('utf-8')
st.download_button(
    label="📥 Download Market Data (CSV)",
    data=csv,
    file_name=f"{coin}_data.csv",
    mime="text/csv",
)

# --- PLOTLY CHART (CANDLES + VOLUME) ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.03, row_width=[0.2, 0.7])

# 1. Candlestick (Top)
fig.add_trace(go.Candlestick(
    x=price_df.index, open=price_df['Open'], high=price_df['High'],
    low=price_df['Low'], close=price_df['Close'], name='Price'
), row=1, col=1)

# 2. Bollinger Bands (Top)
fig.add_trace(go.Scatter(x=price_df.index, y=price_df['BB_Upper'], mode='lines', name='Upper Band', line=dict(color='rgba(0,255,255,0.3)', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=price_df.index, y=price_df['BB_Lower'], mode='lines', name='Lower Band', line=dict(color='rgba(0,255,255,0.3)', width=1), fill='tonexty', fillcolor='rgba(0,255,255,0.05)'), row=1, col=1)

# 3. Moving Average (Top)
fig.add_trace(go.Scatter(x=price_df.index, y=price_df['MA50'], mode='lines', name='50h Avg', line=dict(color='yellow', width=1)), row=1, col=1)

# 4. Volume (Bottom)
vol_colors = ['#00ff00' if row['Close'] >= row['Open'] else '#ff0000' for i, row in price_df.iterrows()]
fig.add_trace(go.Bar(
    x=price_df.index, y=price_df['Volume'],
    name='Volume', marker_color=vol_colors
), row=2, col=1)

fig.update_layout(title=f"{coin} Pro Analysis (Price + Volume)", height=700, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=False)
st.plotly_chart(fig, use_container_width=True)


# --- PART 2: SENTIMENT ANALYSIS ---
st.divider()
st.subheader("📰 AI News Sentiment")

market_mood = "Neutral"

if st.button("📡 Scan News & Run AI Models"):
    with st.spinner('Analyzing Global News...'):
        feed = feedparser.parse(rss_url)
        pos, neg = 0, 0
        results = []

        for entry in feed.entries[:20]:
            score = analyzer.polarity_scores(entry.title)['compound']
            if score > 0.05: pos += 1
            elif score < -0.05: neg += 1
            results.append({"Headline": entry.title, "Score": score})

    # Mood Logic
    if pos > neg: market_mood = "Bullish 🐂"
    elif neg > pos: market_mood = "Bearish 🐻"
    else: market_mood = "Neutral ⚖️"
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Positive News", pos)
    k2.metric("Negative News", neg)
    k3.metric("Overall Mood", market_mood)

    # --- PART 3: AI SIGNAL ENGINE ---
    st.divider()
    st.subheader("🤖 Final Trading Signal")
    
    signal = "HOLD"
    color = "gray"
    reason = "Wait for a clearer setup."

    if rsi_value < 30 and "Bullish" in market_mood:
        signal = "STRONG BUY 🎯"
        color = "#00ff00"
        reason = "Asset is Oversold + News is Positive."
    elif rsi_value < 70 and "Uptrend" in price_trend and "Bullish" in market_mood:
        signal = "BUY 🚀"
        color = "#00cc00"
        reason = "Trend is Up + Sentiment is Good."
    elif rsi_value > 70:
        signal = "SELL 💰"
        color = "#ff0000"
        reason = "RSI Overbought (>70). Correction likely."
    elif "Downtrend" in price_trend and "Bearish" in market_mood:
        signal = "STRONG SELL ⚠️"
        color = "#cc0000"
        reason = "Trend Down + Bad News."

    st.markdown(f"""
        <div style='background-color: #1E1E1E; padding: 20px; border-left: 10px solid {color}; border-radius: 5px;'>
            <h1 style='color: {color}; margin:0;'>{signal}</h1>
            <p style='color: white; margin-top: 10px;'>🧠 <b>Reason:</b> {reason}</p>
        </div>
    """, unsafe_allow_html=True)

    # --- PART 4: MACHINE LEARNING FORECAST ---
    st.divider()
    st.subheader("🔮 Machine Learning Price Forecast (Next 24h)")
    
    with st.spinner('Training Linear Regression Model...'):
        df_ml = price_df.reset_index().dropna()
        date_col = df_ml.columns[0] # Fix for YFinance column naming
        df_ml['Time'] = np.arange(len(df_ml))
        
        X = df_ml[['Time']]
        y = df_ml['Close']

        model = LinearRegression()
        model.fit(X, y)

        future_time = np.arange(len(df_ml), len(df_ml) + 24).reshape(-1, 1)
        future_price = model.predict(future_time)

        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(x=df_ml[date_col], y=df_ml['Close'], mode='lines', name='History', line=dict(color='cyan')))
        
        last_date = df_ml[date_col].iloc[-1]
        future_dates = [last_date + timedelta(hours=x) for x in range(1, 25)]
        
        forecast_fig.add_trace(go.Scatter(x=future_dates, y=future_price, mode='lines', name='AI Prediction', line=dict(color='yellow', dash='dot')))
        forecast_fig.update_layout(title="AI Projected Trajectory (24 Hours)", template="plotly_dark", height=400)
        st.plotly_chart(forecast_fig)

        predicted_price = future_price[-1]
        st.info(f"🧠 The AI predicts the price will move to **${predicted_price:,.2f}** in 24 hours based on the current trend.")

    # --- PART 5: STRATEGY BACKTESTER ---
    st.divider()
    st.subheader("💰 Strategy Simulator (Backtest)")
    st.markdown("If you had started with **$1,000** 14 days ago and followed this AI, how much would you have now?")

    initial_investment = 1000
    balance = initial_investment
    holdings = 0
    in_market = False
    trade_log = []

    for i in range(2, len(price_df)):
        price = price_df['Close'].iloc[i]
        rsi = price_df['RSI'].iloc[i]
        time = price_df.index[i]

        if rsi < 30 and not in_market:
            holdings = balance / price
            balance = 0
            in_market = True
            trade_log.append(f"🟢 BOUGHT at ${price:,.2f} on {time.strftime('%Y-%m-%d %H:%M')}")
        
        elif rsi > 70 and in_market:
            balance = holdings * price
            holdings = 0
            in_market = False
            trade_log.append(f"🔴 SOLD at ${price:,.2f} on {time.strftime('%Y-%m-%d %H:%M')}")

    if in_market:
        final_value = holdings * price_df['Close'].iloc[-1]
    else:
        final_value = balance

    profit_loss = final_value - initial_investment
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Starting Balance", f"${initial_investment:,.2f}")
    m2.metric("Final Balance", f"${final_value:,.2f}", delta=f"{profit_loss:,.2f}")

    if profit_loss > 0:
        st.success(f"✅ Result: This strategy would have made a **${profit_loss:,.2f}** profit.")
    else:
        st.error(f"⚠️ Result: This strategy would have lost **${abs(profit_loss):,.2f}**.")
    
    with st.expander("📜 View Transaction Log"):
        for trade in trade_log:
            st.write(trade)

    # --- PART 6: RISK MANAGEMENT ---
    st.divider()
    st.subheader("🛡️ Smart Trade Plan (Risk Management)")
    
    # ATR Calculation
    high_low = price_df['High'] - price_df['Low']
    high_close = np.abs(price_df['High'] - price_df['Close'].shift())
    low_close = np.abs(price_df['Low'] - price_df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(14).mean().iloc[-1]

    stop_loss = current_price - (atr * 2)
    take_profit = current_price + (atr * 4)

    r1, r2, r3 = st.columns(3)
    r1.metric("📉 STOP LOSS", f"${stop_loss:,.4f}", delta=f"-${(atr*2):,.4f}", delta_color="inverse")
    r2.metric("🎯 TAKE PROFIT", f"${take_profit:,.4f}", delta=f"+${(atr*4):,.4f}", delta_color="normal")
    r3.metric("⚖️ Risk/Reward", "1 : 2.0")

    st.info(f"**Advice:** Set your Stop Loss at **${stop_loss:,.4f}** to limit downside risk.")

else:
    st.info("👋 Click the 'Scan News' button above to activate the AI & ML Engines.")