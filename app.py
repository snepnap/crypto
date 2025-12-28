import streamlit as st
import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Crypto AI Predictor", page_icon="🔮", layout="wide")

# --- SIDEBAR (Settings) ---
st.sidebar.header("⚙️ Settings")

# 1. EXPANDED COIN LIST
coin_options = ["Bitcoin", "Ethereum", "Solana", "XRP", "Dogecoin"]
coin = st.sidebar.selectbox("Select Asset", coin_options)

# Map selection to ticker symbols and RSS search terms
coin_map = {
    "Bitcoin":  {"ticker": "BTC-USD",  "keyword": "bitcoin"},
    "Ethereum": {"ticker": "ETH-USD",  "keyword": "ethereum"},
    "Solana":   {"ticker": "SOL-USD",  "keyword": "solana"},
    "XRP":      {"ticker": "XRP-USD",  "keyword": "xrp"},
    "Dogecoin": {"ticker": "DOGE-USD", "keyword": "dogecoin"}
}

ticker_symbol = coin_map[coin]["ticker"]
keyword = coin_map[coin]["keyword"]
rss_url = f"https://www.coindesk.com/arc/outboundfeeds/rss/?keywords={keyword}"

# --- SETUP NLP ENGINE ---
nltk.download('vader_lexicon', quiet=True)
analyzer = SentimentIntensityAnalyzer()

# Custom Financial Lexicon
new_words = {
    'sinks': -4.0, 'slide': -4.0, 'plunge': -4.0, 'crash': -4.0,
    'soar': 2.0, 'surge': 3.5, 'jump': 3.0, 'rebound': 2.5,
    'fades': -2.0, 'slips': -2.0, 'loss': -3.0, 'gain': 3.0,
    'bullish': 3.5, 'bearish': -3.5, 'rally': 3.0, 'dump': -4.0
}
analyzer.lexicon.update(new_words)

# --- APP HEADER ---
st.title(f"🔮 {coin} AI Prediction Dashboard")
st.markdown(f"Generating signals for **{coin}** based on live Sentiment & Price Momentum.")

# --- PART 1: PRICE DATA (The Trend) ---
st.subheader(f"📉 Market Trend (7 Days)")

price_trend = "Neutral" # Default
change_pct = 0.0

try:
    with st.spinner(f'Fetching live price data for {ticker_symbol}...'):
        # Get 7 days of data
        ticker_data = yf.Ticker(ticker_symbol)
        price_df = ticker_data.history(period="7d", interval="1h")

        if not price_df.empty:
            start_price = price_df['Close'].iloc[0]
            end_price = price_df['Close'].iloc[-1]
            change_pct = ((end_price - start_price) / start_price) * 100
            
            # Determine Price Trend
            if change_pct > 1.0:
                price_trend = "Uptrend 📈"
                chart_color = "#00ff00"
            elif change_pct < -1.0:
                price_trend = "Downtrend 📉"
                chart_color = "#ff0000"
            else:
                price_trend = "Sideways ➡️"
                chart_color = "#888888"

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Price", f"${end_price:,.4f}")
            m2.metric("7-Day Change", f"{change_pct:.2f}%", delta_color="normal")
            m3.metric("Price Momentum", price_trend)

            st.line_chart(price_df['Close'], color=chart_color)
        else:
            st.error("Error: No price data found.")

except Exception as e:
    st.error(f"Error loading charts: {e}")

# --- PART 2: SENTIMENT DATA (The Mood) ---
st.divider()
st.subheader("📰 Live Sentiment Analysis")

market_mood = "Neutral" # Default

if st.button("📡 Scan News & Predict"):
    with st.spinner('Analyzing Global News...'):
        feed = feedparser.parse(rss_url)
        
        positive = 0
        negative = 0
        results = []

        # Analyze top 20 articles for better accuracy
        for entry in feed.entries[:20]:
            headline = entry.title
            score = analyzer.polarity_scores(headline)['compound']
            
            if score > 0.05:
                verdict = "Positive"
                positive += 1
            elif score < -0.05:
                verdict = "Negative"
                negative += 1
            else:
                verdict = "Neutral"
            
            results.append({"Headline": headline, "Verdict": verdict})

    # Display Sentiment Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Bullish Articles", positive)
    c2.metric("Bearish Articles", negative)

    # Determine Sentiment Mood
    if positive > negative:
        market_mood = "Bullish 🐂"
    elif negative > positive:
        market_mood = "Bearish 🐻"
    else:
        market_mood = "Uncertain ⚖️"
    
    c3.metric("Overall Media Mood", market_mood)

    # --- PART 3: THE AI PREDICTION ENGINE 🤖 ---
    st.divider()
    st.subheader("🤖 AI Trading Signal")

    # LOGIC: Combining Price Trend + Sentiment Mood
    signal = "HOLD / WAIT"
    reason = "Conflicting signals. Market direction is unclear."
    signal_color = "gray"

    # 1. STRONG BUY (Both Agree)
    if "Uptrend" in price_trend and "Bullish" in market_mood:
        signal = "STRONG BUY 🚀"
        reason = "Price is rising AND news sentiment is positive. Momentum is strong."
        signal_color = "green"
    
    # 2. STRONG SELL (Both Agree)
    elif "Downtrend" in price_trend and "Bearish" in market_mood:
        signal = "STRONG SELL 🩸"
        reason = "Price is falling AND news is negative. Panic selling likely."
        signal_color = "red"

    # 3. CONTRARIAN / DIVERGENCE (They Disagree)
    elif "Downtrend" in price_trend and "Bullish" in market_mood:
        signal = "WATCH LIST (Dip Buy?)"
        reason = "Price is down, but news is Positive. Possible oversold bounce coming."
        signal_color = "orange"
        
    elif "Uptrend" in price_trend and "Bearish" in market_mood:
        signal = "CAUTION (Bull Trap?)"
        reason = "Price is up, but news is Negative. Market might reverse soon."
        signal_color = "orange"

    # Display the Signal Card
    st.markdown(f"""
        <div style="padding: 20px; background-color: #262730; border-radius: 10px; border-left: 10px solid {signal_color};">
            <h2 style="color: {signal_color}; margin:0;">SIGNAL: {signal}</h2>
            <p style="font-size: 18px; margin-top: 10px;">🧠 <b>AI Reasoning:</b> {reason}</p>
        </div>
    """, unsafe_allow_html=True)

    # Show raw data at the bottom
    with st.expander("View Analyzed Headlines"):
        st.dataframe(pd.DataFrame(results))

else:
    st.info("Click the button above to generate a prediction.")