import streamlit as st
import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd # You might need: pip install pandas

# --- PAGE CONFIG ---
st.set_page_config(page_title="Crypto Sentiment AI", page_icon="🧠")

# --- SIDEBAR (Settings) ---
st.sidebar.header("⚙️ Settings")
coin = st.sidebar.selectbox("Select Asset", ["Bitcoin", "Ethereum"])
keywords = "bitcoin" if coin == "Bitcoin" else "ethereum"
rss_url = f"https://www.coindesk.com/arc/outboundfeeds/rss/?keywords={keywords}"

# --- SETUP ANALYZER ---
nltk.download('vader_lexicon', quiet=True)
analyzer = SentimentIntensityAnalyzer()

# Custom Financial Lexicon
new_words = {
    'sinks': -4.0, 'slide': -4.0, 'plunge': -4.0, 'crash': -4.0,
    'soar': 2.0, 'surge': 3.5, 'jump': 3.0, 'rebound': 2.5,
    'fades': -2.0, 'slips': -2.0
}
analyzer.lexicon.update(new_words)

# --- APP TITLE ---
st.title(f"🧠 {coin} AI Sentiment Tracker")
st.markdown("Real-time analysis of crypto news using **VADER** and **NLP**.")

# --- MAIN LOGIC ---
if st.button("📡 Scan Market Now"):
    with st.spinner('Fetching live data from CoinDesk...'):
        feed = feedparser.parse(rss_url)
        
        positive = 0
        negative = 0
        neutral = 0
        
        results = []

        # Analyze
        for entry in feed.entries[:15]:
            headline = entry.title
            score = analyzer.polarity_scores(headline)['compound']
            
            if score > 0.05:
                verdict = "Positive 🟢"
                positive += 1
            elif score < -0.05:
                verdict = "Negative 🔴"
                negative += 1
            else:
                verdict = "Neutral ⚪"
                neutral += 1
            
            results.append({"Headline": headline, "Score": score, "Verdict": verdict})

    # --- DISPLAY METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Bullish News", positive, delta_color="normal")
    col2.metric("Bearish News", negative, delta_color="inverse")
    
    # Calculate Mood
    if positive > negative:
        market_mood = "🐂 BULLISH"
        color = "green"
    elif negative > positive:
        market_mood = "🐻 BEARISH"
        color = "red"
    else:
        market_mood = "⚖️ NEUTRAL"
        color = "gray"

    col3.metric("Overall Mood", market_mood)

    # --- VISUAL ALERT ---
    if market_mood == "🐻 BEARISH":
        st.error("⚠️ ALERT: Selling pressure detected in the news!")
    elif market_mood == "🐂 BULLISH":
        st.success("✅ STATUS: Market looks optimistic.")

    # --- DATA TABLE ---
    st.subheader("📰 Latest Headlines Analyzed")
    df = pd.DataFrame(results)
    st.dataframe(df)

else:
    st.info("Click the button above to start the scan.")