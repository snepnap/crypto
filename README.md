# 🔮 Crypto AI Oracle & Financial Advisor

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B) ![Finance](https://img.shields.io/badge/Finance-Algo_Trading-green)

### A Full-Stack Financial Dashboard that predicts market moves using AI, NLP, and Machine Learning.

**Crypto AI Oracle** is a sophisticated trading terminal built for the modern investor. Unlike standard charts that only show *past* prices, this engine analyzes **News Sentiment (NLP)**, **Technical Indicators (RSI, Bollinger Bands)**, and uses **Linear Regression (ML)** to forecast the next 24 hours of price action.

---

## 🚀 Key Features

### 🧠 1. Artificial Intelligence Core
* **Sentiment Analysis:** Scrapes global news (CoinDesk) in real-time and uses `NLTK VADER` to determine if the market mood is Bullish 🐂 or Bearish 🐻.
* **ML Price Forecasting:** Uses `Scikit-Learn` Linear Regression to draw a "Future Trajectory" line, predicting price direction for the next 24 hours.

### 📊 2. Professional Technical Analysis
* **Interactive Charts:** Built with `Plotly` for zoom/pan capabilities.
* **Indicators:** * **RSI (Relative Strength Index):** Detects Oversold/Overbought conditions.
    * **Bollinger Bands:** Visualizes volatility and breakout zones.
    * **Volume Analysis:** Verifies trend strength.

### 🛡️ 3. Risk Management Engine
* **Smart Stop-Loss:** Automatically calculates safe exit prices using **ATR (Average True Range)** volatility math.
* **Strategy Backtester:** Simulates a "What if?" scenario on the last 14 days of data to prove if the strategy is profitable before you invest.

### 💎 4. Fundamental Health Check
* **FOMO Meter:** Instantly shows if an asset is trading near its yearly high (Expensive) or yearly low (Cheap).
* **Asset Classification:** Categorizes coins as "Safe Havens" or "Speculative" based on Market Cap.

---

## 🛠️ Tech Stack

* **Language:** Python 3.11
* **Frontend:** Streamlit
* **Data Visualization:** Plotly & Plotly Graph Objects
* **Machine Learning:** Scikit-Learn (Linear Regression)
* **Natural Language Processing:** NLTK (VADER Sentiment)
* **Data Source:** YFinance API & RSS Feeds

---

## 💿 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/snepnap/crypto.git](https://github.com/snepnap/crypto.git)
2.**Install dependencies:**
    
    pip install -r requirements.txt
3.Run the application:
    
    python -m streamlit run app.py
