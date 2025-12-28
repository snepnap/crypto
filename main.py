import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import winsound  # 🔊 This allows us to make system beeps (Windows only)
from datetime import datetime

# --- SETUP ---
nltk.download('vader_lexicon', quiet=True)
analyzer = SentimentIntensityAnalyzer()

# Custom Dictionary
new_words = {
    'sinks': -4.0, 'slide': -4.0, 'plunge': -4.0, 'crash': -4.0,
    'soar': 2.0, 'surge': 3.5, 'jump': 3.0, 'rebound': 2.5,
    'fades': -2.0, 'slips': -2.0
}
analyzer.lexicon.update(new_words)

rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/?keywords=bitcoin"

print("🤖 CRYPTO SENTIMENT BOT ACTIVATED.")
print("Press 'Ctrl + C' in this terminal to stop the bot.\n")

# --- THE INFINITE LOOP ---
while True:
    try:
        # 1. Get the time
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] 📡 Scanning markets...")

        # 2. Fetch Data
        feed = feedparser.parse(rss_url)
        
        positive = 0
        negative = 0

        # Analyze top 10 articles
        for entry in feed.entries[:10]:
            score = analyzer.polarity_scores(entry.title)['compound']
            if score > 0.05:
                positive += 1
            elif score < -0.05:
                negative += 1

        # 3. Decision Logic
        print(f"   ↳ Positives: {positive} | Negatives: {negative}")

        if negative > positive:
            print("   ⚠️ ALERT: Market is BEARISH! (Selling Pressure)")
            # 🔊 Sound the Alarm: (Frequency=1000Hz, Duration=1000ms)
            winsound.Beep(1000, 1000) 
            winsound.Beep(1000, 1000) 
            
        elif positive > negative:
            print("   ✅ STATUS: Market is BULLISH (Buying Pressure)")
            
        else:
            print("   ⚖️ STATUS: Market is Neutral")

        print("-" * 40)
        
        # 4. Wait for 60 seconds before next scan
        time.sleep(60)

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(60) # Wait and try again even if it crashes
    except KeyboardInterrupt:
        print("\n🛑 Bot Stopped by User.")
        break