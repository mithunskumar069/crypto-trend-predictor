import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from textblob import TextBlob
from newsapi import NewsApiClient
from prophet import Prophet
import requests
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Crypto Trend Predictor", layout="wide")

NEWSAPI_KEY = st.secrets["NEWSAPI_KEY"]  # <<< YOUR KEY HERE
LSTM_EPOCHS = 10
LOOKBACK = 60

# -------------------------------
# FETCH TOP 20 COINS AUTOMATICALLY
# -------------------------------
@st.cache_data(ttl=3600)
def get_top20_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 20,
        "page": 1,
    }
    res = requests.get(url, params=params).json()
    coin_map = {}
    for c in res:
        symbol = c["symbol"].upper()
        coin_map[f"{c['name']} ({symbol})"] = f"{symbol}-USD"
    return coin_map

COINS = get_top20_coins()

# -------------------------------
# DOWNLOAD & CLEAN PRICE DATA
# -------------------------------
@st.cache_data(ttl=3600)
def download_clean_price(ticker, period):
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df is None or len(df) == 0:
        raise ValueError("Data download failed or empty.")

    df = df.reset_index()

    # Fix Multi-Index columns if needed
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("Invalid Yahoo Finance structure.")

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).reset_index(drop=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)

    return df

# -------------------------------
# LSTM UTILITIES
# -------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_sequences(data, lookback=LOOKBACK):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# -------------------------------
# NEWS + SENTIMENT
# -------------------------------
def fetch_news_sentiment(query):
    newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
    articles = newsapi.get_everything(
        q=query, language='en', sort_by='publishedAt', page_size=40
    )

    rows = []
    for a in articles.get("articles", []):
        text = (a.get("title", "") or "") + " " + (a.get("description", "") or "")
        s = TextBlob(text).sentiment.polarity
        rows.append([a.get("publishedAt"), text, s])

    df = pd.DataFrame(rows, columns=["Date", "Text", "Sentiment"])
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna().reset_index(drop=True)
    return df

def compute_daily_sent(df_news):
    if df_news.empty:
        return 0.0
    s = df_news.set_index("Date").resample("D")["Sentiment"].mean().fillna(0)
    return float(s.iloc[-1])

# -------------------------------
# PROPHET FORECAST
# -------------------------------
def prophet_forecast(df):
    dfp = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    dfp["ds"] = pd.to_datetime(dfp["ds"], errors="coerce")
    dfp["y"] = pd.to_numeric(dfp["y"], errors="coerce")
    dfp = dfp.dropna().reset_index(drop=True)

    if len(dfp) < 30:
        raise ValueError("Insufficient data for Prophet.")

    model = Prophet()
    model.fit(dfp)
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)

    return float(forecast["yhat"].iloc[-1])

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("🔮 Crypto Trend Predictor (Top 20 Coins – Daily Data)")

with st.sidebar:
    st.header("⚙ Settings")

    coin_name = st.selectbox("Select Cryptocurrency", list(COINS.keys()))
    ticker = COINS[coin_name]

    period = st.selectbox(
        "Select Data Period",
        ["1y", "2y", "3y", "5y", "max"],
        index=3
    )

    run_btn = st.button("Run Prediction")

st.write(f"### Selected: *{coin_name}* ・ Ticker: {ticker} ・ Period: {period}")

if not run_btn:
    st.info("Click *Run Prediction* to start analysis.")
    st.stop()

# -------------------------------
# LOAD PRICE DATA
# -------------------------------
with st.spinner("Downloading and cleaning price data..."):
    df = download_clean_price(ticker, period)
st.success(f"Loaded {len(df)} rows of clean price data.")

# Price Chart
st.subheader("📈 Price History")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df["Date"], df["Close"])
ax.set_xlabel("Date"), ax.set_ylabel("Close Price")
st.pyplot(fig)

# -------------------------------
# LSTM TRAINING
# -------------------------------
st.subheader("🤖 LSTM Prediction")
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["Close"]])

X, y_vals = create_sequences(scaled)
if len(X) < 100:
    st.error("Not enough data to train LSTM.")
    st.stop()

X = X.reshape(X.shape[0], X.shape[1], 1)

model = build_lstm_model((LOOKBACK,1))
with st.spinner("Training LSTM (10 epochs)..."):
    model.fit(X, y_vals, epochs=LSTM_EPOCHS, batch_size=32, verbose=0)

last_seq = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
lstm_scaled = model.predict(last_seq)
lstm_pred = float(scaler.inverse_transform(lstm_scaled)[0][0])

st.metric("LSTM Next-Day Prediction", f"${lstm_pred:,.4f}")

# -------------------------------
# PROPHET
# -------------------------------
st.subheader("📉 Prophet Forecast")
try:
    prophet_pred = prophet_forecast(df)
    st.metric("Prophet Next-Day Prediction", f"${prophet_pred:,.4f}")
except Exception as e:
    st.error(f"Prophet Error: {e}")
    prophet_pred = lstm_pred

# -------------------------------
# NEWS SENTIMENT
# -------------------------------
st.subheader("📰 News Sentiment Analysis")
df_news = fetch_news_sentiment(coin_name.split()[0])
latest_sent = compute_daily_sent(df_news)
st.metric("Latest Sentiment Score", f"{latest_sent:.4f}")
st.dataframe(df_news.head(10))

# -------------------------------
# CLUSTERING
# -------------------------------
st.subheader("🔍 Price Clustering (K-Means)")
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df[["Close"]])

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.scatter(df["Date"], df["Close"], c=df["Cluster"])
ax2.set_xlabel("Date"), ax2.set_ylabel("Close")
st.pyplot(fig2)

last_cluster = int(df["Cluster"].iloc[-1])
st.metric("Last Price Cluster", last_cluster)

# -------------------------------
# FINAL ENSEMBLE
# -------------------------------
st.subheader("🎯 Final Ensemble Prediction")

final_pred = (
    0.5 * lstm_pred +
    0.3 * prophet_pred +
    0.1 * (lstm_pred * latest_sent) +
    0.1 * (lstm_pred * (last_cluster / 2))
)

trend = "📈 UP" if final_pred > df["Close"].iloc[-1] else "📉 DOWN"
confidence = round(
    abs(final_pred - df["Close"].iloc[-1]) / df["Close"].iloc[-1] * 100, 2
)

st.metric("Final Predicted Price", f"${final_pred:,.4f}")
st.metric("Trend", trend)
st.metric("Confidence", f"{confidence}%")


st.success("✔ Analysis complete – ready for interview demo!")
