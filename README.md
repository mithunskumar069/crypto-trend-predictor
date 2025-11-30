# 🚀 Crypto Trend Predictor  
A machine learning project that predicts *next-day cryptocurrency prices* using:  
- 🧠 *LSTM deep learning model*  
- 📈 *Meta Prophet forecasting model*  
- 📰 *Live news sentiment analysis (NewsAPI + TextBlob)*  
- 🔍 *KMeans clustering-based market behavior adjustment*  
- 🌐 *Interactive Streamlit web application*

---

## 📌 Features
### 🔮 1. LSTM Next-Day Prediction  
Uses the last 60 days of closing prices to estimate the next day's price.
![LSTM Screenshot](screenshots/Screenshot (3).png)

### 📊 2. Prophet Forecast  
A statistical model that captures trend + seasonality to predict next-day value.
![Prophet Screenshot](screenshots/Screenshot (3).png)

### 📰 3. Sentiment Analysis  
Fetches the latest cryptocurrency-related news using *NewsAPI*,  
analyzes sentiment using *TextBlob*, and adjusts prediction accordingly.
![Sentiment Analysis Screenshot](screenshots/Screenshot (4).png)

### 🎯 4. Final Ensemble Prediction  
Combines all models into a single final prediction with:  
- Final Price  
- Trend (UP 📈 / DOWN 📉)  
- Confidence Score
![Final Prediction Screenshot](screenshots/Screenshot (6).png)  

---

## 🛠️ Tech Stack
- Python  
- TensorFlow / Keras  
- Prophet  
- Scikit-Learn  
- Pandas / NumPy  
- Streamlit  
- NewsAPI  
- TextBlob  

---

## 📂 Project Structure
