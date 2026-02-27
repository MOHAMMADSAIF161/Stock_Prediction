import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
st.title("📈 AI Stock Price Predictor")

@st.cache_data
def load_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)

    df = pd.read_csv(StringIO(response.text))

    df["Yahoo"] = df["SYMBOL"] + ".NS"

    nse_list = df["Yahoo"].tolist()

    global_stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "META", "NVDA", "NFLX", "INTC", "IBM"
    ]

    return sorted(nse_list + global_stocks)
    

stock_list = load_stock_list()

ticker = st.selectbox(
    "🔎 Search & Select Stock",
    stock_list,
    index=stock_list.index("RELIANCE.NS") if "RELIANCE.NS" in stock_list else 0
)
# ==============================
# CACHE FUNCTIONS
# =============================

@st.cache_data
def load_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)

    # FIX: Flatten multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


@st.cache_data
def preprocess_data(df):
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()
    return df


@st.cache_resource
def train_model(X_train, y_train, features):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# ==============================
# USER INPUT
# ==============================

start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime(date.today()))

# ==============================
# FEATURE SELECTION INPUT
# ==============================

st.subheader(" Select Target Feautre To Make Prediction")

all_features = ['Open','Close','High','Low','Volume']

selected_features = st.selectbox(
    "Choose Target Features",
    all_features,
    )
input_features = [col for col in all_features if col != selected_features]


# ==============================
# MAIN LOGIC
# ==============================

if st.button("Predict Price"):

    st.cache_data.clear()

    df = load_data(ticker, start_date, end_date)

    if df.empty:
        st.error("❌ No data found for this stock!")
        st.stop()

    st.subheader("📊 Raw Data")
    st.write(df.tail())

    df = preprocess_data(df)

    if df.empty:
        st.error("❌ Not enough data after preprocessing to plot chart")
        st.stop()

    X = df[input_features]
    y = df[selected_features].shift(-1)

    y = y[:-1]
    X = X[:-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = train_model(X_train, y_train, tuple(selected_features))

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    st.subheader("📊 Model Performance")
    st.write("MSE:", mse)
    st.write("R² Score:", r2)

    last_row = X.iloc[-1]
    future_price = model.predict(np.array(last_row).reshape(1, -1))

    st.subheader(f"🔮 Next Day Predicted {selected_features} Price")
    st.success(f"${int(future_price[0])}")

    st.subheader(f"📉 {selected_features} Price Chart")

    chart_data = df[[selected_features]].copy()
    chart_data = chart_data.dropna()

    st.line_chart(chart_data)






