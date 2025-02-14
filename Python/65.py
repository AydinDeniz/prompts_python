
import numpy as np
import pandas as pd
import yfinance as yf
import openai
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load OpenAI API key
API_KEY = "your_openai_api_key"

# Fetch stock data
def fetch_stock_data(tickers, start="2020-01-01", end="2024-01-01"):
    print("Fetching stock data...")
    stock_data = yf.download(tickers, start=start, end=end)["Adj Close"]
    return stock_data

# Compute expected returns and covariance matrix
def compute_risk_return(stock_data):
    print("Calculating risk and returns...")
    returns = stock_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

# Optimize portfolio using Markowitz Efficient Frontier
def optimize_portfolio(mean_returns, cov_matrix, risk_aversion=0.5):
    print("Optimizing portfolio using Markowitz Efficient Frontier...")
    num_assets = len(mean_returns)

    def objective(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        return -portfolio_return + risk_aversion * portfolio_risk

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = [(0, 1) for _ in range(num_assets)]
    initial_guess = np.array([1.0 / num_assets] * num_assets)

    result = minimize(objective, initial_guess, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x

# Predict stock prices using LSTM
def train_lstm_model(stock_data):
    print("Training LSTM model for stock price prediction...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_data.values.reshape(-1, 1))

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i - 60:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    model = keras.Sequential([
        layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        layers.LSTM(50, return_sequences=False),
        layers.Dense(25),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, batch_size=16, epochs=20)

    return model, scaler

# Generate investment advice using AI
def generate_investment_advice(portfolio, predicted_prices):
    print("Generating AI-powered investment advice...")
    
    prompt = (
        "Given the following optimized portfolio weights and predicted stock prices, generate investment recommendations.

"
        f"Portfolio Allocation: {portfolio}
"
        f"Predicted Prices: {predicted_prices}
"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a financial expert providing investment advice."},
                  {"role": "user", "content": prompt}],
        max_tokens=500
    )
    
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    tickers = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN"]
    stock_data = fetch_stock_data(tickers)

    mean_returns, cov_matrix = compute_risk_return(stock_data)
    optimized_portfolio = optimize_portfolio(mean_returns, cov_matrix)

    print("Training LSTM model for future stock price predictions...")
    lstm_model, scaler = train_lstm_model(stock_data["AAPL"])

    future_data = stock_data["AAPL"].values[-60:].reshape(-1, 1)
    scaled_future_data = scaler.transform(future_data)
    X_future = np.array([scaled_future_data])
    predicted_price = scaler.inverse_transform(lstm_model.predict(X_future))[0][0]

    investment_advice = generate_investment_advice(optimized_portfolio, predicted_price)
    
    with open("investment_report.txt", "w", encoding="utf-8") as f:
        f.write(investment_advice)

    print("Investment advice saved to 'investment_report.txt'.")
