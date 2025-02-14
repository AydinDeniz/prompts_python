
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load stock market data
def load_stock_data(ticker, period="1y"):
    print(f"Fetching stock data for {ticker}...")
    stock_data = yf.download(ticker, period=period)
    stock_data["Return"] = stock_data["Close"].pct_change()
    stock_data = stock_data.dropna()
    return stock_data

# Prepare quantum feature map
def quantum_feature_map(num_qubits):
    qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
    circuit = cirq.Circuit()

    for i in range(num_qubits):
        circuit.append(cirq.H(qubits[i]))

    sympy_vars = [sympy.Symbol(f"x{i}") for i in range(num_qubits)]
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.rx(sympy_vars[i])(qubit))

    return circuit, sympy_vars

# Build quantum neural network
def build_quantum_model(num_qubits):
    circuit, sympy_vars = quantum_feature_map(num_qubits)
    qlayer = tfq.layers.PQC(circuit, tf.keras.layers.Dense(1))

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        qlayer
    ])
    
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# Train quantum AI model
def train_quantum_model(ticker):
    data = load_stock_data(ticker)
    X = data["Return"].values.reshape(-1, 1)
    y = data["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    num_qubits = 4
    model = build_quantum_model(num_qubits)
    
    print("Training quantum AI model...")
    model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

    predictions = model.predict(X_test)
    plt.plot(y_test, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.title("Quantum AI Stock Prediction")
    plt.savefig("quantum_stock_prediction.png")
    plt.show()

if __name__ == "__main__":
    stock_ticker = "AAPL"
    train_quantum_model(stock_ticker)
