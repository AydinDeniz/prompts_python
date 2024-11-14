import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error

# Load the dataset
# Replace 'your_stock_data.csv' with the path to your dataset file
data = pd.read_csv('your_stock_data.csv', usecols=['Date', 'Close'])
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Preprocess the data for time series analysis
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define sequence length
sequence_length = 60  # 60 days of historical data to predict the next day

# Create sequences for training and testing
X, y = create_sequences(scaled_data, sequence_length)

# Split into training and testing sets
split = int(0.8 * len(X))  # 80% for training, 20% for testing
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=10)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Reverse scaling of predictions
train_predictions = scaler.inverse_transform(train_predictions)
y_train_actual = scaler.inverse_transform(y_train)
test_predictions = scaler.inverse_transform(test_predictions)
y_test_actual = scaler.inverse_transform(y_test)

# Calculate Mean Absolute Error (MAE)
train_mae = mean_absolute_error(y_train_actual, train_predictions)
test_mae = mean_absolute_error(y_test_actual, test_predictions)
print(f'Training MAE: {train_mae}')
print(f'Testing MAE: {test_mae}')

# Plot the predictions vs. actual values
plt.figure(figsize=(14, 5))
plt.plot(data.index[sequence_length:split + sequence_length], y_train_actual, label="Actual Train Prices")
plt.plot(data.index[sequence_length:split + sequence_length], train_predictions, label="Train Predictions")
plt.plot(data.index[split + sequence_length:], y_test_actual, label="Actual Test Prices")
plt.plot(data.index[split + sequence_length:], test_predictions, label="Test Predictions")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("LSTM Model - Stock Price Prediction")
plt.legend()
plt.show()
