import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import math  # Ensure you have this import for sqrt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def calculate_rmse(predictions, actual):
    """Calculate the Root Mean Square Error between predictions and actual values."""
    predictions = np.array(predictions)
    actual = np.array(actual)
    return np.sqrt(np.mean((predictions - actual) ** 2))

# Fetch data
nvda_data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')
nvda_data.drop(['Adj Close'], axis=1, inplace=True)
nvda_data.reset_index(inplace=True)
nvda_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
specific_df = nvda_data[(nvda_data['Date'] >= '2020-01-01') & (nvda_data['Date'] <= '2020-12-31')]
specific_df['Date'] = pd.to_datetime(specific_df['Date'])

# Normalize close prices
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(specific_df['Close'].values.reshape(-1,1))
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Prepare sequences
total_data_points = len(scaled_data)
n_past = max(10, int(total_data_points * 0.15))  # Ensure at least 10, but no more than 15% of the total data
train_size = int(total_data_points * 0.75)  # Adjusting to 75% training data
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

print("Using n_past:", n_past)
print("Total data points:", total_data_points)
print("Training data points:", len(train_data))
print("Testing data points:", len(test_data))

X_train, y_train = [], []
for i in range(n_past, len(train_data)):
    X_train.append(train_data[i - n_past:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = [], []
for i in range(n_past, len(test_data)):
    X_test.append(test_data[i - n_past:i, 0])
    y_test.append(test_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
print("Total data points:", len(scaled_data))
print("Training data points:", len(train_data))
print("Testing data points:", len(test_data))
print("Generated training sequences:", X_train.shape)
print("Generated testing sequences:", X_test.shape)

# Check and reshape
if X_train.size > 0:
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
else:
    print("X_train is empty.")

if X_test.size > 0:
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
else:
    print("X_test is empty. Adjust 'n_past' or the data range.")

# Assuming the previous parts of the script are the same and 
# X_train, y_train, X_test, y_test are already defined and properly scaled.

# Build an improved LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping, reduce_lr],
    shuffle=False
)

# Predicting and inverse transforming the results
test_predict = model.predict(X_test)
test_predict = scaler.inverse_transform(test_predict)

# Assuming y_test has been inverse transformed to true_test
true_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculating RMSE
test_rmse = calculate_rmse(true_test, test_predict)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(true_test, label='True Values')
plt.plot(test_predict, label='Predicted Values', alpha=0.7)
plt.title('Test Data: Actual vs. Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Output RMSE
print(f'Test RMSE: {test_rmse}')
# Calculate MAPE
mape = np.mean(np.abs((true_test - test_predict) / true_test)) * 100

# Output MAPE
print(f'Test MAPE: {mape:.2f}%')


