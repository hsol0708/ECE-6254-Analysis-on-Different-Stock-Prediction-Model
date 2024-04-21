import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Other imports...
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Fetching data for NVDA from Yahoo Finance
nvda_data = yf.download('NVDA', start='2020-01-01', end='2020-12-31')

# Dropping unnecessary columns and resetting index
nvda_data.drop(['Adj Close'], axis=1, inplace=True)
nvda_data.reset_index(inplace=True)
nvda_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# Defining a function to filter data based on dates
def specific_data(data, start, end):
    date_filtered_data = data[(data['Date'] >= start) & (data['Date'] <= end)]
    return date_filtered_data

# Using the function
specific_df = specific_data(nvda_data, '2020-01-01', '2020-12-31')
specific_df['Date'] = pd.to_datetime(specific_df['Date'])

# Normalizing close prices
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(specific_df['Close'].values.reshape(-1,1))

# Splitting into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Preparing sequences
n_past = 1000
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

# Ensure data is sufficient
if X_test.size > 0:
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Model definition
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    # Compile and fit model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Prediction and performance metrics
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse scaling for a proper comparison
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # RMSE calculation
    train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
    test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))

    print(f'Train RMSE: {train_rmse}')
    print(f'Test RMSE: {test_rmse}')
else:
    print("Insufficient data for testing after considering 'n_past'. Adjust 'n_past' or increase data size.")
