import pandas as pd
import numpy as np
import lightgbm as lgbm
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
import math

# Function to fetch and prepare data
def fetch_and_prepare_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.drop(['Adj Close'], axis=1, inplace=True)
    stock_data.reset_index(inplace=True)
    stock_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    stock_data['daily_return'] = stock_data['Close'].pct_change()
    stock_data['daily_volatility'] = stock_data['Close'].pct_change().rolling(window=5).std()
    stock_data['lagged_close'] = stock_data['Close'].shift(1)  # New feature: previous day's close
    stock_data.fillna(stock_data.mean(), inplace=True)  # Handle NaNs
    return stock_data

# Fetch and prepare the data
prepared_data = fetch_and_prepare_data('AAPL', '2021-01-01', '2023-12-31')

# Convert 'Date' to datetime for better plotting
prepared_data['Date'] = pd.to_datetime(prepared_data['Date'])

# Prepare the features and labels for regression
features = ['Open', 'High', 'Low', 'Volume', 'daily_return', 'daily_volatility', 'lagged_close']
X = prepared_data[features]
y = prepared_data['Close']  # Predicting closing price

# Split the dataset
X_train, X_val, y_train, y_val, dates_train, dates_val = train_test_split(X, y, prepared_data['Date'], test_size=0.2, random_state=42)

# LightGBM parameters for regression
params = {
    "num_leaves": 31,
    "objective": "regression",
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "metric": "rmse",
    "verbose": 1
}

# Create datasets for LightGBM
d_train = lgbm.Dataset(X_train, label=y_train)
d_eval = lgbm.Dataset(X_val, label=y_val, reference=d_train)

# Train the regression model
try:
    reg_model = lgbm.train(
        params,
        d_train,
        num_boost_round=1000,
        valid_sets=[d_eval],
        callbacks=[lgbm.callback.early_stopping(stopping_rounds=50)]
    )
    print("Regression model trained successfully!")
except Exception as e:
    print("Error during model training:", e)

# Predicting on the validation set
val_predictions = reg_model.predict(X_val)
rmse = math.sqrt(mean_squared_error(y_val, val_predictions))
print("RMSE on validation set:", rmse)

# Combine the validation dates and predictions into a DataFrame
validation_df = pd.DataFrame({
    'Date': dates_val,
    'Actual_Close': y_val,
    'Predicted_Close': val_predictions
}).sort_values('Date')

# Plot the predictions against the actual values
plt.figure(figsize=(14, 7))
plt.plot(validation_df['Date'], validation_df['Actual_Close'], label='Actual Close Price', color='blue', alpha=0.5)
plt.plot(validation_df['Date'], validation_df['Predicted_Close'], label='Predicted Close Price', color='red', linestyle='--', alpha=0.5)
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
