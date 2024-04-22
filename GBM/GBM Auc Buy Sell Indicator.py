import pandas as pd
import numpy as np
import lightgbm as lgbm
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    stock_data.fillna(stock_data.mean(), inplace=True)  # Handle NaNs
    return stock_data

# Fetch and prepare the data
prepared_data = fetch_and_prepare_data('AAPL', '2021-01-01', '2023-12-31')

# Prepare the features and labels
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'daily_return', 'daily_volatility']
X = prepared_data[features]
y = (prepared_data['daily_return'] > 0).astype(int)
dates = prepared_data['Date']  # Save the dates for later

# Split the dataset
X_train, X_val, y_train, y_val, dates_train, dates_val = train_test_split(X, y, dates, test_size=0.2, random_state=42)

# LightGBM parameters
params = {
    "num_leaves": 31,
    "objective": "binary",
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "metric": "auc",
    "verbose": 1  # Set verbosity here
}

# Create datasets for LightGBM
d_train = lgbm.Dataset(X_train, label=y_train)
d_eval = lgbm.Dataset(X_val, label=y_val, reference=d_train)

# Train the model using early_stopping callback
try:
    clf = lgbm.train(
        params,
        d_train,
        num_boost_round=1000,
        valid_sets=[d_eval],
        callbacks=[lgbm.callback.early_stopping(stopping_rounds=50)]
    )
    print("Model trained successfully!")
except Exception as e:
    print("Error during model training:", e)

preds = clf.predict(X_val)
pred_labels = np.rint(preds)
accuracy = accuracy_score(y_val, pred_labels)
print("Accuracy:", accuracy)

# Merge predictions with validation dates
predictions_df = pd.DataFrame({
    'Date': dates_val.reset_index(drop=True),
    'Predicted_Movement': pred_labels,
    'Close': X_val['Close'].reset_index(drop=True)
})

# Now merge this DataFrame with the original data
full_data_with_predictions = prepared_data.merge(predictions_df, on='Date', how='left')

plt.figure(figsize=(14, 7))

# Plot actual close prices
plt.plot(full_data_with_predictions['Date'], full_data_with_predictions['Close_x'], label='Actual Close Price', color='blue')

# Plot 'buy' signals for predicted up movement
buy_signals = full_data_with_predictions[full_data_with_predictions['Predicted_Movement'] == 1]
plt.scatter(buy_signals['Date'], buy_signals['Close_x'], label='Buy Signal', color='green', marker='^', alpha=0.5)

# Plot 'sell' signals for predicted down movement
sell_signals = full_data_with_predictions[full_data_with_predictions['Predicted_Movement'] == 0]
plt.scatter(sell_signals['Date'], sell_signals['Close_x'], label='Sell Signal', color='red', marker='v', alpha=0.5)

plt.title('Stock Price and Model Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
