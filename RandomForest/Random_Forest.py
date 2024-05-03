import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

# Fetch AAPL stock data
ticker_symbol = "AAPL"
start_date = "2021-01-01"
end_date = "2023-12-31"
aapl_data = yf.download(ticker_symbol, start=start_date, end=end_date)
aapl_data.reset_index(inplace=True)

# 1. Plot the closing prices over time
plt.figure(figsize=(14, 7))
plt.plot(aapl_data['Date'], aapl_data['Close'], label='Close Price')
plt.title('Closing Price of AAPL')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.show()

# 2. Candlestick chart
plt.figure(figsize=(14, 7))
ohlc = aapl_data.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]
ohlc['Date'] = mdates.date2num(ohlc['Date'])
ax = plt.gca()
candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='g', colordown='r')
ax.xaxis_date()
plt.title('Candlestick Chart')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# 3. Closing price and 30-days moving average
aapl_data['30_MA'] = aapl_data['Close'].rolling(window=30).mean()
plt.figure(figsize=(14, 7))
plt.plot(aapl_data.index, aapl_data['Close'], label='Close Price')
plt.plot(aapl_data.index, aapl_data['30_MA'], label='30-Day MA', color='r')
plt.title('Close Price and 30-Day Moving Average of AAPL')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# 4. Monthly seasonality
plt.figure(figsize=(14, 7))
aapl_data['Month'] = aapl_data['Date'].dt.month
monthly_avg = aapl_data.groupby('Month')['Close'].mean()
plt.plot(monthly_avg.index, monthly_avg.values, marker='o')
plt.title('Monthly Seasonality')
plt.xlabel('Month')
plt.ylabel('Avg. Close Price')
plt.xticks(range(1, 13))
plt.show()

# Prepare data for Random Forest regression
data = aapl_data[['Close']].copy()
data['Target'] = data['Close'].shift(-1)
data = data[:-1]  # Remove the last NaN
X = data[['Close']]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Multiple trials for accuracy and variance calculations
accuracies = []
mses = []
rmses = []

for _ in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    model = RandomForestRegressor(n_estimators=100, random_state=43)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracies.append(model.score(X_test, y_test))
    mse = mean_squared_error(y_test, predictions)
    mses.append(mse)
    rmses.append(np.sqrt(mse))

# 5. Plot actual vs predicted
plt.figure(figsize=(14, 7))
plt.scatter(y_test.index, y_test, label='Actual')
plt.scatter(y_test.index, predictions, label='Predicted', color='r')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Calculate and display statistics
mean_accuracy = np.mean(accuracies)
variance = np.var(accuracies)
mean_mse = np.mean(mses)
mean_rmse = np.mean(rmses)

print(f"Mean Accuracy: {mean_accuracy:.6f}")
print(f"Variance of Accuracies: {variance:.6f}")
print(f"Mean MSE: {mean_mse:.6f}")
print(f"Mean RMSE: {mean_rmse:.6f}")