import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

# Fetch AAPL stock data from Yahoo Finance
data = yf.download('AAPL', start='2021-01-01', end='2023-12-31')
data['Date'] = data.index

# Prepare data for SVM
data['Days'] = (data['Date'] - data['Date'].min()).dt.days
X = data[['Days']].values
y = data['Close'].values

# Split the data into training and testing sets (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Initialize SVM with RBF kernel
model = SVR(kernel='rbf', C=100, gamma=0.1)

# Fit the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate RMSE, MSE, and accuracy (R^2 score)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
ss_total = np.sum((y_test - np.mean(y_train)) ** 2)  # Use training mean for fairness in comparison
ss_res = np.sum((y_test - y_pred) ** 2)
accuracy = 1 - ss_res / ss_total

# Repeat trials for mean accuracy and variance
accuracies = []
rmses = []
mses = []

for i in range(100):
    # Split the data into training and testing sets (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Initialize SVM with RBF kernel
    model = SVR(kernel='rbf', C=100, gamma=0.1)

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    ss_total = np.sum((y_test - np.mean(y_train)) ** 2)
    ss_res = np.sum((y_test - y_pred) ** 2)
    accuracy = 1 - ss_res / ss_total

    accuracies.append(accuracy)
    rmses.append(rmse)
    mses.append(mse)

# Calculate mean and variance of accuracy
mean_accuracy = np.mean(accuracies)
variance_accuracy = np.var(accuracies)
mean_rmse = np.mean(rmses)
mean_mse = np.mean(mses)

plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.title('Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

plt.figure(figsize=(14, 7))
ohlc = data.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]
ohlc['Date'] = mdates.date2num(ohlc['Date'])
ax = plt.gca()
candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='g', colordown='r')
ax.xaxis_date()
plt.title('Candlestick Chart')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

plt.figure(figsize=(14, 7))
data['30_MA'] = data['Close'].rolling(window=30).mean()
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.plot(data['Date'], data['30_MA'], label='30-Day MA', linestyle='--')
plt.title('Close and 30-Day MA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
data['Month'] = data['Date'].dt.month
monthly_avg = data.groupby('Month')['Close'].mean()
plt.plot(monthly_avg.index, monthly_avg.values, marker='o')
plt.title('Monthly Seasonality')
plt.xlabel('Month')
plt.ylabel('Avg. Close Price')
plt.xticks(range(1, 13))
plt.show()

plt.figure(figsize=(14, 7))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted', alpha=0.5)
plt.title('Actual vs. Predicted Prices (SVM)')
plt.xlabel('Days from Start')
plt.ylabel('Price')
plt.legend()
plt.show()

# Output results
print(f'Accuracy of prediction on test data (R^2 score): {accuracy:.6f}')
print(f'RMSE of the model: {rmse:.6f}')
print(f'MSE of the model: {mse:.6f}')
print(f'Mean RMSE of the model: {mean_rmse:.6f}')
print(f'Mean MSE of the model: {mean_mse:.6f}')
print(f'Mean Accuracy over 100 trials: {mean_accuracy:.6f}')
print(f'Variance of Accuracies over 100 trials: {variance_accuracy:.6f}')