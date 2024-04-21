import lightgbm as lgbm
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import numpy as np

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
prepared_data = fetch_and_prepare_data('NVDA', '2020-01-01', '2020-12-31')

# Prepare the features and labels
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'daily_return', 'daily_volatility']
X = prepared_data[features]
y = (prepared_data['daily_return'] > 0).astype(int)

# Split the dataset
xtr, xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM parameters
params = {
    "num_leaves": 31,
    "objective": "binary",
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "metric": "auc"
}

# Create datasets for LightGBM
d_train = lgbm.Dataset(xtr, label=ytr)
d_eval = lgbm.Dataset(xval, label=yval, reference=d_train)

# Train the model
try:
    clf = lgbm.train(
        params,
        d_train,
        num_boost_round=1000,
        valid_sets=[d_eval],  # Specify at least one validation set
        early_stopping_rounds=50,  # Use early stopping
        verbose_eval=50
    )
    print("Model trained successfully!")

    # Save the model
    model_name = 'lgb_model.bin'
    pickle.dump(clf, open(model_name, 'wb'))

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    lgbm.plot_importance(clf, ax=ax, max_num_features=10)
    plt.show()
except Exception as e:
    print("Error during model training:", e)


#preds = clf.predict(xtr)
#pred_labels = np.rint(preds)


    
#accuracy = sklearn.metrics.accuracy_score(ytr, pred_labels)