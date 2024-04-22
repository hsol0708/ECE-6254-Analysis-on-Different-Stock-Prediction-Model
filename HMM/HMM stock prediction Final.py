import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc


np.random.seed(43)
# Fetch historical stock price data
def fetch_stock_data(ticker, start_date, end_date, num_states):
    data = yf.download(ticker, start=start_date, end=end_date)
    pct_change = data['Close'].pct_change().dropna().values
    
    # Discretize the percentage changes into states
    min_change = np.min(pct_change)
    max_change = np.max(pct_change)
    step = (max_change - min_change) / num_states
    states = np.round((pct_change - min_change) / step).astype(int)
    
    return states, data

# Define function to calculate HMM parameters
def calculate_hmm_params(observations, num_states):
    # Calculate prior probabilities
    prior = np.ones(num_states) / num_states

    # Calculate transition matrix
    transition = np.random.rand(num_states, num_states)
    transition /= transition.sum(axis=1)[:, np.newaxis]

    # Calculate emission probabilities (assuming Gaussian emissions)
    means = np.random.rand(num_states)
    stds = np.random.rand(num_states)
    emission = np.array([[np.random.normal(means[i], stds[i]) for _ in range(num_states)] for i in range(num_states)])

    return prior, transition, emission

# Forward-Backward algorithm adjusted for numerical stability
def forward_backward(observations, prior, transition, emission):
    T = len(observations)
    K = len(prior)
    alpha = np.zeros((T, K))
    beta = np.zeros((T, K))
    
    alpha[0] = prior * emission[:, observations[0]-1]
    for t in range(1, T):
        for k in range(K):
            alpha[t, k] = emission[k, observations[t]-1] * np.sum(alpha[t-1] * transition[:, k])

    beta[-1] = 1
    for t in range(T-2, -1, -1):
        for k in range(K):
            beta[t, k] = np.sum(beta[t+1] * transition[k] * emission[:, observations[t+1]-1])

    posteriors = (alpha * beta) / np.sum(alpha * beta, axis=1, keepdims=True)
    return posteriors

# Viterbi algorithm with log probabilities
def viterbi(observations, prior, transition, emission):
    T = len(observations)
    K = len(prior)
    V_t = np.zeros((T, K))
    V_t_arg = np.zeros((T, K), dtype=int)
    
    V_t[0] = np.log(prior) + np.log(emission[:, observations[0]-1])
    for t in range(1, T):
        for k in range(K):
            seq_probs = V_t[t-1] + np.log(transition[:, k]) + np.log(emission[k, observations[t]-1])
            V_t[t, k] = np.max(seq_probs)
            V_t_arg[t, k] = np.argmax(seq_probs)
    
    # Decode the sequence
    best_path = np.zeros(T, dtype=int)
    best_path[-1] = np.argmax(V_t[-1])
    for t in range(T-2, -1, -1):
        best_path[t] = V_t_arg[t+1, best_path[t+1]]
    
    return best_path + 1  # Adjust index to match 1-based state indices

def calculate_rmse(predictions, actual):
    """Calculate the Root Mean Square Error between predictions and actual values."""
    predictions = np.array(predictions)
    actual = np.array(actual)
    return np.sqrt(np.mean((predictions - actual) ** 2))

def run_hmm_model(ticker, start_date, end_date, num_states, train_size_ratio):
    # Fetch data
    observations, _ = fetch_stock_data(ticker, start_date, end_date, num_states)
    
    # Split data into training and testing
    train_size = int(len(observations) * train_size_ratio)
    train_observations = observations[:train_size]
    test_observations = observations[train_size:]
    
    # Train model on the training data
    prior, transition, emission = calculate_hmm_params(train_observations, num_states)
    
    # Apply Viterbi algorithm to the test data
    predicted_states = viterbi(test_observations, prior, transition, emission)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_states == test_observations)
    return accuracy

def main_simulation(ticker, start_date, end_date, num_states, num_trials, train_size_ratio):
    accuracies = []
    for _ in range(num_trials):
        accuracy = run_hmm_model(ticker, start_date, end_date, num_states, train_size_ratio)
        accuracies.append(accuracy)
    
    mean_accuracy = np.mean(accuracies)
    variance_accuracy = np.var(accuracies)
    
    return mean_accuracy, variance_accuracy

# Example usage
ticker = 'AAPL'
start_date = '2021-01-01'
end_date = '2023-12-31'
num_states = 3
num_trials = 100  # Number of trials to run the simulation
train_size_ratio = 0.8  # 80% training data

# Fetch stock data
observations, specific_df = fetch_stock_data(ticker, start_date, end_date, num_states)
prior, transition, emission = calculate_hmm_params(observations, num_states)

# Use forward-backward and Viterbi algorithms
posteriors = forward_backward(observations, prior, transition, emission)
most_likely_states = viterbi(observations, prior, transition, emission)

# Line Chart of Closing Prices Over Time
plt.figure(figsize=(15, 6))
plt.plot(specific_df.index, specific_df['Close'], marker='.')
plt.title('Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Candlestick Chart
matplotlib_date = mdates.date2num(specific_df.index)
ohlc = np.vstack((matplotlib_date, specific_df['Open'], specific_df['High'], specific_df['Low'], specific_df['Close'])).T

plt.figure(figsize=(15, 6))
ax = plt.subplot()
candlestick_ohlc(ax, ohlc, width=0.6, colorup='g', colordown='r')
ax.xaxis_date()
plt.title('Candlestick Chart')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Closing Prices and Moving Average plot
window = 30
plt.figure(figsize=(15, 6))
plt.plot(specific_df.index, specific_df['Close'], label='Closing Price', linewidth=2)
plt.plot(specific_df.index, specific_df['Close'].rolling(window=window).mean(), label=f'{window}-Day Moving Avg', linestyle='--')
plt.title(f'Closing Prices and {window}-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# Monthly Seasonality of Closing Prices
specific_df['Month'] = specific_df.index.month
monthly_average = specific_df.groupby('Month')['Close'].mean()

plt.figure(figsize=(15, 6))
plt.plot(monthly_average.index, monthly_average.values, marker='o')
plt.title(f'Monthly Seasonality of {ticker}')
plt.xlabel('Months')
plt.ylabel('Average Closing Price')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.show()



# Split data into training and testing
train_size = int(len(observations) * 0.8)
train_observations = observations[:train_size]
test_observations = observations[train_size:]

# Train model on the training data
prior, transition, emission = calculate_hmm_params(train_observations, num_states)

# Apply Viterbi algorithm to the test data
def viterbi(observations, prior, transition, emission):
    T = len(observations)
    K = len(prior)
    V_t = np.zeros((T, K))
    V_t_arg = np.zeros((T, K), dtype=int)
    
    # Initialization with logarithm to prevent underflow
    V_t[0] = np.log(prior) + np.log(emission[:, observations[0]-1])
    for t in range(1, T):
        for k in range(K):
            seq_probs = V_t[t-1] + np.log(transition[:, k]) + np.log(emission[k, observations[t]-1])
            V_t[t, k] = np.max(seq_probs)
            V_t_arg[t, k] = np.argmax(seq_probs)
    
    # Backtrace to find the most likely state sequence
    best_path = np.zeros(T, dtype=int)
    best_path[-1] = np.argmax(V_t[-1])
    for t in range(T-2, -1, -1):
        best_path[t] = V_t_arg[t+1, best_path[t+1]]
    
    return best_path + 1  # Adjust index to match 1-based state indices

most_likely_states = viterbi(test_observations, prior, transition, emission)

# Evaluate predictions against the actual test data
def evaluate_predictions(predicted, actual):
    # Log shapes to verify they match
    print("Predicted shape:", predicted.shape)
    print("Actual shape:", actual.shape)

    # Check if shapes are compatible for comparison
    if predicted.shape != actual.shape:
        raise ValueError("Shape mismatch between predicted and actual data arrays.")

    # Calculate accuracy
    accuracy = np.mean(predicted == actual)
    return accuracy

accuracy = evaluate_predictions(most_likely_states, test_observations)
print("Accuracy of predictions on test data:", accuracy)

def plot_predictions_vs_actual(test_observations, most_likely_states):
    # Create a time index from 0 to the number of test observations
    time_steps = range(len(test_observations))
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, test_observations, label='Actual States', marker='o', linestyle='-', color='b')
    plt.plot(time_steps, most_likely_states - 1, label='Predicted States', marker='x', linestyle='--', color='r')  # Adjust if necessary based on how states are indexed
    plt.title('Comparison of Actual vs. Predicted States')
    plt.xlabel('Time Step')
    plt.ylabel('State')
    plt.legend()
    plt.grid(True)
    plt.show()

# Assuming test_observations and most_likely_states are already defined and available
plot_predictions_vs_actual(test_observations, most_likely_states)
rmse = calculate_rmse(most_likely_states, test_observations)
print("RMSE of the model:", rmse)

# Run the simulation
mean_accuracy, variance_accuracy = main_simulation(ticker, start_date, end_date, num_states, num_trials, train_size_ratio)

print("Mean Accuracy over", num_trials, "trials:", mean_accuracy)
print("Variance of Accuracies over", num_trials, "trials:", variance_accuracy)

