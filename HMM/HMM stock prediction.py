import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

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

# Example usage
ticker = 'NVDA'
start_date = '2020-01-01'
end_date = '2020-12-31'
num_states = 3

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

# Calculate HMM parameters
# prior, transition, emission = calculate_hmm_params(observations, num_states)

# Print or use the parameters as needed
print("Prior probabilities:", prior)
print("Transition matrix:", transition)
print("Emission matrix:", emission)
print(observations)

train_size = int(len(observations) * 0.8)
train_observations = observations[:train_size]
pred_observations = observations[train_size:]
n_samples = int(len(pred_observations) * 0.2)

# Calculate forward and backward matrices
num_states = transition.shape[0]
num_obs = len(observations)

forward = np.zeros((num_obs, num_states))
backward = np.zeros((num_obs, num_states))
# # Define Viterbi algorithm
# def viterbi(observations, states, start_p, trans_p, emit_p):
 
# # Setup empty set 
#     V = [{}]
#     path = {}

#     for y in states:
#         V[0][y] = start_p[y] * emit_p[y][observations[0]]
#         path[y] = [y]

#     for t in range(1, len(observations)):
#         V.append({})
#         new_path = {}

#         for curr_state in states:
#             (prob, state) = max((V[t-1][prev_state] * trans_p[prev_state][curr_state] *emit_p[curr_state][observations[t]], prev_state) 
#                                 for prev_state in states if V[t-1][prev_state] > 0)
            
#             V[t][curr_state] = prob
#             new_path[curr_state] = path[state] + [curr_state]

#         path = new_path

#     (prob, state) = max((V[t][y], y) for y in states)
#     return path[state]

# Define prediction function
def predict(num_steps, start_prob, trans_p, emit_p):
    pred_states = []
    pred_obs = []

    state_distribution = start_prob

    for _ in range(num_steps):
        next_state_disp = np.dot(state_distribution, trans_p)
        pred_state = np.argmax(next_state_disp)
        pred_states.append(pred_state + 1)

        pred_ob = np.argmax(emit_p[pred_state, :])
        pred_obs.append(pred_ob + 1)

        state_distribution = next_state_disp

    return pred_states, pred_obs

# Define evaluation function
def eval_pred(observations, num_reps, num_steps, transition, emission):
    num_correct = []
    last_state_dis = forward[-1, :]
    
    for _ in range(num_reps):
        _, pred_obs = predict(num_steps, last_state_dis, transition, emission)
        
        gt_obs = observations[100:128]
        gt_obs = [int(obs) for obs in gt_obs]
        
        correct = np.array(pred_obs) == np.array(gt_obs)
        temp_correct = np.sum(correct)
        num_correct.append(temp_correct)
    
    return num_correct


for s in range(num_states):
    forward[0, s] = prior[s] * emission[s, int(observations[0])-1]

for t in range(1, num_obs):
    for s in range(num_states):
        forward[t, s] = emission[s, int(observations[t])-1] * np.sum(forward[t-1, :] * transition[:, s])

# Backward pass
backward[-1, :] = 1

for t in range(num_obs - 2, -1, -1):
    for s in range(num_states):
        backward[t, s] = np.sum(backward[t+1, :] * transition[s, :] * emission[:, int(observations[t+1])-1])

# Calculate posteriors
posteriors = np.zeros((num_obs, num_states))
for t in range(num_obs):
    posteriors[t, :] = forward[t, :] * backward[t, :] / np.sum(forward[t, :] * backward[t, :])
t = np.arange(1, num_obs + 1)
plt.figure(figsize=(14, 7))
for s in range(num_states):
    plt.plot(t, posteriors[:, s], label=f'State {s+1}')

plt.title('P(Zt = i|X1, ..., X100) vs. t')
plt.xlabel('t')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()


# Evaluate predictions against ground truth
steps = len(pred_observations)
reps = 100
num_correct = eval_pred(pred_observations, reps, steps, transition, emission)
percent_correct = [count / steps for count in num_correct]

def eval_pred(pred_observations, gt_observations):
    num_correct = 0
    
    for pred, gt in zip(pred_observations, gt_observations):
        if pred == gt:
            num_correct += 1
    
    return num_correct

# Evaluate predictions against ground truth
gt_obs = observations[train_size:]

num_correct = eval_pred(pred_observations, gt_obs)
accuracy = num_correct / len(pred_observations)

print('Evaluation: Predictions against Ground Truth:')
print(f'accuracy: {accuracy * 100:.2f}%')
print('Evaluation: Predictions against Ground Truth:')
print(f'mean: {np.mean(percent_correct)}, \nvariance: {np.var(percent_correct)}')
