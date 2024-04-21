import numpy as np
import pandas as pd
import scipy.io as sio
from matplotlib import pyplot as plt  

# Load the data
data = sio.loadmat("C:/Users/hoonf/Downloads/Solution-codes/Solution-codes/Code-problem 2/hmm_params.mat")
transition = data["transition"]
prior = data["prior"]
emission = data["emission"]
price_change = data["price_change"][0]

#### PART 1: Forward-Backward Algorithm ####
T = 100
K = 3

def forward_backward(price_change, prior, emission, transition):
    # Forward propagation
    alpha = [np.exp(np.log(prior) + np.log(emission[:, price_change[0]-1]).reshape(-1,1))]
    for i in range(1, T):
        non_marg_alpha = np.exp((np.log(transition) + np.log(emission[:, price_change[i]-1])) + np.log(alpha[-1]))
        alpha += [non_marg_alpha.sum(axis=0).reshape(-1,1)]
    
    # Backward propagation
    beta = [np.ones((K, 1))]
    for i in range(1, T):
        non_marg_beta = np.exp(np.log(beta[-1]).T + (np.log(transition) + np.log(emission[:, price_change[T-i]-1])))
        beta += [non_marg_beta.sum(axis=1, keepdims=True)]
    beta = beta[::-1]

    return alpha, beta

alpha, beta = forward_backward(price_change, prior, emission, transition)
cond_z_t = [np.exp(np.log(alpha[i]) + np.log(beta[i])) / np.exp(np.log(alpha[i]) + np.log(beta[i])).sum() for i in range(T)]
cond_z_t = np.hstack(cond_z_t)

# Plot state probabilities
fig, axs = plt.subplots(1)
axs.plot(range(1, T+1), cond_z_t[0], label='$Z_t = 1$')
axs.plot(range(1, T+1), cond_z_t[1], label='$Z_t = 2$')
axs.plot(range(1, T+1), cond_z_t[2], label='$Z_t = 3$')
axs.set_xlabel("Time step (t)")
axs.set_title("$P(Z_t \mid X_1, \ldots, X_{100})$")
plt.legend()
plt.show()

#### PART 2: Viterbi Algorithm ####
V_t = np.zeros((T, K))
V_t_arg = np.zeros((T-1, K))

# Initialization
V_t[0, :] = np.log(emission[:, price_change[0]-1]) + np.log(prior.T)
for t in range(1, T):
    V_t[t, :] = np.log(emission[:,price_change[t]-1]) + (np.log(transition) + V_t[t-1, :].reshape(K, 1)).max(axis=0)
    V_t_arg[t-1, :] = (np.log(transition) + V_t[t-1, :].reshape(K, 1)).argmax(axis=0)

# Decode the most probable final state sequence
cleanSequence = [V_t[-1, :].argmax()]
for t in range(T-2, -1, -1):
    cleanSequence.insert(0, int(V_t_arg[t, cleanSequence[0]]))

# Plot most likely hidden states
fig, axs = plt.subplots(1)
axs.plot(np.array(cleanSequence)+1)
axs.set_title("Most likely hidden states")
axs.set_xlabel("Time step (t)")
plt.show()

#### PART 3: Sample Generation and Evaluation ####
def generate_samples(prior_data, prior, transition, emission, n_samples=28, window_len=100):
    for i in range(n_samples):
        curr_data = prior_data[-window_len:]
        alpha, beta = forward_backward(curr_data, prior, emission, transition)
        p_z_unn = alpha[-1] * beta[-1]
        p_z = p_z_unn / p_z_unn.sum()
        p_zt1 = (p_z * transition).sum(axis=1)
        z = np.random.choice(np.arange(3), p=p_zt1)
        x = np.random.choice(np.arange(1,6), p=emission[z])
        np.append(prior_data, int(x))
    return prior_data[-n_samples:]

acc_list = []
for _ in range(10):
    new_samples = generate_samples(price_change[:100], prior, transition, emission, n_samples=28)
    acc = np.sum(new_samples == price_change[-28:])/28
    acc_list.append(acc)

print("Mean accuracy across 100 trails: {}".format(np.mean(acc_list)))
print("Variance across 100 trials: {}".format(np.var(acc_list)))


def forward_backward(price_change, prior, emission, transition):
    # Forward pass
    alpha = [np.exp(np.log(prior) + np.log(emission[:, price_change[0]-1]).reshape(-1,1))]
    for i in range(1, T):
        non_marg_alpha = np.exp((np.log(transition) + np.log(emission[:, price_change[i]-1])) + np.log(alpha[-1]))
        alpha += [non_marg_alpha.sum(axis=0).reshape(-1,1)]
    
    # Backward pass
    beta = [np.ones((K, 1))]
    for i in range(1,T):
        non_marg_beta = np.exp( np.log(beta[-1]).T + (np.log(transition) + np.log(emission[:, price_change[T-i]-1])) )
        beta += [non_marg_beta.sum(axis=1, keepdims=True)]
    beta = beta[::-1]

    return alpha, beta

alpha, beta = forward_backward(price_change, prior, emission, transition)
cond_z_t = [np.exp(np.log(alpha[i]) + np.log(beta[i])) / np.exp(np.log(alpha[i]) + np.log(beta[i])).sum() for i in range(T)]
cond_z_t = np.hstack(cond_z_t)

V_t = np.zeros((T, K))
V_t_arg = np.zeros((T-1, K))
V_t[0, :] = np.log(emission[:, price_change[0]-1]) + np.log(prior.T) 
for t in range(1, T):
    V_t[t, :] = np.log(emission[:,price_change[t]-1]) + (np.log(transition) + V_t[t-1, :].reshape(K, 1)).max(axis=0)
    V_t_arg[t-1, :] = (np.log(transition) + V_t[t-1, :].reshape(K, 1)).argmax(axis=0)

# Decoding the most probable final state sequence
cleanSequence = [V_t[-1, :].argmax()]
for t in range(T-2, -1, -1):
    cleanSequence.insert(0, int(V_t_arg[t, cleanSequence[0]]))

# Plotting the most likely hidden states and actual states
fig, axs = plt.subplots(1)
axs.plot(np.array(cleanSequence)+1, label='Predicted States')
if isinstance(price_change, np.ndarray) and len(price_change) == T:
    axs.plot(price_change, label='Actual States', linestyle='--')  # Assuming price_change is the actual state sequence
axs.set_title("Most likely hidden states vs Actual States")
axs.set_xlabel("Time step (t)")
plt.legend()
plt.show()