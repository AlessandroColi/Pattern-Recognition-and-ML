import numpy as np
from PattRecClasses import MarkovChain, GaussD, HMM


# Minimal emission distribution class for testing
class FixedEmission:
    def __init__(self, probs):
        self.probs = probs

    def prob(self, obs):
        return self.probs[obs]

# Setup: Toy HMM with 2 states and 2 possible observations
q = np.array([0.6, 0.4])                  # Initial state probs
A = np.array([[0.7, 0.3],                 # Transition matrix
              [0.4, 0.6]])
B = [FixedEmission([0.9, 0.1]),           # State 1: P(obs=0)=0.9, obs=1=0.1
     FixedEmission([0.2, 0.8])]           # State 2: P(obs=0)=0.2, obs=1=0.8

# Observation sequence (e.g., 0 = 'low', 1 = 'high')
obs_seq = [0, 1, 0, 0, 1]

# Build chain
mc = MarkovChain(initial_prob=q, transition_prob=A)
mc.B = B  # Inject emissions

# Run forward-backward
alpha_hat, c = mc.forward(obs_seq)
beta_hat = mc.backward(c, obs_seq)

# --- Verifications ---

print("\nAlpha-hat (scaled forward probabilities):\n", alpha_hat)
print("\nBeta-hat (scaled backward probabilities):\n", beta_hat)
print("\nScaling factors (c):\n", c)

# 1. Sanity check: Each column of alpha_hat should sum to 1
print("\nCheck: sum(alpha_hat[:, t]) == 1?", np.allclose(alpha_hat.sum(axis=0), 1.0))

# 2. Check that sum_i alpha_hat[i,t] * beta_hat[i,t] == 1 for all t
print("Check: sum_i alpha_hat[i,t] * beta_hat[i,t] == 1 / c[t]?", np.allclose((alpha_hat * beta_hat).sum(axis=0), 1.0 / c))

# 3. Compute log-likelihood
log_likelihood = -np.sum(np.log(c))
print("\nLog-likelihood of observation sequence:", log_likelihood)

# 4. Optional: direct likelihood (unscaled, for short sequences only)
# Not recommended for T > 5 due to numerical underflow
def brute_force_likelihood():
    total_prob = 0.0
    for s0 in range(2):
        for s1 in range(2):
            for s2 in range(2):
                for s3 in range(2):
                    for s4 in range(2):
                        states = [s0, s1, s2, s3, s4]
                        prob = q[states[0]] * B[states[0]].prob(obs_seq[0])
                        for t in range(1, 5):
                            prob *= A[states[t-1], states[t]] * B[states[t]].prob(obs_seq[t])
                        total_prob += prob
    return np.log(total_prob)

print("\nBrute-force log-likelihood:", brute_force_likelihood())
