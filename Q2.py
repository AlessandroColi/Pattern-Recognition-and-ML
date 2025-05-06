import numpy as np
from PattRecClasses import MarkovChain, GaussD, HMM

q = np.array([0.75, 0.25])
A = np.array([[0.99, 0.01],
              [0.03, 0.97]])

b1 = GaussD(means=[0], stdevs=[1])
b2 = GaussD(means=[3], stdevs=[2])

mc = MarkovChain(q, A)
hmm = HMM(mc, [b1, b2])

T = 10000
observations, states = hmm.rand(T)

print("Verification Results:\n")

state_counts = np.bincount(states)[1:]
state_freq = state_counts / T
print(f"State frequencies:\nState 1: {state_freq[0]:.4f}\nState 2: {state_freq[1]:.4f}\n")

obs_state1 = observations[states == 1]
obs_state2 = observations[states == 2]

print(f"Observations in State 1:\nMean: {np.mean(obs_state1):.4f}\nStd: {np.std(obs_state1):.4f}\n")
print(f"Observations in State 2:\nMean: {np.mean(obs_state2):.4f}\nStd: {np.std(obs_state2):.4f}\n")