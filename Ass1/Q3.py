import numpy as np
from PattRecClasses import MarkovChain, GaussD, HMM

q = np.array([0.75, 0.25])
A = np.array([[0.99, 0.01],
              [0.03, 0.97]])

b1 = GaussD(means=[0], stdevs=[1])
b2 = GaussD(means=[3], stdevs=[2])

hmm = HMM(MarkovChain(q, A), [b1, b2])

T = 10000
observations, _ = hmm.rand(T)

print(f"Mean of generated sequence: {np.mean(observations):.4f}")
print(f"Variance of generated sequence: {np.var(observations):.4f}")