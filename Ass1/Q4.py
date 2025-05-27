import numpy as np
from PattRecClasses import MarkovChain, GaussD, HMM
import matplotlib
matplotlib.use("TkAgg") #for stability 
import matplotlib.pyplot as plt

q = np.array([0.75, 0.25])
A = np.array([[0.99, 0.01],
              [0.03, 0.97]])

b1 = GaussD(means=[0], stdevs=[1])
b2 = GaussD(means=[3], stdevs=[2])

hmm = HMM(MarkovChain(q, A), [b1, b2])

observations, states = hmm.rand(500)

plt.figure(figsize=(12, 4))
plt.plot(observations, 'b-', linewidth=0.8, label='Observations $X_t$')
plt.plot(states, 'r-', alpha=0.3, label='States')
plt.title('HMM Output Example')
plt.xlabel('Time Step $t$')
plt.ylabel('Observation Value $X_t$')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('hmm_output_example.png', dpi=300)
plt.show()