import numpy as np
from .DiscreteD import DiscreteD
from .GaussD import GaussD
from .MarkovChain import MarkovChain

class HMM:
    def __init__(self, mc, distributions):
        self.stateGen = mc
        self.outputDistr = distributions
        self.nStates = mc.nStates
        self.dataSize = distributions[0].dataSize

    def rand(self, nSamples):
        print(f"[HMM] Generating random sequence of length {nSamples}")
        S = self.stateGen.rand(nSamples)
        nS = len(S)
        
        if self.dataSize == 1:
            X = np.zeros(nS)
        else:
            X = np.zeros((self.dataSize, nS))
        
        for t, state in enumerate(S):
            distr = self.outputDistr[state-1]
            if isinstance(distr, GaussD):
                x = distr.rand(1)
            elif isinstance(distr, DiscreteD):
                x = distr.rand(1)
            else:
                raise ValueError("Unsupported distribution type")
            
            if self.dataSize == 1:
                X[t] = x
            else:
                X[:, t] = x.reshape(-1)
        print("[HMM] Random sequence generation done")
        return X, S

    def forward(self, observations):
        T = len(observations)
        N = self.nStates
        alpha = np.zeros((N, T))
        c = np.zeros(T)

        # Initialization
        for i in range(N):
            alpha[i, 0] = self.stateGen.q[i] * self.outputDistr[i].prob(observations[0])
        c[0] = 1.0 / np.sum(alpha[:, 0])
        alpha[:, 0] *= c[0]

        # Recursion
        for t in range(1, T):
            for j in range(N):
                alpha[j, t] = self.outputDistr[j].prob(observations[t]) * \
                              np.sum(alpha[:, t-1] * self.stateGen.A[:, j])
            c[t] = 1.0 / np.sum(alpha[:, t])
            alpha[:, t] *= c[t]

        return alpha, c

    def backward(self, observations, c):
        T = len(observations)
        N = self.nStates
        beta = np.zeros((N, T))

        # Initialize
        beta[:, T-1] = c[T-1]

        # Recursion
        for t in range(T-2, -1, -1):
            bt1 = np.array([d.prob(observations[t+1]) for d in self.outputDistr])
            for i in range(N):
                beta[i, t] = np.sum(self.stateGen.A[i, :] * bt1 * beta[:, t+1])
            beta[:, t] *= c[t]

        return beta

    def train(self, observations, n_iter=10):
        print("[HMM] Starting training")
        for iteration in range(n_iter):
            print(f"[HMM] Training iteration {iteration+1}/{n_iter}")
            alpha, c = self.forward(observations)
            beta = self.backward(observations, c)
            T = len(observations)
            N = self.nStates

            gamma = alpha * beta
            gamma /= gamma.sum(axis=0, keepdims=True)

            xi = np.zeros((N, N, T-1))
            for t in range(T-1):
                for i in range(N):
                    # Get emission probabilities for ALL states
                    emission_probs = np.array([d.prob(observations[t+1]) for d in self.outputDistr]) 
                    
                    xi[i, :, t] = alpha[i, t] * self.stateGen.A[i, :] * \
                                emission_probs * beta[:, t+1]
                xi[:, :, t] /= xi[:, :, t].sum()

            # Update initial probabilities
            self.stateGen.q = gamma[:, 0]

            # Update transition matrix
            trans_num = np.sum(xi, axis=2)
            trans_den = np.sum(gamma[:, :-1], axis=1, keepdims=True)
            self.stateGen.A = trans_num / np.where(trans_den == 0, 1e-10, trans_den)

            # Update emission parameters
            for j in range(N):
                gamma_j = gamma[j, :]
                total = gamma_j.sum()
                if total == 0:
                    print(f"[HMM] Warning: total gamma for state {j} is zero, skipping update")
                    continue

                self.outputDistr[j].means = np.sum(observations * gamma_j[:, None], axis=0) / total

                diff = observations - self.outputDistr[j].means
                cov = np.dot((diff * gamma_j[:, None]).T, diff) / total
                self.outputDistr[j].cov = cov
                self.outputDistr[j].stdevs = np.sqrt(np.diag(cov))
                self.outputDistr[j].variance = self.outputDistr[j].stdevs ** 2
                print(f"[HMM] Updated emission for state {j}")

        print("[HMM] Training finished")

    def viterbi(self, observations):
        print("[HMM] Starting Viterbi decoding")
        T = len(observations)
        N = self.nStates
        delta = np.zeros((N, T))
        psi = np.zeros((N, T), dtype=int)

        delta[:, 0] = np.log(self.stateGen.q + 1e-10) + \
                     np.array([d.prob(observations[0]) for d in self.outputDistr])
        delta[:, 0] = np.where(delta[:, 0] < -1e20, -1e20, delta[:, 0])

        for t in range(1, T):
            for j in range(N):
                trans_prob = np.log(self.stateGen.A[:, j] + 1e-10)
                prev = delta[:, t-1] + trans_prob
                psi[j, t] = np.argmax(prev)
                delta[j, t] = prev[psi[j, t]] + np.log(self.outputDistr[j].prob(observations[t]) + 1e-10)

        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[:, -1])
        for t in range(T-2, -1, -1):
            states[t] = psi[states[t+1], t+1]

        print("[HMM] Viterbi decoding finished")
        return states + 1  # 1-based states

    def logprob(self, observations):
        _, c = self.forward(observations)
        lp = -np.sum(np.log(c))
        return lp
