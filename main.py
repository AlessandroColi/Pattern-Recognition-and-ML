import os
import numpy as np
import pandas as pd
from PattRecClasses import MarkovChain, GaussD, HMM


def load_sequences(data_folder):
    """Load preprocessed data from directory"""
    sequences = []
    for fname in os.listdir(data_folder):
        filepath = os.path.join(data_folder, fname)
        df = pd.read_csv(filepath, sep='\t', header=None, 
                        names=['timestamp', 'x', 'y', 'z'])
        sequences.append(df[['x', 'y', 'z']].values)
    return sequences

def initialize_hmm(activity_data):
    """Initialize HMM with empirical parameters"""
    distributions = []
    for state_data in activity_data:
        all_obs = np.concatenate(state_data)
        means = np.mean(all_obs, axis=0)
        cov = np.cov(all_obs.T)
        distributions.append(GaussD(means=means, cov=cov))
    
    n_states = len(activity_data)
    initial_prob = np.ones(n_states) / n_states
    trans_prob = np.ones((n_states, n_states)) / n_states
    
    mc = MarkovChain(initial_prob, trans_prob)
    return HMM(mc, distributions)

def train_hmm(model, train_sequences, n_iter=20):
    """Train HMM using multiple sequences"""
    for _ in range(n_iter):
        for seq in train_sequences:
            model.train(seq)

def test_hmm(model, test_sequences, true_labels):
    """Test HMM and return accuracy"""
    correct = 0
    total = len(test_sequences)
    
    for seq, label in zip(test_sequences, true_labels):
        states = model.viterbi(seq)
        pred = np.bincount(states).argmax()
        correct += (pred == label)
    
    return correct / total

def main():
    activities = ['Still', 'Walking', 'Running']
    
    # Load training data
    train_sequences = []
    train_labels = []
    for state_idx, activity in enumerate(activities):
        activity_dir = os.path.join('Data', 'Train', activity)
        seqs = load_sequences(activity_dir)
        train_sequences.extend(seqs)
        train_labels.extend([state_idx]*len(seqs))
    
    # Initialize HMM
    activity_data = []
    for activity in activities:
        activity_dir = os.path.join('Data', 'Train', activity)
        activity_data.append(load_sequences(activity_dir))
    
    hmm = initialize_hmm(activity_data)
    
    # Train HMM
    train_hmm(hmm, train_sequences, n_iter=20)
    
    # Load test data
    test_sequences = []
    test_labels = []
    for state_idx, activity in enumerate(activities):
        activity_dir = os.path.join('Data', 'Test', activity)
        seqs = load_sequences(activity_dir)
        test_sequences.extend(seqs)
        test_labels.extend([state_idx]*len(seqs))
    
    # Evaluate
    accuracy = test_hmm(hmm, test_sequences, test_labels)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()