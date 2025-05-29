import os
import numpy as np
import pandas as pd
import random
from PattRecClasses import MarkovChain, GaussD, HMM

def load_sequences(data_folder):
    print(f"Loading sequences from {data_folder}...")
    sequences = []
    for fname in os.listdir(data_folder):
        filepath = os.path.join(data_folder, fname)
        try:
            df = pd.read_csv(filepath, sep='\t', header=None, 
                           names=['timestamp', 'x', 'y', 'z'])
            if df.shape[0] == 0:
                print(f"  Skipping empty file: {fname}")
                continue
            sequences.append(df[['x', 'y', 'z']].values.astype(float))
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
    print(f"Loaded {len(sequences)} sequences from {data_folder}")
    return sequences

def initialize_hmm(n_states, all_train_obs):
    print("Initializing HMM with random parameters...")
    # Compute global data statistics for initialization scaling
    global_min = np.min(all_train_obs, axis=0)
    global_max = np.max(all_train_obs, axis=0)
    global_std = np.std(all_train_obs, axis=0)
    
    distributions = []
    
    # Initialize Gaussian distributions with random parameters scaled to data range
    for i in range(n_states):
        # Random mean within [global_min, global_max]
        means = global_min + np.random.rand(*global_min.shape) * (global_max - global_min)
        
        # Random variances scaled by global std with random factor (0.5-1.5)
        rand_factor = 0.5 + np.random.rand(*global_std.shape)
        variances = (global_std * rand_factor) ** 2
        
        # Diagonal covariance matrix
        cov = np.diag(variances)
        distributions.append(GaussD(means=means, cov=cov))
    
    # Random initial state probabilities (normalized)
    initial_prob = np.random.rand(n_states)
    initial_prob /= np.sum(initial_prob)
    
    # Random transition matrix (rows normalized to sum to 1)
    trans_prob = np.random.rand(n_states, n_states)
    trans_prob /= np.sum(trans_prob, axis=1, keepdims=True)
    
    mc = MarkovChain(initial_prob, trans_prob)
    print("HMM random initialization done.")
    return HMM(mc, distributions)

def train_hmm(model, train_sequences, n_iter=20):
    print(f"Starting training for {n_iter} iterations...")
    for iter_num in range(1, n_iter + 1):
        print(f"Iteration {iter_num}...")
        model.train(train_sequences, 1)  # Process ALL sequences in one batch
    print("Training completed.")

def test_hmm(model, test_sequences, true_labels):
    print(f"Testing {len(test_sequences)} sequences...")
    correct = 0
    total = len(test_sequences)
    for i, (seq, label) in enumerate(zip(test_sequences, true_labels)):
        try:
            states = model.viterbi(seq)
            # Convert 1-indexed states to 0-indexed labels
            state_counts = np.bincount(states, minlength=model.nStates+1)[1:]
            pred = np.argmax(state_counts)
            if pred == label:
                correct += 1
            else:
                print(f"  Mismatch on sequence #{i+1}: predicted {pred}, actual {label}")
        except Exception as e:
            print(f"  Error decoding sequence #{i+1}: {e}")
    accuracy = correct / total if total > 0 else 0
    print(f"\n\t\t [MAIN] Testing done. Accuracy: {accuracy*100:.2f}%")
    return accuracy

def main():
    pure_activities = ['Still', 'Walking'] #, 'Running']
    mixed_activity = 'Mixed'
    
    # Load pure activity data
    pure_train_sequences = []
    for activity in pure_activities:
        activity_dir = os.path.join('Data', 'Train', activity)
        pure_train_sequences.extend(load_sequences(activity_dir))
    
    # Load mixed activity data
    mixed_dir = os.path.join('Data', 'Train', mixed_activity)
    mixed_sequences = load_sequences(mixed_dir)
    
    # Combine all training data
    train_sequences = pure_train_sequences + mixed_sequences
    
    # Create array of all observations for initialization
    all_train_obs = np.concatenate(train_sequences)
    
    # Initialize HMM with random parameters
    n_states = len(pure_activities)
    hmm = initialize_hmm(n_states, all_train_obs)
    
    # Train HMM with shuffled sequences
    train_hmm(hmm, train_sequences, n_iter=10)
    
    # Load test data (only pure activities)
    test_sequences = []
    test_labels = []
    for state_idx, activity in enumerate(pure_activities):
        activity_dir = os.path.join('Data', 'Test', activity)
        seqs = load_sequences(activity_dir)
        test_sequences.extend(seqs)
        test_labels.extend([state_idx]*len(seqs))
    
    # Evaluate
    accuracy = test_hmm(hmm, test_sequences, test_labels)
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()