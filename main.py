import os
import numpy as np
import pandas as pd
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

def initialize_hmm(activity_data):
    print("Initializing HMM...")
    distributions = []
    for idx, state_data in enumerate(activity_data):
        print(f"  Preparing state {idx+1} with {len(state_data)} sequences")
        all_obs = np.concatenate(state_data)
        means = np.mean(all_obs, axis=0)
        cov = np.cov(all_obs.T)
        distributions.append(GaussD(means=means, cov=cov))
    n_states = len(activity_data)
    initial_prob = np.ones(n_states) / n_states
    trans_prob = np.ones((n_states, n_states)) / n_states
    mc = MarkovChain(initial_prob, trans_prob)
    print("HMM initialization done.")
    return HMM(mc, distributions)

def train_hmm(model, train_sequences, n_iter=20):
    print(f"Starting training for {n_iter} iterations on {len(train_sequences)} sequences...")
    for iter_num in range(1, n_iter + 1):
        print(f"Iteration {iter_num}...")
        for i, seq in enumerate(train_sequences):
            print(f"seq {i}")
            
            if len(seq) < 2:
                print(f"  Skipping too short sequence #{i+1} (length {len(seq)})")
                continue
            try:
                seq = np.array(seq)
                model.train(seq)
            except Exception as e:
                print(f"  Training error on sequence #{i+1}: {e}")
    print("Training completed.")

def test_hmm(model, test_sequences, true_labels):
    print(f"Testing {len(test_sequences)} sequences...")
    correct = 0
    total = len(test_sequences)
    for i, (seq, label) in enumerate(zip(test_sequences, true_labels)):
        print(f"seq {i}")
        try:
            states = model.viterbi(seq)
            pred = np.bincount(states).argmax()
            if pred == label:
                correct += 1
            else:
                print(f"  Mismatch on sequence #{i+1}: predicted {pred}, actual {label}")
        except Exception as e:
            print(f"  Error decoding sequence #{i+1}: {e}")
    accuracy = correct / total if total > 0 else 0
    print(f"Testing done. Accuracy: {accuracy*100:.2f}%")
    return accuracy

def main():
    activities = ['Still', 'Walking']
    
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
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
