import os
import numpy as np
import pandas as pd
import random
from PattRecClasses import MarkovChain, GaussD, HMM
import matplotlib.pyplot as plt  # For plotting if needed

def load_sequences(data_folder, file_extension='.txt'):
    """Load sequences from a directory with specified file extension"""
    print(f"Loading sequences from {data_folder}...")
    sequences = []
    valid_files = []
    
    for fname in os.listdir(data_folder):
        if fname.endswith(file_extension):
            filepath = os.path.join(data_folder, fname)
            try:
                df = pd.read_csv(filepath, sep='\t', header=None, 
                               names=['timestamp', 'x', 'y', 'z'])
                if df.shape[0] == 0:
                    print(f"  Skipping empty file: {fname}")
                    continue
                sequences.append(df[['x', 'y', 'z']].values.astype(float))
                valid_files.append(fname)
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")
    
    print(f"Loaded {len(sequences)} sequences from {data_folder}")
    return sequences, valid_files

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
    print(f"Starting training for {n_iter} iterations...")
    for iter_num in range(1, n_iter + 1):
        print(f"Iteration {iter_num}...")
        model.train(train_sequences, 1)  # Process ALL sequences in one batch
    print("Training completed.")

def test_hmm(model, test_sequences, true_labels):
    """Test HMM on individual sequences (pure activities)"""
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
    print(f"\nPure Activity Testing Accuracy: {accuracy*100:.2f}%")
    return accuracy

def test_mixed_sequence(model, mixed_sequence_path, mixed_labels_path):
    """Test HMM on continuous mixed sequence with state transitions"""
    print("\nTesting on mixed sequence with state transitions...")
    
    # Load mixed sequence data
    try:
        df = pd.read_csv(mixed_sequence_path, sep='\t', header=None,
                         names=['timestamp', 'x', 'y', 'z'])
        obs = df[['x', 'y', 'z']].values.astype(float)
    except Exception as e:
        print(f"Error loading mixed sequence: {e}")
        return 0
        
    # Load ground truth labels
    try:
        true_labels = np.loadtxt(mixed_labels_path, dtype=int)
    except Exception as e:
        print(f"Error loading mixed sequence labels: {e}")
        return 0
        
    # Verify sequence and labels match
    if len(obs) != len(true_labels):
        print(f"Warning: Mixed sequence length ({len(obs)}) doesn't match labels ({len(true_labels)})")
        min_len = min(len(obs), len(true_labels))
        obs = obs[:min_len]
        true_labels = true_labels[:min_len]
    
    # Run Viterbi decoding on entire sequence
    try:
        states = model.viterbi(obs)
        # Convert 1-indexed states to 0-indexed predictions
        pred_labels = np.array(states) - 1
    except Exception as e:
        print(f"Error during Viterbi decoding: {e}")
        return 0
        
    # Calculate accuracy
    accuracy = np.mean(pred_labels == true_labels)
    print(f"Mixed Sequence Accuracy: {accuracy*100:.2f}%")
    
    # Optional: Plot results for visual inspection
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(true_labels, 'b-', label='True States')
        plt.plot(pred_labels, 'r--', alpha=0.7, label='Predicted States')
        plt.yticks([0, 1], ['Still', 'Walking'])
        plt.xlabel('Time Sample')
        plt.ylabel('Activity State')
        plt.title('State Transition Tracking')
        plt.legend()
        plt.savefig('state_transition_comparison.png')
        print("Saved state transition plot to state_transition_comparison.png")
    except Exception as e:
        print(f"Could not generate plot: {e}")
    
    return accuracy

def main():
    pure_activities = ['Still', 'Walking', 'Running']
    mixed_activity = 'Mixed'
    
    # Load pure activity training data per activity
    activity_data = []  
    for activity in pure_activities:
        activity_dir = os.path.join('Data', 'Train', activity)
        seqs, files = load_sequences(activity_dir)
        print(f"  Loaded {len(seqs)} {activity} training files")
        activity_data.append(seqs)
    
    # Load mixed activity data
    mixed_dir = os.path.join('Data', 'Train', mixed_activity)
    mixed_sequences, mixed_files = load_sequences(mixed_dir)
    print(f"  Loaded {len(mixed_sequences)} mixed training files")
    
    # Combine all training data (for training, not initialization)
    train_sequences = []
    for seq_list in activity_data:
        train_sequences.extend(seq_list)
    train_sequences.extend(mixed_sequences)
    
    # Initialize HMM using activity-specific data (only pure activities)
    hmm = initialize_hmm(activity_data)
    
    # Train HMM
    train_hmm(hmm, train_sequences, n_iter=5)
    
    # Load test data for pure activities
    test_sequences = []
    test_labels = []
    for state_idx, activity in enumerate(pure_activities):
        activity_dir = os.path.join('Data', 'Test', activity)
        seqs, files = load_sequences(activity_dir)
        test_sequences.extend(seqs)
        test_labels.extend([state_idx] * len(seqs))
        print(f"  Loaded {len(seqs)} {activity} testing files")
    
    # Test on pure activities
    pure_acc = test_hmm(hmm, test_sequences, test_labels)
    
    # Test on mixed sequence
    mixed_dir = os.path.join('Data', 'Test', 'Mixed')
    mixed_sequence_path = os.path.join(mixed_dir, 'mixed_sequence.txt')
    mixed_labels_path = os.path.join(mixed_dir, 'mixed_sequence_labels.txt')
    
    if os.path.exists(mixed_sequence_path) and os.path.exists(mixed_labels_path):
        mixed_acc = test_mixed_sequence(hmm, mixed_sequence_path, mixed_labels_path)
        print(f"\nFinal Report:")
        print(f"  Pure Activity Accuracy: {pure_acc*100:.2f}%")
        print(f"  Mixed Sequence Accuracy: {mixed_acc*100:.2f}%")
    else:
        print("\nMixed sequence test files not found. Skipping mixed sequence test.")
        print(f"Final Pure Activity Accuracy: {pure_acc*100:.2f}%")

if __name__ == "__main__":
    main()