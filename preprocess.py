import os
import pandas as pd
import numpy as np
from math import ceil
import re

def preprocess_accelerometer_data(input_file, output_file, window_size=5, generate_labels=False):
    try:
        df = pd.read_csv(input_file, delimiter=',', header=None, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"Error reading {input_file}: {e}")
        return

    if df.empty:
        print(f"Empty DataFrame after reading {input_file}")
        return

    n_cols = df.shape[1]
    if n_cols == 5:
        if pd.api.types.is_numeric_dtype(df[1]):
            df = df.iloc[:, :4]
        else:
            df = df.iloc[:, [0, 2, 3, 4]]
        df.columns = ['timestamp', 'x', 'y', 'z']
    elif n_cols == 4:
        df.columns = ['timestamp', 'x', 'y', 'z']
    else:
        print(f"Unexpected columns ({n_cols}) in {input_file}, skipping")
        return

    # Convert to numeric and clean data
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['timestamp', 'x', 'y', 'z'])
    
    if df.empty:
        print(f"No valid data after cleaning in {input_file}")
        return

    # Apply smoothing
    for col in ['x', 'y', 'z']:
        df[col] = df[col].rolling(window=window_size, min_periods=1, center=True).mean().round(3)

    # Save processed data
    df.to_csv(output_file, sep='\t', index=False, header=False)
    
    # Generate labels ONLY if requested (for mixed test sequences)
    if generate_labels:
        labels = generate_heuristic_labels(df)
        label_file = output_file.replace('.txt', '_labels.txt')
        np.savetxt(label_file, labels, fmt='%d')
        print(f"  Generated heuristic labels: {os.path.basename(label_file)}")

def generate_heuristic_labels(df):
    """Generate heuristic labels for mixed sequences based on motion characteristics"""
    labels = []
    abs_acc = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    
    # Calculate moving statistics
    moving_avg = abs_acc.rolling(window=100, min_periods=1).mean()
    moving_std = abs_acc.rolling(window=100, min_periods=1).std()
    
    # Classify based on thresholds
    for i in range(len(df)):
        if moving_avg[i] < 1.0 and moving_std[i] < 0.2:
            labels.append(0)  # Still
        elif moving_avg[i] > 1.5 and moving_std[i] > 0.5:
            labels.append(2)  # Running
        else:
            labels.append(1)  # Walking
    return labels

def create_mixed_test_sequence(test_path, activities, segment_duration=30, sampling_rate=100):
    """Creates synthetic mixed sequence with ground truth labels"""
    print("\nCreating synthetic mixed test sequence with ground truth...")
    mixed_dir = os.path.join(test_path, "Mixed")
    os.makedirs(mixed_dir, exist_ok=True)
    
    mixed_sequence = pd.DataFrame(columns=['timestamp', 'x', 'y', 'z'])
    labels = []
    start_time = 0
    step = 1000 // sampling_rate
    
    # Activity pattern: Still → Walking → Running → Walking → Still
    pattern = [activities[0], activities[1], activities[2], activities[1], activities[0]]
    next_index = {activity: 0 for activity in activities}
    
    for activity in pattern:
        activity_dir = os.path.join(test_path, activity)
        files = sorted([f for f in os.listdir(activity_dir) 
                       if f.endswith('.txt') and not f.endswith('_labels.txt')])
        
        if not files:
            print(f"  No test files found for {activity}, skipping segment")
            continue
            
        # Rotate through files
        idx = next_index[activity] % len(files)
        file_path = os.path.join(activity_dir, files[idx])
        next_index[activity] += 1
        
        df = pd.read_csv(file_path, delimiter='\t', header=None, 
                         names=['timestamp', 'x', 'y', 'z'])
        
        # Take needed duration
        n_samples = segment_duration * sampling_rate
        segment = df.head(n_samples).copy()
        
        # Adjust timestamps
        segment['timestamp'] = np.arange(start_time, start_time + n_samples * step, step)
        start_time = segment['timestamp'].iloc[-1] + step
        
        # Append to sequence
        mixed_sequence = pd.concat([mixed_sequence, segment])
        
        # Create labels (0=Still, 1=Walking, 2=Running)
        label_value = activities.index(activity)
        labels.extend([label_value] * len(segment))
    
    # Save sequence and labels
    sequence_path = os.path.join(mixed_dir, "synthetic_mixed_sequence.txt")
    mixed_sequence.to_csv(sequence_path, sep='\t', index=False, header=False)
    
    labels_path = os.path.join(mixed_dir, "synthetic_mixed_sequence_labels.txt")
    np.savetxt(labels_path, labels, fmt='%d')
    
    print(f"  Created synthetic mixed sequence ({len(mixed_sequence)} samples)")
    print(f"  Ground truth labels: {os.path.basename(labels_path)}")
    return sequence_path, labels_path

def process_all_data(base_path="Data", window_size=5):
    raw_path = os.path.join(base_path, "Raw")
    train_path = os.path.join(base_path, "Train")
    test_path = os.path.join(base_path, "Test")
    
    activities = ['Still', 'Walking', 'Running']

    for class_folder in activities + ['Mixed']:
        class_folder_path = os.path.join(raw_path, class_folder)

        if not os.path.isdir(class_folder_path):
            print(f"Skipping missing folder: {class_folder_path}")
            continue

        files = sorted([
            f for f in os.listdir(class_folder_path)
            if os.path.isfile(os.path.join(class_folder_path, f))
            and not f.endswith('_labels.txt') 
        ])

        if not files:
            print(f"No data files found in {class_folder_path}")
            continue

        split_index = ceil(len(files) * 0.8)
        train_files = files[:split_index]
        test_files = files[split_index:]

        train_output_dir = os.path.join(train_path, class_folder)
        test_output_dir = os.path.join(test_path, class_folder)
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)

        print(f"\nProcessing {class_folder}:")
        print(f"  {len(train_files)} train files, {len(test_files)} test files")

        # Process train files - NO LABELS for any type
        for f in train_files:
            input_file = os.path.join(class_folder_path, f)
            base_name = os.path.splitext(f)[0]
            output_file = os.path.join(train_output_dir, base_name + '.txt')
            preprocess_accelerometer_data(input_file, output_file, window_size)

        # Process test files
        for f in test_files:
            input_file = os.path.join(class_folder_path, f)
            base_name = os.path.splitext(f)[0]
            output_file = os.path.join(test_output_dir, base_name + '.txt')
            
            # Generate labels ONLY for mixed test sequences
            should_generate = (class_folder == 'Mixed')
            preprocess_accelerometer_data(
                input_file, 
                output_file, 
                window_size,
                generate_labels=should_generate
            )
    
    # Create synthetic mixed sequence with ground truth labels
    create_mixed_test_sequence(test_path, activities)

if __name__ == "__main__":
    process_all_data()