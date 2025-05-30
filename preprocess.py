import os
import pandas as pd
import numpy as np
from math import ceil

def preprocess_accelerometer_data(input_file, output_file, window_size=5):
    try:
        # Read raw data with flexible column handling
        df = pd.read_csv(
            input_file,
            delimiter='\t',
            header=None,
            on_bad_lines='skip'
        )
    except pd.errors.ParserError as e:
        print(f"Error reading {input_file}: {e}")
        return

    if df.empty:
        print(f"Empty DataFrame after reading {input_file}")
        return

    n_cols = df.shape[1]

    # Handle different input formats
    if n_cols == 5:
        if pd.api.types.is_numeric_dtype(df[1]):
            # New format: timestamp,x,y,z,abs_acc → keep first 4 columns
            df = df.iloc[:, :4]
        else:
            # Original format: timestamp,type,x,y,z → keep timestamp,x,y,z
            df = df.iloc[:, [0, 2, 3, 4]]
        df.columns = ['timestamp', 'x', 'y', 'z']
    elif n_cols == 4:
        # Already in target format
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
        df[col] = df[col].rolling(
            window=window_size,
            min_periods=1,
            center=True
        ).mean().round(3)

    # Save processed data
    df.to_csv(output_file, sep='\t', index=False, header=False)

# (Rest of the code remains unchanged below this point)

def create_mixed_test_sequence(test_path, activities, segment_duration=30, sampling_rate=100):
    """
    Creates a continuous test sequence with state transitions and ground truth labels
    """
    print("\nCreating mixed test sequence with state transitions...")
    mixed_dir = os.path.join(test_path, "Mixed")
    os.makedirs(mixed_dir, exist_ok=True)
    
    # Initialize empty DataFrames for sequence and labels
    mixed_sequence = pd.DataFrame(columns=['timestamp', 'x', 'y', 'z'])
    labels = []
    
    start_time = 0
    step = 1000 // sampling_rate  # Time step in milliseconds
    
    # Create activity pattern: Still → Walking → Running → Walking → Still
    pattern = [activities[0], activities[1], activities[2], activities[1], activities[0]]
    
    # Track next file index for each activity
    next_index = {activity: 0 for activity in activities}
    
    # Build the sequence segment by segment
    for activity in pattern:
        activity_dir = os.path.join(test_path, activity)
        # Look for TXT files instead of CSV
        files = sorted([f for f in os.listdir(activity_dir) if f.endswith('.txt')])
        
        if not files:
            print(f"  No test files found for {activity}, skipping segment")
            continue
            
        # Get next file in rotation for this activity
        idx = next_index[activity] % len(files)
        file_path = os.path.join(activity_dir, files[idx])
        next_index[activity] += 1  # Move to next file for next occurrence
        
        df = pd.read_csv(file_path, delimiter='\t', header=None, 
                         names=['timestamp', 'x', 'y', 'z'])
        
        # Take only the needed duration
        n_samples = segment_duration * sampling_rate
        segment = df.head(n_samples).copy()
        
        # Adjust timestamps for continuity
        segment['timestamp'] = np.arange(start_time, start_time + n_samples * step, step)
        start_time = segment['timestamp'].iloc[-1] + step
        
        # Append to mixed sequence
        mixed_sequence = pd.concat([mixed_sequence, segment])
        
        # Create labels (0=Still, 1=Walking, 2=Running)
        label_value = activities.index(activity)
        labels.extend([label_value] * len(segment))
    
    sequence_path = os.path.join(mixed_dir, "mixed_sequence.txt")
    mixed_sequence.to_csv(sequence_path, sep='\t', index=False, header=False)
    print(f"  Saved mixed sequence to {sequence_path} ({len(mixed_sequence)} samples)")
    
    labels_path = os.path.join(mixed_dir, "mixed_sequence_labels.txt")
    np.savetxt(labels_path, labels, fmt='%d')
    print(f"  Saved ground truth labels to {labels_path}")
    
    return sequence_path, labels_path

def process_all_data(base_path="Data", window_size=5):
    raw_path = os.path.join(base_path, "Raw")
    train_path = os.path.join(base_path, "Train")
    test_path = os.path.join(base_path, "Test")
    
    activities = ['Still', 'Walking', 'Running']

    # First process all pure activities
    for class_folder in activities + ['Mixed']:
        class_folder_path = os.path.join(raw_path, class_folder)

        if not os.path.isdir(class_folder_path):
            print(f"Skipping missing folder: {class_folder_path}")
            continue

        files = sorted([
            f for f in os.listdir(class_folder_path)
            if os.path.isfile(os.path.join(class_folder_path, f))
        ])

        if not files:
            print(f"No files found in {class_folder_path}")
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

        # Convert to TXT files during preprocessing
        for f in train_files:
            input_file = os.path.join(class_folder_path, f)
            base_name = os.path.splitext(f)[0]  # Remove original extension
            output_file = os.path.join(train_output_dir, base_name + '.txt')
            preprocess_accelerometer_data(input_file, output_file, window_size)

        for f in test_files:
            input_file = os.path.join(class_folder_path, f)
            base_name = os.path.splitext(f)[0]  # Remove original extension
            output_file = os.path.join(test_output_dir, base_name + '.txt')
            preprocess_accelerometer_data(input_file, output_file, window_size)
    
    create_mixed_test_sequence(test_path, activities)

if __name__ == "__main__":
    process_all_data()