import os
import pandas as pd
from math import ceil

def preprocess_accelerometer_data(input_file, output_file, window_size=5):
    try:
        df = pd.read_csv(
            input_file,
            delimiter='\t',
            header=None,
            names=['timestamp', 'type', 'x', 'y', 'z'],
            on_bad_lines='skip'  # Skip malformed lines
        )
    except pd.errors.ParserError as e:
        print(f"Error reading {input_file}: {e}")
        return

    df = df.drop(columns=['type'])

    for col in ['x', 'y', 'z']:
        df[col] = df[col].rolling(window=window_size, min_periods=1, center=True).mean().round(3)

    df.to_csv(output_file, sep='\t', index=False, header=False)

def process_all_data(base_path="Data", window_size=5):
    '''
        Preprocess the data using a simple moving average and round the values to 3 decimals to avoid overinterpretation.
        It salso splits the data with a 80-20 ration between train and test files.
        
        It assumes a folder structure:
        preprocess.py
        Data
            Raw
                Still
                Walking
                Running
            Test
                Still
                Walking
                Running
            Train
                Still
                Walking
                Running
    '''
    raw_path = os.path.join(base_path, "Raw")
    train_path = os.path.join(base_path, "Train")
    test_path = os.path.join(base_path, "Test")

    for class_folder in os.listdir(raw_path):
        class_folder_path = os.path.join(raw_path, class_folder)

        if not os.path.isdir(class_folder_path):
            continue

        files = sorted([
            f for f in os.listdir(class_folder_path)
            if os.path.isfile(os.path.join(class_folder_path, f))
        ])

        split_index = ceil(len(files) * 0.8)

        train_files = files[:split_index]
        test_files = files[split_index:]

        train_output_dir = os.path.join(train_path, class_folder)
        test_output_dir = os.path.join(test_path, class_folder)
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)

        for f in train_files:
            input_file = os.path.join(class_folder_path, f)
            output_file = os.path.join(train_output_dir, f)
            preprocess_accelerometer_data(input_file, output_file, window_size)

        for f in test_files:
            input_file = os.path.join(class_folder_path, f)
            output_file = os.path.join(test_output_dir, f)
            preprocess_accelerometer_data(input_file, output_file, window_size)

if __name__ == "__main__":
    process_all_data()
