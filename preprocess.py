import os
import argparse
import pandas as pd

def preprocess_accelerometer_data(input_file, output_file, window_size=5):
    try:
        df = pd.read_csv(
            input_file,
            delimiter='\t',
            header=None,
            names=['timestamp', 'type', 'x', 'y', 'z'],
            on_bad_lines='skip'
        )
    except pd.errors.ParserError as e:
        print(f"Error reading {input_file}: {e}")
        return

    # Remove the 'type' column
    df = df.drop(columns=['type'])

    # Apply moving average filter and round to 3 decimal places
    for col in ['x', 'y', 'z']:
        df[col] = df[col].rolling(window=window_size, min_periods=1, center=True).mean().round(3)

    # Save the processed data
    df.to_csv(output_file, sep='\t', index=False, header=False)


def preprocess_folder(input_folder, output_folder, window_size=5):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)

        if os.path.isfile(input_file_path):
            preprocess_accelerometer_data(input_file_path, output_file_path, window_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process accelerometer data files.")
    parser.add_argument("input_folder", help="Path to the folder containing raw data files.")
    parser.add_argument("output_folder", help="Path to the folder where processed files will be saved.")
    parser.add_argument("--window_size", type=int, default=5, help="Window size for moving average (default: 5)")

    args = parser.parse_args()

    preprocess_folder(args.input_folder, args.output_folder, args.window_size)
