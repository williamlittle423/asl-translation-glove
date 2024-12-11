import os
import numpy as np
import matplotlib.pyplot as plt
import string

# Define constants
LETTERS = list(string.ascii_uppercase[:10])  # Letters A-J
NAMES = ['will', 'eric']
DATA_DIR = 'datasets'  # Directory where .npy files are stored
SENSOR = 'Index'       # Selected finger: 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky'
AXES = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']  # All six axes

# Define sensor list for index mapping
SENSORS = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

# Initialize a dictionary to hold data for each axis and letter
# Structure: data_per_axis[axis][letter] = list of sum_values
data_per_axis = {axis: {letter: [] for letter in LETTERS} for axis in AXES}

# Function to load and extract data for the selected finger and all axes
def load_sensor_data():
    for name in NAMES:
        for letter in LETTERS:
            file_path = os.path.join(DATA_DIR, f'asl_data_{letter}_{name}.npy')
            if not os.path.isfile(file_path):
                print(f"Warning: File {file_path} does not exist. Skipping.")
                continue
            try:
                data = np.load(file_path)  # Shape: (N_training, 960)
            except Exception as e:
                print(f"Error loading {file_path}: {e}. Skipping.")
                continue
            for sample_idx, sample in enumerate(data):
                # Ensure the sample has the expected number of data points
                if sample.shape[0] != 960:
                    print(f"Warning: Sample {sample_idx} in {file_path} has incorrect shape {sample.shape}. Skipping.")
                    continue
                # Extract sum_values for each axis
                for axis in AXES:
                    sensor_idx = SENSORS.index(SENSOR)  # e.g., 'Index' -> 1
                    axis_idx = AXES.index(axis)          # e.g., 'Accel_X' -> 0
                    position = sensor_idx * 6 + axis_idx # Calculate position
                    if position >= len(sample):
                        print(f"Warning: Position {position} out of bounds for sample {sample_idx} in {file_path}. Skipping this axis.")
                        continue
                    sum_value = sample[position]
                    data_per_axis[axis][letter].append(sum_value)

# Function to plot and save histograms for each axis
def plot_and_save_histograms():
    # Create a directory to save histograms if it doesn't exist
    output_dir = 'histograms'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define colors for each letter for consistency across histograms
    colors = plt.cm.get_cmap('tab10', len(LETTERS))
    
    for axis in AXES:
        plt.figure(figsize=(10, 6))
        for idx, letter in enumerate(LETTERS):
            data = data_per_axis[axis][letter]
            if not data:
                print(f"No data available for Letter '{letter}' - Axis '{axis}'. Skipping.")
                continue
            plt.hist(data, bins=20, alpha=0.5, label=letter, color=colors(idx))
        
        plt.title(f'Distribution of {SENSOR} Sensor - {axis}')
        plt.xlabel(f'{axis} Sum over 32 Data Points')
        plt.ylabel('Frequency')
        plt.legend(title='Letter')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the histogram as a PNG file
        filename = f'histogram_{SENSOR}_{axis}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        print(f"Saved histogram for {axis} at {filepath}")
        plt.close()

# Main execution
if __name__ == "__main__":
    print(f"Loading data for {SENSOR} sensor across all axes...")
    load_sensor_data()
    print("Data loading complete.")
    
    print("Generating and saving histograms...")
    plot_and_save_histograms()
    print("All histograms have been saved successfully.")
