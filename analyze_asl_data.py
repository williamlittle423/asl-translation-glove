import os
import numpy as np
import matplotlib.pyplot as plt
import string

# Define constants
LETTERS = list(string.ascii_uppercase[:10])  # A-J
NAMES = ['will', 'eric']
DATA_DIR = 'datasets'
SENSOR = 'Index'  # Selected sensor
AXIS = 'Accel_X'   # Selected axis

# Initialize a dictionary to hold Accel_X data for Index sensor per letter
accel_x_data = {letter: [] for letter in LETTERS}

# Function to load and extract Accel_X data for Index sensor
def load_accel_x_data():
    for name in NAMES:
        for letter in LETTERS:
            file_path = os.path.join(DATA_DIR, f'asl_data_{letter}_{name}.npy')
            if not os.path.isfile(file_path):
                print(f"Warning: File {file_path} does not exist. Skipping.")
                continue
            try:
                data = np.load(file_path)  # Shape: (N_training, N_v_max)
            except Exception as e:
                print(f"Error loading {file_path}: {e}. Skipping.")
                continue
            for sample_idx, sample in enumerate(data):
                # Reshape the sample to (32 times, 5 sensors, 6 axes)
                try:
                    reshaped = sample.reshape(32, 5, 6)
                except ValueError:
                    print(f"Warning: Sample {sample_idx} in {file_path} cannot be reshaped. Skipping this sample.")
                    continue
                # Sum over the 32 times for each sensor and axis is already done
                # Since the data saved is the sum, we can directly extract it
                # Find the index for the selected sensor and axis
                sensor_idx = SENSORS.index(SENSOR)  # Assuming SENSORS list is defined
                AXES = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
                axis_idx = AXES.index(AXIS)
                sum_value = reshaped[:, sensor_idx, axis_idx].sum()
                accel_x_data[letter].append(sum_value)

# Alternatively, based on the data collection script, the saved data is the sum already.
# Hence, we need to extract the specific feature from the saved data.

def load_accel_x_data_correctly():
    # According to the data collection script, each saved .npy file has shape (N_letters, N_training, N_v_max)
    # But in previous assistant scripts, it might have been saved as (N_training, N_v_max) per letter per file
    # So adjust accordingly
    for name in NAMES:
        for letter in LETTERS:
            file_path = os.path.join(DATA_DIR, f'asl_data_{letter}_{name}.npy')
            if not os.path.isfile(file_path):
                print(f"Warning: File {file_path} does not exist. Skipping.")
                continue
            try:
                data = np.load(file_path)  # Shape: (N_training, N_v_max)
            except Exception as e:
                print(f"Error loading {file_path}: {e}. Skipping.")
                continue
            for sample_idx, sample in enumerate(data):
                # Reshape the sample to (32 times, 5 sensors, 6 axes)
                try:
                    reshaped = sample.reshape(32, 5, 6)
                except ValueError:
                    print(f"Warning: Sample {sample_idx} in {file_path} cannot be reshaped. Skipping this sample.")
                    continue
                # Sum over the 32 times for each sensor and axis is already done during data collection
                # So here, 'sample' is already the summed data across 32 times
                # Extract the Accel_X for Index sensor
                SENSOR_IDX = 1  # Index sensor (Thumb=0, Index=1, Middle=2, Ring=3, Pinky=4)
                AXIS_IDX = 0    # Accel_X is first axis
                sum_value = reshaped[:, SENSOR_IDX, AXIS_IDX].sum(axis=0)  # Sum over the 32 times
                # However, since in data collection, the sum over 32 times is already stored in data
                # The script above might be double summing
                # To correctly extract, assume data is already summed
                sum_value = sample[SENSOR_IDX * 6 + AXIS_IDX]
                accel_x_data[letter].append(sum_value)

# Correct data loading based on data collection script
def load_accel_x_data_final():
    # Each .npy file is saved per letter and name, shape (N_training, N_v_max)
    # N_v_max = 32 * 5 * 6 = 960
    # To extract Accel_X for Index sensor:
    # Index sensor is sensor 1 (0-based)
    # Accel_X is axis 0
    # So position = sensor_idx * 6 + axis_idx = 1 * 6 + 0 = 6
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
                # Extract the Accel_X for Index sensor
                SENSOR_IDX = 1  # Index sensor
                AXIS_IDX = 0    # Accel_X
                position = SENSOR_IDX * 6 + AXIS_IDX  # 1*6 + 0 = 6
                if position >= len(sample):
                    print(f"Warning: Position {position} out of bounds for sample {sample_idx} in {file_path}. Skipping.")
                    continue
                sum_value = sample[position]
                accel_x_data[letter].append(sum_value)

# Load the data
load_accel_x_data_final()

# Plotting the histograms
def plot_histograms(accel_x_data, sensor, axis):
    num_letters = len(LETTERS)
    cols = 5
    rows = (num_letters + cols - 1) // cols  # Ceiling division
    plt.figure(figsize=(20, 10))
    plt.suptitle(f'Histogram of {sensor} Sensor - {axis} Across Letters A-J', fontsize=16)
    
    for idx, letter in enumerate(LETTERS):
        plt.subplot(rows, cols, idx + 1)
        data = accel_x_data[letter]
        if not data:
            plt.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
            plt.title(f'Letter {letter}')
            plt.xlabel('Sum of Accel_X')
            plt.ylabel('Frequency')
            continue
        plt.hist(data, bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Letter {letter}')
        plt.xlabel('Sum of Accel_X')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Optionally, save the figure
    plt.savefig('histograms_index_accel_x.png')
    print('Histograms saved to histograms_index_accel_x.png')
    plt.show()

# Define sensor and axis
SELECTED_SENSOR = 'Index'
SELECTED_AXIS = 'Accel_X'

# Plot the histograms
plot_histograms(accel_x_data, SELECTED_SENSOR, SELECTED_AXIS)
