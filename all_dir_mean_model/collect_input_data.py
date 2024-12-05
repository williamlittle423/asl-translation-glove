import smbus2
import time
import numpy as np
import argparse
import math
import pandas as pd
import sys
import os

# Step 1: Create the parser
parser = argparse.ArgumentParser(description="ASL training script with enhanced error handling and velocity features.")

# Step 2: Add arguments
parser.add_argument("--name", type=str, help="Name of person collecting data", default='noname')
parser.add_argument("--labels", type=str, nargs='+', help="List of labels to collect data for (e.g., A B C D or milk want)", default=['A', 'B', 'C', 'D'])
parser.add_argument("--training_samples", type=int, help="Number of training samples per label", default=40)
parser.add_argument("--time_steps", type=int, help="Number of time steps per sample", default=32)
parser.add_argument("--retry_limit", type=int, help="Maximum number of retries for I2C communication", default=20)
parser.add_argument("--start_label", type=str, help="Label to start data collection from", default=None)

# Step 3: Parse arguments
args = parser.parse_args()

# Validate start_label if provided
if args.start_label and args.start_label not in args.labels:
    print(f"Error: Start label '{args.start_label}' is not in the list of labels {args.labels}.")
    sys.exit(1)

# Initialize I2C bus
try:
    bus = smbus2.SMBus(1)
except Exception as e:
    print(f"Failed to initialize I2C bus: {e}")
    sys.exit(1)

# I2C addresses
TCA9548A_ADDRESS = 0x70
MPU6050_ADDRESS = 0x68

# CSV file path
CSV_FILENAME = 'asl_data_WILL.csv'

# Function to select the channel on the multiplexer with retry mechanism
def select_channel(channel, retry_count=0):
    if channel < 0 or channel > 7:
        raise ValueError("Invalid channel. Must be between 0 and 7.")
    try:
        bus.write_byte(TCA9548A_ADDRESS, 1 << channel)
        return True
    except OSError as e:
        if e.errno in [121, 5, 110]:  # Remote I/O error, Input/output error, Connection timed out
            if retry_count < args.retry_limit:
                print(f"OSError {e.errno}: {e.strerror} while selecting channel {channel}. Retrying ({retry_count + 1}/{args.retry_limit})...")
                time.sleep(0.1)  # Wait before retrying
                return select_channel(channel, retry_count + 1)
            else:
                print(f"Failed to select channel {channel} after {args.retry_limit} retries.")
                return False
        else:
            print(f"Unexpected OSError: {e}")
            return False

# Function to initialize the MPU6050 sensor
def initialize_mpu6050():
    try:
        # Write to power management register to wake up the MPU6050
        bus.write_byte_data(MPU6050_ADDRESS, 0x6B, 0x00)
    except OSError as e:
        print(f"OSError {e.errno}: Error initializing MPU6050.")
        return False
    except Exception as e:
        print(f"Unexpected error initializing MPU6050: {e}")
        return False
    return True

# Function to read two bytes of data and convert it to a signed value
def read_word_2c(addr, reg):
    try:
        high = bus.read_byte_data(addr, reg)
        low = bus.read_byte_data(addr, reg + 1)
    except OSError as e:
        print(f"OSError {e.errno}: Error reading from address {addr}, register {reg}.")
        return None
    except Exception as e:
        print(f"Unexpected error reading from address {addr}, register {reg}: {e}")
        return None
    value = (high << 8) + low
    if value >= 0x8000:
        return -((65535 - value) + 1)
    else:
        return value

# Function to read raw data from MPU6050
def read_mpu6050():
    try:
        # Read accelerometer and gyroscope data from MPU6050
        accel_x = read_word_2c(MPU6050_ADDRESS, 0x3B)
        accel_y = read_word_2c(MPU6050_ADDRESS, 0x3D)
        accel_z = read_word_2c(MPU6050_ADDRESS, 0x3F)
        gyro_x = read_word_2c(MPU6050_ADDRESS, 0x43)
        gyro_y = read_word_2c(MPU6050_ADDRESS, 0x45)
        gyro_z = read_word_2c(MPU6050_ADDRESS, 0x47)
    except Exception as e:
        print(f"Unexpected error reading MPU6050 data: {e}")
        return None
    # Check if any of the readings failed
    if None in [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]:
        return None
    return {
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z
    }

def read_channel(channel, previous_data, retry_count=0):
    if not select_channel(channel, retry_count):
        print(f"Skipping sensor {channel} due to channel selection failure.")
        return None, None
    if not initialize_mpu6050():
        print(f"Skipping sensor {channel} due to MPU6050 initialization failure.")
        return None, None
    data = read_mpu6050()
    if data is None:
        print(f"Skipping sensor {channel} due to data read failure.")
        return None, None
    if previous_data is not None:
        delta = {key: data[key] - previous_data[key] for key in data}
    else:
        delta = {key: 0 for key in data}  # For the first read, use zero delta
    return delta, data

# Map labels to indices
labels = args.labels

print(f'Collecting data for American Sign Language (ASL) labels: {labels}')

# Prepare dataset list
# dataset = []  # Removed as we're writing incrementally

# Parameters
N_labels = len(labels)
N_training = args.training_samples  # Number of samples per label
N_time_steps = args.time_steps      # Number of time steps per sample
N_v_max = N_time_steps * 5 * 6      # time_steps * 5 sensors * 6 data points (delta values)
TIME_DELTA = 0.02                   # 20 ms between reads

# Maximum number of retries for a sample
MAX_RETRIES = args.retry_limit

def collect_single_label(label, input_idx, retry_count=0):
    if retry_count >= MAX_RETRIES:
        print(f"Max retries exceeded for label '{label}' ({input_idx+1}/{N_training}). Skipping this sample.")
        return None
    input(f"Press Enter and perform the ASL movement for label '{label}' ({input_idx+1}/{N_training})...")
    collected_data = []
    base_data_list = [None]*5  # For each sensor

    # Step 1: Collect base data from all sensors
    print("Collecting base data from all sensors...")
    for sensor_idx in range(5):  # Sensors 0 to 4
        delta, data = read_channel(sensor_idx, None, retry_count=0)
        if delta is None:
            print('Attempting label again due to sensor read error during base data collection')
            return collect_single_label(label, input_idx, retry_count=retry_count+1)
        base_data_list[sensor_idx] = data
    print("Base data collection complete.")

    previous_data_list = base_data_list.copy()  # Initialize previous_data with base data

    # Step 2: Collect delta data during the sample period
    collected_time_steps = 0
    while collected_time_steps < N_time_steps:
        for sensor_idx in range(5):  # Sensors 0 to 4
            delta, data = read_channel(sensor_idx, previous_data_list[sensor_idx], retry_count=0)
            if delta is None:
                print('Attempting label again due to sensor read error during delta collection')
                return collect_single_label(label, input_idx, retry_count=retry_count+1)
            previous_data_list[sensor_idx] = data
            # Collect delta data
            delta_values = list(delta.values())  # 6 data points
            collected_data.extend(delta_values)
        collected_time_steps += 1
        time.sleep(TIME_DELTA)  # Wait for the next time step
    #print(f"Number of times data was read from all five sensors: {collected_time_steps}")
    return collected_data

# Initialize table header and write to CSV if not exists
header = ["Label", "Input"]
for sensor in range(1, 6):
    header.extend([
        f"S{sensor}_MeanAccelX",
        f"S{sensor}_StdAccelX",
        f"S{sensor}_MeanAccelY",
        f"S{sensor}_StdAccelY",
        f"S{sensor}_MeanAccelZ",
        f"S{sensor}_StdAccelZ",
        f"S{sensor}_MeanGyroX",
        f"S{sensor}_StdGyroX",
        f"S{sensor}_MeanGyroY",
        f"S{sensor}_StdGyroY",
        f"S{sensor}_MeanGyroZ",
        f"S{sensor}_StdGyroZ"
    ])

# Check if CSV file exists
file_exists = os.path.isfile(CSV_FILENAME)

# If starting at a specific label, find its index
start_idx = 0
if args.start_label:
    try:
        start_idx = labels.index(args.start_label)
    except ValueError:
        print(f"Error: Start label '{args.start_label}' not found in labels list.")
        sys.exit(1)

print("\nData Collection Summary:")
print(" | ".join(f"{h:<20}" for h in header))
print("-" * (20 * len(header) + (3 * (len(header)-1))))

try:
    for label_idx, label in enumerate(labels[start_idx:], start=start_idx):
        print(f"\nCollecting data for label '{label}'")
        for input_idx in range(N_training):
            collected_data = collect_single_label(label, input_idx)
            if collected_data is None:
                print(f"Skipping sample {input_idx+1}/{N_training} for label '{label}' due to repeated errors.")
                continue
            # Ensure we have exactly N_v_max data points
            collected_length = len(collected_data)
            if collected_length > N_v_max:
                print(f"Warning: Collected {collected_length} data points exceeds maximum of {N_v_max}. Truncating data.")
                collected_data = collected_data[:N_v_max]
            elif collected_length < N_v_max:
                print(f"Warning: Collected {collected_length} data points is less than maximum of {N_v_max}. Padding data.")
                collected_data.extend([0]*(N_v_max - collected_length))
            
            # Initialize per-sensor delta lists for each axis
            accel_delta_lists = [ {'x': [], 'y': [], 'z': []} for _ in range(5)]
            gyro_delta_lists = [ {'x': [], 'y': [], 'z': []} for _ in range(5)]

            # Process collected_data per sensor
            # Each time step adds 5 sensors * 6 delta values
            for read_idx in range(N_time_steps):
                for sensor in range(5):
                    base_idx = (read_idx * 5 * 6) + (sensor * 6)
                    accel_x = collected_data[base_idx + 0]
                    accel_y = collected_data[base_idx + 1]
                    accel_z = collected_data[base_idx + 2]
                    gyro_x = collected_data[base_idx + 3]
                    gyro_y = collected_data[base_idx + 4]
                    gyro_z = collected_data[base_idx + 5]

                    # Append delta values to corresponding lists
                    accel_delta_lists[sensor]['x'].append(accel_x)
                    accel_delta_lists[sensor]['y'].append(accel_y)
                    accel_delta_lists[sensor]['z'].append(accel_z)

                    gyro_delta_lists[sensor]['x'].append(gyro_x)
                    gyro_delta_lists[sensor]['y'].append(gyro_y)
                    gyro_delta_lists[sensor]['z'].append(gyro_z)

            # Calculate mean and std for each axis per sensor
            mean_accel = []
            std_accel = []
            mean_gyro = []
            std_gyro = []

            for sensor in range(5):
                # Accelerometer
                mean_accel_x = np.mean(accel_delta_lists[sensor]['x']) if accel_delta_lists[sensor]['x'] else 0.0
                std_accel_x = np.std(accel_delta_lists[sensor]['x']) if accel_delta_lists[sensor]['x'] else 0.0

                mean_accel_y = np.mean(accel_delta_lists[sensor]['y']) if accel_delta_lists[sensor]['y'] else 0.0
                std_accel_y = np.std(accel_delta_lists[sensor]['y']) if accel_delta_lists[sensor]['y'] else 0.0

                mean_accel_z = np.mean(accel_delta_lists[sensor]['z']) if accel_delta_lists[sensor]['z'] else 0.0
                std_accel_z = np.std(accel_delta_lists[sensor]['z']) if accel_delta_lists[sensor]['z'] else 0.0

                mean_accel.extend([mean_accel_x, std_accel_x, mean_accel_y, std_accel_y, mean_accel_z, std_accel_z])

                # Gyroscope
                mean_gyro_x = np.mean(gyro_delta_lists[sensor]['x']) if gyro_delta_lists[sensor]['x'] else 0.0
                std_gyro_x = np.std(gyro_delta_lists[sensor]['x']) if gyro_delta_lists[sensor]['x'] else 0.0

                mean_gyro_y = np.mean(gyro_delta_lists[sensor]['y']) if gyro_delta_lists[sensor]['y'] else 0.0
                std_gyro_y = np.std(gyro_delta_lists[sensor]['y']) if gyro_delta_lists[sensor]['y'] else 0.0

                mean_gyro_z = np.mean(gyro_delta_lists[sensor]['z']) if gyro_delta_lists[sensor]['z'] else 0.0
                std_gyro_z = np.std(gyro_delta_lists[sensor]['z']) if gyro_delta_lists[sensor]['z'] else 0.0

                mean_gyro.extend([mean_gyro_x, std_gyro_x, mean_gyro_y, std_gyro_y, mean_gyro_z, std_gyro_z])

            # Prepare the row data
            row = [label, input_idx + 1]
            for sensor in range(5):
                row.extend([
                    np.mean(accel_delta_lists[sensor]['x']) if accel_delta_lists[sensor]['x'] else 0.0,
                    np.std(accel_delta_lists[sensor]['x']) if accel_delta_lists[sensor]['x'] else 0.0,
                    np.mean(accel_delta_lists[sensor]['y']) if accel_delta_lists[sensor]['y'] else 0.0,
                    np.std(accel_delta_lists[sensor]['y']) if accel_delta_lists[sensor]['y'] else 0.0,
                    np.mean(accel_delta_lists[sensor]['z']) if accel_delta_lists[sensor]['z'] else 0.0,
                    np.std(accel_delta_lists[sensor]['z']) if accel_delta_lists[sensor]['z'] else 0.0,
                    np.mean(gyro_delta_lists[sensor]['x']) if gyro_delta_lists[sensor]['x'] else 0.0,
                    np.std(gyro_delta_lists[sensor]['x']) if gyro_delta_lists[sensor]['x'] else 0.0,
                    np.mean(gyro_delta_lists[sensor]['y']) if gyro_delta_lists[sensor]['y'] else 0.0,
                    np.std(gyro_delta_lists[sensor]['y']) if gyro_delta_lists[sensor]['y'] else 0.0,
                    np.mean(gyro_delta_lists[sensor]['z']) if gyro_delta_lists[sensor]['z'] else 0.0,
                    np.std(gyro_delta_lists[sensor]['z']) if gyro_delta_lists[sensor]['z'] else 0.0
                ])

            # Create a DataFrame row
            df_row = pd.DataFrame([row], columns=header)

            # Append to CSV
            if not file_exists:
                df_row.to_csv(CSV_FILENAME, index=False)
                file_exists = True  # Update the flag after creating the file
                print(f"Created new CSV file: {CSV_FILENAME}")
            else:
                df_row.to_csv(CSV_FILENAME, mode='a', header=False, index=False)

            # Prepare and print the table row
            table_row = []
            for item in row:
                if isinstance(item, float):
                    table_row.append(f"{item:<20.2f}")
                else:
                    table_row.append(f"{item:<20}")
            print(" | ".join(table_row))

except KeyboardInterrupt:
    print("\nProgram terminated by user.")
finally:
    bus.close()
    print("\nData collection complete.")
