import smbus2
import time
import numpy as np
import string
import argparse
import math
import pandas as pd  # Added for data handling and saving
import sys

# Step 1: Create the parser
parser = argparse.ArgumentParser(description="ASL training script with enhanced error handling and velocity features.")

# Step 2: Add arguments
parser.add_argument("--name", type=str, help="Name of person collecting data", default='noname')
parser.add_argument("--labels", type=str, nargs='+', help="List of labels to collect data for (e.g., A B C D or milk want)", default=['A', 'B', 'C', 'D'])
parser.add_argument("--training_samples", type=int, help="Number of training samples per label", default=40)
parser.add_argument("--time_steps", type=int, help="Number of time steps per sample", default=32)
parser.add_argument("--retry_limit", type=int, help="Maximum number of retries for I2C communication", default=5)

# Step 3: Parse arguments
args = parser.parse_args()

# Initialize I2C bus
try:
    bus = smbus2.SMBus(1)
except Exception as e:
    print(f"Failed to initialize I2C bus: {e}")
    sys.exit(1)

# I2C addresses
TCA9548A_ADDRESS = 0x70
MPU6050_ADDRESS = 0x68

# Function to select the channel on the multiplexer with retry mechanism
def select_channel(channel, retry_count=0):
    if channel < 0 or channel > 7:
        raise ValueError("Invalid channel. Must be between 0 and 7.")
    try:
        bus.write_byte(TCA9548A_ADDRESS, 1 << channel)
        return True
    except OSError as e:
        if e.errno == 121:  # Remote I/O error
            if retry_count < args.retry_limit:
                print(f"OSError 121: Remote I/O error while selecting channel {channel}. Retrying ({retry_count + 1}/{args.retry_limit})...")
                time.sleep(0.1)  # Wait before retrying
                return select_channel(channel, retry_count + 1)
            else:
                print(f"Failed to select channel {channel} after {args.retry_limit} retries.")
                return False
        elif e.errno == 5:  # Input/output error
            if retry_count < args.retry_limit:
                print(f"OSError 5: Input/output error while selecting channel {channel}. Retrying ({retry_count + 1}/{args.retry_limit})...")
                time.sleep(0.1)
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
        delta = {key: 0 for key in data}  # For the first read, use raw data as delta
    return delta, data

# Map labels to indices
labels = args.labels

print(f'Collecting data for American Sign Language (ASL) labels: {labels}')

# Prepare dataset list
dataset = []

# Parameters
N_labels = len(labels)
N_training = args.training_samples  # Number of samples per label
N_time_steps = args.time_steps      # Number of time steps per sample
N_v_max = N_time_steps * 5 * 6      # time_steps * 5 sensors * 6 data points (delta values)
TIME_DELTA = 0.02                     # 20 ms between reads

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

# Initialize table header
print("\nData Collection Summary:")
header = ["Label", "Input"]
for sensor in range(1, 6):
    header.extend([
        f"S{sensor}_MeanAccelMag",
        f"S{sensor}_StdAccelMag",
        f"S{sensor}_MeanGyroMag",
        f"S{sensor}_StdGyroMag",
        f"S{sensor}_MeanVelMag",
        f"S{sensor}_StdVelMag"
    ])
print(" | ".join(f"{h:<20}" for h in header))
print("-" * (20 * len(header) + (3 * (len(header)-1))))

try:
    for label_idx, label in enumerate(labels):
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
            
            # Initialize per-sensor magnitude lists
            accel_mag_lists = [[] for _ in range(5)]
            gyro_mag_lists = [[] for _ in range(5)]
            vel_mag_lists = [[] for _ in range(5)]
            current_velocity = [0.0 for _ in range(5)]  # Initialize velocity for each sensor
            
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
                    
                    # Calculate magnitudes
                    accel_mag = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
                    gyro_mag = math.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
                    
                    accel_mag_lists[sensor].append(accel_mag)
                    gyro_mag_lists[sensor].append(gyro_mag)
                    
                    # Update velocity using numerical integration (Euler method)
                    # velocity += acceleration * delta_t
                    current_velocity[sensor] += accel_mag * TIME_DELTA
                    vel_mag = current_velocity[sensor]
                    vel_mag_lists[sensor].append(vel_mag)
            
            # Calculate mean and std for each sensor
            mean_accel_mag = [np.mean(mags) if mags else 0.0 for mags in accel_mag_lists]
            std_accel_mag = [np.std(mags) if mags else 0.0 for mags in accel_mag_lists]
            
            mean_gyro_mag = [np.mean(mags) if mags else 0.0 for mags in gyro_mag_lists]
            std_gyro_mag = [np.std(mags) if mags else 0.0 for mags in gyro_mag_lists]
            
            mean_vel_mag = [np.mean(mags) if mags else 0.0 for mags in vel_mag_lists]
            std_vel_mag = [np.std(mags) if mags else 0.0 for mags in vel_mag_lists]
            
            # Prepare the row data
            row = [label, input_idx + 1]
            for sensor in range(5):
                row.extend([
                    mean_accel_mag[sensor],
                    std_accel_mag[sensor],
                    mean_gyro_mag[sensor],
                    std_gyro_mag[sensor],
                    mean_vel_mag[sensor],
                    std_vel_mag[sensor]
                ])
            
            # Append the row to the dataset
            dataset.append(row)
            
            # Prepare and print the table row
            table_row = []
            for item in row:
                if isinstance(item, float):
                    table_row.append(f"{item:<20.2f}")
                else:
                    table_row.append(f"{item:<20}")
            print(" | ".join(table_row))
            
    # After collecting all data, create a DataFrame and save to CSV
    columns = header  # Use the previously defined header
    df = pd.DataFrame(dataset, columns=columns)
    
    # Save the DataFrame to a CSV file
    csv_filename = f'asl_data_{args.name}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nDataset saved to {csv_filename}")
        
except KeyboardInterrupt:
    print("\nProgram terminated by user.")
finally:
    bus.close()
    print("\nData collection complete.")
