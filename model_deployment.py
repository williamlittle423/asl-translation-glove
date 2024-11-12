import smbus2
import time
import numpy as np
import torch
import torch.nn as nn
import string
import argparse
from asl_network import ASLNetwork

# Step 1: Create the parser
parser = argparse.ArgumentParser(description="ASL model deployment script.")

# Step 2: Add arguments
parser.add_argument("--input_file", type=str, help="Hidden layer dimension size", default="asl_model.pth")
parser.add_argument("--hidden_dim", type=int, help="Hidden layer dimension size", default=128)
parser.add_argument("--hidden_layers", type=int, help="Number of hidden layers", default=1)

# Step 3: Parse arguments
args = parser.parse_args()

# Device and Model Parameters
MODEL_PATH = 'asl_model.pth'
INPUT_SIZE = 32 * 5 * 6  # 32 reads, 5 sensors, 6 data points each
NUM_CLASSES = 4  # Number of letters/classes

# Map indices to letters
letters = ['A', 'B', 'C', 'D']

# Initialize I2C bus
bus = smbus2.SMBus(1)

# I2C addresses
TCA9548A_ADDRESS = 0x70
MPU6050_ADDRESS = 0x68

# Function to select the channel on the multiplexer
def select_channel(channel):
    if channel < 0 or channel > 7:
        raise ValueError("Invalid channel. Must be between 0 and 7.")
    try:
        bus.write_byte(TCA9548A_ADDRESS, 1 << channel)
    except OSError as e:
        if e.errno == 5:
            print(f"OSError 5: Input/output error while selecting channel {channel}")
            return False
        else:
            raise
    return True

# Function to initialize the MPU6050 sensor
def initialize_mpu6050():
    # Write to power management register to wake up the MPU6050
    bus.write_byte_data(MPU6050_ADDRESS, 0x6B, 0x00)

# Function to read two bytes of data and convert it to a signed value
def read_word_2c(addr, reg):
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg + 1)
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
    except OSError as e:
        if e.errno == 121:
            print("OSError 121: Remote I/O error while reading MPU6050 data")
            return None
        else:
            raise
    return {
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z
    }

def read_channel(channel, previous_data):
    if not select_channel(channel):
        print(f"Failed to select channel {channel}")
        return None, None
    try:
        initialize_mpu6050()
    except Exception as e:
        print(f'ERROR initializing MPU6050 from channel {channel}: {e}')
        return None, None
    data = read_mpu6050()
    if data is None:
        print(f'ERROR reading data from MPU6050 on channel {channel}')
        return None, None
    if previous_data is not None:
        delta = {key: data[key] - previous_data[key] for key in data}
    else:
        delta = data  # For the first read, use raw data as delta
    return delta, data

# Load the trained model
model = ASLNetwork(INPUT_SIZE, NUM_CLASSES, args.hidden_dim, args.hidden_layers)
model.load_state_dict(torch.load(args.input_file))
model.eval()  # Set model to evaluation mode

# Load mean and std for standardization
stats = np.load('asl_data_stats.npy', allow_pickle=True).item()
mean = stats['mean']
std = stats['std']

# Initialize data buffer
BUFFER_SIZE = 32  # Number of past reads to keep
data_buffer = []

# Start real-time data collection and prediction
try:
    print("Starting real-time ASL recognition...")
    previous_data_list = [None]*5  # For each sensor
    while True:
        collected_data = []
        # Read data from all sensors
        for sensor_idx in range(5):  # Sensors 0 to 4
            delta, data = read_channel(sensor_idx, previous_data_list[sensor_idx])
            if delta is None:
                print('Sensor read error. Skipping this read.')
                continue
            previous_data_list[sensor_idx] = data
            # Collect delta data
            delta_values = list(delta.values())  # 6 data points
            collected_data.extend(delta_values)
        if len(collected_data) != 5 * 6:
            # Not all sensors provided data
            continue
        # Add to buffer
        data_buffer.append(collected_data)
        if len(data_buffer) > BUFFER_SIZE:
            data_buffer.pop(0)  # Remove oldest entry to maintain buffer size
        if len(data_buffer) == BUFFER_SIZE:
            # Flatten the buffer data
            input_data = np.array(data_buffer).flatten()
            # Standardize the data
            standardized_data = (input_data - mean) / std
            # Convert to torch tensor
            input_tensor = torch.tensor(standardized_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            # Get model output
            outputs = model(input_tensor)
            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            print(probabilities)
            prob_values, predicted_class = torch.max(probabilities, dim=1)
            prob_value = prob_values.item()
            predicted_class = predicted_class.item()
            if prob_value >= 0.60:
                print(f"Detected letter '{letters[predicted_class]}' with probability {prob_value*100:.2f}%")
        time.sleep(0.02)  # 20 ms delay between reads
except KeyboardInterrupt:
    print("\nReal-time ASL recognition stopped by user.")
finally:
    bus.close()