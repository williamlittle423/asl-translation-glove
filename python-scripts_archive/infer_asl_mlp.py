#!/usr/bin/env python3
import smbus2
import time
import numpy as np
import argparse
import math
import pandas as pd
import torch
import subprocess
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import joblib
import sys

# Step 1: Define the MLP Model Architecture (Must match the trained model)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        
        out = self.fc3(out)
        return out

def play_audio_on_pi(label):
    """
    Plays the corresponding audio file for the predicted label.
    
    Args:
        label (str): The predicted label (e.g., 'A' or 'milk').
    """
    # Replace spaces or special characters in labels to match file naming conventions
    sanitized_label = label.replace(" ", "_").replace("-", "_")
    audio_path = f"/home/rp/copied_repo/audio_files/{sanitized_label}.wav"
    try:
        subprocess.run(["aplay", audio_path], check=True)
    except FileNotFoundError:
        print(f"Audio file for '{label}' not found at {audio_path}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to play audio on Raspberry Pi: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Step 2: Define Command-Line Arguments
parser = argparse.ArgumentParser(description="ASL Inference Script using Trained MLP Model.")
parser.add_argument("--model_path", type=str, default='asl_mlp_model_will_A-Z.pth', help="Path to the trained MLP model (.pth file)")
parser.add_argument("--scaler_path", type=str, default='asl_scaler_eric.save', help="Path to the saved StandardScaler (.save file)")
parser.add_argument("--label_encoder_path", type=str, default='asl_label_encoder_eric.save', help="Path to the saved LabelEncoder (.save file)")
parser.add_argument("--hidden_sizes", type=int, nargs='+', help="Hidden layer sizes matching the trained model", default=[64, 32])
parser.add_argument("--input_size", type=int, default=30, help="Number of input features")
parser.add_argument("--num_classes", type=int, default=22, help="Number of classes (letters + words)")
parser.add_argument("--time_steps", type=int, default=32, help="Number of time steps to collect data")
parser.add_argument("--num_sensors", type=int, default=5, help="Number of MPU6050 sensors")
parser.add_argument("--retry_limit", type=int, default=5, help="Maximum number of retries for I2C communication")
parser.add_argument("--time_delta", type=float, default=0.02, help="Time delay between sensor reads in seconds (default 20ms)")
parser.add_argument("--threshold", type=float, default=0.2, help="Probability threshold for prediction (default 80%)")

args = parser.parse_args()

# Step 3: Initialize I2C Bus
try:
    bus = smbus2.SMBus(1)
except Exception as e:
    print(f"Failed to initialize I2C bus: {e}")
    sys.exit(1)

# I2C addresses
TCA9548A_ADDRESS = 0x70
MPU6050_ADDRESS = 0x68

# Step 4: Load the StandardScaler
try:
    scaler = joblib.load(args.scaler_path)
    print(f"Scaler loaded from {args.scaler_path}")
except Exception as e:
    print(f"Failed to load scaler from {args.scaler_path}: {e}")
    sys.exit(1)

# Step 5: Load the LabelEncoder
try:
    label_encoder = joblib.load(args.label_encoder_path)
    print(f"Label encoder loaded from {args.label_encoder_path}")
except Exception as e:
    print(f"Failed to load label encoder from {args.label_encoder_path}: {e}")
    sys.exit(1)

# Step 6: Load the Trained Model
model = MLP(input_size=args.input_size, hidden_sizes=args.hidden_sizes, num_classes=args.num_classes)
try:
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    print(f"Model loaded from {args.model_path}")
except Exception as e:
    print(f"Failed to load model from {args.model_path}: {e}")
    sys.exit(1)

model.eval()  # Set model to evaluation mode

# Check if GPU is available and move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

# Step 7: Define I2C Communication Functions with Error Handling
def select_channel(channel, retry_count=0):
    if channel < 0 or channel > 7:
        raise ValueError("Invalid channel. Must be between 0 and 7.")
    try:
        bus.write_byte(TCA9548A_ADDRESS, 1 << channel)
        return True
    except OSError as e:
        if e.errno in [121, 5]:  # Remote I/O error or Input/output error
            if retry_count < args.retry_limit:
                print(f"OSError {e.errno}: Error selecting channel {channel}. Retrying ({retry_count + 1}/{args.retry_limit})...")
                time.sleep(0.1)  # Wait before retrying
                return select_channel(channel, retry_count + 1)
            else:
                print(f"Failed to select channel {channel} after {args.retry_limit} retries.")
                return False
        else:
            print(f"Unexpected OSError {e.errno}: {e}")
            return False

def initialize_mpu6050():
    try:
        # Wake up MPU6050
        bus.write_byte_data(MPU6050_ADDRESS, 0x6B, 0x00)
        return True
    except OSError as e:
        print(f"OSError {e.errno}: Error initializing MPU6050.")
        return False
    except Exception as e:
        print(f"Unexpected error initializing MPU6050: {e}")
        return False

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

def read_mpu6050():
    try:
        accel_x = read_word_2c(MPU6050_ADDRESS, 0x3B)
        accel_y = read_word_2c(MPU6050_ADDRESS, 0x3D)
        accel_z = read_word_2c(MPU6050_ADDRESS, 0x3F)
        gyro_x = read_word_2c(MPU6050_ADDRESS, 0x43)
        gyro_y = read_word_2c(MPU6050_ADDRESS, 0x45)
        gyro_z = read_word_2c(MPU6050_ADDRESS, 0x47)
    except Exception as e:
        print(f"Unexpected error reading MPU6050 data: {e}")
        return None
    # Check for None values
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

# Step 8: Feature Extraction Function
def collect_features():
    """
    Collects features from all sensors over defined time steps.
    
    Returns:
        list: A feature vector containing mean and std deviations of accelerometer and gyroscope magnitudes,
              as well as mean and std of velocity magnitudes for each sensor.
    """
    collected_data = []
    base_data_list = [None] * args.num_sensors  # Initialize base data for each sensor
    
    # Step 1: Collect base data from all sensors
    print("Collecting base data from all sensors...")
    for sensor_idx in range(args.num_sensors):
        delta, data = read_channel(sensor_idx, None, retry_count=0)
        if delta is None:
            print(f"Failed to read from sensor {sensor_idx}. Aborting this collection.")
            return None
        base_data_list[sensor_idx] = data
    print("Base data collection complete.")
    
    previous_data_list = base_data_list.copy()  # Initialize previous data
    
    # Step 2: Collect data over defined time steps
    current_velocity = [0.0 for _ in range(args.num_sensors)]  # Initialize velocity for each sensor
    accel_mag_lists = [[] for _ in range(args.num_sensors)]
    gyro_mag_lists = [[] for _ in range(args.num_sensors)]
    vel_mag_lists = [[] for _ in range(args.num_sensors)]
    
    print(f"Collecting data over {args.time_steps} time steps...")
    for step in range(args.time_steps):
        for sensor_idx in range(args.num_sensors):
            delta, data = read_channel(sensor_idx, previous_data_list[sensor_idx], retry_count=0)
            if delta is None:
                print(f"Failed to read from sensor {sensor_idx} at time step {step + 1}. Aborting this collection.")
                return None
            previous_data_list[sensor_idx] = data
            # Calculate magnitudes
            accel_mag = math.sqrt(delta['accel_x']**2 + delta['accel_y']**2 + delta['accel_z']**2)
            gyro_mag = math.sqrt(delta['gyro_x']**2 + delta['gyro_y']**2 + delta['gyro_z']**2)
            accel_mag_lists[sensor_idx].append(accel_mag)
            gyro_mag_lists[sensor_idx].append(gyro_mag)
            # Update velocity using Euler integration
            current_velocity[sensor_idx] += accel_mag * args.time_delta
            vel_mag = current_velocity[sensor_idx]
            vel_mag_lists[sensor_idx].append(vel_mag)
        print(f"Time step {step + 1}/{args.time_steps} collected.")
        time.sleep(args.time_delta)  # Wait before next time step
    
    print("Data collection complete.")
    
    # Step 3: Compute mean and std for each sensor's features
    feature_vector = []
    for sensor_idx in range(args.num_sensors):
        mean_accel = np.mean(accel_mag_lists[sensor_idx]) if accel_mag_lists[sensor_idx] else 0.0
        std_accel = np.std(accel_mag_lists[sensor_idx]) if accel_mag_lists[sensor_idx] else 0.0
        mean_gyro = np.mean(gyro_mag_lists[sensor_idx]) if gyro_mag_lists[sensor_idx] else 0.0
        std_gyro = np.std(gyro_mag_lists[sensor_idx]) if gyro_mag_lists[sensor_idx] else 0.0
        mean_vel = np.mean(vel_mag_lists[sensor_idx]) if vel_mag_lists[sensor_idx] else 0.0
        std_vel = np.std(vel_mag_lists[sensor_idx]) if vel_mag_lists[sensor_idx] else 0.0
        feature_vector.extend([mean_accel, std_accel, mean_gyro, std_gyro, mean_vel, std_vel])
    
    return feature_vector

# Step 9: Prediction Function
def predict_label(feature_vector):
    """
    Predicts the ASL label based on the input feature vector.
    
    Args:
        feature_vector (list): The extracted feature vector.
    
    Returns:
        tuple: (predicted_class_index, max_probability, probabilities_array)
    """
    # Convert to numpy array and reshape
    feature_array = np.array(feature_vector).reshape(1, -1)
    # Scale features
    feature_scaled = scaler.transform(feature_array)
    # Convert to tensor
    input_tensor = torch.tensor(feature_scaled, dtype=torch.float32).to(device)
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
    # Get the highest probability
    max_prob = np.max(probabilities)
    predicted_class = np.argmax(probabilities)
    return predicted_class, max_prob, probabilities

# Step 10: Main Inference Loop
def main():
    print("\n--- ASL Label Inference Started ---\n")
    print("Instructions:")
    print("1. Press Enter to perform an ASL gesture.")
    print(f"2. The system will collect data and attempt to predict the label.")
    print(f"3. Prediction will be displayed if confidence > {args.threshold * 100:.0f}%.")
    print("4. Press Ctrl+C to exit.\n")
    
    try:
        while True:
            input("Press Enter to perform an ASL gesture...")
            feature_vector = collect_features()
            if feature_vector is None:
                print("Data collection failed. Please try again.\n")
                continue
            predicted_class, max_prob, probabilities = predict_label(feature_vector)
            if max_prob >= args.threshold:
                predicted_label = label_encoder.inverse_transform([predicted_class])[0]
                print(f"Predicted Label: {predicted_label} (Confidence: {max_prob * 100:.2f}%)\n")
                # Play the corresponding audio
                play_audio_on_pi(predicted_label)
            else:
                print(f"No confident prediction made (Highest Confidence: {max_prob * 100:.2f}%)\n")
    except KeyboardInterrupt:
        print("\nInference terminated by user.")
    finally:
        bus.close()
        print("I2C bus closed. Goodbye!")

if __name__ == "__main__":
    main()
