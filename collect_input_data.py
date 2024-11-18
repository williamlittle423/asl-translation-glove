import smbus2
import time
import numpy as np
import string
import argparse
import math

# Step 1: Create the parser
parser = argparse.ArgumentParser(description="ASL training script.")

# Step 2: Add arguments
parser.add_argument("--name", type=str, help="Name of person collecting data", default='noname')

# Step 3: Parse arguments
args = parser.parse_args()

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
        delta = {key: 0 for key in data}  # For the first read, use raw data as delta
    return delta, data


# Map letters to indices
letters = string.ascii_uppercase[:2]

print(f'Collecting data for American Sign Language (ASL) letters: {letters}')

# Prepare data array: shape (2 letters, 40 inputs per letter, 160 data points)
N_letters = len(letters)
N_training = 40
N_v_max = 32 * 5 * 6  # Estimated maximum number of data points
data_array = np.zeros((N_letters, N_training, N_v_max), dtype=int)

# Maximum number of retries for a sample
MAX_RETRIES = 10

def collect_single_letter(letter, input_idx, retry_count=0):
    if retry_count >= MAX_RETRIES:
        print(f"Max retries exceeded for letter '{letter}' ({input_idx+1}/{N_training}). Skipping this sample.")
        return None
    input(f"Press Enter and perform the ASL movement for letter '{letter}' ({input_idx+1}/{N_training})...")
    collected_data = []
    previous_data_list = [None]*5  # For each sensor
    start_time = time.time()
    read_counts = 0
    while time.time() - start_time < 1.5:
        for sensor_idx in range(5):  # Sensors 0 to 4
            delta, data = read_channel(sensor_idx, previous_data_list[sensor_idx])
            if delta is None:
                print('Attempting letter again due to sensor read error')
                return collect_single_letter(letter, input_idx, retry_count=retry_count+1)
            previous_data_list[sensor_idx] = data
            # Collect delta data
            delta_values = list(delta.values())  # 6 data points
            collected_data.extend(delta_values)
        read_counts += 1
        time.sleep(0.02)  # 20 ms delay
    print(f"Number of times data was read from all five sensors: {read_counts}")
    return collected_data

# Initialize table header
print("\nData Collection Summary:")
print(f"{'Letter':<8} {'Input':<6} {'Sum Delta':<12} {'Sum Accel Mag':<18} {'Sum Gyro Mag':<17}")
print("-" * 65)

try:
    for letter_idx, letter in enumerate(letters):
        print(f"\nCollecting data for letter '{letter}'")
        for input_idx in range(N_training):
            collected_data = collect_single_letter(letter, input_idx)
            if collected_data is None:
                print(f"Skipping sample {input_idx+1}/{N_training} for letter '{letter}' due to repeated errors.")
                continue
            # Ensure we have exactly N_v_max data points
            collected_length = len(collected_data)
            if collected_length > N_v_max:
                print(f"Warning: Collected {collected_length} data points exceeds maximum of {N_v_max}. Truncating data.")
                collected_data = collected_data[:N_v_max]
            elif collected_length < N_v_max:
                print(f"Warning: Collected {collected_length} data points is less than maximum of {N_v_max}. Padding data.")
                collected_data.extend([0]*(N_v_max - collected_length))
            data_array[letter_idx, input_idx, :] = collected_data
            print(f"Collected data for letter '{letter}' input {input_idx+1}/{N_training}")
            
            # Calculate Sum Delta
            sum_delta = sum(collected_data)
            
            # Calculate Sum Acceleration Magnitude and Sum Gyroscope Magnitude
            sum_accel_mag = 0.0
            sum_gyro_mag = 0.0
            for sensor in range(5):
                base_idx = sensor * 6
                accel_x = collected_data[base_idx + 0]
                accel_y = collected_data[base_idx + 1]
                accel_z = collected_data[base_idx + 2]
                gyro_x = collected_data[base_idx + 3]
                gyro_y = collected_data[base_idx + 4]
                gyro_z = collected_data[base_idx + 5]
                
                accel_mag = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
                gyro_mag = math.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
                
                sum_accel_mag += accel_mag
                sum_gyro_mag += gyro_mag
            
            # Print the table row
            print(f"{letter:<8} {input_idx+1:<6} {sum_delta:<12} {sum_accel_mag:<18.2f} {sum_gyro_mag:<17.2f}")
            
        # Save data for each letter to a file
        print(f"Finished collecting data for letter '{letter}'")
        letter_data = data_array[letter_idx, :, :]
        np.save(f'asl_data_{letter}_{args.name}.npy', letter_data)
        print(f'Saving letter {letter} data to file asl_data_{letter}_{args.name}.npy')
        
except KeyboardInterrupt:
    print("\nProgram terminated by user.")
finally:
    bus.close()
    # Save data_array to a file if desired
    print("\nData collection complete.")
