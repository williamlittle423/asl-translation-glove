import smbus2
import time
import numpy as np
import string

# Initialize I2C bus
bus = smbus2.SMBus(1)

# I2C addresses
TCA9548A_ADDRESS = 0x70
MPU6050_ADDRESS = 0x68

# Function to select the channel on the multiplexer
def select_channel(channel):
    if channel < 0 or channel > 7:
        raise ValueError("Invalid channel. Must be between 0 and 7.")
    bus.write_byte(TCA9548A_ADDRESS, 1 << channel)

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
    # Read accelerometer and gyroscope data from MPU6050
    accel_x = read_word_2c(MPU6050_ADDRESS, 0x3B)
    accel_y = read_word_2c(MPU6050_ADDRESS, 0x3D)
    accel_z = read_word_2c(MPU6050_ADDRESS, 0x3F)
    gyro_x = read_word_2c(MPU6050_ADDRESS, 0x43)
    gyro_y = read_word_2c(MPU6050_ADDRESS, 0x45)
    gyro_z = read_word_2c(MPU6050_ADDRESS, 0x47)

    return {
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z
    }
    
def read_channel(channel, previous_data):
    select_channel(channel)
    initialize_mpu6050()
    data = read_mpu6050()
    if previous_data is not None:
        delta = {key: data[key] - previous_data[key] for key in data}
    else:
        delta = data  # For the first read, use raw data as delta
    return delta, data


# Map letters to indices
letters = ['A', 'B', 'C', 'D']

# Prepare data array: shape (4 letters, 50 inputs per letter, variable data points)
# Initialize with an estimated maximum size (you may need to adjust N_v_max based on your system's performance)
N_letters = 4
N_training = 50
N_v_max = 5000  # Estimated maximum number of data points
data_array = np.zeros((N_letters, N_training, N_v_max), dtype=int)

def collect_single_letter(letter, input_idx):
    input(f"Press Enter and perform the ASL movement for letter '{letter}' ({input_idx+1}/{N_training})...")
    collected_data = []
    previous_data_list = [None]*5  # For each sensor
    start_time = time.time()
    read_counts = 0
    while time.time() - start_time < 1.5:
        for sensor_idx in range(5):  # Sensors 0 to 4
            delta, data = read_channel(sensor_idx, previous_data_list[sensor_idx])
            previous_data_list[sensor_idx] = data
            # Collect delta data
            delta_values = list(delta.values())  # 6 data points
            collected_data.extend(delta_values)
        read_counts += 1
        time.sleep(0.02)  # 20 ms delay
    print(f"Number of times data was read from all five sensors: {read_counts}")
    return collected_data


try:
    for letter_idx, letter in enumerate(letters):
        print(f"\nCollecting data for letter '{letter}'")
        for input_idx in range(N_training):
            collected_data = collect_single_letter(letter, input_idx)
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
        print(f"Finished collecting data for letter '{letter}'")
except KeyboardInterrupt:
    print("\nProgram terminated by user.")
finally:
    bus.close()
    # Save data_array to a file if desired
    np.save('asl_data.npy', data_array)
    print("\nData collection complete. Data saved to 'asl_data.npy'")
