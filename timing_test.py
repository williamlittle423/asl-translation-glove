import smbus2
import time

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

def read_channel(channel):
    select_channel(channel)
    initialize_mpu6050()
    return read_mpu6050()

# Main data collection loop for 1.5 seconds
def count_sensor_collections(duration=1.5):
    start_time = time.time()
    end_time = start_time + duration
    collection_count = 0

    try:
        while time.time() < end_time:
            successful = True
            for sensor_idx in range(5):  # Attempt to collect data from all 5 sensors
                try:
                    read_channel(sensor_idx)
                except Exception as e:
                    print(f"Failed to collect data from sensor {sensor_idx}: {e}")
                    successful = False
                    break
            
            # If data collection was successful for all 5 sensors, increment the count
            if successful:
                collection_count += 1

    finally:
        bus.close()
        print(f"\nData collection complete. Successfully collected data from all five sensors {collection_count} times in {duration} seconds.")

# Run the count function
count_sensor_collections()
