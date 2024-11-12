import smbus2
import time

# Initialize I2C bus
bus = smbus2.SMBus(1)

# I2C addresses
TCA9548A_ADDRESS = 0x70
MPU6050_ADDRESS = 0x68

# Global variable to store previous data for each channel
previous_data = {}
initialized_channels = set()

# Function to select the channel on the multiplexer
def select_channel(channel):
    if channel < 0 or channel > 7:
        raise ValueError("Invalid channel. Must be between 0 and 7.")
    bus.write_byte(TCA9548A_ADDRESS, 1 << channel)
    time.sleep(0.01)  # Small delay to allow channel selection to settle

# Function to initialize the MPU6050 sensor
def initialize_mpu6050():
    # Write to power management register to wake up the MPU6050
    bus.write_byte_data(MPU6050_ADDRESS, 0x6B, 0x00)
    time.sleep(0.1)  # Allow time for the sensor to stabilize

# Function to read raw data from MPU6050
def read_mpu6050():
    # Read 14 bytes of data starting from register 0x3B
    data = bus.read_i2c_block_data(MPU6050_ADDRESS, 0x3B, 14)

    # Combine the bytes to get raw values
    accel_x = (data[0] << 8) | data[1]
    accel_y = (data[2] << 8) | data[3]
    accel_z = (data[4] << 8) | data[5]
    temp    = (data[6] << 8) | data[7]
    gyro_x  = (data[8] << 8) | data[9]
    gyro_y  = (data[10] << 8) | data[11]
    gyro_z  = (data[12] << 8) | data[13]

    # Convert to signed 16-bit integers
    def twos_complement(val):
        return val - 65536 if val >= 0x8000 else val

    accel_x = twos_complement(accel_x)
    accel_y = twos_complement(accel_y)
    accel_z = twos_complement(accel_z)
    temp    = twos_complement(temp)
    gyro_x  = twos_complement(gyro_x)
    gyro_y  = twos_complement(gyro_y)
    gyro_z  = twos_complement(gyro_z)

    return {
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "temp": temp,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z
    }

def read_channel(channel):
    while True:
        try:
            select_channel(channel)
            if channel not in initialized_channels:
                initialize_mpu6050()
                initialized_channels.add(channel)
            data = read_mpu6050()
            if channel in previous_data:
                delta = {key: data[key] - previous_data[channel][key] for key in data}
                print(f"Channel {channel} Data Change: {delta}")
            else:
                print(f"Channel {channel} Initial Data: {data}")
            previous_data[channel] = data
            time.sleep(0.01)  # Small delay to prevent bus overload
            break  # Exit the loop if the read was successful

        except IOError as e:
            print(f"IOError on channel {channel}: {e}. Retrying in 1 second...")
            time.sleep(1)  # Wait before retrying

# Main loop to continuously read data from each MPU6050
try:
    while True:
        read_channel(0)
        read_channel(1)
        read_channel(2)
        read_channel(3)
        read_channel(4)

except KeyboardInterrupt:
    print("\nProgram terminated by user.")

finally:
    bus.close()


    
# Construct a data collection script that collects data using the method above and follows the procceeding algorithm
# Each input data is (5 sensors) * (10 measurements / 1 sensor) * (20 ms / 1 measurement) = 1s total collection time
# (5 sensors) * (10 measurements / 1 sensor) * (6 data points / measurement) = 300 data points / input
# I want to press enter and perform an ASL movement and collect the 300 delta data points and store them in an array
# Once 50 inputs have been collected, switch to the next letter in the American alphabet
# I want to do this for each letter - therefore the array should be (N_letters, N_training, N_v) = (26, 50, 300)

