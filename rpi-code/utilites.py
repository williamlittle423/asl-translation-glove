import smbus2
import time

# MPU6050 Registers
MPU6050_ADDRESS = 0x68  # I2C address when AD0 is low
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

# I2C Bus Configuration
I2C_BUS_NUMBER = 1  # I2C-1 on Raspberry Pi
bus = smbus2.SMBus(I2C_BUS_NUMBER)

# TCA9548A I2C Address
MUX_I2C_ADDRESS = 0x70

def select_mux_channel(channel):
    """
    Selects a specific channel on the TCA9548A multiplexer.
    """
    if channel < 0 or channel > 7:
        raise ValueError("Channel must be between 0 and 7")

    control_byte = 1 << channel
    try:
        bus.write_byte(MUX_I2C_ADDRESS, control_byte)
        print(f"[MUX] Selected channel {channel} (Control Byte: {hex(control_byte)})")
    except IOError as e:
        print(f"[MUX] Error selecting channel {channel}: {e}")

def initialize_mpu6050():
    """
    Initializes the MPU6050 sensor by waking it up.
    """
    try:
        bus.write_byte_data(MPU6050_ADDRESS, PWR_MGMT_1, 0)
        print("[MPU6050] Initialized (woken up).")
    except IOError as e:
        print(f"[MPU6050] Initialization error: {e}")

def read_raw_data(register):
    """
    Reads two bytes of raw data from the MPU6050 and converts them to a signed integer.
    """
    try:
        high = bus.read_byte_data(MPU6050_ADDRESS, register)
        low = bus.read_byte_data(MPU6050_ADDRESS, register + 1)
        value = (high << 8) | low
        if value > 32767:
            value -= 65536
        return value
    except IOError as e:
        print(f"[MPU6050] Read error at register {hex(register)}: {e}")
        return None

def read_mpu6050_data():
    """
    Reads accelerometer and gyroscope data from the MPU6050.
    """
    accel_x = read_raw_data(ACCEL_XOUT_H)
    accel_y = read_raw_data(ACCEL_XOUT_H + 2)
    accel_z = read_raw_data(ACCEL_XOUT_H + 4)

    gyro_x = read_raw_data(GYRO_XOUT_H)
    gyro_y = read_raw_data(GYRO_XOUT_H + 2)
    gyro_z = read_raw_data(GYRO_XOUT_H + 4)

    if None in [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]:
        print("[MPU6050] Incomplete data read.")
        return None

    data = {
        'accel_x': accel_x,
        'accel_y': accel_y,
        'accel_z': accel_z,
        'gyro_x': gyro_x,
        'gyro_y': gyro_y,
        'gyro_z': gyro_z
    }
    return data

def main():
    # Channels where MPU6050 sensors are connected
    mpu_channels = [0, 1, 2, 3, 4]

    for channel in mpu_channels:
        print(f"\n--- MPU6050 on Channel {channel} ---")
        select_mux_channel(channel)
        time.sleep(0.1)  # Allow time for channel selection

        initialize_mpu6050()
        time.sleep(0.1)  # Allow time for sensor initialization

        data = read_mpu6050_data()
        if data:
            # Placeholder for calibration (optional)
            # Example: Convert raw data to physical units
            # accel_x_g = data['accel_x'] / 16384.0
            # gyro_x_dps = data['gyro_x'] / 131.0

            print(f"Accelerometer: X={data['accel_x']} | Y={data['accel_y']} | Z={data['accel_z']}")
            print(f"Gyroscope:    X={data['gyro_x']} | Y={data['gyro_y']} | Z={data['gyro_z']}")
        else:
            print(f"[MPU6050] Failed to read data on channel {channel}.")

        time.sleep(0.5)  # Delay before next channel

    bus.close()
    print("\n[INFO] Completed reading from all MPU6050 sensors.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted by user. Exiting...")
        bus.close()
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        bus.close()
