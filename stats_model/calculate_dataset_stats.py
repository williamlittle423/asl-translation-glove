import numpy as np
import glob
import os
import argparse
from scipy.stats import skew, kurtosis

def compute_normalization_stats(data_dir, output_file='asl_data_stats.npy', window_size=32, num_fingers=5, num_axes=3):
    """
    Computes the mean and standard deviation for each feature across the entire dataset.
    
    Args:
        data_dir (str): Directory containing the `.npy` data files.
        output_file (str, optional): Filename to save the computed statistics. Default is 'asl_data_stats.npy'.
        window_size (int, optional): Number of time steps per window. Default is 32.
        num_fingers (int, optional): Number of fingers/sensors. Default is 5.
        num_axes (int, optional): Number of accelerometer axes (X, Y, Z). Default is 3.
    """
    # Define the pattern to match all relevant `.npy` files (across all names and letters)
    pattern = os.path.join(data_dir, 'asl_data_*_*.npy')
    file_list = glob.glob(pattern)

    if not file_list:
        raise ValueError(f"No data files found in '{data_dir}' matching pattern 'asl_data_*_*.npy'.")

    print(f"Found {len(file_list)} data files.")

    # Initialize lists to hold all features
    all_features = []

    # Iterate over each file and extract features
    for f in file_list:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) < 4:
            print(f"Skipping file '{basename}' due to unexpected naming convention.")
            continue  # Skip files that do not match the expected pattern

        letter = parts[2]  # Extract the letter part
        # Load the data
        try:
            sample_data = np.load(f)  # Shape: (N_training, N_v_max)
        except Exception as e:
            print(f"Error loading file '{basename}': {e}. Skipping this file.")
            continue  # Skip files that cannot be loaded

        # Iterate over each sample in the file
        for i in range(sample_data.shape[0]):
            sample = sample_data[i]  # Shape: (N_v_max,)
            # Reshape to (num_fingers, 6, window_size)
            try:
                reshaped = sample.reshape(num_fingers, 6, window_size)
            except ValueError as e:
                print(f"Skipping sample {i} in file '{basename}' due to incorrect shape: {e}")
                continue  # Skip samples that cannot be reshaped

            # Extract accelerometer data (first 3 data points)
            accel_data = reshaped[:, 0:3, :]  # Shape: (num_fingers, num_axes, window_size)

            # Flatten raw accelerometer data
            raw = accel_data.flatten()  # Shape: (num_fingers * num_axes * window_size,) = 480

            # Compute aggregate sums for each finger and axis
            sum_features = accel_data.sum(axis=2).flatten()  # Shape: (num_fingers * num_axes,) = 15

            # Compute statistical features: mean, std, skewness, kurtosis
            mean = accel_data.mean(axis=2).flatten()          # Shape: (num_fingers * num_axes,) = 15
            std = accel_data.std(axis=2).flatten()            # Shape: (num_fingers * num_axes,) = 15
            skewness = skew(accel_data, axis=2).flatten()      # Shape: (num_fingers * num_axes,) = 15
            kurt = kurtosis(accel_data, axis=2).flatten()      # Shape: (num_fingers * num_axes,) = 15

            # Concatenate statistical features
            stats = np.stack([mean, std, skewness, kurt], axis=1).flatten()  # Shape: 60

            # Concatenate all features into a single vector
            features = np.concatenate([raw, sum_features, stats], axis=0)  # Shape: 555

            all_features.append(features)

    if not all_features:
        raise ValueError("No valid samples found to compute statistics.")

    # Convert to NumPy array
    all_features = np.array(all_features)  # Shape: (total_samples, 555)

    print(f"Total samples extracted: {all_features.shape[0]}")

    # Compute mean and std per feature
    mean = np.mean(all_features, axis=0)  # Shape: (555,)
    std = np.std(all_features, axis=0)    # Shape: (555,)

    # Replace any std values that are zero to avoid division by zero
    std_replaced = np.where(std == 0, 1.0, std)

    # Save the mean and std to a file
    stats_dict = {'mean': mean, 'std': std_replaced}
    np.save(output_file, stats_dict)

    print(f"Data mean shape: {mean.shape}, Data std shape: {std_replaced.shape}")
    print(f"Mean and standard deviation saved to '{output_file}'.")

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Compute ASL Data Normalization Statistics.")
    parser.add_argument("--data_dir", type=str, help="Directory containing ASL `.npy` files.", required=True)
    parser.add_argument("--output_file", type=str, help="Filename to save the computed statistics.", default='asl_data_stats.npy')
    parser.add_argument("--window_size", type=int, help="Number of time steps per window.", default=32)
    parser.add_argument("--num_fingers", type=int, help="Number of fingers/sensors.", default=5)
    parser.add_argument("--num_axes", type=int, help="Number of accelerometer axes (X, Y, Z).", default=3)
    
    args = parser.parse_args()
    
    # Execute the computation
    compute_normalization_stats(
        data_dir=args.data_dir,
        output_file=args.output_file,
        window_size=args.window_size,
        num_fingers=args.num_fingers,
        num_axes=args.num_axes
    )
