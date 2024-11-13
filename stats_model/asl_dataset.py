import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
from scipy.stats import skew, kurtosis

class ASLDataset(Dataset):
    def __init__(self, data_dir, stats_path, window_size=32, num_fingers=5, num_axes=3, transform=None):
        """
        Initializes the ASLDataset by aggregating data from all individuals and applying normalization.
        
        Args:
            data_dir (str): Directory containing the `.npy` data files.
            stats_path (str): Path to the `.npy` file containing normalization statistics.
            window_size (int, optional): Number of time steps per window. Default is 32.
            num_fingers (int, optional): Number of fingers/sensors. Default is 5.
            num_axes (int, optional): Number of accelerometer axes (X, Y, Z). Default is 3.
            transform (callable, optional): Optional transform to be applied on a sample after normalization.
        """
        self.data = []
        self.labels = []
        self.transform = transform

        # Load normalization statistics
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"Statistics file '{stats_path}' not found.")
        
        stats = np.load(stats_path, allow_pickle=True).item()
        if 'mean' not in stats or 'std' not in stats:
            raise ValueError("Statistics file must contain 'mean' and 'std' keys.")
        
        self.mean = stats['mean']
        self.std = stats['std']
        
        # Replace zeros in std to avoid division by zero
        self.std[self.std == 0] = 1.0
        
        # Define the pattern to match all relevant `.npy` files (across all names and letters)
        pattern = os.path.join(data_dir, 'asl_data_*_*.npy')
        file_list = glob.glob(pattern)

        if not file_list:
            raise ValueError(f"No data files found in '{data_dir}' matching pattern 'asl_data_*_*.npy'.")

        # Extract unique letters from filenames
        letters = sorted(list(set([os.path.basename(f).split('_')[2].split('.')[0] for f in file_list])))
        print('Found letters:', letters)
        self.label_map = {letter: idx for idx, letter in enumerate(letters)}
        print('Label mapping:', self.label_map)

        # Iterate over each file and load the data
        for f in file_list:
            basename = os.path.basename(f)
            parts = basename.split('_')
            if len(parts) < 4:
                print(f"Skipping file '{basename}' due to unexpected naming convention.")
                continue  # Skip files that do not match the expected pattern

            letter = parts[2]  # Extract the letter part
            if letter not in self.label_map:
                print(f"Skipping file '{basename}' with unrecognized letter '{letter}'.")
                continue  # Skip if the letter is not recognized

            label = self.label_map[letter]
            try:
                sample_data = np.load(f)  # Shape: (N_training, N_v_max)
            except Exception as e:
                print(f"Error loading file '{basename}': {e}. Skipping this file.")
                continue  # Skip files that cannot be loaded

            for i in range(sample_data.shape[0]):
                self.data.append(sample_data[i])  # Each sample: (N_v_max,)
                self.labels.append(label)

        # Set parameters for feature extraction
        self.window_size = window_size
        self.num_fingers = num_fingers
        self.num_axes = num_axes  # Typically 3 for X, Y, Z accelerometer axes

        # Calculate feature dimensions
        self.raw_features = window_size * num_fingers * num_axes  # e.g., 32*5*3=480
        self.sum_features = num_fingers * num_axes              # e.g., 5*3=15
        self.stat_features = num_fingers * num_axes * 4        # e.g., 5*3*4=60 (mean, std, skew, kurtosis)
        self.total_features = self.raw_features + self.sum_features + self.stat_features  # e.g., 555

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the processed and normalized sample along with its label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (features, label) where features is a tensor of shape (total_features,)
                   and label is an integer.
        """
        sample = self.data[idx]  # Shape: (N_v_max,)
        label = self.labels[idx]

        # Reshape to (num_fingers, 6 data points, window_size)
        # Assuming data is ordered as [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z] per sensor
        try:
            reshaped = sample.reshape(self.num_fingers, 6, self.window_size)
        except ValueError as e:
            raise ValueError(f"Sample at index {idx} has incorrect shape: {e}")

        # Extract accelerometer data (first 3 data points)
        accel_data = reshaped[:, 0:3, :]  # Shape: (num_fingers, num_axes, window_size)

        # Flatten raw accelerometer data
        raw = accel_data.flatten()  # Shape: (num_fingers * num_axes * window_size,)

        # Compute aggregate sums for each finger and axis
        sum_features = accel_data.sum(axis=2).flatten()  # Shape: (num_fingers * num_axes,)

        # Compute statistical features: mean, std, skewness, kurtosis
        mean = accel_data.mean(axis=2).flatten()          # Shape: (num_fingers * num_axes,)
        std = accel_data.std(axis=2).flatten()            # Shape: (num_fingers * num_axes,)
        skewness = skew(accel_data, axis=2).flatten()      # Shape: (num_fingers * num_axes,)
        kurt = kurtosis(accel_data, axis=2).flatten()      # Shape: (num_fingers * num_axes,)

        # Concatenate statistical features
        stats = np.stack([mean, std, skewness, kurt], axis=1).flatten()  # Shape: (num_fingers * num_axes * 4,)

        # Concatenate all features into a single vector
        features = np.concatenate([raw, sum_features, stats], axis=0)  # Shape: (total_features,)

        # Normalize the features
        features = (features - self.mean) / self.std  # Element-wise normalization

        # Apply any transformations if provided
        if self.transform:
            features = self.transform(features)
        else:
            # Convert to PyTorch tensor and ensure data type is float32
            features = torch.tensor(features, dtype=torch.float32)

        return features, label

def get_dataloader(data_dir='data', stats_path='all_data_stats.npy', batch_size=32, shuffle=True, num_workers=4, transform=None):
    """
    Initializes the DataLoader for the ASLDataset with normalization.

    Args:
        data_dir (str): Directory containing the `.npy` data files.
        stats_path (str): Path to the `.npy` file containing normalization statistics.
        batch_size (int, optional): Number of samples per batch. Default is 32.
        shuffle (bool, optional): Whether to shuffle the data. Default is True.
        num_workers (int, optional): Number of subprocesses for data loading. Default is 4.
        transform (callable, optional): Optional transform to be applied on a sample after normalization.

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    dataset = ASLDataset(data_dir=data_dir, stats_path=stats_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
