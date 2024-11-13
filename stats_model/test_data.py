import torch
from torch.utils.data import DataLoader
import argparse
from asl_dataset import ASLDataset, get_dataloader  # Ensure this import matches your project structure
import numpy as np

def test_dataloader(data_dir, batch_size=32, num_workers=4, num_batches=5):
    """
    Tests the ASLDataset and DataLoader by iterating through a specified number of batches
    and performing various checks on the data.

    Args:
        data_dir (str): Directory containing the `.npy` data files.
        batch_size (int, optional): Number of samples per batch. Default is 32.
        num_workers (int, optional): Number of subprocesses for data loading. Default is 4.
        num_batches (int, optional): Number of batches to test. Default is 5.
    """
    try:
        # Initialize the DataLoader without any transformations
        dataloader = get_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        print(f"DataLoader initialized successfully with batch size {batch_size} and {num_workers} workers.")
    except Exception as e:
        print(f"Error initializing DataLoader: {e}")
        return

    # Fetch the label map from the dataset
    dataset = dataloader.dataset
    label_map = dataset.label_map
    inverse_label_map = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    print(f"Number of classes (ASL letters): {num_classes}")
    print(f"Label mapping: {label_map}")

    # Iterate through a few batches to perform checks
    for batch_idx, (features, labels) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break  # Only test the specified number of batches

        print(f"\n--- Batch {batch_idx + 1} ---")
        print(f"Features shape: {features.shape}")  # Expected: (batch_size, 555)
        print(f"Labels shape: {labels.shape}")      # Expected: (batch_size,)
        print(f"Features data type: {features.dtype}")  # Expected: torch.float32
        print(f"Labels data type: {labels.dtype}")      # Expected: torch.int64

        # Check for NaNs or infinite values in features
        if torch.isnan(features).any():
            print("Warning: NaN values found in features!")
        if torch.isinf(features).any():
            print("Warning: Infinite values found in features!")

        # Check label values are within the expected range
        if labels.min() < 0 or labels.max() >= num_classes:
            print(f"Warning: Labels out of range! Labels should be between 0 and {num_classes - 1}.")
        else:
            print("Labels are within the expected range.")

        # Optionally, print some statistics of the features
        features_np = features.numpy()
        print(f"Features - min: {features_np.min():.4f}, max: {features_np.max():.4f}, mean: {features_np.mean():.4f}, std: {features_np.std():.4f}")

        # Optionally, display a sample feature vector and its label
        sample_idx = 0
        sample_feature = features[sample_idx].numpy()
        sample_label = labels[sample_idx].item()
        print(f"Sample {sample_idx + 1} - Label: {inverse_label_map[sample_label]} ({sample_label})")
        print(f"Sample {sample_idx + 1} - Feature Vector (first 10 values): {sample_feature[:10]}")

    print("\nDataLoader test completed successfully.")

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="ASL DataLoader Test Script")
    parser.add_argument("--data_dir", type=str, help="Directory containing ASL `.npy` files.", required=True)
    parser.add_argument("--batch_size", type=int, help="Batch size for DataLoader.", default=32)
    parser.add_argument("--num_workers", type=int, help="Number of worker threads.", default=4)
    parser.add_argument("--num_batches", type=int, help="Number of batches to test.", default=5)
    
    args = parser.parse_args()
    
    # Run the DataLoader test
    test_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_batches=args.num_batches
    )
