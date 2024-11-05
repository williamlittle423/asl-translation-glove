import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ASLDataset(Dataset):
    def __init__(self, data_path='asl_data.npy', stats_path='asl_data_stats.npy', transform=None):
        # Load the data
        self.data_array = np.load(data_path)
        
        # Load mean and std for standardization
        stats = np.load(stats_path, allow_pickle=True).item()
        self.mean = stats['mean']
        self.std = stats['std']
        
        # Apply standardization
        self.standardized_data = (self.data_array - self.mean) / self.std
        
        # Prepare labels
        # Assuming letters are ['A', 'B', 'C', 'D'] mapped to [0, 1, 2, 3]
        N_letters, N_samples, N_features = self.standardized_data.shape
        self.labels = np.repeat(np.arange(N_letters), N_samples)
        
        # Flatten data_array to shape (N_letters * N_samples, N_features)
        self.standardized_data = self.standardized_data.reshape(-1, N_features)
        
        # Store any transformations (if needed)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get the sample and label
        sample = self.standardized_data[idx]
        label = self.labels[idx]
        
        # Apply any transformations (if provided)
        if self.transform:
            sample = self.transform(sample)
        
        # Convert to torch tensors
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return sample, label

# Create the dataset
dataset = ASLDataset()

# Create the DataLoader
batch_size = 32  # You can adjust this value based on your system's capabilities
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
