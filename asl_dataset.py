import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# dataset = (N_letters, N_samples, N_v)
# labels = (N_letters, N_samples)


class ASLDataset(Dataset):
    def __init__(self, data_path='asl_data.npy', stats_path='asl_data_stats.npy', transform=None):
        # Load the data
        self.data_array = np.load(data_path)
        
        # Load mean and std for standardization
        stats = np.load(stats_path, allow_pickle=True).item()
        self.mean = stats['mean']
        self.std = stats['std']
        
        print('Length of letter A?', len(self.data_array[0]))
        print('Length of letter B?', len(self.data_array[1]))
        print('Length of letter C?', len(self.data_array[2]))
        print('Length of letter D?', len(self.data_array[3]))

        # Apply standardization
        self.standardized_data = (self.data_array - self.mean) / self.std
        
        print('Mean of standardized', np.mean(self.standardized_data[0]))
        
        # Get dataset dimensions
        N_letters, N_samples, N_features = self.standardized_data.shape
        self.num_classes = N_letters

        # Prepare labels as one-hot encoded vectors
        # Initialize labels array with zeros
        self.labels = np.zeros((N_letters, N_samples, N_letters))
        
        # Set the appropriate position to 1 for one-hot encoding
        for i in range(N_letters):  # Loop over letters
            self.labels[i, :, i] = 1  # Set the ith position to 1 for all samples of letter i

        # Flatten data_array to shape (N_letters * N_samples, N_features)
        self.standardized_data = self.standardized_data.reshape(-1, N_features)
        self.labels = self.labels.reshape(-1, N_letters)
        
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
        label = torch.tensor(label, dtype=torch.float32)  # Use float32 for BCELoss
        
        return sample, label

# Create the dataset
dataset = ASLDataset()

print(f'Length of dataset: {dataset.__len__()}')
print('Random item: ', dataset.__getitem__(30)[1])
