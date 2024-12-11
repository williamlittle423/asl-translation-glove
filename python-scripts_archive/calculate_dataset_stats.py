import numpy as np

# Load the collected data
data_array = np.load('asl_data.npy')

# Data has shape (N_letters, N_training, N_v_max)
# We need to find the mean and standard deviation over N_training and N_v_max

# Reshape the data to combine all samples and letters
# This flattens the data to a 1D array for global mean and std calculation
data_flat = data_array.flatten()

# Compute the overall mean and standard deviation
mean = np.mean(data_flat)
std = np.std(data_flat)

# Alternatively, if you want to compute the mean and std per data point across all samples:
# Reshape data to (total_samples, N_v_max)
# total_samples = data_array.shape[0] * data_array.shape[1]
# data_reshaped = data_array.reshape(total_samples, data_array.shape[2])

# Compute mean and std for each data point (column-wise)
# mean_per_feature = np.mean(data_reshaped, axis=0)
# std_per_feature = np.std(data_reshaped, axis=0)

# Save the mean and std to a file
np.save('asl_data_stats.npy', {'mean': mean, 'std': std})

print(f"Data mean: {mean}, Data standard deviation: {std}")
print("Mean and standard deviation saved to 'asl_data_stats.npy'")
