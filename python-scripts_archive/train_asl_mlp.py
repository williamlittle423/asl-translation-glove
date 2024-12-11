import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import joblib  # Import moved to the top for better practice

# Step 1: Load and preprocess the data

# Path to the merged CSV file
DATA_PATH = 'asl_data_eric.csv'  # Replace with your actual merged file path

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Display the first few rows to verify
print("First five rows of the dataset:")
print(df.head())

# OPTIONAL: If you need to filter specific labels, uncomment and modify the following lines
# For example, to include letters A-L and words 'milk', 'want'
# labels_to_include = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'milk', 'want']
# df = df[df['Label'].isin(labels_to_include)]

# If your merged dataset already contains only the desired labels, you can skip filtering

# Reset index after filtering (if any)
df.reset_index(drop=True, inplace=True)

print(f"\nDataset shape after loading: {df.shape}")

# Check if 'Label' column exists
if 'Label' not in df.columns:
    print("Error: The dataset does not contain a 'Label' column.")
    print(f"Available columns: {df.columns.tolist()}")
    sys.exit(1)

# Encode the labels (Letters A-L and Words) to numerical values
label_encoder = LabelEncoder()
df['Encoded_Label'] = label_encoder.fit_transform(df['Label'])

# Display the mapping
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nLabel Encoding Mapping:")
for label, encoded in label_mapping.items():
    print(f"{label}: {encoded}")

# Features: All columns except 'Label', 'Input', and 'Encoded_Label'
feature_columns = [col for col in df.columns if col not in ['Label', 'Input', 'Encoded_Label']]
X = df[feature_columns].values
y = df['Encoded_Label'].values

print(f"\nNumber of samples: {X.shape[0]}")
print(f"Number of features per sample: {X.shape[1]}")

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define PyTorch Dataset
class ASLDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create Dataset instances
train_dataset = ASLDataset(X_train_scaled, y_train)
test_dataset = ASLDataset(X_test_scaled, y_test)

# Create DataLoader instances
BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nNumber of batches in training loader: {len(train_loader)}")
print(f"Number of batches in testing loader: {len(test_loader)}")

# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)  # num_classes will be dynamic
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        
        out = self.fc3(out)
        return out

# Define model parameters dynamically based on the dataset
INPUT_SIZE = X_train_scaled.shape[1]  # Number of features
HIDDEN_SIZES = [64, 32]                # Two hidden layers
NUM_CLASSES = len(label_encoder.classes_)  # Dynamic number of classes based on labels

# Instantiate the model
model = MLP(INPUT_SIZE, HIDDEN_SIZES, NUM_CLASSES)

print("\nModel Architecture:")
print(model)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Move the model to the device
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define Training Function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=100):
    model.train()  # Set model to training mode
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_features, batch_labels in train_loader:
            # Move data to device
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * batch_features.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == batch_labels).sum().item()
            total_predictions += batch_labels.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / total_predictions * 100
        
        # Print progress every 10 epochs and the first epoch
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
    
    print("\nTraining complete.")

# Define Evaluation Function
def evaluate_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Define Training and Evaluation
NUM_EPOCHS = 100

# Train the model
train_model(model, train_loader, criterion, optimizer, device, num_epochs=NUM_EPOCHS)

# Evaluate the model
test_accuracy = evaluate_model(model, test_loader, device)

# Save the trained model
MODEL_PATH = 'asl_mlp_model_eric_A-L-Words.pth'  # Updated filename to reflect inclusion of words
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

# Save the scaler to disk
print("Saving the scaler to disk...")
joblib.dump(scaler, 'asl_scaler_eric.save')  # Updated filename for clarity
print("Scaler saved to 'asl_scaler_eric.save'")

# Save the label encoder for future use
joblib.dump(label_encoder, 'asl_label_encoder_eric.save')
print("Label encoder saved to 'asl_label_encoder_eric.save'")
