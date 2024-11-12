import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from asl_dataset import ASLDataset
from asl_network import ASLNetwork
import argparse

# Step 1: Create the parser
parser = argparse.ArgumentParser(description="ASL training script.")

# Step 2: Add arguments
parser.add_argument("--hidden_dim", type=int, help="Hidden layer dimension size", default=128)
parser.add_argument("--hidden_layers", type=int, help="Number of hidden layers", default=1)
parser.add_argument("--batch_size", type=int, help="Batch size for training", default=4)
parser.add_argument("--num_epochs", type=int, help="Number of epochs for training", default=100)
parser.add_argument("--lr", type=float, help="Learning rate for training", default=0.001)
parser.add_argument("--dropout_prob", type=float, help="Dropout probability", default=0.5)

# Step 3: Parse arguments
args = parser.parse_args()

print(f'Network Parameters: hidden_dim={args.hidden_dim}, hidden_layers={args.hidden_layers}, '
      f'batch_size={args.batch_size}, num_epochs={args.num_epochs}, lr={args.lr}, '
      f'dropout_prob={args.dropout_prob}')

# Create the dataset
dataset = ASLDataset()

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Instantiate the model
input_size = dataset.standardized_data.shape[1]  # Number of features
num_classes = dataset.num_classes  # Number of classes (letters)
model = ASLNetwork(input_size, num_classes, args.hidden_dim, args.hidden_layers, args.dropout_prob)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Training loop
for epoch in range(args.num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (samples, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(samples)
        loss = criterion(outputs, labels)
        
        # Backward and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print loss every 10 batches
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{args.num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # Calculate average loss and accuracy over the epoch
    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{args.num_epochs}], Average Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
    
    # Evaluate the model on the test set after each epoch
    model.eval()  # Set model to evaluation mode
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for samples, labels in test_loader:
            outputs = model(samples)
            _, predicted = torch.max(outputs.data, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_accuracy = 100 * test_correct / test_total
    print(f'Accuracy on test data after epoch {epoch+1}: {test_accuracy:.2f}%')

# Save the trained model
file_name = f'asl_model_hd-{args.hidden_dim}_hl-{args.hidden_layers}_lr-{args.lr}.pth'
torch.save(model.state_dict(), file_name)
print(f'Model saved to {file_name}')
