import torch
from torch.utils.data import DataLoader, random_split
import argparse
from asl_dataset import get_dataloader
from asl_classifier import ASLClassifier
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import os
from scipy.stats import skew, kurtosis

def train(model, train_loader, criterion, optimizer, device):
    """
    Trains the ASLClassifier model for one epoch.
    
    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to train on.
    
    Returns:
        tuple: (epoch_loss, epoch_acc, all_preds, all_labels)
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item() * features.size(0)
        
        # Compute accuracy
        _, preds = torch.max(outputs, 1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        # Collect predictions and labels for confusion matrix
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Compute average loss and accuracy for the epoch
    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions / total_samples) * 100
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def evaluate(model, val_loader, criterion, device):
    """
    Evaluates the ASLClassifier model on the validation set.
    
    Args:
        model (nn.Module): The neural network model to evaluate.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to evaluate on.
    
    Returns:
        tuple: (val_loss, val_acc, all_preds, all_labels)
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(val_loader):
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Accumulate loss
            running_loss += loss.item() * features.size(0)
            
            # Compute accuracy
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # Collect predictions and labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute average loss and accuracy for the validation set
    val_loss = running_loss / total_samples
    val_acc = (correct_predictions / total_samples) * 100
    
    return val_loss, val_acc, all_preds, all_labels

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs, save_dir='plots'):
    """
    Plots training and validation loss and accuracy over epochs.
    
    Args:
        train_losses (list): List of training loss values per epoch.
        val_losses (list): List of validation loss values per epoch.
        train_accuracies (list): List of training accuracy values per epoch.
        val_accuracies (list): List of validation accuracy values per epoch.
        num_epochs (int): Total number of training epochs.
        save_dir (str, optional): Directory to save the plots. Default is 'plots'.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    epochs = range(1, num_epochs + 1)
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_curves.png'))
    plt.close()
    
    print(f"Loss and accuracy plots saved to '{save_dir}' directory.")

def plot_confusion_matrix(all_labels, all_preds, classes, title='Confusion Matrix', save_path='plots/confusion_matrix.png'):
    """
    Plots and saves the confusion matrix.
    
    Args:
        all_labels (list): List of true labels.
        all_preds (list): List of predicted labels.
        classes (list): List of class names.
        title (str, optional): Title of the confusion matrix. Default is 'Confusion Matrix'.
        save_path (str, optional): Path to save the confusion matrix plot. Default is 'plots/confusion_matrix.png'.
    """
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    plt.figure(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    
    print(f"{title} saved to '{save_path}'.")

def generate_classification_report(all_labels, all_preds, classes, save_path='plots/classification_report.txt'):
    """
    Generates and saves the classification report.
    
    Args:
        all_labels (list): List of true labels.
        all_preds (list): List of predicted labels.
        classes (list): List of class names.
        save_path (str, optional): Path to save the classification report. Default is 'plots/classification_report.txt'.
    """
    report = classification_report(all_labels, all_preds, target_names=classes)
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"Classification report saved to '{save_path}'.")

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Train ASLClassifier Model with Validation Metrics")
    parser.add_argument("--data_dir", type=str, help="Directory containing ASL `.npy` files.", default='data')
    parser.add_argument("--stats_path", type=str, help="Path to the normalization stats `.npy` file.", default='all_data_stats.npy')
    parser.add_argument("--batch_size", type=int, help="Batch size for training.", default=16)
    parser.add_argument("--num_workers", type=int, help="Number of worker threads for DataLoader.", default=4)
    parser.add_argument("--learning_rate", type=float, help="Learning rate for optimizer.", default=0.0005)
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=50)
    parser.add_argument("--dropout_p", type=float, help="Dropout probability in ASLClassifier.", default=0.1)
    parser.add_argument("--validation_split", type=float, help="Fraction of data to use for validation.", default=0.2)
    parser.add_argument("--save_dir", type=str, help="Directory to save plots and reports.", default='plots')
    
    args = parser.parse_args()
    
    # Define model save path
    model_save_path = f'asl_classifier_{args.num_epochs}-epchs_{args.learning_rate}-lr_{args.dropout_p}-drp.pth'
    
    # Initialize DataLoader
    # First, load the entire dataset
    full_dataset = get_dataloader(
        data_dir=args.data_dir,
        stats_path=args.stats_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        transform=None  # Or pass any additional transforms if needed
    ).dataset  # Assuming get_dataloader returns a DataLoader
    
    # Determine sizes for training and validation
    val_size = int(args.validation_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Dataset split into {train_size} training samples and {val_size} validation samples.")
    
    # Create DataLoaders for training and validation
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize the model
    num_features = 555  # As per your ASLDataset's total_features
    num_classes = 10    # As per your label mapping (A-J)
    model = ASLClassifier(input_size=num_features, num_classes=num_classes, dropout_p=args.dropout_p)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Train the model
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    all_train_preds = []
    all_train_labels = []
    
    all_val_preds = []
    all_val_labels = []
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch [{epoch+1}/{args.num_epochs}]")
        
        # Training phase
        epoch_train_loss, epoch_train_acc, train_preds, train_labels = train(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        all_train_preds.extend(train_preds)
        all_train_labels.extend(train_labels)
        
        # Validation phase
        epoch_val_loss, epoch_val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        all_val_preds.extend(val_preds)
        all_val_labels.extend(val_labels)
        
        print(f"Training - Loss: {epoch_train_loss:.4f} - Accuracy: {epoch_train_acc:.2f}%")
        print(f"Validation - Loss: {epoch_val_loss:.4f} - Accuracy: {epoch_val_acc:.2f}%")
    
    print("\nTraining complete.")
    
    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to '{model_save_path}'.")
    
    # Define class names (ensure this matches your label mapping)
    classes = [chr(i) for i in range(65, 65 + num_classes)]  # ['A', 'B', ..., 'J']
    
    # Plot training and validation loss and accuracy
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, args.num_epochs, save_dir=args.save_dir)
    
    # Plot confusion matrix for training set
    plot_confusion_matrix(
        all_train_labels,
        all_train_preds,
        classes,
        title='Confusion Matrix - Training Set',
        save_path=os.path.join(args.save_dir, 'confusion_matrix_training.png')
    )
    
    # Plot confusion matrix for validation set
    plot_confusion_matrix(
        all_val_labels,
        all_val_preds,
        classes,
        title='Confusion Matrix - Validation Set',
        save_path=os.path.join(args.save_dir, 'confusion_matrix_validation.png')
    )
    
    # Generate classification report for training set
    generate_classification_report(
        all_train_labels,
        all_train_preds,
        classes,
        save_path=os.path.join(args.save_dir, 'classification_report_training.txt')
    )
    
    # Generate classification report for validation set
    generate_classification_report(
        all_val_labels,
        all_val_preds,
        classes,
        save_path=os.path.join(args.save_dir, 'classification_report_validation.txt')
    )
    
if __name__ == "__main__":
    main()
