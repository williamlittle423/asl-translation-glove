import torch
from torch.utils.data import DataLoader
import argparse
from asl_dataset import get_dataloader
from asl_classifier import ASLClassifier
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import os

def train(model, dataloader, criterion, optimizer, device, num_epochs=25):
    """
    Trains the ASLClassifier model and collects metrics for analysis.

    Args:
        model (nn.Module): The neural network model to train.
        dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to train on.
        num_epochs (int, optional): Number of training epochs. Default is 25.

    Returns:
        tuple: Contains lists of epoch losses, epoch accuracies, all predictions, and all true labels.
    """
    model.to(device)

    epoch_losses = []
    epoch_accuracies = []
    all_preds = []
    all_labels = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (features, labels) in enumerate(dataloader):
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

        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

    print("Training complete.")
    return model, epoch_losses, epoch_accuracies, all_preds, all_labels

def plot_metrics(epoch_losses, epoch_accuracies, num_epochs, save_dir='plots'):
    """
    Plots training loss and accuracy over epochs.

    Args:
        epoch_losses (list): List of loss values per epoch.
        epoch_accuracies (list): List of accuracy values per epoch.
        num_epochs (int): Total number of training epochs.
        save_dir (str, optional): Directory to save the plots. Default is 'plots'.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epochs = range(1, num_epochs + 1)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, epoch_losses, 'b-', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, epoch_accuracies, 'r-', label='Training Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_accuracy.png'))
    plt.close()

    print(f"Loss and accuracy plots saved to '{save_dir}' directory.")

def plot_confusion_matrix(all_labels, all_preds, classes, save_dir='plots'):
    """
    Plots and saves the confusion matrix.

    Args:
        all_labels (list): List of true labels.
        all_preds (list): List of predicted labels.
        classes (list): List of class names.
        save_dir (str, optional): Directory to save the plot. Default is 'plots'.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    
    plt.figure(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    print(f"Confusion matrix plot saved to '{save_dir}' directory.")

def generate_classification_report(all_labels, all_preds, classes, save_dir='plots'):
    """
    Generates and saves the classification report.

    Args:
        all_labels (list): List of true labels.
        all_preds (list): List of predicted labels.
        classes (list): List of class names.
        save_dir (str, optional): Directory to save the report. Default is 'plots'.
    """
    report = classification_report(all_labels, all_preds, target_names=classes)
    report_path = os.path.join(save_dir, 'classification_report.txt')
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Classification report saved to '{report_path}'.")

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Train ASLClassifier Model")
    parser.add_argument("--data_dir", type=str, help="Directory containing ASL `.npy` files.", default='data')
    parser.add_argument("--stats_path", type=str, help="Path to the normalization stats `.npy` file.", default='all_data_stats.npy')
    parser.add_argument("--batch_size", type=int, help="Batch size for training.", default=16)
    parser.add_argument("--num_workers", type=int, help="Number of worker threads for DataLoader.", default=4)
    parser.add_argument("--learning_rate", type=float, help="Learning rate for optimizer.", default=0.0005)
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=50)
    parser.add_argument("--dropout_p", type=float, help="Dropout probability in ASLClassifier.", default=0.1)
    parser.add_argument("--save_dir", type=str, help="Directory to save plots and reports.", default='plots')

    args = parser.parse_args()
    
    # Define model save path
    model_save_path = f'asl_classifier_{args.num_epochs}-epchs_{args.learning_rate}-lr_{args.dropout_p}-drp.pth'

    # Initialize DataLoader
    dataloader = get_dataloader(
        data_dir=args.data_dir,
        stats_path=args.stats_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        transform=None  # Or pass any additional transforms if needed
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
    trained_model, epoch_losses, epoch_accuracies, all_preds, all_labels = train(
        model, dataloader, criterion, optimizer, device, num_epochs=args.num_epochs
    )

    # Save the trained model
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Trained model saved to '{model_save_path}'.")

    # Define class names (ensure this matches your label mapping)
    classes = [chr(i) for i in range(65, 65 + num_classes)]  # ['A', 'B', ..., 'J']

    # Plot training loss and accuracy
    plot_metrics(epoch_losses, epoch_accuracies, args.num_epochs, save_dir=args.save_dir)

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, classes, save_dir=args.save_dir)

    # Generate classification report
    generate_classification_report(all_labels, all_preds, classes, save_dir=args.save_dir)

if __name__ == "__main__":
    main()
