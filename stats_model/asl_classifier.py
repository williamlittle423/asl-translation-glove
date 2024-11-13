import torch
import torch.nn as nn
import torch.nn.functional as F

class ASLClassifier(nn.Module):
    def __init__(self, input_size=555, num_classes=10, dropout_p=0.5):
        """
        Initializes the ASLClassifier model.
        
        Args:
            input_size (int): Number of input features. Default is 555.
            num_classes (int): Number of output classes. Default is 26 for ASL letters.
            dropout_p (float): Dropout probability. Default is 0.5.
        """
        super(ASLClassifier, self).__init__()
        
        # First Fully Connected Layer
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(p=dropout_p)
        
        # Second Fully Connected Layer
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(p=dropout_p)
        
        # Third Fully Connected Layer
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(p=dropout_p)
        
        # Output Layer
        self.fc4 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # First Layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second Layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Third Layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output Layer
        x = self.fc4(x)
        
        return x  # Note: CrossEntropyLoss in PyTorch applies Softmax internally
