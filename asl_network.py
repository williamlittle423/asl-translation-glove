import torch
import torch.nn as nn

class ASLNetwork(nn.Module):
    def __init__(self, input_size, num_classes, hidden_dim, hidden_layers, dropout_prob=0.5):
        super(ASLNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)  # First hidden layer
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Initialize hidden layers using nn.ModuleList
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob)
            ))

        self.fc2 = nn.Linear(hidden_dim, num_classes)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        for layer in self.hidden_layers:
            x = layer(x)  # Each layer includes Linear + BatchNorm + ReLU + Dropout

        x = self.fc2(x)
        return x
