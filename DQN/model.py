import torch
import torch.nn as nn
import torch.nn.functional as F

class GobangNet(nn.Module):
    """
    Neural network architecture for the Gobang game.
    Takes a board state as input and outputs Q-values for all possible actions.
    
    Architecture:
    1. Four convolutional layers with batch normalization
    2. Two fully connected layers with dropout
    """
    
    def __init__(self, board_size=15):
        """
        Initialize the network architecture.
        Args:
            board_size (int): Size of the Gobang board (default: 15x15)
        """
        super(GobangNet, self).__init__()
        self.board_size = board_size
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Fully connected layers with dropout
        self.fc1 = nn.Linear(256 * board_size * board_size, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, board_size * board_size)

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (tensor): Input tensor of shape (batch_size, 1, board_size, board_size)
        Returns:
            tensor: Q-values for each possible action
        """
        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten and apply fully connected layers
        x = x.view(-1, 256 * self.board_size * self.board_size)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x 