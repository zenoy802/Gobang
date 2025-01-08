import torch
import torch.nn as nn
import torch.nn.functional as F

class GobangNet(nn.Module):
    """
    Neural network architecture for the Gobang game.
    Takes a board state as input and outputs Q-values for all possible actions.
    
    Architecture:
    1. Three convolutional layers for spatial feature extraction
    2. Two fully connected layers for action value prediction
    """
    
    def __init__(self, board_size=15):
        """
        Initialize the network architecture.
        Args:
            board_size (int): Size of the Gobang board (default: 15x15)
        """
        super(GobangNet, self).__init__()
        self.board_size = board_size
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input -> 32 features
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 -> 64 features
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 -> 128 features
        
        # Fully connected layers for Q-value prediction
        self.fc1 = nn.Linear(128 * board_size * board_size, 512)  # Flatten -> 512 neurons
        self.fc2 = nn.Linear(512, board_size * board_size)  # 512 -> action space

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (tensor): Input tensor of shape (batch_size, 1, board_size, board_size)
        Returns:
            tensor: Q-values for each possible action
        """
        # Apply convolutions with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten and apply fully connected layers
        x = x.view(-1, 128 * self.board_size * self.board_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation on final layer (Q-values can be negative)
        
        return x 