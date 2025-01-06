import torch
import torch.nn as nn
import torch.nn.functional as F

class GobangNet(nn.Module):
    def __init__(self, board_size=15):
        super(GobangNet, self).__init__()
        self.board_size = board_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * board_size * board_size, 512)
        self.fc2 = nn.Linear(512, board_size * board_size)

    def forward(self, x):
        # Input shape: (batch_size, 1, board_size, board_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(-1, 128 * self.board_size * self.board_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x 