import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PPOGobangNet(nn.Module):
    """
    Neural network architecture for the Gobang game using PPO.
    Outputs both policy (action probabilities) and value function.
    """
    
    def __init__(self, board_size=15):
        super(PPOGobangNet, self).__init__()
        self.board_size = board_size
        
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, board_size * board_size)
        
        # Value head
        self.value_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Shared layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * self.board_size * self.board_size)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = self.value_fc2(value)
        
        return policy, value

    def get_action(self, state, valid_moves, device):
        """Get action and its log probability"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        policy, value = self(state)
        
        # Mask invalid moves
        mask = torch.ones_like(policy) * float('-inf')
        mask[0, valid_moves] = 0
        policy = policy + mask
        
        # Get action distribution
        probs = F.softmax(policy, dim=-1)
        dist = Categorical(probs)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value 