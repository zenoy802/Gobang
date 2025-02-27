"""
DQN Agent implementation for the Gobang game.
Implements Deep Q-Learning with experience replay and target network.
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
from DQN.model import GobangNet

class DQNAgent:
    """
    Deep Q-Learning agent with experience replay and target network.
    Implements epsilon-greedy exploration and automatic device selection (CUDA/MPS/CPU).
    """
    
    def __init__(self, board_size=15, memory_size=10000, batch_size=64, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        """
        Initialize the DQN agent.
        Args:
            board_size (int): Size of the game board
            memory_size (int): Size of replay buffer
            batch_size (int): Size of training batch
            gamma (float): Discount factor for future rewards
            epsilon (float): Initial exploration rate
            epsilon_min (float): Minimum exploration rate
            epsilon_decay (float): Rate at which epsilon decays
            learning_rate (float): Learning rate for optimizer
        """
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Check available devices in order: CUDA -> MPS -> CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA device:", torch.cuda.get_device_name(0))
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS device")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device")
        
        self.model = GobangNet(board_size).to(self.device)
        self.target_model = GobangNet(board_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Enable automatic mixed precision for faster training
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.update_target_model()

    def update_target_model(self):
        """Copy weights from training model to target model"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store a single transition in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def remember_batch(self, states, actions, rewards, next_states, dones):
        """Store a batch of transitions in replay memory"""
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.memory.append((s, a, r, ns, d))

    def act(self, state, valid_moves):
        """
        Choose an action using epsilon-greedy policy.
        Args:
            state: Current game state
            valid_moves: List of valid moves
        Returns:
            int: Chosen action
        """
        if random.random() <= self.epsilon:
            return random.choice(valid_moves)
        
        with torch.cuda.amp.autocast():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor).cpu().detach().numpy()[0]
        
        # Mask invalid moves with very low values
        valid_moves_mask = np.ones_like(q_values) * float('-inf')
        valid_moves_mask[valid_moves] = 0
        q_values = q_values + valid_moves_mask
        
        return np.argmax(q_values)

    def train(self):
        """
        Train the network on a batch of experiences.
        Uses automatic mixed precision for faster training.
        Returns:
            float: The training loss value
        """
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        actions = torch.LongTensor(np.array([x[1] for x in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([x[2] for x in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([x[4] for x in batch])).to(self.device)

        # Use automatic mixed precision for faster training
        with torch.cuda.amp.autocast():
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_model(next_states).max(1)[0].detach()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

    def save(self, filename):
        """Save model checkpoint"""
        os.makedirs("checkpoint", exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, os.path.join("checkpoint", filename))

    def load(self, filename):
        """Load model checkpoint with proper device mapping"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_target_model() 