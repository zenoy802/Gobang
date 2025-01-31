"""
PPO Agent implementation for the Gobang game.
Implements Proximal Policy Optimization with clipped objective.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import os
from ppo_model import PPOGobangNet

class PPOMemory:
    """Memory buffer for PPO experiences"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def clear(self):
        """Clear all stored experiences"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)

class PPOAgent:
    """
    PPO Agent implementation with:
    - Clipped surrogate objective
    - Value function clipping
    - Generalized Advantage Estimation (GAE)
    """
    
    def __init__(self, board_size=15, learning_rate=3e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_epsilon=0.2, c1=1.0, c2=0.01,
                 batch_size=64, n_epochs=10):
        """
        Initialize PPO agent.
        Args:
            board_size (int): Size of the game board
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter
            clip_epsilon (float): PPO clip parameter
            c1 (float): Value loss coefficient
            c2 (float): Entropy coefficient
            batch_size (int): Mini-batch size for updates
            n_epochs (int): Number of epochs to train on each batch
        """
        self.board_size = board_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA device:", torch.cuda.get_device_name(0))
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS device")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device")
        
        # Initialize network and optimizer
        self.model = PPOGobangNet(board_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # Add epsilon decay for exploration
        self.epsilon_start = 0.3
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.9995
        self.epsilon = self.epsilon_start
        
        # Initialize memory
        self.memory = PPOMemory()
    
    def act(self, state, valid_moves):
        """
        Select action using the current policy with epsilon-greedy exploration.
        Args:
            state: Current game state
            valid_moves: List of valid moves
        Returns:
            tuple: (action, log_prob, value)
        """
        self.model.eval()
        with torch.no_grad():
            action, log_prob, value = self.model.get_action(
                state, valid_moves, self.device, epsilon=self.epsilon
            )
        self.model.train()
        
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        
        return action, log_prob, value
    
    def remember(self, state, action, reward, next_state, log_prob, value, done):
        """Store transition in memory"""
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.rewards.append(reward)
        self.memory.next_states.append(next_state)
        self.memory.log_probs.append(log_prob)
        self.memory.values.append(value)
        self.memory.dones.append(done)
    
    def compute_gae(self, values, rewards, dones):
        """
        Compute Generalized Advantage Estimation.
        Args:
            values: List of value estimates
            rewards: List of rewards
            dones: List of done flags
        Returns:
            tensor: Advantage estimates
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        return torch.tensor(advantages, dtype=torch.float32)
    
    def train(self):
        """
        Update policy and value function using PPO.
        Returns:
            float: Average loss value
        """
        if len(self.memory) < self.batch_size:
            return None
        
        memory_size = len(self.memory)
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_log_probs = torch.stack(self.memory.log_probs).to(self.device)
        values = torch.stack(self.memory.values).squeeze().to(self.device)
        
        # Normalize rewards for better training stability
        rewards = np.array(self.memory.rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Compute advantages with normalized rewards
        advantages = self.compute_gae(
            values.cpu().numpy(),
            rewards,
            self.memory.dones
        ).to(self.device)
        
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        for _ in range(self.n_epochs):
            indices = np.random.permutation(len(self.memory))
            
            for start_idx in range(0, len(self.memory), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                policy, value = self.model(batch_states)
                dist = torch.distributions.Categorical(logits=policy)
                curr_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Calculate ratios and surrogate losses with dynamic clipping
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * batch_advantages
                
                # Use PPO-clip with additional KL divergence penalty
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value.view(-1), batch_returns.view(-1))
                
                # Add KL divergence penalty
                kl_div = (batch_log_probs - curr_log_probs).mean()
                loss = policy_loss + self.c1 * value_loss - self.c2 * entropy + 0.01 * kl_div.abs()
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        # Update learning rate and clip epsilon
        self.scheduler.step()
        
        self.memory.clear()
        return total_loss / (memory_size * self.n_epochs)
    
    def state_dict(self):
        """Get state dictionary of the agent"""
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon
        }
    
    def load_state_dict(self, state_dict):
        """Load state dictionary into the agent with backward compatibility"""
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        
        # Handle new fields with backward compatibility
        if 'scheduler_state_dict' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        else:
            print("Warning: Loading checkpoint without scheduler state. Resetting scheduler.")
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        if 'epsilon' in state_dict:
            self.epsilon = state_dict['epsilon']
        else:
            print("Warning: Loading checkpoint without epsilon value. Using default.")
            self.epsilon = self.epsilon_start
    
    def save(self, filename):
        """Save model checkpoint with all necessary states"""
        os.makedirs("checkpoint", exist_ok=True)
        state_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'version': 2  # Add version number for future compatibility
        }
        torch.save(state_dict, os.path.join("checkpoint", filename))
    
    def load(self, filename):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(
                filename, 
                map_location=self.device,
                weights_only=True
            )
            self.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint {filename}: {e}")
            raise 