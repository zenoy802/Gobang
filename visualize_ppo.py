"""
Visualization tool for PPO training results and model comparison.
Supports visual gameplay for model vs model and human vs model matches.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from ppo_agent import PPOAgent
from environment import GobangEnv
import pygame
import time

class PPOGameVisualizer:
    def __init__(self, board_size=15, cell_size=40):
        pygame.init()
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = 40
        
        # Calculate window size
        self.window_size = self.board_size * self.cell_size + 2 * self.margin
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("PPO Gobang")
        
        # Colors
        self.BACKGROUND = (219, 176, 102)
        self.LINE_COLOR = (0, 0, 0)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        
        # Initialize environment
        self.env = GobangEnv(board_size)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                 "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
                                 else "cpu")
    
    def draw_board(self):
        """Draw the game board with grid lines"""
        self.screen.fill(self.BACKGROUND)
        
        # Draw grid lines
        for i in range(self.board_size):
            # Vertical lines
            start_pos = (self.margin + i * self.cell_size, self.margin)
            end_pos = (self.margin + i * self.cell_size, self.window_size - self.margin)
            pygame.draw.line(self.screen, self.LINE_COLOR, start_pos, end_pos, 1)
            
            # Horizontal lines
            start_pos = (self.margin, self.margin + i * self.cell_size)
            end_pos = (self.window_size - self.margin, self.margin + i * self.cell_size)
            pygame.draw.line(self.screen, self.LINE_COLOR, start_pos, end_pos, 1)
    
    def draw_stones(self):
        """Draw the stones on the board"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.env.board[i][j] != 0:
                    color = self.BLACK if self.env.board[i][j] == 1 else self.WHITE
                    center = (self.margin + j * self.cell_size, 
                            self.margin + i * self.cell_size)
                    pygame.draw.circle(self.screen, color, center, self.cell_size // 2 - 2)
    
    def get_move_from_pos(self, pos):
        """Convert mouse position to board position"""
        x, y = pos
        x = (x - self.margin + self.cell_size // 2) // self.cell_size
        y = (y - self.margin + self.cell_size // 2) // self.cell_size
        
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            return y * self.board_size + x
        return None
    
    def load_model(self, model_path):
        """Load a trained PPO model"""
        state_dim = self.board_size * self.board_size
        action_dim = state_dim
        hidden_dim = 128
        
        agent = PPOAgent(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            actor_lr=1e-3,
            critic_lr=1e-2,
            lmbda=0.95,
            epochs=10,
            eps=0.2,
            gamma=0.98,
            device=self.device
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        return agent
    
    def model_vs_model_visual(self, model1_path, model2_path, delay=1.0):
        """Visual gameplay between two PPO models"""
        agent1 = self.load_model(model1_path)
        agent2 = self.load_model(model2_path)
        state = self.env.reset()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            if not self.env.done:
                mask = state == 0
                if self.env.current_player == 1:
                    current_agent = agent1
                else:
                    current_agent = agent2
                    state = -state
                action = current_agent.take_action(state, mask)
                state, reward, done = self.env.step(action)
                
                self.draw_board()
                self.draw_stones()
                pygame.display.flip()
                time.sleep(delay)
            else:
                time.sleep(2)
                running = False
        
        pygame.quit()
    
    def human_vs_model(self, model_path):
        """Visual gameplay between human and PPO model"""
        agent = self.load_model(model_path)
        state = self.env.reset()
        running = True
        
        while running:
            self.draw_board()
            self.draw_stones()
            pygame.display.flip()
            
            if self.env.done:
                time.sleep(2)
                running = False
                continue
            
            if self.env.current_player == 1:  # Human's turn
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        pos = pygame.mouse.get_pos()
                        action = self.get_move_from_pos(pos)
                        if action is not None and action in self.env.get_valid_moves():
                            state, reward, done = self.env.step(action)
            else:
                mask = state == 0  # Model's turn
                action = agent.take_action(state, mask)
                state, reward, done = self.env.step(action)
        
        pygame.quit()

def main():
    visualizer = PPOGameVisualizer()
    
    # Example paths to model checkpoints
    model1_path = "training_results/run_20250219_222613/model_episode_5000.pth"
    model2_path = "training_results/run_20250219_222613/model_episode_5000.pth"
    
    # Uncomment one of these to run different modes:
    
    # Watch two models play against each other
    visualizer.model_vs_model_visual(model1_path, model2_path, delay=0.1)
    
    # Play against the model
    # visualizer.human_vs_model(model2_path)

if __name__ == "__main__":
    main() 