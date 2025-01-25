"""
Visualization tool for Gobang game using Pygame.
Supports both agent vs agent and human vs agent gameplay modes.
"""

import pygame
import numpy as np
from agent import DQNAgent
from environment import GobangEnv
import time
import os

class GobangVisualizer:
    """
    Visualizer class for Gobang game using Pygame.
    Handles rendering and user interaction.
    """
    
    def __init__(self, board_size=15, cell_size=40):
        """
        Initialize the visualizer.
        Args:
            board_size (int): Size of the game board
            cell_size (int): Size of each cell in pixels
        """
        pygame.init()
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = 40
        
        # Calculate window size
        self.window_size = self.board_size * self.cell_size + 2 * self.margin
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Gobang AI")
        
        # Colors
        self.BACKGROUND = (219, 176, 102)
        self.LINE_COLOR = (0, 0, 0)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        
        # Initialize game environment
        self.env = GobangEnv(board_size)
        
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

    def agent_vs_agent(self, agent1_path, agent2_path, delay=1.0):
        """
        Run a game between two trained agents.
        Args:
            agent1_path (str): Path to first agent's checkpoint
            agent2_path (str): Path to second agent's checkpoint
            delay (float): Delay between moves in seconds
        """
        agent1 = DQNAgent(self.board_size)
        agent2 = DQNAgent(self.board_size)
        agent1.load(agent1_path)
        agent2.load(agent2_path)
        
        state = self.env.reset()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            if not self.env.done:
                current_agent = agent1 if self.env.current_player == 1 else agent2
                valid_moves = self.env.get_valid_moves()
                action = current_agent.act(state, valid_moves)
                state, reward, done = self.env.step(action)
                
                self.draw_board()
                self.draw_stones()
                pygame.display.flip()
                time.sleep(delay)
        
        pygame.quit()

    def human_vs_agent(self, agent_path):
        """
        Run a game between human player and trained agent.
        Args:
            agent_path (str): Path to agent's checkpoint
        """
        agent = DQNAgent(self.board_size)
        agent.load(os.path.join("checkpoint", agent_path))
        
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
            else:  # Agent's turn
                valid_moves = self.env.get_valid_moves()
                action = agent.act(state, valid_moves)
                state, reward, done = self.env.step(action)
        
        pygame.quit()

if __name__ == "__main__":
    visualizer = GobangVisualizer()
    
    # Uncomment one of these to run different modes:
    # visualizer.agent_vs_agent("model_episode_1000.pth", 
    #                          "model_episode_2000.pth")
    visualizer.human_vs_agent("best_model_r87.5_e4413.pth") 