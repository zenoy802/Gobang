"""
Gobang (Five in a Row) game environment implementation.
Provides game state management and rule enforcement.
"""

import numpy as np

class GobangEnv:
    """
    Gobang game environment that follows gym-like interface.
    Manages game state and implements game rules.
    """
    
    def __init__(self, board_size=15):
        """
        Initialize the game environment.
        Args:
            board_size (int): Size of the game board (default: 15x15)
        """
        self.board_size = board_size
        self.reset()

    def reset(self):
        """Reset the game to initial state"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # 1 for first player, -1 for second player
        self.done = False
        return self.get_state()

    def get_state(self):
        """Return current board state in the format expected by the neural network"""
        return self.board.reshape(1, self.board_size, self.board_size)

    def get_valid_moves(self):
        """Return indices of empty positions on the board"""
        return np.where(self.board.flatten() == 0)[0]

    def step(self, action):
        """
        Execute one step in the environment.
        Args:
            action (int): Position to place the stone (0 to board_size^2 - 1)
        Returns:
            tuple: (next_state, reward, done)
        """
        if self.done:
            return self.get_state(), 0, True

        row = action // self.board_size
        col = action % self.board_size

        if self.board[row, col] != 0:
            return self.get_state(), -10, True

        self.board[row, col] = self.current_player

        if self._check_win(row, col):
            self.done = True
            return self.get_state(), 1, True

        if len(self.get_valid_moves()) == 0:
            self.done = True
            return self.get_state(), 0, True

        self.current_player = -self.current_player
        return self.get_state(), 0, False

    def _check_win(self, row, col):
        """
        Check if the current move results in a win.
        Args:
            row (int): Row of the last move
            col (int): Column of the last move
        Returns:
            bool: True if the current player has won
        """
        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # Check forward direction
            r, c = row + dr, col + dc
            while (0 <= r < self.board_size and 
                   0 <= c < self.board_size and 
                   self.board[r, c] == player):
                count += 1
                r += dr
                c += dc
            
            # Check backward direction
            r, c = row - dr, col - dc
            while (0 <= r < self.board_size and 
                   0 <= c < self.board_size and 
                   self.board[r, c] == player):
                count += 1
                r -= dr
                c -= dc
            
            if count >= 5:
                return True
        return False 