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

        # TODO: check the reason avg reward decrease
        if self.board[row, col] != 0:
            return self.get_state(), -10, True

        self.board[row, col] = self.current_player

        if self._check_win(row, col):
            self.done = True
            return self.get_state(), 10, True  # Increased reward for winning

        if len(self.get_valid_moves()) == 0:
            self.done = True
            return self.get_state(), 0, True

        # Add reward for creating potential winning positions
        potential_score = self._evaluate_position(row, col)
        # TODO: check if the player setup valid
        self.current_player = -self.current_player
        return self.get_state(), potential_score, False
    
    def _get_row_col(self, action):
        row = action // self.board_size
        col = action % self.board_size
        return row, col

    def _check_win(self, row, col):
        """
        Check if the current move results in a win.
        Args:
            row (int): Row of the last move
            col (int): Column of the last move
        Returns:
            bool: True if the current player has won (exactly 5 consecutive stones)
        """
        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1  # Count the current stone
            # Check forward direction
            r, c = row + dr, col + dc
            while (0 <= r < self.board_size and 
                   0 <= c < self.board_size):
                if self.board[r, c] == player:
                    count += 1
                    r += dr
                    c += dc
                else:
                    break
            
            # Check backward direction
            r, c = row - dr, col - dc
            while (0 <= r < self.board_size and 
                   0 <= c < self.board_size):
                if self.board[r, c] == player:
                    count += 1
                    r -= dr
                    c -= dc
                else:
                    break
            
            if count >= 5:
                return True
        return False

    def _evaluate_position(self, row, col):
        """Evaluate the potential of a position"""
        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        total_score = 0
        
        for dr, dc in directions:
            count = 1
            spaces = 0
            # Count in both directions
            for direction in [1, -1]:
                r, c = row + dr * direction, col + dc * direction
                while (0 <= r < self.board_size and 
                       0 <= c < self.board_size and 
                       (self.board[r, c] == player or self.board[r, c] == 0) and
                       spaces <= 2):
                    if self.board[r, c] == player:
                        count += 1
                    else:
                        spaces += 1
                    r += dr * direction
                    c += dc * direction
            
            # Score based on consecutive stones and spaces
            if count >= 4:
                total_score += 0.5
            elif count >= 3:
                total_score += 0.3
            elif count >= 2:
                total_score += 0.1
        
        return total_score 