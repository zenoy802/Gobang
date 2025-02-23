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
    
    def __init__(self, board_size=16):
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
        # return self.board.reshape(1, self.board_size, self.board_size)
        return self.board.flatten()

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
            return self.get_state(), -0.5, True

        self.board[row, col] = self.current_player

        if self._get_max_consecutive_num(row, col) >= 5:
            self.done = True
            return self.get_state(), 5.0, True  # Increased reward for winning

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

    def _get_max_consecutive_num(self, row, col):
        """
        Check if the current move results in a win.
        Args:
            row (int): Row of the last move
            col (int): Column of the last move
        Returns:
            max_count (int): Maximum consecutive stones count
        """
        # player = self.board[row, col]
        player = self.current_player
        if self.board[row, col] != player:
            return False
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        max_count = 0
        
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
            
            max_count = max(max_count, count)
        return max_count

    def _evaluate_position(self, row, col):
        """
        Evaluate the potential of a position based on:
        1. Offensive potential (consecutive stones within 5 steps)
        2. Defensive necessity (blocking opponent's winning moves)
        3. Prioritize winning moves when available
        """
        # player = self.board[row, col]
        player = self.current_player
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        total_score = 0.0
        
        # First check if there are any winning moves or good moves available
        winning_positions = set()
        consecutive_4_positions = set()
        consecutive_3_positions = set()
        consecutive_2_positions = set()
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] == 0:  # Empty position
                    self.board[r, c] = player
                    max_count = self._get_max_consecutive_num(r, c)
                    if max_count >= 5:
                        winning_positions.add((r, c))
                    elif max_count == 4:
                        consecutive_4_positions.add((r, c))
                    elif max_count == 3:
                        consecutive_3_positions.add((r, c))
                    elif max_count == 2:
                        consecutive_2_positions.add((r, c))
                    self.board[r, c] = 0  # Reset position
        
        # If winning moves exist, only give high reward to those positions
        if winning_positions:
            current_pos = (row, col)
            if current_pos in winning_positions:
                return 5.0  # High reward for winning move
            else:
                return 0.0  # Penalize non-winning moves when win is possible
        # elif consecutive_4_positions:
        #     current_pos = (row, col)
        #     if current_pos in consecutive_4_positions:
        #         total_score += 25.0
        # elif consecutive_3_positions:
        #     current_pos = (row, col)
        #     if current_pos in consecutive_3_positions:
        #         total_score += 15.0
        # elif consecutive_2_positions:
        #     current_pos = (row, col)
        #     if current_pos in consecutive_2_positions:
        #         total_score += 5.0
        
        # # add rewards for putting stones in groups
        # def count_stones_in_direction(r, c, dr, dc, player, max_steps=2):
        #     """Count consecutive stones and empty spaces within max_steps"""
        #     count = 0
        #     empty_spots = []
        #     steps = 0
        #     curr_r, curr_c = r, c
            
        #     while steps < max_steps and 0 <= curr_r < self.board_size and 0 <= curr_c < self.board_size:
        #         if self.board[curr_r, curr_c] == player:
        #             count += 1
        #         elif self.board[curr_r, curr_c] == 0:
        #             empty_spots.append((curr_r, curr_c))
        #         else:  # Opponent stone
        #             break
        #         steps += 1
        #         curr_r += dr
        #         curr_c += dc
            
        #     return count, empty_spots
        
        # # Check each direction for both offensive and defensive potential
        # for dr, dc in directions:
        #     # Check forward and backward for offensive potential
        #     forward_count, forward_empty = count_stones_in_direction(row+dr, col+dc, dr, dc, player)
        #     backward_count, backward_empty = count_stones_in_direction(row-dr, col-dc, -dr, -dc, player)
        #     total_stones = forward_count + backward_count + 1  # +1 for current stone
            
        #     # Offensive scoring
        #     if total_stones >= 4:
        #         total_score += 5.0  # In a group of more than 4 stones
            
        #     # Check opponent's threats
        #     opponent = -player
        #     fwd_opponent_count, fwd_opponent_empty = count_stones_in_direction(row+dr, col+dc, dr, dc, opponent)
        #     bwd_opponent_count, bwd_opponent_empty = count_stones_in_direction(row-dr, col-dc, -dr, -dc, opponent)
        #     opponent_stones = fwd_opponent_count + bwd_opponent_count
            
        #     # Defensive scoring
        #     if opponent_stones >= 3:
        #         total_score += 1.5  # Critical defensive move
        #     elif opponent_stones == 2 and (len(fwd_opponent_empty) + len(bwd_opponent_empty)) >= 2:
        #         total_score += 0.8  # Potential defensive necessity
        
        return total_score 