"""
Test script for GobangEnv class.
Allows testing different board configurations and environment functions.
"""

import numpy as np
from environment import GobangEnv

class GobangEnvTester:
    def __init__(self, board_size=15):
        self.env = GobangEnv(board_size)
    
    def set_board(self, board_array):
        """
        Set the board to a specific configuration.
        Args:
            board_array: 2D numpy array or list of lists with board configuration
                        (1 for black, -1 for white, 0 for empty)
        """
        board_array = np.array(board_array)
        if board_array.shape != (self.env.board_size, self.env.board_size):
            raise ValueError(f"Board must be {self.env.board_size}x{self.env.board_size}")
        self.env.board = board_array
        return self.print_board()
    
    def print_board(self):
        """Print current board state in a readable format"""
        symbols = {0: ".", 1: "●", -1: "○"}
        print("\n  " + " ".join([f"{i:2d}" for i in range(self.env.board_size)]))
        for i in range(self.env.board_size):
            print(f"{i:2d}", end=" ")
            for j in range(self.env.board_size):
                print(f"{symbols[self.env.board[i,j]]} ", end=" ")
            print()
        print()
    
    def test_valid_moves(self):
        """Test and display valid moves"""
        valid_moves = self.env.get_valid_moves()
        print("Valid moves:", valid_moves)
        print("Number of valid moves:", len(valid_moves))
        return valid_moves
    
    def test_step(self, action):
        """
        Test step function with a specific action.
        Args:
            action: Position to place stone (0 to board_size^2 - 1)
        """
        print(f"Taking action: {action}")
        print(f"Row: {action // self.env.board_size}, Col: {action % self.env.board_size}")
        
        state, reward, done = self.env.step(action)
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Current player: {self.env.current_player}")
        self.print_board()
        return state, reward, done
    
    def test_check_win(self, row, col):
        """
        Test if a position results in a win.
        Args:
            row: Row to check
            col: Column to check
        """
        result = self.env._check_win(row, col)
        print(f"Win check at ({row}, {col}): {result}")
        return result
    
    def test_evaluate_position(self, row, col):
        """
        Test position evaluation.
        Args:
            row: Row to evaluate
            col: Column to evaluate
        """
        score = self.env._evaluate_position(row, col)
        print(f"Position evaluation at ({row}, {col}): {score}")
        return score

def main():
    # Initialize tester
    tester = GobangEnvTester(board_size=15)
    
    # Example 1: Test empty board
    print("=== Testing Empty Board ===")
    tester.env.reset()
    tester.print_board()
    tester.test_valid_moves()
    
    # Example 2: Test custom board configuration
    print("\n=== Testing Custom Board ===")
    custom_board = np.zeros((15, 15))
    custom_board[7:10, 7] = 1  # Three black stones in a row
    custom_board[7, 8] = -1    # One white stone
    tester.set_board(custom_board)
    tester.test_valid_moves()
    
    # Example 3: Test step function
    print("\n=== Testing Step Function ===")
    tester.test_step(112)  # Place stone at position 112
    
    # Example 4: Test win condition
    print("\n=== Testing Win Condition ===")
    win_board = np.zeros((15, 15))
    win_board[7:12, 7] = 1  # Five black stones in a row
    tester.set_board(win_board)
    tester.test_check_win(9, 7)  # Check middle stone
    
    # Example 5: Test position evaluation
    print("\n=== Testing Position Evaluation ===")
    eval_board = np.zeros((15, 15))
    eval_board[7:10, 7] = 1  # Three black stones in a row
    tester.set_board(eval_board)
    tester.test_evaluate_position(8, 7)  # Evaluate middle stone

if __name__ == "__main__":
    main() 