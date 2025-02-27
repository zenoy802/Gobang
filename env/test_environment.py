"""
Test script for GobangEnv class.
Allows testing different board configurations and environment functions.
"""

import numpy as np
from env.environment import GobangEnv

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
    
    def print_board(self, show_scores=False):
        """
        Print current board state in a readable format.
        Args:
            show_scores (bool): If True, show evaluation scores for empty spots
        """
        if not show_scores:
            print("\n  " + " ".join([f"{i:2d}" for i in range(self.env.board_size)]))
            for i in range(self.env.board_size):
                print(f"{i:2d}", end=" ")
                for j in range(self.env.board_size):
                    if self.env.board[i,j] == 1:
                        print(" ● ", end="")
                    elif self.env.board[i,j] == -1:
                        print(" ○ ", end="")
                    else:
                        print(" . ", end="")
                print()
            print()
        else:
            print("\n  " + "    ".join([f"{i:2d}" for i in range(self.env.board_size)]))
            for i in range(self.env.board_size):
                print(f"{i:2d}", end=" ")
                for j in range(self.env.board_size):
                    if self.env.board[i,j] == 1:
                        print(" ● ", end="")
                    elif self.env.board[i,j] == -1:
                        print(" ○ ", end="")
                    else:
                        score = self.env._evaluate_position(i, j)
                        print(f" {score:3.1f} ", end="")
                print()
            print()
    
    def test_valid_moves(self):
        """Test and display valid moves"""
        valid_moves = self.env.get_valid_moves()
        print("Valid moves:", valid_moves)
        print("Number of valid moves:", len(valid_moves))
        for i in range(self.env.board_size**2):
            if i not in valid_moves:
                print(f"invalid_move: {self.env._get_row_col(i)}")
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
        self.print_board(show_scores=True)  # Show scores for all empty positions
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
    custom_board[0, 0:5] = -1    # One white stone
    tester.set_board(custom_board)
    tester.test_valid_moves()
    
    # Example 3: Test step function
    print("\n=== Testing Step Function ===")
    tester.test_step(112)  # Place stone at position 112
    tester.test_step(113)  # Place stone at position 113
    tester.test_step(111)  # Place stone at position 111
    tester.test_step(100)  # Place stone at position 112
    
    # Example 4: Test win condition
    print("\n=== Testing Win Condition ===")
    win_board = np.zeros((15, 15))
    win_board[7:12, 7] = 1  # Five black stones in a row
    win_board[9, 7] = 0
    tester.set_board(win_board)
    tester.test_check_win(9, 7)  # Should return True
    
    # Test case 2: More than 5 in a row (should not win)
    win_board = np.zeros((15, 15))
    win_board[7:13, 7] = 1  # Six black stones in a row
    tester.set_board(win_board)
    tester.test_check_win(9, 7)  # Should return False
    
    # Test case 3: 5 stones with gap
    win_board = np.zeros((15, 15))
    win_board[7:10, 7] = 1  # Three black stones
    win_board[11:13, 7] = 1  # Two more black stones with gap
    tester.set_board(win_board)
    tester.test_check_win(8, 7)  # Should return False
    
    # Test case 4: Diagonal win
    win_board = np.zeros((15, 15))
    for i in range(5):  # Five black stones diagonally
        win_board[7+i, 7+i] = 1
    win_board[7, 7] = 0
    tester.set_board(win_board)
    tester.test_check_win(7, 7)  # Should return True
    tester.test_check_win(12, 12)  # Should return True
    
    # Test case 5: Anti-diagonal win
    win_board = np.zeros((15, 15))
    for i in range(5):  # Five black stones in anti-diagonal
        win_board[7+i, 11-i] = 1
    tester.set_board(win_board)
    tester.test_check_win(9, 9)  # Should return True
    
    # Test case 6: Near board edge
    win_board = np.zeros((15, 15))
    win_board[0:5, 0] = 1  # Five black stones at edge
    tester.set_board(win_board)
    tester.test_check_win(2, 0)  # Should return True
    
    # Example 5: Test position evaluation
    print("\n=== Testing Position Evaluation ===")
    
    # Test case 1: Offensive potential - three in a row
    eval_board = np.zeros((15, 15))
    eval_board[7:10, 7] = 1  # Three black stones in a row
    tester.set_board(eval_board)
    print("\nOffensive - Three in a row:")
    tester.test_evaluate_position(6, 7)  # Position before three
    tester.test_evaluate_position(10, 7)  # Position after three
    
    # Test case 2: Defensive necessity - block opponent's three
    eval_board = np.zeros((15, 15))
    eval_board[7:10, 7] = -1  # Three white stones in a row
    tester.set_board(eval_board)
    print("\nDefensive - Block opponent's three:")
    tester.test_evaluate_position(6, 7)  # Block one end
    tester.test_evaluate_position(10, 7)  # Block other end
    
    # Test case 3: Mixed potential
    eval_board = np.zeros((15, 15))
    eval_board[7:9, 7] = 1   # Two black stones
    eval_board[7:10, 9] = -1 # Three white stones
    tester.set_board(eval_board)
    print("\nMixed - Offense vs Defense:")
    tester.test_evaluate_position(9, 7)  # Continue own line
    tester.test_evaluate_position(6, 9)  # Block opponent
    
    # Test case 4: Diagonal threats
    eval_board = np.zeros((15, 15))
    for i in range(3):
        eval_board[7+i, 7+i] = -1  # Three white stones diagonal
    tester.set_board(eval_board)
    print("\nDiagonal threats:")
    tester.test_evaluate_position(6, 6)   # Block one end
    tester.test_evaluate_position(10, 10) # Block other end
    
    # Test case 5: Multiple directions
    eval_board = np.zeros((15, 15))
    eval_board[7:9, 7] = 1    # Two black vertical
    eval_board[7, 7:9] = 1    # Two black horizontal
    tester.set_board(eval_board)
    print("\nMultiple directions:")
    tester.test_evaluate_position(9, 7)  # Continue vertical
    tester.test_evaluate_position(7, 9)  # Continue horizontal
    
    # Test case 6: Negative tests - Blocked positions
    print("\n=== Testing Blocked Positions ===")
    
    # Test 6.1: Blocked on both ends
    eval_board = np.zeros((15, 15))
    eval_board[7:10, 7] = 1    # Three black stones
    eval_board[6, 7] = -1      # Blocked by white
    eval_board[10, 7] = -1     # Blocked by white
    tester.set_board(eval_board)
    print("\nBlocked on both ends:")
    tester.test_evaluate_position(5, 7)  # Should have low score
    tester.test_evaluate_position(8, 6)
    
    # Test 6.2: No space for five
    eval_board = np.zeros((15, 15))
    eval_board[0:3, 0] = 1     # Three black stones at edge
    tester.set_board(eval_board)
    print("\nNo space for five:")
    tester.test_evaluate_position(1, 0)  # Should have low score
    
    # Test 6.3: Scattered stones
    eval_board = np.zeros((15, 15))
    eval_board[7, 7] = 1
    eval_board[7, 9] = 1
    eval_board[7, 11] = 1      # Scattered stones with gaps
    tester.set_board(eval_board)
    print("\nScattered stones:")
    tester.test_evaluate_position(7, 8)  # Should have low score
    tester.test_evaluate_position(7, 10)  # Should have low score
    
    # Test 6.4: Wrong defensive move
    eval_board = np.zeros((15, 15))
    eval_board[7:10, 7] = -1   # Three opponent stones
    eval_board[8, 8] = 1       # Wrong blocking position
    tester.set_board(eval_board)
    print("\nWrong defensive move:")
    tester.test_evaluate_position(8, 8)  # Should have low score
    tester.test_evaluate_position(10, 7)  # Should have high score (correct block)
    
    # Test 6.5: Mixed ineffective positions
    eval_board = np.zeros((15, 15))
    eval_board[7, 7] = 1
    eval_board[8, 8] = -1
    eval_board[9, 9] = 1       # Mixed stones without clear pattern
    tester.set_board(eval_board)
    print("\nMixed ineffective positions:")
    tester.test_evaluate_position(6, 6)  # Should have low score
    tester.test_evaluate_position(10, 10)  # Should have low score
    
    # Test 6.6: Opponent winning move vs non-critical position
    eval_board = np.zeros((15, 15))
    eval_board[7:11, 7] = -1   # Four opponent stones
    eval_board[7:10, 9] = 1    # Three own stones
    tester.set_board(eval_board)
    print("\nOpponent winning move vs own progress:")
    tester.test_evaluate_position(11, 7)  # Should have very high score (block win)
    tester.test_evaluate_position(10, 9)  # Should have lower score (own progress)

if __name__ == "__main__":
    main() 