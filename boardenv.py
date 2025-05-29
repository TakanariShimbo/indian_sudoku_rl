import time
import os

import numpy as np

from data_loader import SudokuDataLoader
from config import SUDOKU_SIZE, AGENT_START_POSITION


class SudokuEnvironment:
    """Sudoku game environment for RL agent"""

    def __init__(self):
        # Action space: movement (up, down, left, right) + number insertion (1-9)
        self.action_space = ["u", "d", "l", "r", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.n_actions = len(self.action_space)

        # State features: agent position (row, col)
        self.n_features = 2

        # Data loader
        self.data_loader = SudokuDataLoader()
        self.total_puzzles = self.data_loader.get_puzzle_count()

        # Current puzzle index
        self.current_puzzle_index = 0

        # Initialize first puzzle
        self._load_puzzle()

    def _load_puzzle(self):
        """Load the current puzzle and its solution"""
        if self.current_puzzle_index >= self.total_puzzles:
            self.current_puzzle_index = 0  # Wrap around

        self.puzzle_array, self.solution_array, self.binary_mask = self.data_loader.get_puzzle_and_solution(self.current_puzzle_index)

        # Agent starts at the configured position
        self.agent_position = np.array(AGENT_START_POSITION.copy())

        # Move to next puzzle for next reset
        self.current_puzzle_index += 1

    def reset(self):
        """Reset environment and return initial state"""
        time.sleep(0.0001)
        self.agent_position = np.array(AGENT_START_POSITION.copy())
        return self.agent_position.copy()

    def step(self, action):
        """
        Execute action and return next state, reward, and done flag

        Args:
            action: Integer action index

        Returns:
            tuple: (next_state, reward, done)
        """
        current_pos = self.agent_position.copy()

        # Movement actions (0-3)
        if action == 0:  # up
            if current_pos[0] > 0:
                current_pos[0] -= 1
        elif action == 1:  # down
            if current_pos[0] < SUDOKU_SIZE - 1:
                current_pos[0] += 1
        elif action == 2:  # left
            if current_pos[1] > 0:
                current_pos[1] -= 1
        elif action == 3:  # right
            if current_pos[1] < SUDOKU_SIZE - 1:
                current_pos[1] += 1

        # Number insertion actions (4-12)
        elif 4 <= action <= 12:
            number = str(action - 3)  # Convert action to number (1-9)
            row, col = self.agent_position

            # Only insert if the cell was originally empty
            if self.binary_mask[row, col] == "0":
                self.puzzle_array[row, col] = number

        # Calculate reward
        reward, done = self._calculate_reward()

        # Update agent position for movement actions
        if action < 4:
            self.agent_position = current_pos

        # Display current state (optional)
        self._display_state()

        return self.agent_position.copy(), reward, done

    def _calculate_reward(self):
        """Calculate reward based on current puzzle state"""
        if np.array_equal(self.puzzle_array, self.solution_array):
            return 1.0, True  # Puzzle solved
        else:
            return 0.0, False  # Puzzle not yet solved

    def _display_state(self):
        """Display current puzzle state (optional)"""
        if hasattr(self, "_display_enabled") and self._display_enabled:
            os.system("cls" if os.name == "nt" else "clear")
            print("Current Sudoku State:")
            print("-" * 19)
            for i, row in enumerate(self.puzzle_array):
                if i % 3 == 0 and i != 0:
                    print("-" * 19)
                row_str = ""
                for j, cell in enumerate(row):
                    if j % 3 == 0 and j != 0:
                        row_str += "| "
                    # Highlight agent position
                    if [i, j] == self.agent_position.tolist():
                        row_str += f"[{cell}]"
                    else:
                        row_str += f" {cell} "
                print(row_str)
            print("-" * 19)
            time.sleep(0.001)

    def enable_display(self, enabled=True):
        """Enable or disable visual display of the puzzle"""
        self._display_enabled = enabled

    def get_current_state(self):
        """Get current state information"""
        return {
            "puzzle": self.puzzle_array.copy(),
            "solution": self.solution_array.copy(),
            "agent_position": self.agent_position.copy(),
            "binary_mask": self.binary_mask.copy(),
        }

    def next_puzzle(self):
        """Load the next puzzle"""
        self._load_puzzle()
