import pandas as pd
import numpy as np

from config import SUDOKU_CSV_PATH, CSV_COLUMNS, SUDOKU_SIZE


class SudokuDataLoader:
    """Handles loading and preprocessing of Sudoku data"""

    def __init__(self):
        self.df = pd.read_csv(SUDOKU_CSV_PATH, skipinitialspace=True)
        self.quizzes = self.df["quizzes"].astype(str)
        self.solutions = self.df["solutions"].astype(str)

    def get_puzzle_count(self):
        """Returns the total number of puzzles"""
        return len(self.quizzes)

    def get_puzzle_and_solution(self, index):
        """
        Returns a specific puzzle and its solution as numpy arrays

        Args:
            index: Index of the puzzle to retrieve

        Returns:
            tuple: (puzzle_array, solution_array, binary_mask)
        """
        # Get puzzle and solution strings
        quiz_str = str(self.quizzes.iloc[index])
        solution_str = str(self.solutions.iloc[index])

        # Convert to character arrays
        quiz_chars = [char for char in quiz_str]
        solution_chars = [char for char in solution_str]

        # Create binary mask (1 for fixed cells, 0 for empty cells)
        binary_mask = ["1" if char != "0" else "0" for char in quiz_chars]

        # Reshape to 9x9 grids
        puzzle_array = np.array(quiz_chars).reshape(SUDOKU_SIZE, SUDOKU_SIZE)
        solution_array = np.array(solution_chars).reshape(SUDOKU_SIZE, SUDOKU_SIZE)
        binary_mask_array = np.array(binary_mask).reshape(SUDOKU_SIZE, SUDOKU_SIZE)

        return puzzle_array, solution_array, binary_mask_array
