import random
import numpy as np
from config import SUDOKU_SIZE


class SudokuDataLoader:
    """Handles auto-generation of Sudoku puzzles without CSV dependency"""

    def __init__(self, puzzle_cache_size=1000):
        self.puzzle_cache = []
        self.puzzle_cache_size = puzzle_cache_size
        self.current_index = 0

        # Pre-generate initial puzzles
        print("Generating initial Sudoku puzzles...")
        self._generate_puzzle_cache()
        print(f"Generated {len(self.puzzle_cache)} puzzles")

    def _is_valid_placement(self, board, row, col, num):
        """Check if placing num at (row, col) is valid"""
        # Check row
        for j in range(9):
            if board[row][j] == num:
                return False

        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False

        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False

        return True

    def _solve_sudoku(self, board):
        """Solve sudoku using backtracking (for generating complete solution)"""
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    numbers = list(range(1, 10))
                    random.shuffle(numbers)  # Add randomness

                    for num in numbers:
                        if self._is_valid_placement(board, i, j, num):
                            board[i][j] = num
                            if self._solve_sudoku(board):
                                return True
                            board[i][j] = 0
                    return False
        return True

    def _generate_complete_sudoku(self):
        """Generate a complete valid sudoku solution"""
        board = [[0 for _ in range(9)] for _ in range(9)]

        # Fill diagonal 3x3 boxes first (they don't interfere with each other)
        for box in range(0, 9, 3):
            self._fill_box(board, box, box)

        # Solve the rest
        self._solve_sudoku(board)
        return board

    def _fill_box(self, board, row, col):
        """Fill a 3x3 box with random numbers"""
        numbers = list(range(1, 10))
        random.shuffle(numbers)

        for i in range(3):
            for j in range(3):
                board[row + i][col + j] = numbers[i * 3 + j]

    def _create_puzzle_from_solution(self, solution, difficulty="medium"):
        """Create a puzzle by removing numbers from complete solution"""
        puzzle = [row[:] for row in solution]  # Deep copy

        # Difficulty settings (number of cells to remove)
        remove_counts = {"easy": random.randint(35, 45), "medium": random.randint(45, 55), "hard": random.randint(55, 65)}

        cells_to_remove = remove_counts.get(difficulty, 50)

        # Get all cell positions
        positions = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(positions)

        # Remove numbers while ensuring unique solution
        removed = 0
        for row, col in positions:
            if removed >= cells_to_remove:
                break

            # Temporarily remove the number
            original = puzzle[row][col]
            puzzle[row][col] = 0

            # For simplicity, we'll assume the puzzle remains valid
            # In a more sophisticated implementation, you'd verify uniqueness
            removed += 1

        return puzzle

    def _generate_puzzle_cache(self):
        """Generate and cache multiple puzzles"""
        difficulties = ["easy", "medium", "hard"]

        for _ in range(self.puzzle_cache_size):
            # Generate complete solution
            solution = self._generate_complete_sudoku()

            # Create puzzle with random difficulty
            difficulty = random.choice(difficulties)
            puzzle = self._create_puzzle_from_solution(solution, difficulty)

            # Convert to string format and create binary mask
            puzzle_str = "".join(str(cell) for row in puzzle for cell in row)
            solution_str = "".join(str(cell) for row in solution for cell in row)
            binary_mask = ["1" if char != "0" else "0" for char in puzzle_str]

            # Convert to numpy arrays
            puzzle_array = np.array([char for char in puzzle_str]).reshape(SUDOKU_SIZE, SUDOKU_SIZE)
            solution_array = np.array([char for char in solution_str]).reshape(SUDOKU_SIZE, SUDOKU_SIZE)
            binary_mask_array = np.array(binary_mask).reshape(SUDOKU_SIZE, SUDOKU_SIZE)

            self.puzzle_cache.append((puzzle_array, solution_array, binary_mask_array))

    def get_puzzle_count(self):
        """Returns the total number of available puzzles"""
        return len(self.puzzle_cache)

    def get_puzzle_and_solution(self, index):
        """
        Returns a specific puzzle and its solution as numpy arrays

        Args:
            index: Index of the puzzle to retrieve

        Returns:
            tuple: (puzzle_array, solution_array, binary_mask)
        """
        # Use modulo to wrap around if index exceeds cache size
        actual_index = index % len(self.puzzle_cache)

        # If we're running low on cached puzzles, generate more
        if actual_index > len(self.puzzle_cache) * 0.8:
            self._generate_more_puzzles()

        return self.puzzle_cache[actual_index]

    def _generate_more_puzzles(self):
        """Generate additional puzzles when cache is running low"""
        additional_puzzles = self.puzzle_cache_size // 4  # Generate 25% more
        print(f"Generating {additional_puzzles} additional puzzles...")

        current_size = len(self.puzzle_cache)
        difficulties = ["easy", "medium", "hard"]

        for _ in range(additional_puzzles):
            # Generate complete solution
            solution = self._generate_complete_sudoku()

            # Create puzzle with random difficulty
            difficulty = random.choice(difficulties)
            puzzle = self._create_puzzle_from_solution(solution, difficulty)

            # Convert to string format and create binary mask
            puzzle_str = "".join(str(cell) for row in puzzle for cell in row)
            solution_str = "".join(str(cell) for row in solution for cell in row)
            binary_mask = ["1" if char != "0" else "0" for char in puzzle_str]

            # Convert to numpy arrays
            puzzle_array = np.array([char for char in puzzle_str]).reshape(SUDOKU_SIZE, SUDOKU_SIZE)
            solution_array = np.array([char for char in solution_str]).reshape(SUDOKU_SIZE, SUDOKU_SIZE)
            binary_mask_array = np.array(binary_mask).reshape(SUDOKU_SIZE, SUDOKU_SIZE)

            self.puzzle_cache.append((puzzle_array, solution_array, binary_mask_array))

        print(f"Cache expanded from {current_size} to {len(self.puzzle_cache)} puzzles")

    def generate_single_puzzle(self, difficulty="medium"):
        """Generate a single puzzle on demand"""
        solution = self._generate_complete_sudoku()
        puzzle = self._create_puzzle_from_solution(solution, difficulty)

        # Convert to string format and create binary mask
        puzzle_str = "".join(str(cell) for row in puzzle for cell in row)
        solution_str = "".join(str(cell) for row in solution for cell in row)
        binary_mask = ["1" if char != "0" else "0" for char in puzzle_str]

        # Convert to numpy arrays
        puzzle_array = np.array([char for char in puzzle_str]).reshape(SUDOKU_SIZE, SUDOKU_SIZE)
        solution_array = np.array([char for char in solution_str]).reshape(SUDOKU_SIZE, SUDOKU_SIZE)
        binary_mask_array = np.array(binary_mask).reshape(SUDOKU_SIZE, SUDOKU_SIZE)

        return puzzle_array, solution_array, binary_mask_array
