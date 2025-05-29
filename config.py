"""
Configuration file for Sudoku RL training
"""

# Puzzle generation configuration
PUZZLE_CACHE_SIZE = 1000  # Number of puzzles to pre-generate
DEFAULT_DIFFICULTY = "medium"  # 'easy', 'medium', 'hard'

# Environment configuration
SUDOKU_SIZE = 9
AGENT_START_POSITION = [0, 0]

# Neural network configuration
LEARNING_RATE = 0.01
REWARD_DECAY = 0.9
EPSILON_GREEDY = 0.9
EPSILON_INCREMENT = None
REPLACE_TARGET_ITER = 200
MEMORY_SIZE = 2000
BATCH_SIZE = 32
HIDDEN_SIZE = 20

# GPU configuration
ENABLE_CUDA = True
CUDA_DEVICE = 0
GRADIENT_CLIP = 1.0

# Training configuration
EPISODES_PER_PUZZLE = 10
WARMUP_STEPS = 200
LEARN_FREQUENCY = 5
