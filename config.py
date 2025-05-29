"""
Configuration file for Sudoku RL training
"""

# Data configuration
SUDOKU_CSV_PATH = "./sudoku-3m.csv"
CSV_COLUMNS = ["quizzes", "solutions"]

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

# Training configuration
EPISODES_PER_PUZZLE = 10
WARMUP_STEPS = 200
LEARN_FREQUENCY = 5
