import time
import pandas as pd
import numpy as np
import os


# Data Pre-processing
def split(word):
    return [char for char in word]


# Load the puzzles and solutions
COLUMNS = ["quizzes", "solutions"]
PATH = r".\sudoku.csv"
df_train = pd.read_csv(PATH, skipinitialspace=True, names=COLUMNS, index_col=False)

quizzes = df_train["quizzes"].astype(str)
solutions = df_train["solutions"].astype(str)


class board(object):
    def __init__(self):
        super(board, self).__init__()
        # Choose Actions: up, down, left, right, then insert 1–9
        self.action_space = ["u", "d", "l", "r", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.n_actions = len(self.action_space)
        # Features in Neural Network
        self.n_features = 2
        # Which puzzle we're on
        self.mazecount = 1
        self._build_maze()

    def _build_maze(self):
        # Making Environment
        self.currentquiz = quizzes.iloc[self.mazecount]
        self.quizreshaped = np.asarray(self.currentquiz)
        self.quizarray = split(str(self.quizreshaped))
        # Generating binary sudoku array for fixed-cell mask
        self.binaryquiz = []
        for i in self.quizarray:
            if i == "0":
                self.binaryquiz.append("0")
            else:
                self.binaryquiz.append("1")
        self.quizarray = np.array(self.quizarray).reshape(9, 9)
        self.binaryquizarray = np.array(self.binaryquiz).reshape(9, 9)
        # Agent starts at (0,0)
        self.agent = np.array([0, 0])

        # Extract solution for the current puzzle
        self.currentsolution = solutions.iloc[self.mazecount]
        self.solutionreshaped = np.asarray(self.currentsolution)
        self.solutionarray = split(str(self.solutionreshaped))
        self.solutionarray = np.array(self.solutionarray).reshape(9, 9)

        # Move on to the next puzzle index for next reset
        self.mazecount += 1

    def reset(self):
        time.sleep(0.0001)
        # On reset, agent goes back to (0,0)
        return np.array([0, 0])

    def step(self, action):
        s = self.agent
        stemp = s

        # Movement actions
        if action == 0:  # up
            if stemp[0] > 0:
                stemp[0] -= 1
        elif action == 1:  # down
            if stemp[0] < 8:
                stemp[0] += 1
        elif action == 2:  # right
            if stemp[1] < 8:
                stemp[1] += 1
        elif action == 3:  # left
            if stemp[1] > 0:
                stemp[1] -= 1

        # Insert-number actions (only on originally empty cells)
        elif action == 4:  # insert 1
            if self.binaryquizarray[s[0], s[1]] == "0":
                self.quizarray[s[0], s[1]] = "1"
        elif action == 5:  # insert 2
            if self.binaryquizarray[s[0], s[1]] == "0":
                self.quizarray[s[0], s[1]] = "2"
        elif action == 6:  # insert 3
            if self.binaryquizarray[s[0], s[1]] == "0":
                self.quizarray[s[0], s[1]] = "3"
        elif action == 7:  # insert 4
            if self.binaryquizarray[s[0], s[1]] == "0":
                self.quizarray[s[0], s[1]] = "4"
        elif action == 8:  # insert 5
            if self.binaryquizarray[s[0], s[1]] == "0":
                self.quizarray[s[0], s[1]] = "5"
        elif action == 9:  # insert 6
            if self.binaryquizarray[s[0], s[1]] == "0":
                self.quizarray[s[0], s[1]] = "6"
        elif action == 10:  # insert 7
            if self.binaryquizarray[s[0], s[1]] == "0":
                self.quizarray[s[0], s[1]] = "7"
        elif action == 11:  # insert 8
            if self.binaryquizarray[s[0], s[1]] == "0":
                self.quizarray[s[0], s[1]] = "8"
        elif action == 12:  # insert 9
            if self.binaryquizarray[s[0], s[1]] == "0":
                self.quizarray[s[0], s[1]] = "9"

        # Reward logic: puzzle completed?
        if (self.quizarray == self.solutionarray).all():
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        # Next state
        if action < 4:
            s_ = stemp
        else:
            s_ = s

        # (Optional) clear console and display current grid
        cls = lambda: os.system("cls")
        cls()
        print(self.quizarray)
        time.sleep(0.0001)

        return s_, reward, done
