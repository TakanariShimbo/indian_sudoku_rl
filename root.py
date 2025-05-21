import pandas as pd
from boardenv import board
from neural import DeepQNetwork

# --- Load the Sudoku quizzes/solutions CSV ---
COLUMNS = ["quizzes", "solutions"]
PATH = r".\sudoku.csv"

df_train = pd.read_csv(PATH, skipinitialspace=True, names=COLUMNS, index_col=False)
quizzescount = df_train["quizzes"].count()


def run_maze():
    step = 0
    episodes = 10
    for episode in range(episodes):
        currentstate = env.reset()
        while True:
            # choose an action based on the current state
            action = neuralbrain.choose_action(currentstate)

            # step the environment
            futurestate_, reward, done = env.step(action)

            # store the transition for replay
            neuralbrain.store_transition(currentstate, action, reward, futurestate_)

            print(f"Episode {episode}")

            # after a warm-up period, learn every 5 steps
            if (step > 200) and (step % 5 == 0):
                neuralbrain.learn()

            # advance to the next state
            currentstate = futurestate_

            if done:
                break

            step += 1

    print("game over")


if __name__ == "__main__":
    # initialize environment and DQN
    env = board()
    neuralbrain = DeepQNetwork(env.n_actions, env.n_features, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, replace_target_iter=200, memory_size=2000)

    # run one training session per quiz in the CSV
    for i in range(1, quizzescount):
        run_maze()
        # rebuild the maze for the next quiz
        env._build_maze()
