import numpy as np
import matplotlib.pyplot as plt

from boardenv import SudokuEnvironment
from neural import DeepQNetwork
from data_loader import SudokuDataLoader
from config import *


class SudokuTrainer:
    """Handles training of the Sudoku RL agent"""

    def __init__(self):
        self.env = SudokuEnvironment()
        self.agent = DeepQNetwork(
            n_actions=self.env.n_actions,
            n_features=self.env.n_features,
            learning_rate=LEARNING_RATE,
            reward_decay=REWARD_DECAY,
            e_greedy=EPSILON_GREEDY,
            replace_target_iter=REPLACE_TARGET_ITER,
            memory_size=MEMORY_SIZE,
            batch_size=BATCH_SIZE,
            e_greedy_increment=EPSILON_INCREMENT,
        )
        self.data_loader = SudokuDataLoader()

        # Training statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.solved_puzzles = 0

    def train_single_puzzle(self, puzzle_index=None, episodes=EPISODES_PER_PUZZLE):
        """
        Train on a single puzzle for specified number of episodes

        Args:
            puzzle_index: Specific puzzle index (None for current)
            episodes: Number of episodes to train
        """
        if puzzle_index is not None:
            self.env.current_puzzle_index = puzzle_index
            self.env._load_puzzle()

        puzzle_rewards = []
        puzzle_steps = []

        for episode in range(episodes):
            episode_reward = 0
            episode_step = 0
            current_state = self.env.reset()

            while True:
                # Choose action
                action = self.agent.choose_action(current_state)

                # Take step
                next_state, reward, done = self.env.step(action)

                # Store transition
                self.agent.store_transition(current_state, action, reward, next_state)

                # Learn after warmup period
                if (self.agent.memory_counter > WARMUP_STEPS) and (episode_step % LEARN_FREQUENCY == 0):
                    self.agent.learn()

                # Update state and statistics
                current_state = next_state
                episode_reward += reward
                episode_step += 1

                if done:
                    self.solved_puzzles += 1
                    print(f"Episode {episode + 1}: Solved in {episode_step} steps! " f"Total solved: {self.solved_puzzles}")
                    break

                # Prevent infinite episodes
                if episode_step > 1000:
                    print(f"Episode {episode + 1}: Terminated after {episode_step} steps")
                    break

            puzzle_rewards.append(episode_reward)
            puzzle_steps.append(episode_step)

        self.episode_rewards.extend(puzzle_rewards)
        self.episode_steps.extend(puzzle_steps)

        return np.mean(puzzle_rewards), np.mean(puzzle_steps)

    def train_all_puzzles(self, max_puzzles=None):
        """
        Train on all puzzles in the dataset

        Args:
            max_puzzles: Maximum number of puzzles to train on (None for all)
        """
        total_puzzles = self.data_loader.get_puzzle_count()
        if max_puzzles:
            total_puzzles = min(total_puzzles, max_puzzles)

        print(f"Starting training on {total_puzzles} puzzles...")
        print(f"Episodes per puzzle: {EPISODES_PER_PUZZLE}")
        print(f"Agent epsilon: {self.agent.epsilon:.3f}")
        print("-" * 50)

        for puzzle_idx in range(total_puzzles):
            print(f"\n=== Puzzle {puzzle_idx + 1}/{total_puzzles} ===")

            avg_reward, avg_steps = self.train_single_puzzle(puzzle_idx)

            print(f"Average reward: {avg_reward:.3f}")
            print(f"Average steps: {avg_steps:.1f}")
            print(f"Current epsilon: {self.agent.epsilon:.3f}")

            # Save model periodically
            if (puzzle_idx + 1) % 100 == 0:
                self.save_model(f"sudoku_model_puzzle_{puzzle_idx + 1}.pth")

        print(f"\nTraining completed!")
        print(f"Total puzzles solved: {self.solved_puzzles}")
        print(f"Final epsilon: {self.agent.epsilon:.3f}")

    def evaluate(self, num_puzzles=10):
        """
        Evaluate the trained agent

        Args:
            num_puzzles: Number of puzzles to evaluate on
        """
        print(f"\n=== Evaluation on {num_puzzles} puzzles ===")

        # Set epsilon to 0 for pure exploitation
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0

        solved_count = 0
        total_steps = []

        for i in range(num_puzzles):
            current_state = self.env.reset()
            steps = 0

            while steps < 1000:  # Max steps limit
                action = self.agent.choose_action(current_state)
                next_state, reward, done = self.env.step(action)
                current_state = next_state
                steps += 1

                if done:
                    solved_count += 1
                    total_steps.append(steps)
                    print(f"Puzzle {i + 1}: Solved in {steps} steps")
                    break

            if steps >= 1000:
                print(f"Puzzle {i + 1}: Not solved within 1000 steps")

            # Move to next puzzle
            self.env.next_puzzle()

        # Restore original epsilon
        self.agent.epsilon = original_epsilon

        success_rate = solved_count / num_puzzles
        avg_steps = np.mean(total_steps) if total_steps else 0

        print(f"\nEvaluation Results:")
        print(f"Success rate: {success_rate:.2%} ({solved_count}/{num_puzzles})")
        if total_steps:
            print(f"Average steps for solved puzzles: {avg_steps:.1f}")

    def plot_training_progress(self):
        """Plot training progress"""
        if not self.episode_rewards:
            print("No training data to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.grid(True)

        # Plot steps
        ax2.plot(self.episode_steps)
        ax2.set_title("Episode Steps")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("training_progress.png")
        plt.show()

    def save_model(self, filepath):
        """Save the trained model"""
        self.agent.save_model(filepath)

    def load_model(self, filepath):
        """Load a trained model"""
        self.agent.load_model(filepath)


def main():
    """Main training function"""
    trainer = SudokuTrainer()

    # Enable visual display (optional)
    trainer.env.enable_display(False)  # Set to True to see the game

    try:
        # Train on first 50 puzzles as example (change as needed)
        trainer.train_all_puzzles(max_puzzles=50)

        # Save final model
        trainer.save_model("sudoku_final_model.pth")

        # Evaluate the trained agent
        trainer.evaluate(num_puzzles=10)

        # Plot training progress
        trainer.plot_training_progress()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_model("sudoku_interrupted_model.pth")

    print("Training session completed!")


if __name__ == "__main__":
    main()
