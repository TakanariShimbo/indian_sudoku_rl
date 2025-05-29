from collections import deque
import random
import os
import shutil
import csv
import datetime
import argparse
import glob

import numpy as np

from boardenv import SudokuEnvironment
from neural import DeepQNetwork
from data_loader import SudokuDataLoader
from config import *


class SudokuTrainer:
    """Enhanced Sudoku trainer with resume functionality, validation, detailed learning curves, smart model saving, and CSV logging"""

    def __init__(self, train_split=0.8, validation_freq=50, enable_logging=True, resume_from=None):
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

        # Split dataset into train/validation
        self._split_dataset(train_split)
        self.validation_freq = validation_freq

        # Enhanced training statistics
        self.training_stats = {
            "episode_rewards": [],
            "episode_steps": [],
            "loss_history": [],
            "puzzle_success_rates": [],
            "validation_success_rates": [],
            "validation_avg_steps": [],
            "validation_episodes": [],
            "epsilon_history": [],
            "solved_puzzles": 0,
        }

        # Smoothing windows
        self.reward_window = deque(maxlen=100)
        self.step_window = deque(maxlen=100)
        self.success_window = deque(maxlen=100)

        # Best model management
        self.best_validation_score = 0.0
        self.best_model_path = "sudoku_best_model.pth"
        self.latest_model_path = "sudoku_latest_model.pth"

        # Resume functionality
        self.resume_info = {
            "total_episodes_trained": 0,
            "total_puzzles_trained": 0,
            "best_validation_score": 0.0,
            "training_start_time": datetime.datetime.now(),
        }

        # CSV Logging
        self.enable_logging = enable_logging
        self.global_episode_count = 0
        self.cumulative_solved = 0

        # Try to resume from previous training
        if resume_from:
            self._resume_training(resume_from)
        elif self._should_auto_resume():
            self._auto_resume()

        if self.enable_logging:
            self._setup_csv_logging()

    def _should_auto_resume(self):
        """Check if we should automatically resume from the latest model"""
        return os.path.exists(self.latest_model_path) or os.path.exists(self.best_model_path)

    def _auto_resume(self):
        """Automatically resume from the most recent model"""
        # Prefer latest model, fall back to best model
        if os.path.exists(self.latest_model_path):
            self._resume_training(self.latest_model_path)
            print(f"🔄 Auto-resumed from latest model: {self.latest_model_path}")
        elif os.path.exists(self.best_model_path):
            self._resume_training(self.best_model_path)
            print(f"🔄 Auto-resumed from best model: {self.best_model_path}")

    def _resume_training(self, model_path):
        """Resume training from a saved model"""
        if not os.path.exists(model_path):
            print(f"⚠️  Resume model not found: {model_path}")
            return False

        try:
            # Load the model
            self.agent.load_model(model_path)

            # Try to load additional resume information
            resume_info_path = model_path.replace(".pth", "_resume_info.txt")
            if os.path.exists(resume_info_path):
                self._load_resume_info(resume_info_path)

            # Try to restore best validation score from existing logs
            self._restore_best_validation_score()

            print(f"✅ Successfully resumed training from: {model_path}")
            print(f"   Epsilon: {self.agent.epsilon:.3f}")
            print(f"   Learn steps: {self.agent.learn_step_counter}")
            print(f"   Best validation score: {self.best_validation_score:.3f}")
            return True

        except Exception as e:
            print(f"❌ Failed to resume from {model_path}: {e}")
            return False

    def _load_resume_info(self, info_path):
        """Load additional resume information"""
        try:
            with open(info_path, "r") as f:
                for line in f:
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        if key == "total_episodes_trained":
                            self.resume_info["total_episodes_trained"] = int(value)
                        elif key == "total_puzzles_trained":
                            self.resume_info["total_puzzles_trained"] = int(value)
                        elif key == "best_validation_score":
                            self.resume_info["best_validation_score"] = float(value)
                            self.best_validation_score = float(value)
                        elif key == "cumulative_solved":
                            self.cumulative_solved = int(value)
        except Exception as e:
            print(f"Warning: Could not load resume info: {e}")

    def _save_resume_info(self, model_path):
        """Save additional resume information"""
        try:
            info_path = model_path.replace(".pth", "_resume_info.txt")
            with open(info_path, "w") as f:
                f.write(f"total_episodes_trained:{self.global_episode_count}\n")
                f.write(f"total_puzzles_trained:{self.resume_info['total_puzzles_trained']}\n")
                f.write(f"best_validation_score:{self.best_validation_score}\n")
                f.write(f"cumulative_solved:{self.cumulative_solved}\n")
                f.write(f"training_start_time:{self.resume_info['training_start_time'].isoformat()}\n")
                f.write(f"last_update:{datetime.datetime.now().isoformat()}\n")
        except Exception as e:
            print(f"Warning: Could not save resume info: {e}")

    def _restore_best_validation_score(self):
        """Try to restore best validation score from existing CSV logs"""
        if not os.path.exists("logs"):
            return

        # Find the most recent log file
        log_files = glob.glob("logs/training_log_*.csv")
        if not log_files:
            return

        try:
            # Get the most recent log file
            latest_log = max(log_files, key=os.path.getctime)

            # Read the log and find the best validation score
            import pandas as pd

            df = pd.read_csv(latest_log)

            # Filter non-empty validation scores
            validation_data = df[df["validation_success_rate"] != ""]
            if len(validation_data) > 0:
                validation_data = validation_data.copy()
                validation_data["validation_success_rate"] = pd.to_numeric(validation_data["validation_success_rate"])
                best_score = validation_data["validation_success_rate"].max()

                if best_score > self.best_validation_score:
                    self.best_validation_score = best_score
                    print(f"📊 Restored best validation score from logs: {best_score:.3f}")

        except Exception as e:
            print(f"Warning: Could not restore validation score from logs: {e}")

    def _setup_csv_logging(self):
        """Setup CSV logging with resume support"""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Generate log filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"logs/training_log_{timestamp}.csv"

        # CSV field names (matching viz_training.py expectations)
        self.csv_fieldnames = [
            "timestamp",
            "global_episode",
            "puzzle_index",
            "episode_in_puzzle",
            "reward",
            "steps",
            "solved",
            "epsilon",
            "loss",
            "avg_reward_100",
            "avg_steps_100",
            "success_rate_100",
            "cumulative_solved",
            "validation_success_rate",
            "validation_avg_steps",
        ]

        # Initialize CSV file with header
        with open(self.log_filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_fieldnames)
            writer.writeheader()

        print(f"📝 CSV logging enabled: {self.log_filename}")

    def _log_episode(self, puzzle_index, episode_in_puzzle, reward, steps, solved, epsilon, loss, validation_success_rate=None, validation_avg_steps=None):
        """Log episode data to CSV"""
        if not self.enable_logging:
            return

        self.global_episode_count += 1
        if solved:
            self.cumulative_solved += 1

        # Update success window
        self.success_window.append(1 if solved else 0)

        # Calculate moving averages
        avg_reward_100 = np.mean(list(self.reward_window)) if self.reward_window else 0
        avg_steps_100 = np.mean(list(self.step_window)) if self.step_window else 0
        success_rate_100 = np.mean(list(self.success_window)) if self.success_window else 0

        # Create log entry
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "global_episode": self.global_episode_count,
            "puzzle_index": puzzle_index,
            "episode_in_puzzle": episode_in_puzzle,
            "reward": reward,
            "steps": steps,
            "solved": 1 if solved else 0,
            "epsilon": epsilon,
            "loss": loss if loss is not None else 0,
            "avg_reward_100": avg_reward_100,
            "avg_steps_100": avg_steps_100,
            "success_rate_100": success_rate_100,
            "cumulative_solved": self.cumulative_solved,
            "validation_success_rate": validation_success_rate if validation_success_rate is not None else "",
            "validation_avg_steps": validation_avg_steps if validation_avg_steps is not None else "",
        }

        # Write to CSV
        with open(self.log_filename, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_fieldnames)
            writer.writerow(log_entry)

    def _split_dataset(self, train_split):
        """Split dataset into training and validation sets"""
        total_puzzles = self.data_loader.get_puzzle_count()
        train_size = int(total_puzzles * train_split)

        # Random split
        all_indices = list(range(total_puzzles))
        random.shuffle(all_indices)

        self.train_indices = all_indices[:train_size]
        self.validation_indices = all_indices[train_size:]

        print(f"📚 Dataset split: {len(self.train_indices)} training, {len(self.validation_indices)} validation")

    def validate(self, num_puzzles=None):
        """Evaluate on validation set"""
        if num_puzzles is None:
            num_puzzles = min(20, len(self.validation_indices))

        # Set epsilon to 0 for pure exploitation
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0

        success_count = 0
        total_steps = []

        # Randomly select validation puzzles
        val_puzzles = random.sample(self.validation_indices, num_puzzles)

        for puzzle_idx in val_puzzles:
            # Load puzzle
            self.env.current_puzzle_index = puzzle_idx
            self.env._load_puzzle()

            current_state = self.env.reset()
            steps = 0
            max_steps = 500  # Shorter time limit for validation

            while steps < max_steps:
                action = self.agent.choose_action(current_state)
                next_state, reward, done = self.env.step(action)
                current_state = next_state
                steps += 1

                if done:
                    success_count += 1
                    total_steps.append(steps)
                    break

        # Restore original epsilon
        self.agent.epsilon = original_epsilon

        success_rate = success_count / num_puzzles
        avg_steps = np.mean(total_steps) if total_steps else max_steps

        return success_rate, avg_steps

    def _save_models(self, validation_score, force_save=False):
        """Manage and save best and latest models with resume info"""
        # Always save latest model
        self.agent.save_model(self.latest_model_path)
        self._save_resume_info(self.latest_model_path)

        # Update best model if score improved
        if validation_score > self.best_validation_score or force_save:
            if validation_score > self.best_validation_score:
                self.best_validation_score = validation_score

            try:
                # Copy latest to best model
                shutil.copy2(self.latest_model_path, self.best_model_path)
                # Also copy resume info
                latest_info = self.latest_model_path.replace(".pth", "_resume_info.txt")
                best_info = self.best_model_path.replace(".pth", "_resume_info.txt")
                if os.path.exists(latest_info):
                    shutil.copy2(latest_info, best_info)

                if not force_save:
                    print(f"🏆 New best model saved! Validation score: {validation_score:.3f}")
                else:
                    print(f"💾 Model saved (forced)")
            except Exception as e:
                print(f"Warning: Could not update best model: {e}")

    def train_single_puzzle(self, puzzle_index=None, episodes=EPISODES_PER_PUZZLE):
        """Train on a single puzzle with enhanced statistics collection and CSV logging"""
        if puzzle_index is not None:
            self.env.current_puzzle_index = puzzle_index
            self.env._load_puzzle()

        puzzle_rewards = []
        puzzle_steps = []
        puzzle_solved = 0

        for episode in range(episodes):
            episode_reward = 0
            episode_step = 0
            episode_loss = None
            current_state = self.env.reset()

            while True:
                action = self.agent.choose_action(current_state)
                next_state, reward, done = self.env.step(action)

                self.agent.store_transition(current_state, action, reward, next_state)

                if (self.agent.memory_counter > WARMUP_STEPS) and (episode_step % LEARN_FREQUENCY == 0):
                    self.agent.learn()
                    # Record loss history
                    if self.agent.cost_history:
                        episode_loss = self.agent.cost_history[-1]
                        self.training_stats["loss_history"].append(episode_loss)

                current_state = next_state
                episode_reward += reward
                episode_step += 1

                if done:
                    puzzle_solved += 1
                    self.training_stats["solved_puzzles"] += 1
                    break

                if episode_step > 1000:
                    break

            puzzle_rewards.append(episode_reward)
            puzzle_steps.append(episode_step)

            # Add to smoothing windows
            self.reward_window.append(episode_reward)
            self.step_window.append(episode_step)

            # Log episode to CSV
            self._log_episode(
                puzzle_index=puzzle_index if puzzle_index is not None else -1,
                episode_in_puzzle=episode + 1,
                reward=episode_reward,
                steps=episode_step,
                solved=done,
                epsilon=self.agent.epsilon,
                loss=episode_loss,
            )

        # Update statistics
        self.training_stats["episode_rewards"].extend(puzzle_rewards)
        self.training_stats["episode_steps"].extend(puzzle_steps)
        self.training_stats["epsilon_history"].append(self.agent.epsilon)

        # Record puzzle success rate
        puzzle_success_rate = puzzle_solved / episodes
        self.training_stats["puzzle_success_rates"].append(puzzle_success_rate)

        # Store last episode data for validation logging
        self._last_episode_data = {
            "puzzle_index": puzzle_index if puzzle_index is not None else -1,
            "episode_in_puzzle": episodes,
            "reward": puzzle_rewards[-1] if puzzle_rewards else 0,
            "steps": puzzle_steps[-1] if puzzle_steps else 0,
            "solved": puzzle_solved > 0,
            "epsilon": self.agent.epsilon,
            "loss": episode_loss,
        }

        return np.mean(puzzle_rewards), np.mean(puzzle_steps), puzzle_success_rate

    def train_all_puzzles(self, max_puzzles=None):
        """Train on all puzzles with validation"""
        train_puzzles = len(self.train_indices)
        if max_puzzles:
            train_puzzles = min(train_puzzles, max_puzzles)
            self.train_indices = self.train_indices[:train_puzzles]

        print(f"🚀 Starting training on {train_puzzles} puzzles...")
        print(f"📊 Validation every {self.validation_freq} puzzles")
        print(f"🎯 Agent epsilon: {self.agent.epsilon:.3f}")
        print(f"📈 Best validation score: {self.best_validation_score:.3f}")
        if self.enable_logging:
            print(f"📝 Logging to: {self.log_filename}")
        print("-" * 60)

        for i, puzzle_idx in enumerate(self.train_indices[:train_puzzles]):
            print(f"\n=== Training Puzzle {i + 1}/{train_puzzles} (ID: {puzzle_idx}) ===")

            avg_reward, avg_steps, success_rate = self.train_single_puzzle(puzzle_idx)
            self.resume_info["total_puzzles_trained"] += 1

            print(f"Success rate: {success_rate:.2%}")
            print(f"Average reward: {avg_reward:.3f}")
            print(f"Average steps: {avg_steps:.1f}")
            print(f"Current epsilon: {self.agent.epsilon:.3f}")
            print(f"Global episodes: {self.global_episode_count}")

            # Run validation periodically
            if (i + 1) % self.validation_freq == 0 or (i + 1) == train_puzzles:
                print(f"\n--- Validation after {i + 1} puzzles ---")
                val_success_rate, val_avg_steps = self.validate()

                # Record statistics
                self.training_stats["validation_success_rates"].append(val_success_rate)
                self.training_stats["validation_avg_steps"].append(val_avg_steps)
                self.training_stats["validation_episodes"].append(i + 1)

                print(f"Validation success rate: {val_success_rate:.2%}")
                print(f"Validation avg steps: {val_avg_steps:.1f}")

                # Log validation result to CSV (use last episode data)
                if hasattr(self, "_last_episode_data"):
                    self._log_episode(
                        puzzle_index=self._last_episode_data["puzzle_index"],
                        episode_in_puzzle=self._last_episode_data["episode_in_puzzle"],
                        reward=self._last_episode_data["reward"],
                        steps=self._last_episode_data["steps"],
                        solved=self._last_episode_data["solved"],
                        epsilon=self._last_episode_data["epsilon"],
                        loss=self._last_episode_data["loss"],
                        validation_success_rate=val_success_rate,
                        validation_avg_steps=val_avg_steps,
                    )

                # Save models (with best model check)
                self._save_models(val_success_rate)

        print(f"\n🎉 Training completed!")
        print(f"Total puzzles solved: {self.training_stats['solved_puzzles']}")
        print(f"Best validation score: {self.best_validation_score:.3f}")
        print(f"Final epsilon: {self.agent.epsilon:.3f}")
        if self.enable_logging:
            print(f"Training log saved to: {self.log_filename}")

    def evaluate(self, num_puzzles=20, use_best_model=True):
        """Evaluate the trained agent"""
        print(f"\n=== Evaluation on {num_puzzles} puzzles ===")

        if use_best_model and os.path.exists(self.best_model_path):
            print("Loading best model for evaluation...")
            self.agent.load_model(self.best_model_path)

        # Set epsilon to 0 for pure exploitation
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0

        solved_count = 0
        total_steps = []

        # Select test puzzles randomly from validation set
        test_puzzles = random.sample(self.validation_indices, min(num_puzzles, len(self.validation_indices)))

        for i, puzzle_idx in enumerate(test_puzzles):
            self.env.current_puzzle_index = puzzle_idx
            self.env._load_puzzle()

            current_state = self.env.reset()
            steps = 0
            max_steps = 1000

            while steps < max_steps:
                action = self.agent.choose_action(current_state)
                next_state, reward, done = self.env.step(action)
                current_state = next_state
                steps += 1

                if done:
                    solved_count += 1
                    total_steps.append(steps)
                    print(f"Puzzle {i + 1}: Solved in {steps} steps")
                    break

            if steps >= max_steps:
                print(f"Puzzle {i + 1}: Not solved within {max_steps} steps")

        # Restore original epsilon
        self.agent.epsilon = original_epsilon

        success_rate = solved_count / len(test_puzzles)
        avg_steps = np.mean(total_steps) if total_steps else max_steps

        print(f"\nEvaluation Results:")
        print(f"Success rate: {success_rate:.2%} ({solved_count}/{len(test_puzzles)})")
        if total_steps:
            print(f"Average steps for solved puzzles: {avg_steps:.1f}")

        return success_rate, avg_steps

    def cleanup_old_models(self):
        """Clean up old model files, keep only best and latest"""
        current_dir = "."
        model_files = [f for f in os.listdir(current_dir) if f.endswith(".pth")]
        keep_files = {self.best_model_path, self.latest_model_path}

        deleted_count = 0
        for model_file in model_files:
            if model_file not in keep_files and "sudoku" in model_file:
                try:
                    os.remove(model_file)
                    deleted_count += 1
                    print(f"Deleted old model: {model_file}")
                except Exception as e:
                    print(f"Could not delete {model_file}: {e}")

        if deleted_count > 0:
            print(f"🧹 Cleanup completed. Removed {deleted_count} old model files.")

    def save_model(self, filepath):
        """Save the trained model (for backward compatibility)"""
        self.agent.save_model(filepath)
        self._save_resume_info(filepath)

    def load_model(self, filepath):
        """Load a trained model (for backward compatibility)"""
        self.agent.load_model(filepath)


def main():
    """Main training function with command line argument support"""
    parser = argparse.ArgumentParser(description="Sudoku RL Training with Resume Support")
    parser.add_argument("--resume", type=str, help="Path to model file to resume from")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable automatic resume")
    parser.add_argument("--max-puzzles", type=int, default=100000, help="Maximum number of puzzles to train on")
    parser.add_argument("--eval-puzzles", type=int, default=50, help="Number of puzzles for final evaluation")
    parser.add_argument("--no-logging", action="store_true", help="Disable CSV logging")

    args = parser.parse_args()

    # Determine resume behavior
    resume_from = None
    if args.resume:
        resume_from = args.resume
    elif args.no_auto_resume:
        resume_from = False  # Explicitly disable auto-resume

    trainer = SudokuTrainer(train_split=0.8, validation_freq=25, enable_logging=not args.no_logging, resume_from=resume_from)

    # Clean up old model files
    trainer.cleanup_old_models()

    # Disable visual display for faster training
    trainer.env.enable_display(False)

    try:
        # Train on specified number of puzzles
        trainer.train_all_puzzles(max_puzzles=args.max_puzzles)

        # Final evaluation with best model
        trainer.evaluate(num_puzzles=args.eval_puzzles, use_best_model=True)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save model even when interrupted
        trainer._save_models(0.0, force_save=True)

    print("\n🎉 Training session completed!")
    print(f"🏆 Best model saved as: {trainer.best_model_path}")
    print(f"💾 Latest model saved as: {trainer.latest_model_path}")
    if trainer.enable_logging:
        print(f"📝 Training log saved as: {trainer.log_filename}")
        print(f"📊 Use 'python viz_training.py' to visualize training progress")


if __name__ == "__main__":
    main()
