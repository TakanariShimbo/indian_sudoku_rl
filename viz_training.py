import os
import argparse
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class TrainingVisualizer:
    """Visualize training logs from CSV files"""

    def __init__(self, log_file=None, log_dir="logs"):
        self.log_dir = log_dir

        if log_file:
            self.log_file = log_file
        else:
            # Find the most recent log file
            log_files = glob(os.path.join(log_dir, "training_log_*.csv"))
            if not log_files:
                raise FileNotFoundError(f"No training log files found in {log_dir}")
            self.log_file = max(log_files, key=os.path.getctime)
            print(f"Using most recent log file: {self.log_file}")

        # Load data
        self.df = pd.read_csv(self.log_file)
        print(f"Loaded {len(self.df)} episodes from {self.log_file}")

        # Set up plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def plot_basic_metrics(self, save_path=None):
        """Plot basic training metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Episode rewards with moving average
        ax1.plot(self.df["global_episode"], self.df["reward"], alpha=0.3, color="lightblue", label="Episode Reward")
        ax1.plot(self.df["global_episode"], self.df["avg_reward_100"], "b-", linewidth=2, label="Moving Average (100)")
        ax1.set_title("Episode Rewards Over Time", size=14, fontweight="bold")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Episode steps with moving average
        ax2.plot(self.df["global_episode"], self.df["steps"], alpha=0.3, color="lightgreen", label="Episode Steps")
        ax2.plot(self.df["global_episode"], self.df["avg_steps_100"], "g-", linewidth=2, label="Moving Average (100)")
        ax2.set_title("Episode Steps Over Time", size=14, fontweight="bold")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Success rate
        ax3.plot(self.df["global_episode"], self.df["success_rate_100"], "r-", linewidth=2)
        ax3.set_title("Success Rate (100-Episode Moving Average)", size=14, fontweight="bold")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Success Rate")
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)

        # Cumulative solved puzzles
        ax4.plot(self.df["global_episode"], self.df["cumulative_solved"], "purple", linewidth=2)
        ax4.set_title("Cumulative Solved Puzzles", size=14, fontweight="bold")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Total Solved")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Basic metrics plot saved to {save_path}")

        plt.show()

    def plot_learning_dynamics(self, save_path=None):
        """Plot learning dynamics (epsilon, loss)"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

        # Epsilon decay
        ax1.plot(self.df["global_episode"], self.df["epsilon"], "b-", linewidth=2)
        ax1.set_title("Epsilon Decay Over Time", size=14, fontweight="bold")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Epsilon")
        ax1.grid(True, alpha=0.3)

        # Loss (filter out zero values)
        loss_data = self.df[self.df["loss"] > 0]
        if len(loss_data) > 0:
            ax2.plot(loss_data["global_episode"], loss_data["loss"], "r-", alpha=0.7, linewidth=1)
            # Add smoothed loss line
            if len(loss_data) > 50:
                window = min(50, len(loss_data) // 10)
                smoothed_loss = loss_data["loss"].rolling(window=window, center=True).mean()
                ax2.plot(loss_data["global_episode"], smoothed_loss, "darkred", linewidth=2, label=f"Smoothed (window={window})")
                ax2.legend()
        ax2.set_title("Training Loss Over Time", size=14, fontweight="bold")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Loss")
        ax2.grid(True, alpha=0.3)

        # Learning efficiency (reward per step)
        efficiency = []
        for _, row in self.df.iterrows():
            if row["steps"] > 0:
                efficiency.append(row["reward"] / row["steps"])
            else:
                efficiency.append(0)

        ax3.plot(self.df["global_episode"], efficiency, alpha=0.3, color="orange")
        # Smooth efficiency
        if len(self.df) > 100:
            efficiency_smooth = pd.Series(efficiency).rolling(window=100, center=True).mean()
            ax3.plot(self.df["global_episode"], efficiency_smooth, "darkorange", linewidth=2, label="Smoothed (100)")
            ax3.legend()
        ax3.set_title("Learning Efficiency (Reward per Step)", size=14, fontweight="bold")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Reward/Steps")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Learning dynamics plot saved to {save_path}")

        plt.show()

    def plot_puzzle_performance(self, save_path=None):
        """Plot performance per puzzle"""
        if "puzzle_index" not in self.df.columns:
            print("Puzzle index not found in data")
            return

        # Group by puzzle and calculate statistics
        puzzle_stats = self.df.groupby("puzzle_index").agg({"reward": ["mean", "std", "sum"], "steps": ["mean", "std"], "solved": "sum"}).reset_index()

        # Flatten column names
        puzzle_stats.columns = ["puzzle_index", "avg_reward", "std_reward", "total_reward", "avg_steps", "std_steps", "times_solved"]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Average reward per puzzle
        ax1.bar(puzzle_stats["puzzle_index"], puzzle_stats["avg_reward"], alpha=0.7, color="skyblue")
        ax1.set_title("Average Reward per Puzzle", size=14, fontweight="bold")
        ax1.set_xlabel("Puzzle Index")
        ax1.set_ylabel("Average Reward")
        ax1.grid(True, alpha=0.3)

        # Average steps per puzzle
        ax2.bar(puzzle_stats["puzzle_index"], puzzle_stats["avg_steps"], alpha=0.7, color="lightgreen")
        ax2.set_title("Average Steps per Puzzle", size=14, fontweight="bold")
        ax2.set_xlabel("Puzzle Index")
        ax2.set_ylabel("Average Steps")
        ax2.grid(True, alpha=0.3)

        # Times solved per puzzle
        ax3.bar(puzzle_stats["puzzle_index"], puzzle_stats["times_solved"], alpha=0.7, color="coral")
        ax3.set_title("Times Solved per Puzzle", size=14, fontweight="bold")
        ax3.set_xlabel("Puzzle Index")
        ax3.set_ylabel("Times Solved")
        ax3.grid(True, alpha=0.3)

        # Difficulty ranking (based on average steps when solved)
        solved_puzzles = puzzle_stats[puzzle_stats["times_solved"] > 0].copy()
        if len(solved_puzzles) > 0:
            solved_puzzles = solved_puzzles.sort_values("avg_steps", ascending=False)
            ax4.barh(range(len(solved_puzzles)), solved_puzzles["avg_steps"], alpha=0.7, color="gold")
            ax4.set_yticks(range(len(solved_puzzles)))
            ax4.set_yticklabels([f"Puzzle {int(idx)}" for idx in solved_puzzles["puzzle_index"]])
            ax4.set_title("Puzzle Difficulty Ranking (by avg steps)", size=14, fontweight="bold")
            ax4.set_xlabel("Average Steps")
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Puzzle performance plot saved to {save_path}")

        plt.show()

    def plot_heatmap_analysis(self, save_path=None):
        """Plot heatmap analysis of training progress"""
        # Create heatmap data: puzzle vs episode range
        n_puzzles = self.df["puzzle_index"].nunique()
        n_episodes = len(self.df)

        # Divide episodes into bins
        n_bins = min(50, n_episodes // 100) if n_episodes >= 100 else 10
        self.df["episode_bin"] = pd.cut(self.df["global_episode"], bins=n_bins, labels=False)

        # Create pivot table
        heatmap_data = self.df.pivot_table(values="success_rate_100", index="puzzle_index", columns="episode_bin", aggfunc="mean")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Success rate heatmap
        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, cmap="RdYlBu_r", ax=ax1, cbar_kws={"label": "Success Rate"})
            ax1.set_title("Success Rate Heatmap: Puzzle vs Training Progress", size=14, fontweight="bold")
            ax1.set_xlabel("Training Progress (Episode Bins)")
            ax1.set_ylabel("Puzzle Index")

        # Steps heatmap
        heatmap_steps = self.df.pivot_table(values="avg_steps_100", index="puzzle_index", columns="episode_bin", aggfunc="mean")

        if not heatmap_steps.empty:
            sns.heatmap(heatmap_steps, cmap="RdYlGn_r", ax=ax2, cbar_kws={"label": "Avg Steps"})
            ax2.set_title("Average Steps Heatmap: Puzzle vs Training Progress", size=14, fontweight="bold")
            ax2.set_xlabel("Training Progress (Episode Bins)")
            ax2.set_ylabel("Puzzle Index")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Heatmap analysis saved to {save_path}")

        plt.show()

    def generate_summary_report(self):
        """Generate a summary report of the training"""
        print("=" * 60)
        print("TRAINING SUMMARY REPORT")
        print("=" * 60)

        print(f"Log file: {self.log_file}")
        print(f"Total episodes: {len(self.df):,}")
        print(f"Unique puzzles: {self.df['puzzle_index'].nunique()}")
        print(f"Training duration: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")

        print("\n--- PERFORMANCE METRICS ---")
        print(f"Total puzzles solved: {self.df['cumulative_solved'].max()}")
        print(f"Final success rate (100-ep avg): {self.df['success_rate_100'].iloc[-1]:.2%}")
        print(f"Best success rate achieved: {self.df['success_rate_100'].max():.2%}")
        print(f"Average reward per episode: {self.df['reward'].mean():.3f}")
        print(f"Average steps per episode: {self.df['steps'].mean():.1f}")

        print("\n--- LEARNING DYNAMICS ---")
        print(f"Initial epsilon: {self.df['epsilon'].iloc[0]:.3f}")
        print(f"Final epsilon: {self.df['epsilon'].iloc[-1]:.3f}")

        loss_data = self.df[self.df["loss"] > 0]
        if len(loss_data) > 0:
            print(f"Average loss: {loss_data['loss'].mean():.4f}")
            print(f"Final loss (last 100): {loss_data['loss'].tail(100).mean():.4f}")

        print("\n--- PUZZLE ANALYSIS ---")
        puzzle_stats = self.df.groupby("puzzle_index")["solved"].sum().sort_values(ascending=False)
        print(f"Most solved puzzle: Index {puzzle_stats.index[0]} ({puzzle_stats.iloc[0]} times)")
        print(f"Least solved puzzles: {sum(puzzle_stats == 0)} puzzles never solved")

        solved_puzzles = self.df[self.df["solved"] == True]
        if len(solved_puzzles) > 0:
            avg_steps_to_solve = solved_puzzles["steps"].mean()
            print(f"Average steps to solve (when successful): {avg_steps_to_solve:.1f}")

        print("=" * 60)

    def create_comprehensive_report(self, output_dir=None):
        """Create a comprehensive visualization report"""
        if output_dir is None:
            timestamp = self.log_file.split("_")[-1].replace(".csv", "")
            output_dir = os.path.join(self.log_dir, f"analysis_{timestamp}")

        os.makedirs(output_dir, exist_ok=True)

        print(f"Creating comprehensive report in {output_dir}...")

        # Generate all plots
        self.plot_basic_metrics(os.path.join(output_dir, "basic_metrics.png"))
        self.plot_learning_dynamics(os.path.join(output_dir, "learning_dynamics.png"))
        self.plot_puzzle_performance(os.path.join(output_dir, "puzzle_performance.png"))
        self.plot_heatmap_analysis(os.path.join(output_dir, "heatmap_analysis.png"))

        # Generate summary report
        summary_file = os.path.join(output_dir, "summary_report.txt")
        with open(summary_file, "w") as f:
            # Redirect print output to file
            import sys

            original_stdout = sys.stdout
            sys.stdout = f
            self.generate_summary_report()
            sys.stdout = original_stdout

        print(f"Comprehensive report created in {output_dir}")
        print(f"Summary report saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Sudoku RL training logs")
    parser.add_argument("--log_file", type=str, help="Path to specific log file")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory containing log files")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive report")
    parser.add_argument("--basic", action="store_true", help="Show basic metrics only")
    parser.add_argument("--dynamics", action="store_true", help="Show learning dynamics only")
    parser.add_argument("--puzzles", action="store_true", help="Show puzzle performance only")
    parser.add_argument("--heatmap", action="store_true", help="Show heatmap analysis only")

    args = parser.parse_args()

    try:
        visualizer = TrainingVisualizer(args.log_file, args.log_dir)

        if args.report:
            visualizer.create_comprehensive_report()
        elif args.basic:
            visualizer.plot_basic_metrics()
        elif args.dynamics:
            visualizer.plot_learning_dynamics()
        elif args.puzzles:
            visualizer.plot_puzzle_performance()
        elif args.heatmap:
            visualizer.plot_heatmap_analysis()
        else:
            # Show summary and basic plots by default
            visualizer.generate_summary_report()
            visualizer.plot_basic_metrics()

            # Ask if user wants to see more plots
            response = input("\nShow additional analysis? (y/n): ").lower()
            if response in ["y", "yes"]:
                visualizer.plot_learning_dynamics()
                visualizer.plot_puzzle_performance()
                visualizer.plot_heatmap_analysis()

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have training log files in the specified directory.")


if __name__ == "__main__":
    main()
