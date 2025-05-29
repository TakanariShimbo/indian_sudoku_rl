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
        """Plot basic training metrics with overlapped graphs"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Episode rewards and success rate (overlapped)
        ax1_twin = ax1.twinx()
        line1 = ax1.plot(self.df["global_episode"], self.df["avg_reward_100"], "b-", linewidth=2, label="Avg Reward (100)")
        line2 = ax1_twin.plot(self.df["global_episode"], self.df["success_rate_100"], "r-", linewidth=2, label="Success Rate (100)")

        ax1.set_title("Learning Progress: Reward & Success Rate", size=14, fontweight="bold")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Average Reward", color="b")
        ax1_twin.set_ylabel("Success Rate", color="r")
        ax1.tick_params(axis="y", labelcolor="b")
        ax1_twin.tick_params(axis="y", labelcolor="r")
        ax1_twin.set_ylim(0, 1)

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Episode steps and epsilon (overlapped)
        ax2_twin = ax2.twinx()
        line3 = ax2.plot(self.df["global_episode"], self.df["avg_steps_100"], "g-", linewidth=2, label="Avg Steps (100)")
        line4 = ax2_twin.plot(self.df["global_episode"], self.df["epsilon"], "orange", linewidth=2, label="Epsilon")

        ax2.set_title("Training Dynamics: Steps & Exploration", size=14, fontweight="bold")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Average Steps", color="g")
        ax2_twin.set_ylabel("Epsilon", color="orange")
        ax2.tick_params(axis="y", labelcolor="g")
        ax2_twin.tick_params(axis="y", labelcolor="orange")

        # Combine legends
        lines = line3 + line4
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="upper right")
        ax2.grid(True, alpha=0.3)

        # Training loss with smoothing
        loss_data = self.df[self.df["loss"] > 0]
        if len(loss_data) > 0:
            ax3.plot(loss_data["global_episode"], loss_data["loss"], alpha=0.3, color="red", label="Raw Loss")
            if len(loss_data) > 50:
                window = min(100, len(loss_data) // 10)
                smoothed_loss = loss_data["loss"].rolling(window=window, center=True).mean()
                ax3.plot(loss_data["global_episode"], smoothed_loss, "darkred", linewidth=2, label=f"Smoothed ({window})")
        ax3.set_title("Training Loss", size=14, fontweight="bold")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Loss")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Validation results (when available)
        validation_data = self.df[self.df["validation_success_rate"] != ""]
        if len(validation_data) > 0:
            # Convert validation data to numeric
            validation_data = validation_data.copy()
            validation_data["validation_success_rate"] = pd.to_numeric(validation_data["validation_success_rate"])
            validation_data["validation_avg_steps"] = pd.to_numeric(validation_data["validation_avg_steps"])

            ax4_twin = ax4.twinx()
            line5 = ax4.plot(
                validation_data["global_episode"], validation_data["validation_success_rate"], "purple", marker="o", linewidth=2, markersize=4, label="Val Success Rate"
            )
            line6 = ax4_twin.plot(
                validation_data["global_episode"], validation_data["validation_avg_steps"], "brown", marker="s", linewidth=2, markersize=4, label="Val Avg Steps"
            )

            ax4.set_ylabel("Validation Success Rate", color="purple")
            ax4_twin.set_ylabel("Validation Avg Steps", color="brown")
            ax4.tick_params(axis="y", labelcolor="purple")
            ax4_twin.tick_params(axis="y", labelcolor="brown")
            ax4.set_ylim(0, 1)

            # Combine legends
            lines = line5 + line6
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc="upper left")
        else:
            # Cumulative solved if no validation data
            ax4.plot(self.df["global_episode"], self.df["cumulative_solved"], "purple", linewidth=2)
            ax4.set_ylabel("Total Solved")

        ax4.set_title("Validation Performance", size=14, fontweight="bold")
        ax4.set_xlabel("Episode")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Basic metrics plot saved to {save_path}")

        plt.show()

    def generate_summary_report(self):
        """Generate a summary report of the training"""
        print("=" * 60)
        print("TRAINING SUMMARY REPORT")
        print("=" * 60)

        print(f"Log file: {self.log_file}")
        print(f"Total episodes: {len(self.df):,}")
        print(f"Unique puzzles: {self.df['puzzle_index'].nunique()}")

        print("\n--- PERFORMANCE METRICS ---")
        print(f"Total puzzles solved: {self.df['cumulative_solved'].max()}")
        print(f"Final success rate (100-ep avg): {self.df['success_rate_100'].iloc[-1]:.2%}")
        print(f"Best success rate achieved: {self.df['success_rate_100'].max():.2%}")

        print("\n--- LEARNING DYNAMICS ---")
        print(f"Initial epsilon: {self.df['epsilon'].iloc[0]:.3f}")
        print(f"Final epsilon: {self.df['epsilon'].iloc[-1]:.3f}")

        # Validation results if available
        validation_data = self.df[self.df["validation_success_rate"] != ""]
        if len(validation_data) > 0:
            validation_data = validation_data.copy()
            validation_data["validation_success_rate"] = pd.to_numeric(validation_data["validation_success_rate"])
            print(f"\n--- VALIDATION RESULTS ---")
            print(f"Best validation success rate: {validation_data['validation_success_rate'].max():.2%}")
            print(f"Final validation success rate: {validation_data['validation_success_rate'].iloc[-1]:.2%}")

        print("=" * 60)

    def create_comprehensive_report(self, output_dir=None):
        """Create a comprehensive visualization report"""
        if output_dir is None:
            timestamp = self.log_file.split("_")[-1].replace(".csv", "")
            output_dir = os.path.join(self.log_dir, f"analysis_{timestamp}")

        os.makedirs(output_dir, exist_ok=True)

        print(f"Creating comprehensive report in {output_dir}...")

        # Generate basic plot only
        self.plot_basic_metrics(os.path.join(output_dir, "basic_metrics.png"))

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

    args = parser.parse_args()

    try:
        visualizer = TrainingVisualizer(args.log_file, args.log_dir)

        if args.report:
            visualizer.create_comprehensive_report()
        else:
            # Show summary and basic metrics
            visualizer.generate_summary_report()
            visualizer.plot_basic_metrics()

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have training log files in the specified directory.")


if __name__ == "__main__":
    main()
