# test.py
"""
Saved model evaluator for Sudoku DQN (Gymnasium + Stable-Baselines3).

$ python test.py                          # デフォルト (20 episodes, render on)
$ python test.py --model models/best_model.zip --episodes 100 --no-render
"""
from pathlib import Path
import argparse
import sys

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN

from sudoku_env import SudokuEnv


def run_episode(model: DQN, env: gym.Env, render: bool = False) -> bool:
    """単一エピソードを実行して solved / failed を返す"""
    obs, _ = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if render:
            env.render()
    return terminated  # True → 完成


def evaluate(model_path: str, n_episodes: int, render: bool):
    env = FlattenObservation(SudokuEnv(render_mode="human" if render else None))
    model = DQN.load(model_path, env=env)

    solved = 0
    for ep in range(1, n_episodes + 1):
        ok = run_episode(model, env, render)
        solved += int(ok)
        status = "Solved" if ok else "Failed"
        print(f"Episode {ep:>3}/{n_episodes}: {status}")

    rate = solved / n_episodes
    print(f"\nSuccess rate: {solved}/{n_episodes} = {rate:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="sudoku_dqn_sb3.zip",
        help="Path to the saved SB3 model (.zip)",
    )
    parser.add_argument(
        "--episodes",
        "-n",
        type=int,
        default=20,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable console rendering",
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        sys.exit(f"❌ Model file not found: {args.model}")

    evaluate(args.model, args.episodes, render=not args.no_render)
