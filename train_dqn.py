# train_dqn.py
from sudoku_env import SudokuEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import FlattenObservation

# ---- 環境生成 ----
train_env = FlattenObservation(SudokuEnv(render_mode=None))
eval_env = FlattenObservation(SudokuEnv(render_mode=None))

# ---- モデル ----
model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=64,
    target_update_interval=1_000,
    gamma=0.99,
    exploration_fraction=0.1,
    verbose=1,
)

# ---- 評価コールバック ----
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="models/",
    eval_freq=5_000,
    n_eval_episodes=20,
    deterministic=True,
    render=False,
)

# ---- 学習 ----
model.learn(total_timesteps=1_000_000, callback=eval_cb, progress_bar=True)
model.save("sudoku_dqn_sb3")
