# sudoku_env.py
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import time

# ---- データセット ----
DATA_PATH = Path(__file__).with_name("sudoku-3m.csv")  # ← お使いの CSV 名に合わせて

_df = pd.read_csv(DATA_PATH, usecols=["puzzle", "solution"], dtype=str).dropna()  # ← puzzle / solution 列だけ読む

# 81 文字の行だけ残す
_df = _df[_df["puzzle"].str.len() == 81]

assert not _df.empty, f"dataset is empty or invalid at {DATA_PATH}"

QUIZZES = _df["puzzle"].tolist()
SOLUTIONS = _df["solution"].tolist()


class SudokuEnv(gym.Env):
    """
    行動 0–3 : カーソル移動  (up, down, right, left)
    行動 4–12: (1…9) の数字を現在セルに書き込む（元が 0 のマスのみ）
    観測     : Dict:
        - coords : (row,col) 0-8
        - grid   : 9×9 整数配列 (0→空白, 1-9 は数字)
    報酬     : 完成 =  +1, それ以外 0
    エピソード長制限 : 300 ステップ（解けなければ打ち切り）
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        # --- spaces ---
        self.action_space = spaces.Discrete(13)
        self.observation_space = spaces.Dict(
            {
                "coords": spaces.Box(0, 8, shape=(2,), dtype=np.int8),
                "grid": spaces.Box(0, 9, shape=(9, 9), dtype=np.int8),
            }
        )

        # --- 内部状態 ---
        self._puzzle_idx = 0  # CSV 内の現在行
        self.agent_pos = np.array([0, 0], dtype=np.int8)
        self.grid = np.zeros((9, 9), dtype=np.int8)
        self._fixed_mask = np.zeros_like(self.grid, dtype=bool)  # True = 元から数字あり
        self.solution = np.zeros_like(self.grid, dtype=np.int8)
        self._steps = 0  # ステップカウンタ

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 新しいパズルをロード
        q = QUIZZES[self._puzzle_idx]
        sol = SOLUTIONS[self._puzzle_idx]
        self._puzzle_idx = (self._puzzle_idx + 1) % len(QUIZZES)

        self.grid = self._str_to_grid(q)
        self.solution = self._str_to_grid(sol)
        self._fixed_mask = self.grid != 0
        self.agent_pos = np.array([0, 0], dtype=np.int8)
        self._steps = 0

        return self._get_obs(), {}

    def step(self, action):
        self._steps += 1

        r, c = self.agent_pos
        if action == 0 and r > 0:  # up
            self.agent_pos[0] -= 1
        elif action == 1 and r < 8:  # down
            self.agent_pos[0] += 1
        elif action == 2 and c < 8:  # right
            self.agent_pos[1] += 1
        elif action == 3 and c > 0:  # left
            self.agent_pos[1] -= 1
        elif 4 <= action <= 12:  # insert number (1-9)
            num = action - 3  # 4→1, …, 12→9
            if not self._fixed_mask[r, c]:
                self.grid[r, c] = num

        # 報酬 & 終了判定
        terminated = np.array_equal(self.grid, self.solution)
        truncated = self._steps >= 300  # タイムアウト
        reward = 1.0 if terminated else 0.0

        if self.render_mode == "human":
            self._render_text()

        return self._get_obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------ helpers
    def _get_obs(self):
        return {"coords": self.agent_pos.copy(), "grid": self.grid.copy()}

    @staticmethod
    def _str_to_grid(s: str) -> np.ndarray:
        s = s.replace(".", "0")  # ① 空マス '.' → '0' に変換
        assert len(s) == 81, f"Puzzle length must be 81, got {len(s)}"
        arr = np.frombuffer(s.encode(), dtype=np.uint8) - ord("0")
        return arr.reshape(9, 9).astype(np.int8)

    # optional
    def render(self):
        self._render_text()

    # sudoku_env.py 内
    def _render_text(self):
        import time, sys

        r, c = self.agent_pos
        lines = []
        for i in range(9):
            row_elems = []
            for j in range(9):
                val = self.grid[i, j]
                char = "." if val == 0 else str(val)

                # --- カーソル位置を [] で囲う ---
                if (i, j) == (r, c):
                    char = f"[{char}]"
                else:
                    char = f" {char} "

                row_elems.append(char)
            lines.append("".join(row_elems))

        sys.stdout.write("\033c")  # clear
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        time.sleep(0.2)  # ← 0.05 → 0.2 秒に
