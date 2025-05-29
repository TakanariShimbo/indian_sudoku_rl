# Sudoku Reinforcement Learning Agent

深層強化学習を用いて数独パズルを解くエージェントです。

## ファイル構成

```
├── config.py              # 設定の一元化
├── data_loader.py          # データ読み込み
├── boardenv.py            # 数独環境
├── neural_pytorch.py      # PyTorch DQN実装
├── main.py               # メインのトレーニングスクリプト
├── requirements.txt      # 依存関係
└── README.md            # このファイル
```

## セットアップ

1. 必要なパッケージをインストール:

```bash
# GPUサポート付きPyTorchをインストール
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy>=1.21.0 pandas>=1.3.0 matplotlib>=3.4.0
```

2. 数独データセット (`sudoku.csv`) を同じディレクトリに配置

3. トレーニング実行:

```bash
python main.py
```

## GPU 対応について

このプロジェクトは GPU を自動的に検出し、利用可能な場合は GPU を使用します：

- CUDA が利用可能な場合、自動的に GPU で実行
- GPU メモリ使用量の最適化
- cuDNN による自動チューニング
- 勾配クリッピングによる学習の安定化

GPU の設定は`config.py`で調整可能：

```python
# GPU configuration
ENABLE_CUDA = True  # GPU使用の有効/無効
CUDA_DEVICE = 0     # 使用するGPUデバイス
GRADIENT_CLIP = 1.0 # 勾配クリッピングの閾値
```

## 使用方法

### 基本的なトレーニング

```python
from main import SudokuTrainer

trainer = SudokuTrainer()
trainer.train_all_puzzles(max_puzzles=100)  # 100パズルでトレーニング
trainer.save_model("my_model.pth")
```

### 評価

```python
trainer.evaluate(num_puzzles=20)  # 20パズルで評価
```

### 可視化

```python
trainer.plot_training_progress()  # 学習進捗をプロット
```

## 設定のカスタマイズ

`config.py`で以下を調整できます：

- **学習率**: `LEARNING_RATE`
- **エピソード数**: `EPISODES_PER_PUZZLE`
- **メモリサイズ**: `MEMORY_SIZE`
- **バッチサイズ**: `BATCH_SIZE`
- その他のハイパーパラメータ

## アーキテクチャの特徴

### Deep Q-Network (DQN)

- Experience Replay
- Target Network
- Epsilon-greedy exploration
- RMSprop optimizer
- GPU 加速による高速学習
- 勾配クリッピングによる安定性向上

### 環境

- 9x9 数独グリッド
- エージェントは位置を移動し、数字を挿入
- 完全に解けた時に報酬を獲得

### 状態表現

- エージェントの位置 (row, col)
- 2 次元の状態空間

### 行動空間

- 移動: 上下左右 (4 行動)
- 数字挿入: 1-9 (9 行動)
- 合計 13 行動

## パフォーマンス

- PyTorch + CUDA による高速化
- GPU 自動対応
- メモリ効率の改善
- より安定した学習
- cuDNN による最適化
