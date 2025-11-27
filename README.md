# An implementation of TRM

This is an implementation of the [Tiny Recursive Model (TRM)](https://arxiv.org/pdf/2510.04871v1)

Reference [code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)

# Data

ARC-AGI: You need to download the data from the [kaggle challenge page](https://www.kaggle.com/competitions/arc-prize-2025).

Other datasets are generated locally.

To generate a Sudoku dataset: `./bash/generate_sudoku_data.sh`

# Sudoku

## Training

`uv run python src/nn/train.py experiment=trm_sudoku4x4` (takes a few mins on an A10)

## Evaluation

`uv run python src/nn/evaluate.py +checkpoint=/tmp/ml-experiments/lunar-pine-174/checkpoints/last.ckpt +data_dir=./data/sudoku_4x4_small`

## Visualization

To visualize the results of a model, use `notebooks/neural_viewer.ipynb`

# Self-Play (Connect Four)

Generate curriculum training data by pitching two minimax players against each other: `uv run python scripts/data/generate_c4_curriculum_data.py --n-games 50000 --temp-player1 0.1 --temp-player2 0.3 --depth 4 --to-file minimax_games_.pkl`

## Training & evaluation

Set `enable_selfplay=False` to enable/disable self-play in `experiment/mlp_c4.yaml`

Then run `uv run python src/nn/train_c4.py experiment=mlp_c4.yaml`

Replace `mlp` with `cnn` or `trm` for CNN or TRM.

The training script produces the evaluation metrics automatically.

# Comments / contributions

Follow me on [X](https://x.com/olivkoch)

## Sweeps

### 1. Create sweep
```uv run python scripts/create_sweep.py \
    --sweep-file src/nn/configs/sweeps/trm_sudoku6x6_sweep.yaml \
    --project trm-sudoku
    ```

### 2. Run agent
```uv run python src/nn/train_sweep.py \
    --sweep-id <sweep_id> \
    --project trm-sudoku \
    --experiment trm_sudoku6x6 \
    --count 20
```

