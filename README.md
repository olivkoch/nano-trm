# Fast, easy-to-play-with Tiny Recursive Models

This is an implementation of the [Tiny Recursive Model (TRM)](https://arxiv.org/pdf/2510.04871v1). 

Train a TRM in a few minutes on an A10. Reproduce the official TRM results. Push the envelope.

<p align="center">
<img src="demo/sudoku_thinking_9_steps.gif" width="400" alt="Sudoku thinking">
</p>

# Motivation

Recently, recursive models made a big comeback, notably with Tiny Recursive Models, which [won 1st paper award at the ARC-AGI 2 contest](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.youtube.com/watch%3Fv%3DP9zzUM0PrBM&ved=2ahUKEwiO86mcjfWRAxW7BfsDHeyvHsUQwqsBegQIFhAB&usg=AOvVaw12B_Wsm15MK-AVqrZVbGAS) and maybe more importantly, reached an impressive level of performance on several benchmarks such as ARC-AGI 2 and Sudoku Extreme. TRM brings a lot of simplicity to its ancestor, HRM. However, the codebase inherits much of HRM's legacy.

We propose a clean implementation of TRM. We call it "nano" because it is easy to experiment with, yet incorporates all important implementation details of TRM. The project uses hydra, torch lightning and uv to make experimentation easy. We propose an [in-code introductory video](https://youtu.be/8Gzv5tGmJ1M) and small datasets (Sudoku 4x4 and 6x6) that lets you train a TRM on an A10 in just one hour! 

This repo reproduces the results on Sudoku Extreme and Maze Hard (87% and 75% exact accuracy on validation, respectively). We hope you will find this repo useful for your own experimentation on TRM.

# Installation

This repo comes with `uv`. You just need to run `uv run python ...` commands and everything will be installed automagically on the first run.

# Sudoku Extreme

Generate data: 

`uv run python scripts/data/build_sudoku_extreme_dataset.py --output-dir ./data/sudoku_extreme_1k_aug_1k --subsample-size 1000 --num-aug 1000 --eval-ratio 0.01`

Run a training: 

`uv run python src/nn/train.py experiment=trm_sudoku_extreme_1k_aug_1k`

Training time ~1h on an H100 SXM5. You should get to ~87% exact accuracy on validation (same as the reference implementation)

# Maze Hard

Generate data: 

`uv run python scripts/data/build_maze_dataset.py --output-dir ./data/maze-30x30-hard-1k --num-aug 0 --eval-ratio 1.0`

Run a training: 

`uv run python src/nn/train.py experiment=trm_maze`

Training time ~2h on an H100 SXM5. You should get to ~75% exact accuracy on validation (same as the reference implementation)

# Small Sudoku datasets

Generate data:

`bash bash/generate_sudoku_data.sh` -> choose which dataset you want to generate

Run a training:

`uv run python src/nn/train.py experiment=trm_sudoku_4x4`

This take a few minutes on a A10!

# ARC-AGI

Download the data from the [kaggle challenge page](https://www.kaggle.com/competitions/arc-prize-2025).

# Visualizations

Sudoku:
- Evaluate and generate a gif: `uv run python src/nn/evaluate.py +checkpoint=./checkpoints/smooth-sunset-204.ckpt +data_dir=./data/sudoku-extreme-1k-aug-1k +visualize=true +save_gif=true +min_steps=9`

Maze:

- Generate a dataset with test data (not just val): `uv run python scripts/data/build_maze_dataset.py --output-dir ./data/maze-30x30-hard-1k --num-aug 0 --eval-ratio 0.5`
- Evaluate and generate a gif: `uv run python src/nn/evaluate.py +checkpoint=./checkpoints/stellar-shape-194.ckpt +data_dir=./data/maze-30x30-hard-1k +visualize=true +save_gif=true +min_steps=9 +task=maze`

# Comments / contributions

Follow me on [X](https://x.com/olivkoch)

# Technical Notes

## Codebase structure

- `src/nn/train.py`: main training script
- `src/nn/models/trm.py`: TRM model
- `src/nn/configs/experiments`: main experimentation configurations.

## Installing AdamATan2

This project uses the vanilla AdamW optimizer. If you want AdamATan2 and struggle to install it, here is how to install it from source:

```
# Clone the repo
cd /tmp
git clone https://github.com/imoneoi/adam-atan2.git

cd adam-atan2/

uv pip install --python /home/ubuntu/nano-trm/.venv/bin/python --verbose --no-cache-dir --no-build-isolation -e .

# Test
cd /home/ubuntu/nano-trm
uv run python
from adam_atan2 import AdamATan2 -> this should work
```

AdamAtan2 is not needed to reproduce the official results on Sudoku Extreme and Maze Hard. It might be needed on harder problems / deeper models (e.g. ARC AGI 2)
