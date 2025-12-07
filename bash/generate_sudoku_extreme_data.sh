#!/bin/bash
uv run python scripts/data/build_sudoku_extreme_dataset.py \
    --output-dir ./data/sudoku_extreme_1k_aug_1k \
    --subsample-size 1000 \
    --num-aug 1000 \
    --eval-ratio 0.01

