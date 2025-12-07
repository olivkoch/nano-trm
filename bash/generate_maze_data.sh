#!/bin/bash
uv run python scripts/data/build_maze_dataset.py \
    --output-dir ./data/maze_30x30 \
    --num-aug 7 \
    --eval-ratio 0.5