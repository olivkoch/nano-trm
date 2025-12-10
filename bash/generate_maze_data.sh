#!/bin/bash
uv run python scripts/data/build_maze_dataset.py \
    --output-dir ./data/maze-30x30-hard-1k \
    --num-aug 0 \
    --eval-ratio 1.0