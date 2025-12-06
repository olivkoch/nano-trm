#!/bin/bash
uv run python src/nn/train.py experiment=trm_sudoku_extreme_1k_aug1k model_tuning.use_sigreg=false model_tuning.use_constant_lr=true model_tuning.use_2d_rope=false logger.wandb.notes="trm extreme 1k-aug use_constant_lr"
uv run python src/nn/train.py experiment=trm_sudoku_extreme_1k_aug1k model_tuning.use_sigreg=false model_tuning.use_constant_lr=true model_tuning.use_2d_rope=true logger.wandb.notes="trm extreme 1k-aug use_constant_lr + use_2d_rope"
uv run python src/nn/train.py experiment=trm_sudoku_extreme_1k_aug1k model_tuning.use_sigreg=true model_tuning.use_constant_lr=true model_tuning.use_2d_rope=false logger.wandb.notes="trm extreme 1k-aug use_sigreg + use_constant_lr"
uv run python src/nn/train.py experiment=trm_sudoku_extreme_1k_aug1k model_tuning.use_sigreg=true model_tuning.use_constant_lr=true model_tuning.use_2d_rope=true logger.wandb.notes="trm extreme 1k-aug use_sigreg + use_constant_lr + use_2d_rope"
uv run python src/nn/train.py experiment=trm_sudoku_extreme_1k_aug1k model_tuning.use_sigreg=false model_tuning.use_constant_lr=false model_tuning.use_2d_rope=false logger.wandb.notes="trm extreme 1k-aug baseline"

# TODO: very small training sets

