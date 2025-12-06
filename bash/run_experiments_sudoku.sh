#!/bin/bash
uv run python src/nn/train.py experiment=trm_sudoku_6x6_medium model_tuning.use_sigreg=false model_tuning.use_constant_lr=false model_tuning.use_2d_rope=false logger.wandb.notes="trm 6xx6 medium baseline"
uv run python src/nn/train.py experiment=trm_sudoku_6x6_medium model_tuning.use_sigreg=true model_tuning.use_constant_lr=false model_tuning.use_2d_rope=false logger.wandb.notes="trm 6xx6 medium use_sigreg"
uv run python src/nn/train.py experiment=trm_sudoku_6x6_medium model_tuning.use_sigreg=false model_tuning.use_constant_lr=true model_tuning.use_2d_rope=false logger.wandb.notes="trm 6xx6 medium use_constant_lr"
uv run python src/nn/train.py experiment=trm_sudoku_6x6_medium model_tuning.use_sigreg=false model_tuning.use_constant_lr=false model_tuning.use_2d_rope=true logger.wandb.notes="trm 6xx6 medium use_2d_rope"
uv run python src/nn/train.py experiment=trm_sudoku_6x6_medium model_tuning.use_sigreg=true model_tuning.use_constant_lr=true model_tuning.use_2d_rope=true logger.wandb.notes="trm 6xx6 medium use_sigreg + use_constant_lr + use_2d_rope"

uv run python src/nn/train.py experiment=trm_sudoku_mixed_6x6_9x9 model_tuning.use_sigreg=false model_tuning.use_constant_lr=false model_tuning.use_2d_rope=false logger.wandb.notes="trm 9x9 medium baseline"
uv run python src/nn/train.py experiment=trm_sudoku_mixed_6x6_9x9 model_tuning.use_sigreg=true model_tuning.use_constant_lr=false model_tuning.use_2d_rope=false logger.wandb.notes="trm 9x9 medium use_sigreg"
uv run python src/nn/train.py experiment=trm_sudoku_mixed_6x6_9x9 model_tuning.use_sigreg=false model_tuning.use_constant_lr=true model_tuning.use_2d_rope=false logger.wandb.notes="trm 9x9 medium use_constant_lr"
uv run python src/nn/train.py experiment=trm_sudoku_mixed_6x6_9x9 model_tuning.use_sigreg=false model_tuning.use_constant_lr=false model_tuning.use_2d_rope=true logger.wandb.notes="trm 9x9 medium use_2d_rope"
uv run python src/nn/train.py experiment=trm_sudoku_mixed_6x6_9x9 model_tuning.use_sigreg=true model_tuning.use_constant_lr=true model_tuning.use_2d_rope=true logger.wandb.notes="trm 9x9 medium use_sigreg + use_constant_lr + use_2d_rope"

# TODO: very small training sets

