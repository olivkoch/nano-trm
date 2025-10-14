# An implementation of TRM

This is an implementation of the [Transformer Reasoning Model (TRM)](https://arxiv.org/pdf/2510.04871v1)

# Data

You need to download the data from the [kaggle challenge page](https://www.kaggle.com/competitions/arc-prize-2025).

# Training

To run a training:

`uv run python src/train/nn/train.py experiment=trm_arc`

# Evaluation

To run an evaluation:

`uv run python src/train/nn/evaluate.py experiment=trm_arc`

# Visualization

To visualize the results of a model, use the `neural_viewer.ipynb`