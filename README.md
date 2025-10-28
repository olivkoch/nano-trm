# An implementation of TRM

This is an implementation of the [Tiny Recursive Model (TRM)](https://arxiv.org/pdf/2510.04871v1)

Reference [code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)

# Data

We are working with ARC-AGI-2 exclusively for now.

You need to download the data from the [kaggle challenge page](https://www.kaggle.com/competitions/arc-prize-2025).

# Setup

Install `uv`: `sudo snap install astral-uv --classic`

# Training

To run a training:

`uv run python src/train/nn/train.py experiment=trm_arc`

# Evaluation

To run an evaluation:

`uv run python src/train/nn/evaluate.py experiment=trm_arc`

# Visualization

To visualize the results of a model, use `notebooks/neural_viewer.ipynb`

# Note to contributors

If you would like to make contributions to this codebase, here are things you can do:

- Help reproduce the results of the original paper
- Implement missing features (carry, puzzle embeddings)