import shutil
import torch
import os
import lightning
import shutil
import wandb
from pathlib import Path
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from src.viewer.arc_notebook_viewer import create_viewer
from lightning.pytorch.loggers import Logger
from typing import Optional
from src.nn.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
log = RankedLogger(__name__, rank_zero_only=True)
import hydra
from omegaconf import DictConfig, OmegaConf

def flatten_config(cfg, parent_key="", sep="."):
    """Flatten a nested config to avoid W&B duplication."""
    items = []
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    def _flatten(obj, parent_key=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, (dict, list)) and not isinstance(v, str):
                    _flatten(v, new_key)
                else:
                    items.append((new_key, v))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                if isinstance(v, (dict, list)) and not isinstance(v, str):
                    _flatten(v, new_key)
                else:
                    items.append((new_key, v))

    _flatten(config_dict)
    return dict(items)

@task_wrapper
def train(cfg: DictConfig) -> Optional[float]:

    # Set seed for random number generators in pytorch, numpy and python.random.
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)

    output_dir = Path(cfg["paths"]["output_dir"])

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, output_dir=output_dir)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers, enable_progress_bar=False
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }

    if loggers:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
        if not cfg.sweep_mode:
            wandb.config.update(flatten_config(cfg), allow_val_change=True)

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    log.info("Training finished!")

    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)

    if cfg.save_dir is not None:
        save_dir = cfg.save_dir
        # Append wandb run name to save_dir if enabled
        if cfg.append_wandb_name_to_save_dir and wandb.run and wandb.run.name:
            save_dir = save_dir.rstrip("/") + "/" + wandb.run.name

        log.info(f"Uploading training output to: {save_dir}")
        shutil.copytree(output_dir, save_dir)

    # Create and train neural solver
    if False:
        neural_solver = NeuralARCSolver(name="SimpleNeuralNet")

        evaluator = ARCEvaluator(data_dir="data")

        # Train on some data (placeholder training)
        training_data = evaluator.datasets['training']['challenges']
        history = neural_solver.train(training_data, epochs=10)
        print(f"Training complete. Final loss: {history['loss'][-1]:.3f}")

        # Evaluate
        result = evaluator.evaluate_solver(neural_solver, dataset='training', max_tasks=20)

        # store model weights
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(neural_solver.model.state_dict(), "checkpoints/neural_arc_solver.pth")

@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """
    Main entry point for training.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    # Apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # Train the model
    return train(cfg)

if __name__ == "__main__":
    main()