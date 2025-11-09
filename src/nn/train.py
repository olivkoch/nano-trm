import warnings

warnings.filterwarnings("ignore")

import shutil
from pathlib import Path
from typing import Optional

import lightning
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger

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


def update_model_config(cfg: DictConfig, datamodule: LightningDataModule):
    # Add num_puzzles to model config
    cfg.model.num_puzzles = datamodule.num_puzzles
    cfg.model.batch_size = datamodule.batch_size
    cfg.model.pad_value = datamodule.pad_value
    cfg.model.max_grid_size = datamodule.max_grid_size
    cfg.model.vocab_size = datamodule.vocab_size
    cfg.model.seq_len = cfg.model.max_grid_size * cfg.model.max_grid_size
    log.info(
        f"Setting model config from data module:  num_puzzles = {datamodule.num_puzzles} batch_size = {datamodule.batch_size} vocab_size = {datamodule.vocab_size}"
    )


@task_wrapper
def train(cfg: DictConfig) -> Optional[float]:
    # Set seed for random number generators in pytorch, numpy and python.random.
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)

    output_dir = Path(cfg["paths"]["output_dir"])

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Setup datamodule to get num_puzzles
    datamodule.setup(stage="fit")

    update_model_config(cfg, datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, output_dir=output_dir)

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
        if not cfg.sweep_mode and wandb.run is not None:
            wandb.config.update(flatten_config(cfg), allow_val_change=True)

    log.info("Starting training!")

    datamodule.setup(stage="fit")
    log.info(f"Val check interval: {cfg.trainer.get('val_check_interval', 'default (1.0)')}")
    log.info(
        f"Check val every n epoch: {cfg.trainer.get('check_val_every_n_epoch', 'default (1.0)')}"
    )
    log.info(f"Steps per epoch: {len(datamodule.train_dataset) // cfg.data.batch_size}")
    log.info(f"Max epochs: {cfg.trainer.max_epochs}")
    log.info(f"Batch size: {cfg.data.batch_size}")

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
