#!/usr/bin/env python3
"""
Train TRM on Connect Four with Self-Play
"""

import warnings
warnings.filterwarnings("ignore")

import shutil
from pathlib import Path
from typing import Optional

import hydra
import lightning
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from src.nn.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


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
    """Train Connect Four TRM with self-play"""
    
    # Set seed for reproducibility
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)
    
    output_dir = Path(cfg["paths"]["output_dir"])
        
    # Instantiate model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, output_dir=output_dir)
    
    # Instantiate callbacks
    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    
    # Instantiate loggers
    log.info("Instantiating loggers...")
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))
    
    # Instantiate trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        callbacks=callbacks, 
        logger=loggers
    )
    
    # Create object dict for hyperparameter logging
    object_dict = {
        "cfg": cfg,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }
    
    # Log hyperparameters
    if loggers:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
        if not cfg.get("sweep_mode", False) and wandb.run is not None:
            wandb.config.update(flatten_config(cfg), allow_val_change=True)
    
    # Training info
    log.info("="*50)
    log.info("CONNECT FOUR SELF-PLAY TRAINING")
    log.info("="*50)
    log.info(f"Training configuration:")
    log.info(f"  Max epochs: {cfg.trainer.max_epochs}")
    log.info(f"  MCTS simulations: {cfg.model.mcts_simulations}")
    log.info(f"  Val check interval: {cfg.trainer.get('val_check_interval', '1.0')}")
    
    # Start training
    log.info("Starting training!")
    trainer.fit(
        model=model, 
        ckpt_path=cfg.get("ckpt_path")
    )
    
    log.info("Training finished!")
    
    # Save config
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    
    # Save to specified directory if provided
    if cfg.get("save_dir") is not None:
        save_dir = Path(cfg.save_dir)
        
        # Append wandb run name if specified
        if cfg.get("append_wandb_name_to_save_dir", False) and wandb.run and wandb.run.name:
            save_dir = save_dir / wandb.run.name
        
        log.info(f"Saving training output to: {save_dir}")
        shutil.copytree(output_dir, save_dir, dirs_exist_ok=True)
    
    # Return best metric if available
    if hasattr(model, 'best_win_rate'):
        return model.best_win_rate
    return None


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """
    Main entry point for Connect Four training.
    
    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    # Apply extra utilities
    extras(cfg)
    
    # Train the model
    return train(cfg)


if __name__ == "__main__":
    main()