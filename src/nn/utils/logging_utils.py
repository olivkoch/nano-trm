from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from .utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Saves the full resolved configuration and model parameters.

    Args:
        object_dict: A dictionary containing the following objects:
            - `"cfg"`: A DictConfig object containing the main config.
            - `"model"`: The Lightning model.
            - `"trainer"`: The Lightning trainer.
    """
    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    # Convert full config to container with resolved values
    resolved_config = OmegaConf.to_container(cfg, resolve=True)

    # Add model parameter counts
    resolved_config["model/params/total"] = sum(p.numel() for p in model.parameters())
    resolved_config["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    resolved_config["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # Send full config to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(resolved_config)

    # # Also update wandb config if wandb run is active
    # # This ensures sweep mode shows the final config after sweep params are applied
    # if wandb.run is not None:
    #     wandb.config.update(resolved_config, allow_val_change=True)
    #     log.info("Updated wandb config with resolved configuration")
