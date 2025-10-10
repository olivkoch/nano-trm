import warnings
from importlib.util import find_spec
from typing import Any, Callable

from omegaconf import DictConfig

from src.nn.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """
    Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    Args:
        cfg: A DictConfig object containing the config tree.
    """
    # Return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # Disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # Prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # Pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """
    Optional decorator that controls the failure behavior when executing task function.

    This wrapper can be used to:
        - Make sure loggers are closed even if the task function raises an exception
          (prevents multirun failure).
        - Save the exception to a `.log` file.
        - Mark the run as failed with a dedicated file in the `logs/` folder (so we can
          find and rerun it later).
        - Etc. (adjust depending on your needs).

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Any:
        ...
        return something
    ```

    Args:
        task_func: The task function to be wrapped.

    Returns
        The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Any:
        # execute the task
        try:
            return_value = task_func(cfg=cfg)

        # Things to do if exception occurs
        except Exception as ex:
            # Save exception to `.log` file
            log.exception("")

            # Some hyperparameter combinations might be invalid or cause out-of-memory
            # errors so when using hparam search plugins like Optuna, you might want to
            # disable raising the below exception to avoid multirun failure.
            raise ex

        # Things to always do after either success or exception
        finally:
            # Display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # Always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # Check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return return_value

    return wrap
