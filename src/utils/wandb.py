import os
import tempfile

import wandb

WANDB_ENTITY_NAME = "iuhoiuhoiuho"
WANDB_PROJECT_NAME = "uihoiuhoihioh"


def init_wandb_logger(args, run_name=None, tags=None):
    """
    Initialize wandb run and log all parameters from args.

    Args:
        args: Parsed arguments from argparse
        run_name (str, optional): Name for this specific run
        tags (list, optional): List of tags for the run

    Returns:
        wandb.Run: The initialized wandb run object
    """
    if os.environ.get("WANDB_DISABLED") == "true":
        print("Wandb logging is disabled.")
        return None

    # Convert args to dictionary for logging
    config = vars(args)

    # Initialize wandb run
    run = wandb.init(
        dir=tempfile.gettempdir(),
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        name=run_name,
        tags=tags,
        # cause gender classifier uses _sconf_data to store config:
        config=config["_sconf_data"] if "_sconf_data" in config else config,
    )

    # Log additional useful information
    wandb.log(
        {
            "total_parameters": len(config),
        }
    )

    print(f"Wandb run initialized: {wandb.run.name}")
    print(f"Run URL: {wandb.run.url}")

    return run
