# src/nn/train_sweep.py
"""
W&B Sweep agent for hyperparameter optimization.
"""

import argparse
from functools import partial

import hydra
import wandb

from src.nn.train_c4 import train
from src.nn.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def agent_train(experiment: str, save_dir: str = None):
    """Function called by wandb.agent for each sweep trial."""
    
    with wandb.init() as run:
        log.info(f"W&B Agent started - Run: {run.name}")
        log.info(f"   Config: {dict(run.config)}")
        
        # Build overrides from sweep config
        overrides = [
            f"experiment={experiment}",
            "paths=sweep"
        ]
        
        if save_dir:
            full_save_dir = f"{save_dir.rstrip('/')}/{run.id}/"
            overrides.append(f"save_dir={full_save_dir}")
            log.info(f"   Save Dir: {full_save_dir}")
        
        # Add sweep parameters as overrides
        for param_name, param_value in run.config.items():
            if param_name.startswith("_"):
                continue
            if param_value is None:
                overrides.append(f"{param_name}=null")
            else:
                overrides.append(f"{param_name}={param_value}")
        
        log.info(f"   Overrides: {overrides}")
        
        # Load and compose config
        with hydra.initialize(config_path="configs", version_base="1.3"):
            cfg = hydra.compose(
                config_name="train.yaml",
                overrides=overrides,
            )
        
        cfg.sweep_mode = True
        
        # Run training
        result = train(cfg)
        
        # Log final metric for sweep optimization
        if result is not None:
            wandb.log({"train/policy_accuracy": result})
            log.info(f"Logged train/policy_accuracy: {result:.4f}")


def main():
    """Main entry point for W&B sweep agent."""
    parser = argparse.ArgumentParser(
        description="W&B Sweep Agent for TRM Training"
    )
    parser.add_argument("--sweep-id", required=True, help="W&B sweep ID")
    parser.add_argument("--experiment", default="trm_sudoku6x6", help="Experiment config")
    parser.add_argument("--count", type=int, default=None, help="Number of trials to run")
    parser.add_argument("--project", default="trm-sudoku", help="W&B project")
    parser.add_argument("--entity", default=None, help="W&B entity")
    parser.add_argument("--save-dir", default=None, help="Directory to save results")
    
    args = parser.parse_args()
    
    log.info("W&B Sweep Agent for TRM Training")
    log.info(f"   Sweep ID: {args.sweep_id}")
    log.info(f"   Experiment: {args.experiment}")
    log.info(f"   Max trials: {args.count or 'unlimited'}")
    if args.save_dir:
        log.info(f"   Save Dir: {args.save_dir}")
    
    agent_fn = partial(agent_train, experiment=args.experiment, save_dir=args.save_dir)
    
    wandb.agent(
        sweep_id=args.sweep_id,
        function=agent_fn,
        count=args.count,
        project=args.project,
        entity=args.entity,
    )
    
    log.info("Sweep agent completed!")


if __name__ == "__main__":
    main()