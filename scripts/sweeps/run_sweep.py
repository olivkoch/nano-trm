# scripts/run_sweep_agent.py
import argparse
import subprocess
import sys

import wandb


def run_training(sweep_config: dict, experiment: str):
    """Run training with sweep parameters as Hydra overrides."""
    overrides = [f"experiment={experiment}", "sweep_mode=true"]
    
    for key, value in sweep_config.items():
        if key.startswith("_") or key in ["wandb_version"]:
            continue
        if value is None:
            overrides.append(f"{key}=null")
        else:
            overrides.append(f"{key}={value}")
    
    cmd = ["uv", "run", "python", "src/nn/train.py"] + overrides
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run W&B sweep agent")
    parser.add_argument("--sweep-id", required=True, help="Sweep ID")
    parser.add_argument("--experiment", default="trm_sudoku6x6", help="Hydra experiment config")
    parser.add_argument("--count", type=int, default=None, help="Number of runs")
    parser.add_argument("--project", default="trm-sudoku", help="W&B project name")
    parser.add_argument("--entity", default=None, help="W&B entity")

    args = parser.parse_args()

    print(f"Starting sweep agent for: {args.sweep_id}")
    print(f"Using experiment: {args.experiment}")

    def train_wrapper():
        wandb.init()
        try:
            run_training(dict(wandb.config), args.experiment)
        finally:
            wandb.finish()

    wandb.agent(
        sweep_id=args.sweep_id,
        function=train_wrapper,
        count=args.count,
        project=args.project,
        entity=args.entity,
    )


if __name__ == "__main__":
    main()