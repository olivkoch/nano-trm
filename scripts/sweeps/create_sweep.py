# scripts/create_sweep.py
"""
Create W&B sweep for TRM hyperparameter optimization.
"""

import argparse
import json
import os
from pathlib import Path

import wandb
import yaml


def create_sweep_from_file(
    sweep_file: Path,
    project_name: str,
    entity: str,
):
    """Create sweep from YAML configuration file."""
    print(f"Loading sweep configuration from: {sweep_file}")

    with open(sweep_file, "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_config["project"] = project_name
    sweep_config["entity"] = entity

    print(f"\nCreating W&B sweep: {sweep_config.get('name', 'Unnamed Sweep')}")
    print(f"   Project: {project_name}")
    print(f"   Entity: {entity}")
    print(f"   Method: {sweep_config['method']}")
    print(f"   Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
    print(f"   Parameters: {len(sweep_config['parameters'])} hyperparameters")

    print("\nSweep configuration:")
    print(json.dumps(sweep_config, indent=2))

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project_name,
        entity=entity,
    )

    return sweep_id


def main():
    parser = argparse.ArgumentParser(
        description="Create W&B sweep for TRM hyperparameter optimization"
    )
    parser.add_argument(
        "--sweep-file",
        type=Path,
        required=True,
        help="YAML file containing sweep configuration",
    )
    parser.add_argument(
        "--project", 
        default=os.environ.get("WANDB_PROJECT", "trm-sudoku"), 
        help="W&B project name"
    )
    parser.add_argument(
        "--entity", 
        default=os.environ.get("WANDB_ENTITY"), 
        help="W&B entity name"
    )
    parser.add_argument(
        "--save-info", 
        type=Path,
        help="Save sweep info to JSON file"
    )

    args = parser.parse_args()

    if not args.entity:
        print("❌ Entity required. Set WANDB_ENTITY or pass --entity")
        return 1

    if not args.sweep_file.exists():
        print(f"❌ Sweep file not found: {args.sweep_file}")
        return 1

    sweep_id = create_sweep_from_file(args.sweep_file, args.project, args.entity)

    full_sweep_id = f"{args.entity}/{args.project}/{sweep_id}"
    dashboard_url = f"https://wandb.ai/{args.entity}/{args.project}/sweeps/{sweep_id}"

    print(f"\n✅ Sweep created: {full_sweep_id}")
    print(f"   Dashboard: {dashboard_url}")

    if args.save_info:
        sweep_info = {
            "sweep_id": sweep_id,
            "full_sweep_id": full_sweep_id,
            "project": args.project,
            "entity": args.entity,
            "dashboard_url": dashboard_url,
        }
        args.save_info.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_info, "w") as f:
            json.dump(sweep_info, f, indent=2)
        print(f"💾 Saved to: {args.save_info}")

    print(f"\nRun agent:")
    print(f"  uv run python src/nn/train_sweep.py --sweep-id {full_sweep_id} --count 20")

    return 0


if __name__ == "__main__":
    exit(main())