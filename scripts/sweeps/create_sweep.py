# scripts/create_sweep.py
import argparse
import json
from pathlib import Path

import wandb
import yaml


def create_sweep_from_file(
    sweep_file: Path,
    project_name: str = "trm-sudoku",
    entity: str = None,
):
    """Create sweep from YAML configuration file."""
    print(f"Loading sweep configuration from: {sweep_file}")

    with open(sweep_file, "r") as f:
        sweep_config = yaml.safe_load(f)

    if "project" not in sweep_config:
        sweep_config["project"] = project_name
    if entity and "entity" not in sweep_config:
        sweep_config["entity"] = entity

    print(f"Creating W&B sweep: {sweep_config.get('name', 'Unnamed Sweep')}")
    print(f"   Project: {sweep_config['project']}")
    print(f"   Method: {sweep_config['method']}")
    print(f"   Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
    print(f"   Parameters: {len(sweep_config['parameters'])} hyperparameters")

    print("\nSweep configuration:")
    print(json.dumps(sweep_config, indent=2))

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=sweep_config["project"],
        entity=sweep_config.get("entity"),
    )

    print(f"\n✅ Sweep created successfully!")
    print(f"   Sweep ID: {sweep_id}")
    
    entity_str = sweep_config.get('entity', wandb.api.default_entity) or ''
    if entity_str:
        print(f"   Dashboard: https://wandb.ai/{entity_str}/{sweep_config['project']}/sweeps/{sweep_id}")

    return sweep_id


def main():
    parser = argparse.ArgumentParser(description="Create W&B sweep")
    parser.add_argument("--sweep-file", type=Path, required=True, help="YAML sweep config")
    parser.add_argument("--project", default="trm-sudoku", help="W&B project name")
    parser.add_argument("--entity", default=None, help="W&B entity name")
    parser.add_argument("--save-info", type=Path, help="Save sweep info to JSON file")

    args = parser.parse_args()

    if not args.sweep_file.exists():
        print(f"❌ Sweep file not found: {args.sweep_file}")
        return 1

    sweep_id = create_sweep_from_file(args.sweep_file, args.project, args.entity)

    if args.save_info:
        sweep_info = {
            "sweep_id": sweep_id,
            "project": args.project,
            "entity": args.entity,
        }
        args.save_info.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_info, "w") as f:
            json.dump(sweep_info, f, indent=2)
        print(f"💾 Sweep info saved to: {args.save_info}")

    print(f"\nRun agent with: python scripts/run_sweep_agent.py --sweep-id {sweep_id}")
    return 0


if __name__ == "__main__":
    exit(main())