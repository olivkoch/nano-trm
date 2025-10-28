"""
ARC-AGI-2 Challenge - Super Baseline Solution
This provides a minimal working solution for the ARC Prize 2025 competition.
Compatible with the new ARC-AGI-2 dataset structure.
"""

import json
from pathlib import Path
from typing import Dict


def load_task(self, task_path: Path) -> Dict:
    """Load a single ARC-AGI-2 task from JSON file.

    Args:
        task_path: Path to task JSON file

    Returns:
        Dictionary containing the task data
    """
    with open(task_path) as f:
        task = json.load(f)
    return task


def load_all_tasks(self, directory: Path) -> Dict[str, Dict]:
    """Load all tasks from a directory.

    Args:
        directory: Directory containing task JSON files

    Returns:
        Dictionary mapping task IDs to task data
    """
    tasks = {}
    if not directory.exists():
        print(f"Directory {directory} does not exist!")
        return tasks

    for task_file in directory.glob("*.json"):
        task_id = task_file.stem
        tasks[task_id] = load_task(task_file)

    return tasks
