"""
Sudoku evaluation script with cross-size support.
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig

from src.nn.sudoku_evaluator import SudokuEvaluator
from src.nn.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("medium")
warnings.filterwarnings("ignore", module="pydantic")


@task_wrapper
def evaluate(cfg: DictConfig) -> Optional[float]:
    """
    Evaluates a Sudoku model. Supports cross-size evaluation.
    
    Args:
        cfg: Hydra configuration with:
            - checkpoint: path to model checkpoint
            - data_dir: path to dataset
            - batch_size: evaluation batch size
            - device: cpu/cuda/mps/auto
            - eval_split: train/val/test (default: val)
            - per_task: whether to run per-puzzle analysis
            - output_dir: where to save results
    
    Returns:
        Task accuracy as float
    """

    # Initialize evaluator
    evaluator = SudokuEvaluator(
        checkpoint_path=cfg.checkpoint,
        data_dir=cfg.data_dir,
        batch_size=cfg.get("batch_size", 256),
        device=cfg.get("device", "auto"),
        eval_split=cfg.get("eval_split", "val"),
    )

    # Run evaluation
    results = evaluator.evaluate_full(print_examples=True)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    if results.get("cross_size", False):
        print(f"Cross-size: {results['train_grid_size']}x{results['train_grid_size']} → "
              f"{results['eval_grid_size']}x{results['eval_grid_size']}")
    print("=" * 60)
    print(f"Cell Accuracy:   {results['pixel_accuracy']:.4f} ({results['pixel_accuracy'] * 100:.2f}%)")
    print(f"Puzzle Accuracy: {results['task_accuracy']:.4f} ({results['task_accuracy'] * 100:.2f}%)")
    print(f"Validity Rate:   {results['validity_rate']:.4f} ({results['validity_rate'] * 100:.2f}%)")
    print(f"Puzzles Solved:  {results['tasks_correct']}/{results['total_tasks']}")
    print(f"Avg Steps:       {results.get('avg_steps', 0):.1f}")
    print("=" * 60)

    # Run all splits if requested
    if cfg.get("all_splits", False):
        print("\nEvaluating all splits...")
        all_results = evaluator.evaluate_all_splits()
        evaluator.print_summary(all_results)

    # Save results
    output_dir = cfg.get("output_dir") or str(
        Path(cfg.checkpoint).parent / "evaluation_results"
    )
    print(f"\nSaving results to {output_dir}")
    evaluator.save_results(results, output_dir)

    return results["task_accuracy"]


@hydra.main(version_base="1.3", config_path="./configs", config_name="evaluate.yaml")
def main(cfg: DictConfig):
    extras(cfg)
    return evaluate(cfg)


if __name__ == "__main__":
    main()