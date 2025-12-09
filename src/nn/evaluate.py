"""
Sudoku evaluation script.

Usage:
    python -m src.nn.evaluate_sudoku checkpoint=/path/to/model.ckpt data_dir=/path/to/data
"""
import warnings
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
    Evaluates a Sudoku model.

    Args:
        cfg: Hydra configuration with:
            - checkpoint: path to model checkpoint
            - data_dir: path to dataset
            - batch_size: evaluation batch size (default: 256)
            - device: cpu/cuda/mps/auto (default: auto)
            - num_workers: dataloader workers (default: 0)
            - eval_split: train/val/test (default: val)
            - all_splits: evaluate all splits (default: False)
            - output_dir: where to save results (default: checkpoint_dir/evaluation_results)

    Returns:
        Task accuracy as float
    """
    # Initialize evaluator
    evaluator = SudokuEvaluator(
        checkpoint_path=cfg.checkpoint,
        data_dir=cfg.data_dir,
        batch_size=cfg.get("batch_size", 256),
        device=cfg.get("device", "auto"),
        num_workers=cfg.get("num_workers", 0),
        eval_split=cfg.get("eval_split", "val"),
    )

    # Run evaluation
    results = evaluator.evaluate_full(print_examples=True)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    if evaluator.cross_size:
        print(f"Cross-size: train={evaluator.train_grid_size}x{evaluator.train_grid_size}, "
              f"eval={evaluator.eval_grid_size}x{evaluator.eval_grid_size}")
    else:
        print(f"Grid size: {evaluator.grid_size}x{evaluator.grid_size}")
    print(f"Split: {cfg.get('eval_split', 'val')}")
    print("=" * 60)
    print(f"Cell Accuracy:   {results['pixel_accuracy']:.4f} ({results['pixel_accuracy'] * 100:.2f}%)")
    print(f"Puzzle Accuracy: {results['task_accuracy']:.4f} ({results['task_accuracy'] * 100:.2f}%)")
    print(f"Validity Rate:   {results['validity_rate']:.4f} ({results['validity_rate'] * 100:.2f}%)")
    print(f"Puzzles Solved:  {results['tasks_correct']}/{results['total_tasks']}")
    print(f"Valid Solutions: {results['valid_solutions']}/{results['total_tasks']}")
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