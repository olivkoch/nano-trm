"""
Evaluation script for Sudoku and Maze models.

Usage:
    # Sudoku evaluation (default)
    python -m src.nn.evaluate checkpoint=/path/to/model.ckpt data_dir=/path/to/data

    # Maze evaluation
    python -m src.nn.evaluate checkpoint=/path/to/model.ckpt data_dir=/path/to/data task=maze

    # Visualization
    python -m src.nn.evaluate checkpoint=/path/to/model.ckpt data_dir=/path/to/data visualize=true save_gif=true
"""
import warnings
from pathlib import Path
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig

from src.nn.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("medium")
warnings.filterwarnings("ignore", module="pydantic")


def get_evaluator(task: str, **kwargs):
    """Get the appropriate evaluator class based on task type."""
    if task == "sudoku":
        from src.nn.sudoku_evaluator import SudokuEvaluator
        return SudokuEvaluator(**kwargs)
    elif task == "maze":
        from src.nn.maze_evaluator import MazeEvaluator
        return MazeEvaluator(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}. Supported tasks: sudoku, maze")


@task_wrapper
def evaluate(cfg: DictConfig) -> Optional[float]:
    """
    Evaluates a Sudoku or Maze model.

    Args:
        cfg: Hydra configuration with:
            - checkpoint: path to model checkpoint
            - data_dir: path to dataset
            - task: "sudoku" or "maze" (default: sudoku)
            - batch_size: evaluation batch size (default: 256 for sudoku, 64 for maze)
            - device: cpu/cuda/mps/auto (default: auto)
            - num_workers: dataloader workers (default: 0)
            - eval_split: train/val/test (default: val)
            - all_splits: evaluate all splits (default: False)
            - output_dir: where to save results (default: checkpoint_dir/evaluation_results)
            - visualize: visualize TRM thinking on sample(s) (default: False)
            - visualize_samples: number of samples to visualize (default: 1)
            - grid_size: grid size for maze (default: 30, only used for maze task)

    Returns:
        Task accuracy as float
    """
    # Determine task type
    task = cfg.get("task", "sudoku").lower()
    
    # Set task-specific defaults
    if task == "maze":
        default_batch_size = 64
        default_gif_size = 600
        default_gif_duration = 500
        gif_prefix = "maze_thinking"
    else:
        default_batch_size = 256
        default_gif_size = 400
        default_gif_duration = 1000
        gif_prefix = "thinking"
    
    # Build evaluator kwargs
    evaluator_kwargs = {
        "checkpoint_path": cfg.checkpoint,
        "data_dir": cfg.data_dir,
        "batch_size": cfg.get("batch_size", default_batch_size),
        "device": cfg.get("device", "auto"),
        "num_workers": cfg.get("num_workers", 0),
        "eval_split": cfg.get("eval_split", "val"),
    }
    
    # Add maze-specific kwargs
    if task == "maze":
        evaluator_kwargs["grid_size"] = cfg.get("grid_size", 30)
    
    # Initialize evaluator
    print(f"\nTask: {task.upper()}")
    evaluator = get_evaluator(task, **evaluator_kwargs)

    # Visualization mode
    if cfg.get("visualize", False):
        num_samples = cfg.get("visualize_samples", 1)
        split = cfg.get("eval_split", "val")
        min_steps = cfg.get("min_steps", None)
        
        # Handle boolean parsing (Hydra might pass strings)
        save_gif = cfg.get("save_gif", False)
        if isinstance(save_gif, str):
            save_gif = save_gif.lower() in ("true", "1", "yes")
        
        save_pngs = cfg.get("save_pngs", True)
        if isinstance(save_pngs, str):
            save_pngs = save_pngs.lower() in ("true", "1", "yes")
        
        gif_size = cfg.get("gif_size", default_gif_size)
        gif_duration = cfg.get("gif_duration", default_gif_duration)
        
        print(f"\nVisualizing TRM thinking on {num_samples} sample(s) from {split} split...")
        if min_steps:
            print(f"Filtering for samples that require at least {min_steps} steps to solve")
        if save_gif:
            print(f"Will save GIF(s) with size={gif_size}px, duration={gif_duration}ms per frame")
            if save_pngs:
                print(f"Will also save individual PNG frames")
        
        for i in range(num_samples):
            sample_idx = cfg.get("visualize_start_idx", 0) + i
            print(f"\n{'#' * 80}")
            print(f"# SAMPLE {i + 1} (starting search from index {sample_idx})")
            print(f"{'#' * 80}")
            
            # Generate unique GIF path for each sample
            gif_path = cfg.get("gif_path", None)
            if save_gif and gif_path is None:
                gif_path = f"{gif_prefix}_sample_{i + 1}.gif"
            elif save_gif and num_samples > 1 and gif_path is not None:
                base, ext = gif_path.rsplit('.', 1) if '.' in gif_path else (gif_path, 'gif')
                gif_path = f"{base}_{i + 1}.{ext}"
            
            evaluator.visualize_sample(
                split=split, 
                sample_idx=sample_idx,
                min_steps=min_steps,
                save_gif=save_gif,
                gif_path=gif_path,
                gif_size=gif_size,
                gif_duration=gif_duration,
                save_pngs=save_pngs,
            )
        
        return None

    # Run evaluation
    if task == "sudoku":
        results = evaluator.evaluate_full(print_examples=True)
        accuracy_key = "task_accuracy"
        cell_accuracy_key = "pixel_accuracy"
        solved_key = "tasks_correct"
        valid_key = "valid_solutions"
        total_key = "total_tasks"
    else:
        results = evaluator.evaluate(print_examples=True)
        accuracy_key = "maze_accuracy"
        cell_accuracy_key = "cell_accuracy"
        solved_key = "mazes_correct"
        valid_key = "valid_mazes"
        total_key = "total_mazes"

    # Print results
    print("\n" + "=" * 60)
    print(f"{task.upper()} EVALUATION RESULTS")
    
    if task == "sudoku" and evaluator.cross_size:
        print(f"Cross-size: train={evaluator.train_grid_size}x{evaluator.train_grid_size}, "
              f"eval={evaluator.eval_grid_size}x{evaluator.eval_grid_size}")
    else:
        print(f"Grid size: {evaluator.grid_size}x{evaluator.grid_size}")
    
    print(f"Split: {cfg.get('eval_split', 'val')}")
    print("=" * 60)
    print(f"Cell Accuracy:   {results[cell_accuracy_key]:.4f} ({results[cell_accuracy_key] * 100:.2f}%)")
    print(f"Task Accuracy:   {results[accuracy_key]:.4f} ({results[accuracy_key] * 100:.2f}%)")
    print(f"Validity Rate:   {results['validity_rate']:.4f} ({results['validity_rate'] * 100:.2f}%)")
    print(f"Solved:          {results[solved_key]}/{results[total_key]}")
    print(f"Valid Solutions: {results[valid_key]}/{results[total_key]}")
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

    return results[accuracy_key]


@hydra.main(version_base="1.3", config_path="./configs", config_name="evaluate.yaml")
def main(cfg: DictConfig):
    extras(cfg)
    return evaluate(cfg)


if __name__ == "__main__":
    main()