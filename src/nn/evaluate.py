import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig

from src.nn.evaluator import ARCEvaluator
from src.nn.utils import (
    RankedLogger,
    extras,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

# eval resolver already registered in train.py which we import from
torch.set_float32_matmul_precision("medium")

warnings.filterwarnings("ignore", module="pydantic")


@task_wrapper
def evaluate(cfg: DictConfig) -> Optional[float]:
    """
    Evaluates a model from LakeFS checkpoint on LakeFS dataset.
    Extracts logits and saves results back to LakeFS.

    Args:
        cfg: A DictConfig configuration composed by Hydra.

    Returns:
        None (evaluation outputs are saved to LakeFS)
    """

    # Initialize evaluator
    evaluator = ARCEvaluator(
        checkpoint_path=cfg.checkpoint,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        device=cfg.device,
    )

    # Run evaluation
    results = evaluator.evaluate_full()

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(
        f"Pixel Accuracy: {results['pixel_accuracy']:.4f} ({results['pixel_accuracy'] * 100:.2f}%)"
    )
    print(f"Task Accuracy: {results['task_accuracy']:.4f} ({results['task_accuracy'] * 100:.2f}%)")
    print(f"Tasks Solved: {results['tasks_correct']}/{results['total_tasks']}")
    print("=" * 50)

    # Run per-task evaluation if requested
    if cfg.per_task:
        per_task_results = evaluator.evaluate_per_task()
        print("\nPer-Task Results:")
        print(per_task_results.head(20))

        # Save per-task results
        output_path = Path(cfg.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        per_task_file = output_path / f"per_task_results_{timestamp}.csv"
        per_task_results.to_csv(per_task_file, index=False)
        print(f"\nPer-task results saved to {per_task_file}")

    # Save results
    output_dir = (
        cfg.output_dir
        if cfg.output_dir is not None
        else str(Path(cfg.checkpoint).parent / "evaluation_results")
    )
    print(f"Saving evaluation results to {output_dir}")
    evaluator.save_results(results, output_dir)


@hydra.main(version_base="1.3", config_path="./configs", config_name="evaluate.yaml")
def main(cfg: DictConfig):
    """
    Main entry point for evaluation.

    Args:
        cfg: DictConfig configuration composed by Hydra.
    """
    # Apply extra utilities
    extras(cfg)

    # Evaluate the model
    return evaluate(cfg)


if __name__ == "__main__":
    main()
