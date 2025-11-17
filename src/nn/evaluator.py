"""
Evaluation script for ARC models
Tests trained models on the evaluation dataset
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your modules (adjust paths as needed)
from src.nn.data.arc_datamodule import ARCTaskDataset, TRMTransform, collate_fn_with_puzzles
from src.nn.models.mlp_module import ConvMLPModule
from src.nn.models.trm_module import TRMModule


class ARCEvaluator:
    """Evaluator for ARC models on the evaluation dataset."""

    def __init__(
        self,
        checkpoint_path: str,
        data_dir: str = "data",
        batch_size: int = 32,
        device: str = "auto",
    ):
        """
        Initialize evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            data_dir: Directory containing ARC data files
            batch_size: Batch size for evaluation
            device: Device to use (auto, cpu, mps, cuda)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size

        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load model
        self.model = self.load_model()

        # Load evaluation data
        self.eval_challenges, self.eval_solutions = self.load_evaluation_data()

    def load_model(self):
        """Load model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path} and device={self.device}")

        # Try to infer model type from checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # Check hyperparameters to determine model type
        if "hyper_parameters" in checkpoint:
            hparams = checkpoint["hyper_parameters"]

            # Determine which model class to use based on hyperparameters
            if "N_supervision" in hparams:
                # TRM model
                model = TRMModule.load_from_checkpoint(
                    self.checkpoint_path, map_location=self.device
                )
            elif "hidden_channels" in hparams:
                # ConvMLP model
                model = ConvMLPModule.load_from_checkpoint(
                    self.checkpoint_path, map_location=self.device
                )
        else:
            raise ValueError("Cannot determine model type from checkpoint")

        model = model.to(self.device)
        model.eval()

        print(f"Model loaded: {model.__class__.__name__}")
        return model

    def load_evaluation_data(self) -> tuple[Dict, Dict]:
        """Load evaluation challenges and solutions."""
        eval_challenges_path = self.data_dir / "arc-agi_evaluation_challenges.json"
        eval_solutions_path = self.data_dir / "arc-agi_evaluation_solutions.json"

        print(f"Loading evaluation data from {self.data_dir}")

        with open(eval_challenges_path) as f:
            challenges = json.load(f)

        with open(eval_solutions_path) as f:
            solutions = json.load(f)

        print(f"Loaded {len(challenges)} evaluation tasks")
        return challenges, solutions

    def create_dataloader(self) -> DataLoader:
        """Create dataloader for evaluation dataset."""
        # Create dataset
        transform = TRMTransform(
            max_grid_size=30,
            augment=False,  # No augmentation for evaluation
        )

        dataset = ARCTaskDataset(
            self.eval_challenges, self.eval_solutions, transform=transform, samples_per_task=1
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Use 0 for evaluation to avoid issues
            pin_memory=True if self.device.type == "cuda" else False,
            collate_fn=collate_fn_with_puzzles,
        )

        return dataloader

    def evaluate_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Evaluate a single batch."""
        inputs = batch["input"].to(self.device)
        targets = batch["output"].to(self.device)

        with torch.no_grad():
            # Forward pass
            outputs = self.model(inputs)

            # Get predictions
            predictions = outputs.argmax(dim=-1)

            # Calculate metrics
            correct = (predictions == targets).float()

            # Mask out padding (0 values)
            mask = (targets != 0).float()

            # Pixel-level accuracy
            if mask.sum() > 0:
                pixel_accuracy = (correct * mask).sum() / mask.sum()
            else:
                pixel_accuracy = correct.mean()

            # Task-level accuracy (all pixels correct)
            task_correct = []
            for i in range(len(inputs)):
                task_mask = mask[i]
                if task_mask.sum() > 0:
                    task_acc = ((correct[i] * task_mask).sum() / task_mask.sum()).item()
                    task_correct.append(task_acc == 1.0)
                else:
                    task_correct.append(False)

        return {
            "pixel_accuracy": pixel_accuracy.item(),
            "task_correct": task_correct,
            "batch_size": len(inputs),
        }

    def evaluate_full(self) -> Dict[str, Any]:
        """Run full evaluation on the dataset."""
        print("\nRunning evaluation...")

        dataloader = self.create_dataloader()

        total_pixel_correct = 0
        total_pixels = 0
        total_tasks_correct = 0
        total_tasks = 0

        all_results = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            results = self.evaluate_batch(batch)

            # Update totals
            batch_size = results["batch_size"]
            total_pixel_correct += results["pixel_accuracy"] * batch_size
            total_pixels += batch_size
            total_tasks_correct += sum(results["task_correct"])
            total_tasks += len(results["task_correct"])

            # Store detailed results if task IDs available
            if "task_ids" in batch:
                for i, task_id in enumerate(batch["task_ids"]):
                    all_results.append(
                        {
                            "task_id": task_id,
                            "correct": results["task_correct"][i],
                            "pixel_accuracy": results["pixel_accuracy"],
                        }
                    )

        # Calculate final metrics
        overall_pixel_accuracy = total_pixel_correct / total_pixels if total_pixels > 0 else 0
        overall_task_accuracy = total_tasks_correct / total_tasks if total_tasks > 0 else 0

        return {
            "pixel_accuracy": overall_pixel_accuracy,
            "task_accuracy": overall_task_accuracy,
            "tasks_correct": total_tasks_correct,
            "total_tasks": total_tasks,
            "detailed_results": all_results,
        }

    def evaluate_per_task(self) -> pd.DataFrame:
        """Evaluate each task individually for detailed analysis."""
        print("\nEvaluating individual tasks...")

        results_list = []

        for task_id, task_data in tqdm(self.eval_challenges.items(), desc="Tasks"):
            # Create single-task dataset
            single_task = {task_id: task_data}
            single_solution = {task_id: self.eval_solutions[task_id]}

            transform = TRMTransform(max_grid_size=30, augment=False)
            dataset = ARCTaskDataset(
                single_task, single_solution, transform=transform, samples_per_task=1
            )

            dataloader = DataLoader(
                dataset,
                batch_size=len(dataset),  # Process all examples at once
                shuffle=False,
                collate_fn=collate_fn,
            )

            # Evaluate
            for batch in dataloader:
                results = self.evaluate_batch(batch)

                # Store results
                results_list.append(
                    {
                        "task_id": task_id,
                        "num_examples": len(dataset),
                        "pixel_accuracy": results["pixel_accuracy"],
                        "all_correct": all(results["task_correct"]),
                        "num_correct": sum(results["task_correct"]),
                    }
                )

        return pd.DataFrame(results_list)

    def save_results(self, results: Dict[str, Any], output_dir: str = "evaluation_results"):
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary
        summary_file = output_path / f"eval_summary_{timestamp}.json"
        summary = {
            "checkpoint": str(self.checkpoint_path),
            "pixel_accuracy": results["pixel_accuracy"],
            "task_accuracy": results["task_accuracy"],
            "tasks_correct": results["tasks_correct"],
            "total_tasks": results["total_tasks"],
            "timestamp": timestamp,
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {summary_file}")

        # Save detailed results if available
        if "detailed_results" in results and results["detailed_results"]:
            detailed_file = output_path / f"eval_detailed_{timestamp}.json"
            with open(detailed_file, "w") as f:
                json.dump(results["detailed_results"], f, indent=2)
            print(f"Detailed results saved to {detailed_file}")

        return summary_file
