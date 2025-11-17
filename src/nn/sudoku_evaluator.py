"""
Evaluation script for Sudoku 4x4 models
Uses the existing SudokuDataModule for data loading and processing
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from tqdm import tqdm

# Import your modules
from src.nn.data.sudoku_datamodule import SudokuDataModule
from src.nn.models.trm_module import TRMModule


class SudokuEvaluator:
    """Evaluator for Sudoku models using SudokuDataModule."""

    def __init__(
        self,
        checkpoint_path: str,
        data_dir: str,
        batch_size: int = 256,
        device: str = "auto",
        num_workers: int = 0,
    ):
        """
        Initialize evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            data_dir: Path to dataset directory (e.g., data/sudoku_4x4_small)
            batch_size: Batch size for evaluation
            device: Device to use (auto, cpu, mps, cuda)
            num_workers: Number of workers for dataloader
        """
        self.checkpoint_path = Path(checkpoint_path)
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

        # Create data module (loading mode only)
        self.datamodule = SudokuDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Setup datasets
        self.datamodule.setup()

        # Get grid parameters from datamodule
        self.grid_size = self.datamodule.grid_size
        self.max_grid_size = self.datamodule.max_grid_size
        self.vocab_size = self.datamodule.vocab_size

        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        print(f"Max grid size: {self.max_grid_size}x{self.max_grid_size}")
        print(f"Vocab size: {self.vocab_size}")

    def load_model(self):
        """Load TRM model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}")

        model = TRMModule.load_from_checkpoint(self.checkpoint_path, map_location=self.device)

        model = model.to(self.device)
        model.eval()

        print(f"Model loaded: {model.__class__.__name__}")

        # Print model configuration
        if hasattr(model, "hparams"):
            print("Model configuration:")
            for key in ["hidden_dim", "N_L", "N_H", "N_supervision"]:
                if key in model.hparams:
                    print(f"  - {key}: {model.hparams[key]}")

        return model

    def check_sudoku_validity(self, grid: torch.Tensor) -> bool:
        """Check if a Sudoku solution is valid."""
        n = self.grid_size

        # Check if all values are in range [1, n]
        if not torch.all((grid >= 1) & (grid <= n)):
            return False

        # Check rows
        for row in grid:
            if len(torch.unique(row)) != n:
                return False

        # Check columns
        for col in grid.T:
            if len(torch.unique(col)) != n:
                return False

        # Check boxes
        if n == 4:
            box_rows, box_cols = 2, 2
        elif n == 6:
            box_rows, box_cols = 2, 3
        elif n == 9:
            box_rows, box_cols = 3, 3
        else:
            return False

        for br in range(0, n, box_rows):
            for bc in range(0, n, box_cols):
                box = grid[br : br + box_rows, bc : bc + box_cols]
                if len(torch.unique(box)) != n:
                    return False

        return True

    def evaluate_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Evaluate a single batch."""
        # Move batch to device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

        inputs = batch["input"]
        targets = batch["output"]
        batch_size = len(inputs)

        with torch.no_grad():
            # Initialize carry for this batch
            carry = self.model.initial_carry(batch)

            # Run forward passes until all sequences halt
            all_outputs = []
            all_halted = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            while not all_halted.all():
                # Forward pass through TRM with carry
                carry, outputs = self.model.forward(carry, batch)

                # Store outputs for sequences that just halted
                newly_halted = carry.halted & ~all_halted
                if newly_halted.any():
                    all_outputs.append((newly_halted, outputs["logits"]))

                all_halted = all_halted | carry.halted

                # Safety check to avoid infinite loops
                if carry.steps.max() > self.model.hparams.N_supervision_val:
                    break

            # Combine outputs from all halted sequences
            final_logits = torch.zeros_like(all_outputs[0][1]) if all_outputs else None
            for halted_mask, logits in all_outputs:
                final_logits = torch.where(
                    halted_mask.unsqueeze(-1).unsqueeze(-1), logits, final_logits
                )

            # Get predictions
            predictions = final_logits.argmax(dim=-1)

            # Calculate metrics
            # Mask for cells to predict (encoded empty cells = 2)
            mask = (inputs == 2).float()

            # Cell-level accuracy
            correct_cells = (predictions == targets).float()

            if mask.sum() > 0:
                cell_accuracy = (correct_cells * mask).sum() / mask.sum()
            else:
                cell_accuracy = torch.tensor(1.0)

            # Puzzle-level metrics
            puzzle_correct = []
            valid_puzzles = []

            for i in range(batch_size):
                # Reshape to grid
                pred_flat = predictions[i]
                target_flat = targets[i]

                pred_grid = pred_flat.reshape(self.max_grid_size, self.max_grid_size)
                target_grid = target_flat.reshape(self.max_grid_size, self.max_grid_size)

                # Extract actual Sudoku grid
                pred_sudoku = pred_grid[: self.grid_size, : self.grid_size]
                target_sudoku = target_grid[: self.grid_size, : self.grid_size]

                # Check exact match
                exact_match = torch.all(pred_sudoku == target_sudoku).item()
                puzzle_correct.append(exact_match)

                # Decode predictions to check validity
                pred_decoded = pred_sudoku.clone()
                # Decode: values >= 3 are actual Sudoku values (shifted by 2)
                valid_mask = pred_sudoku >= 3
                pred_decoded[valid_mask] = pred_sudoku[valid_mask] - 2
                pred_decoded[~valid_mask] = 0  # Invalid predictions

                # Check if valid Sudoku
                is_valid = self.check_sudoku_validity(pred_decoded)
                valid_puzzles.append(is_valid)

        return {
            "cell_accuracy": cell_accuracy.item(),
            "puzzle_correct": puzzle_correct,
            "valid_puzzles": valid_puzzles,
            "batch_size": batch_size,
        }

    def evaluate(self, split: str = "val") -> Dict[str, Any]:
        """
        Run evaluation on specified split.

        Args:
            split: Which split to evaluate ('train', 'val', or 'test')
        """
        print(f"\nEvaluating on {split} split...")

        # Get appropriate dataloader
        if split == "train":
            dataloader = self.datamodule.train_dataloader()
        elif split == "val":
            dataloader = self.datamodule.val_dataloader()
        elif split == "test":
            dataloader = self.datamodule.test_dataloader()
        else:
            raise ValueError(f"Invalid split: {split}")

        total_cell_correct = 0
        total_cells = 0
        total_puzzles_correct = 0
        total_valid_puzzles = 0
        total_puzzles = 0

        all_results = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split}")):
            results = self.evaluate_batch(batch)

            # Update totals
            batch_size = results["batch_size"]
            total_cell_correct += results["cell_accuracy"] * batch_size
            total_cells += batch_size
            total_puzzles_correct += sum(results["puzzle_correct"])
            total_valid_puzzles += sum(results["valid_puzzles"])
            total_puzzles += len(results["puzzle_correct"])

            # Store detailed results
            for i in range(batch_size):
                all_results.append(
                    {
                        "batch_idx": batch_idx,
                        "sample_idx": i,
                        "exact_match": results["puzzle_correct"][i],
                        "valid_sudoku": results["valid_puzzles"][i],
                    }
                )

        # Calculate final metrics
        overall_cell_accuracy = total_cell_correct / total_cells if total_cells > 0 else 0
        overall_puzzle_accuracy = total_puzzles_correct / total_puzzles if total_puzzles > 0 else 0
        overall_validity_rate = total_valid_puzzles / total_puzzles if total_puzzles > 0 else 0

        return {
            "split": split,
            "cell_accuracy": overall_cell_accuracy,
            "puzzle_accuracy": overall_puzzle_accuracy,
            "validity_rate": overall_validity_rate,
            "puzzles_correct": total_puzzles_correct,
            "valid_puzzles": total_valid_puzzles,
            "total_puzzles": total_puzzles,
            "detailed_results": all_results,
        }

    def evaluate_full(self, split: str = "val") -> Dict[str, Any]:
        """
        Compatibility method that returns results with ARC-style naming.
        This is for compatibility with existing evaluation scripts.
        """
        results = self.evaluate(split)

        # Map Sudoku metrics to ARC-style naming for compatibility
        return {
            "pixel_accuracy": results["cell_accuracy"],  # cell -> pixel
            "task_accuracy": results["puzzle_accuracy"],  # puzzle -> task
            "validity_rate": results["validity_rate"],
            "tasks_correct": results["puzzles_correct"],  # puzzles -> tasks
            "valid_solutions": results["valid_puzzles"],
            "total_tasks": results["total_puzzles"],  # puzzles -> tasks
            "detailed_results": results.get("detailed_results", []),
        }

    def evaluate_all_splits(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate on all available splits."""
        results = {}

        for split in ["train", "val", "test"]:
            try:
                results[split] = self.evaluate(split)
            except Exception as e:
                print(f"Could not evaluate {split} split: {e}")

        return results

    def analyze_difficulty(self, split: str = "val") -> pd.DataFrame:
        """
        Analyze performance by difficulty (number of given cells).
        """
        print(f"\nAnalyzing difficulty for {split} split...")

        # Get dataset
        if split == "train":
            dataset = self.datamodule.train_dataset
        elif split == "val":
            dataset = self.datamodule.val_dataset
        elif split == "test":
            dataset = self.datamodule.test_dataset
        else:
            raise ValueError(f"Invalid split: {split}")

        # Count givens from encoded inputs
        results_list = []

        for idx in range(len(dataset)):
            sample = dataset[idx]
            input_flat = sample["input"]

            # Reshape and extract grid
            input_grid = input_flat.reshape(self.max_grid_size, self.max_grid_size)
            sudoku_grid = input_grid[: self.grid_size, : self.grid_size]

            # Count given cells (values >= 3 in encoded format)
            num_given = torch.sum(sudoku_grid >= 3).item()
            num_empty = self.grid_size * self.grid_size - num_given

            results_list.append(
                {
                    "idx": idx,
                    "num_given": num_given,
                    "num_empty": num_empty,
                }
            )

        df = pd.DataFrame(results_list)

        # Add difficulty categories
        if self.grid_size == 4:
            bins = [0, 4, 8, 12, 16]
        elif self.grid_size == 6:
            bins = [0, 6, 12, 18, 24, 36]
        else:  # 9x9
            bins = [0, 20, 40, 60, 81]

        df["difficulty"] = pd.cut(
            df["num_empty"],
            bins=bins,
            labels=["Easy", "Medium", "Hard", "Very Hard"][: len(bins) - 1],
        )

        return df

    def save_results(self, results: Dict[str, Any], output_dir: str = "evaluation_results"):
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary
        summary_file = output_path / f"sudoku_eval_{timestamp}.json"

        # Create summary based on the results format
        summary = {
            "checkpoint": str(self.checkpoint_path),
            "timestamp": timestamp,
            "grid_size": self.grid_size,
        }

        # Check if results are from evaluate_full (ARC-style) or evaluate (Sudoku-style)
        if "pixel_accuracy" in results:
            # ARC-style naming from evaluate_full
            summary["results"] = {
                "pixel_accuracy": results["pixel_accuracy"],
                "task_accuracy": results["task_accuracy"],
                "validity_rate": results.get("validity_rate", 0),
                "tasks_correct": results.get("tasks_correct", 0),
                "total_tasks": results.get("total_tasks", 0),
            }
        elif "cell_accuracy" in results:
            # Sudoku-style naming from evaluate
            summary["results"] = {
                "cell_accuracy": results["cell_accuracy"],
                "puzzle_accuracy": results["puzzle_accuracy"],
                "validity_rate": results.get("validity_rate", 0),
                "puzzles_correct": results.get("puzzles_correct", 0),
                "total_puzzles": results.get("total_puzzles", 0),
            }
        else:
            # Multi-split results
            summary["results"] = {}
            for split, split_results in results.items():
                if isinstance(split_results, dict):
                    summary["results"][split] = {
                        "cell_accuracy": split_results.get("cell_accuracy", 0),
                        "puzzle_accuracy": split_results.get("puzzle_accuracy", 0),
                        "validity_rate": split_results.get("validity_rate", 0),
                        "puzzles_correct": split_results.get("puzzles_correct", 0),
                        "total_puzzles": split_results.get("total_puzzles", 0),
                    }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {summary_file}")

        # Save detailed results if available
        if "detailed_results" in results and results["detailed_results"]:
            detailed_file = output_path / f"sudoku_eval_detailed_{timestamp}.csv"
            df = pd.DataFrame(results["detailed_results"])
            df.to_csv(detailed_file, index=False)
            print(f"Detailed results saved to {detailed_file}")

        return summary_file

    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("SUDOKU EVALUATION RESULTS")
        print("=" * 60)

        if "split" in results:
            # Single split
            self._print_split_results(results["split"], results)
        else:
            # Multiple splits
            for split, split_results in results.items():
                self._print_split_results(split, split_results)
                print("-" * 60)

    def _print_split_results(self, split: str, results: Dict[str, Any]):
        """Print results for a single split."""
        print(f"\n{split.upper()} Split:")
        print(f"  Total puzzles: {results['total_puzzles']}")
        print(f"  Cell accuracy: {results['cell_accuracy']:.2%}")
        print(f"  Exact match accuracy: {results['puzzle_accuracy']:.2%}")
        print(f"  Valid Sudoku rate: {results['validity_rate']:.2%}")
        print(f"  Puzzles solved: {results['puzzles_correct']}/{results['total_puzzles']}")
        print(f"  Valid solutions: {results['valid_puzzles']}/{results['total_puzzles']}")
