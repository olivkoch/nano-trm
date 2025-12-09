"""
Evaluation script for Sudoku models
Supports standard and cross-size evaluation (train on 6x6, test on 9x9)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from tqdm import tqdm

from src.nn.data.sudoku_datamodule import SudokuDataModule
from src.nn.models.trm import TRMModule


class SudokuEvaluator:
    """Evaluator for Sudoku models using SudokuDataModule."""

    def __init__(
        self,
        checkpoint_path: str,
        data_dir: str,
        batch_size: int = 256,
        device: str = "auto",
        num_workers: int = 0,
        eval_split: str = "val",
    ):
        """
        Initialize evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            data_dir: Path to dataset directory
            batch_size: Batch size for evaluation
            device: Device to use (auto, cpu, mps, cuda)
            num_workers: Number of workers for dataloader
            eval_split: Which split to evaluate ('train', 'val', 'test')
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size
        self.eval_split = eval_split

        if data_dir is None:
            raise ValueError("data_dir is required for evaluation. Cannot generate puzzles on-the-fly.")
    
        if not Path(data_dir).exists():
            raise ValueError(f"data_dir does not exist: {data_dir}")

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

        # Load model first to get its configuration
        self.model = self.load_model()

        # Create data module (loading mode)
        self.datamodule = SudokuDataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Setup datasets
        self.datamodule.setup()

        # Get cross-size info from datamodule (now always set)
        self.cross_size = self.datamodule.cross_size
        self.train_grid_size = self.datamodule.train_grid_size
        self.eval_grid_size = self.datamodule.eval_grid_size
        self.grid_size = self.eval_grid_size  # Use eval grid size for evaluation
        self.max_grid_size = self.datamodule.max_grid_size
        self.vocab_size = self.datamodule.vocab_size

        if self.cross_size:
            print(f"Cross-size mode: trained on {self.train_grid_size}x{self.train_grid_size}, "
                  f"evaluating on {self.eval_grid_size}x{self.eval_grid_size}")
        else:
            print(f"Grid size: {self.grid_size}x{self.grid_size}")
            
        print(f"Max grid size: {self.max_grid_size}x{self.max_grid_size}")
        print(f"Vocab size: {self.vocab_size}")
        
        # Verify model compatibility
        self._verify_model_compatibility()

    def _verify_model_compatibility(self):
        """Check that model config matches data config."""
        model_vocab = self.model.hparams.get('vocab_size', None)
        model_seq_len = self.model.hparams.get('seq_len', None)
        
        data_seq_len = self.max_grid_size * self.max_grid_size
        
        if model_vocab is not None and model_vocab != self.vocab_size:
            print(f"⚠ Warning: Model vocab_size ({model_vocab}) != data vocab_size ({self.vocab_size})")
            
        if model_seq_len is not None and model_seq_len != data_seq_len:
            print(f"⚠ Warning: Model seq_len ({model_seq_len}) != data seq_len ({data_seq_len})")

    def load_model(self):
        """Load TRM model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}")

        model = TRMModule.load_from_checkpoint(
            self.checkpoint_path, 
            map_location=self.device
        )

        model = model.to(self.device)
        model.eval()

        print(f"Model loaded: {model.__class__.__name__}")

        if hasattr(model, "hparams"):
            print("Model configuration:")
            for key in ["hidden_size", "num_layers", "H_cycles", "L_cycles", 
                       "N_supervision", "vocab_size", "seq_len"]:
                if key in model.hparams:
                    print(f"  {key}: {model.hparams[key]}")

        return model

    def _get_box_dims(self, grid_size: int) -> tuple:
        """Get box dimensions for a grid size."""
        if grid_size == 4:
            return 2, 2
        elif grid_size == 6:
            return 2, 3
        elif grid_size == 9:
            return 3, 3
        else:
            raise ValueError(f"Unsupported grid_size: {grid_size}")

    def check_sudoku_validity(self, grid: torch.Tensor, grid_size: int = None) -> bool:
        """Check if a Sudoku solution is valid."""
        if grid_size is None:
            grid_size = self.grid_size
            
        n = grid_size

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
        box_rows, box_cols = self._get_box_dims(n)

        for br in range(0, n, box_rows):
            for bc in range(0, n, box_cols):
                box = grid[br : br + box_rows, bc : bc + box_cols]
                if len(torch.unique(box)) != n:
                    return False

        return True

    def visualize_thinking(
        self, 
        batch: Dict[str, torch.Tensor], 
        sample_idx: int = 0,
        max_steps: int = None,
        show_confidence: bool = True,
    ) -> Dict[str, Any]:
        """
        Visualize the TRM's thinking process for a single sample.
        
        Shows how z_H (decoded to predictions) evolves at each iteration.
        
        Args:
            batch: Batch of samples
            sample_idx: Which sample in the batch to visualize
            max_steps: Maximum steps to run (default: N_supervision_val)
            show_confidence: Whether to show confidence values
            
        Returns:
            Dictionary with step-by-step predictions and metadata
        """
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()
        }
        
        inputs = batch["input"]
        targets = batch["output"]
        batch_size = len(inputs)
        
        if sample_idx >= batch_size:
            raise ValueError(f"sample_idx {sample_idx} >= batch_size {batch_size}")
        
        if max_steps is None:
            max_steps = self.model.hparams.N_supervision_val
        
        # Get puzzle embedding length
        puzzle_emb_len = getattr(self.model, 'puzzle_emb_len', 0)
        
        # Extract single sample info
        inp = inputs[sample_idx].reshape(self.max_grid_size, self.max_grid_size)
        tgt = targets[sample_idx].reshape(self.max_grid_size, self.max_grid_size)
        inp_grid = inp[:self.grid_size, :self.grid_size]
        tgt_grid = tgt[:self.grid_size, :self.grid_size]
        
        # Track predictions at each step
        step_predictions = []
        step_confidences = []
        step_q_halt = []
        step_halted = []
        
        with torch.no_grad():
            carry = self.model.initial_carry(batch)
            
            for step in range(max_steps):
                carry, outputs = self.model.forward(carry, batch)
                
                logits = outputs["logits"]  # [batch, seq_len, vocab]
                q_halt = outputs["q_halt_logits"]  # [batch]
                
                # Get predictions and confidence for this sample
                sample_logits = logits[sample_idx]  # [seq_len, vocab]
                probs = torch.softmax(sample_logits, dim=-1)
                confidence, preds = probs.max(dim=-1)  # [seq_len]
                
                # Reshape to grid
                pred_grid = preds.reshape(self.max_grid_size, self.max_grid_size)
                conf_grid = confidence.reshape(self.max_grid_size, self.max_grid_size)
                
                # Extract actual puzzle region
                pred_sudoku = pred_grid[:self.grid_size, :self.grid_size].clone()
                conf_sudoku = conf_grid[:self.grid_size, :self.grid_size].clone()
                
                step_predictions.append(pred_sudoku.cpu())
                step_confidences.append(conf_sudoku.cpu())
                step_q_halt.append(q_halt[sample_idx].item())
                step_halted.append(carry.halted[sample_idx].item())
                
                # Check if this sample halted
                if carry.halted[sample_idx]:
                    break
        
        # Print visualization
        self._print_thinking_visualization(
            inp_grid.cpu(), 
            tgt_grid.cpu(), 
            step_predictions, 
            step_confidences,
            step_q_halt,
            step_halted,
            show_confidence=show_confidence,
        )
        
        return {
            "input": inp_grid.cpu(),
            "target": tgt_grid.cpu(),
            "step_predictions": step_predictions,
            "step_confidences": step_confidences,
            "step_q_halt": step_q_halt,
            "step_halted": step_halted,
            "num_steps": len(step_predictions),
        }
    
    def _print_thinking_visualization(
        self,
        inp_grid: torch.Tensor,
        tgt_grid: torch.Tensor,
        step_predictions: list,
        step_confidences: list,
        step_q_halt: list,
        step_halted: list,
        show_confidence: bool = True,
    ):
        """Print a nice visualization of the thinking process."""
        
        def decode_cell(val, is_input=False):
            val = val.item() if hasattr(val, 'item') else val
            if val == 0:
                return "."  # PAD
            elif val == 1:
                return "X"  # EOS
            elif val == 2:
                return "_" if is_input else "?"  # Empty
            else:
                return str(val - 2)  # Actual value (1-9)
        
        def format_cell(val, conf=None, prev_val=None, target_val=None, is_empty=False):
            """Format a cell with optional markers."""
            cell_str = decode_cell(val)
            
            # Markers: * = changed, ! = wrong
            prefix = " "
            if prev_val is not None and val != prev_val:
                prefix = "*"  # Changed from previous step
            elif target_val is not None and val != target_val and is_empty:
                prefix = "!"  # Wrong vs target (only for cells that need prediction)
            
            return prefix + cell_str
        
        def grid_to_lines(grid, conf_grid=None, prev_grid=None, target=None, input_mask=None):
            """Convert grid to list of formatted lines."""
            lines = []
            box_rows, box_cols = self._get_box_dims(self.grid_size)
            
            for r in range(self.grid_size):
                if r > 0 and r % box_rows == 0:
                    # Horizontal separator
                    sep_parts = []
                    for s in range(self.grid_size // box_cols):
                        sep_parts.append("-" * (box_cols * 2))
                    lines.append("+".join(sep_parts))
                
                row_str = ""
                for c in range(self.grid_size):
                    if c > 0 and c % box_cols == 0:
                        row_str += "|"
                    
                    is_empty = input_mask[r, c].item() if input_mask is not None else False
                    prev_val = prev_grid[r, c] if prev_grid is not None else None
                    target_val = target[r, c] if target is not None else None
                    
                    row_str += format_cell(grid[r, c], None, prev_val, target_val, is_empty)
                
                lines.append(row_str)
            
            return lines
        
        def compute_metrics(pred, target, mask):
            """Compute accuracy metrics."""
            correct = (pred == target) & mask
            total = mask.sum().item()
            if total == 0:
                return 1.0, 0
            return correct.sum().item() / total, (mask & ~correct).sum().item()
        
        # Mask for cells to predict (empty cells in input, encoded as 2)
        input_mask = (inp_grid == 2)
        
        print("\n" + "=" * 80)
        print("TRM THINKING VISUALIZATION")
        print(f"H_cycles={self.model.hparams.H_cycles}, L_cycles={self.model.hparams.L_cycles}")
        print("=" * 80)
        
        # Print input and target side by side
        inp_lines = grid_to_lines(inp_grid)
        tgt_lines = grid_to_lines(tgt_grid)
        
        width = max(len(line) for line in inp_lines) + 4
        
        print(f"\n{'INPUT':<{width}}TARGET")
        print(f"{'-' * (width - 2):<{width}}{'-' * (width - 2)}")
        for inp_line, tgt_line in zip(inp_lines, tgt_lines):
            print(f"{inp_line:<{width}}{tgt_line}")
        
        print(f"\nEmpty cells to fill: {input_mask.sum().item()}")
        
        print("\n" + "-" * 80)
        print("STEP-BY-STEP REASONING (each step = H×L iterations of reasoning blocks)")
        print("-" * 80)
        
        prev_pred = None
        for step, (pred, conf, q_halt, halted) in enumerate(
            zip(step_predictions, step_confidences, step_q_halt, step_halted)
        ):
            # Compute metrics
            acc, errors = compute_metrics(pred, tgt_grid, input_mask)
            avg_conf = conf[input_mask].mean().item() if input_mask.sum() > 0 else 1.0
            min_conf = conf[input_mask].min().item() if input_mask.sum() > 0 else 1.0
            
            # Count changes from previous step
            if prev_pred is not None:
                changes = ((pred != prev_pred) & input_mask).sum().item()
            else:
                changes = "-"
            
            # Q-halt status
            q_halt_str = f"q={q_halt:+.2f}"
            if q_halt > 0:
                q_halt_str += " (HALT)"
            
            halt_marker = " ← STOPPED" if halted else ""
            
            print(f"\n┌─ Step {step + 1} ─────────────────────────────────────────────────────────")
            print(f"│ Accuracy: {acc:.1%} ({errors} errors) | Changes: {changes} | {q_halt_str}{halt_marker}")
            print(f"│ Confidence: avg={avg_conf:.2f}, min={min_conf:.2f}")
            print("└" + "─" * 70)
            
            # Print grid
            pred_lines = grid_to_lines(pred, conf, prev_pred, tgt_grid, input_mask)
            for line in pred_lines:
                print(f"  {line}")
            
            prev_pred = pred
        
        # Final summary
        final_pred = step_predictions[-1]
        final_correct = torch.all(final_pred[input_mask] == tgt_grid[input_mask]).item() if input_mask.sum() > 0 else True
        
        # Decode for validity check
        final_decoded = final_pred.clone()
        valid_mask = final_pred >= 3
        final_decoded[valid_mask] = final_pred[valid_mask] - 2
        final_decoded[~valid_mask] = 0
        is_valid = self.check_sudoku_validity(final_decoded)
        
        print("\n" + "=" * 80)
        status = "✓ SOLVED" if final_correct else "✗ FAILED"
        valid_str = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"RESULT: {status} | {valid_str} Sudoku | {len(step_predictions)} steps")
        print("=" * 80)
        
        # Show final errors if any
        if not final_correct:
            print("\nErrors (row, col): ", end="")
            errors = []
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if input_mask[r, c] and final_pred[r, c] != tgt_grid[r, c]:
                        pred_val = decode_cell(final_pred[r, c])
                        tgt_val = decode_cell(tgt_grid[r, c])
                        errors.append(f"({r},{c}): {pred_val}→{tgt_val}")
            print(", ".join(errors[:10]))
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        
        print("\nLegend: *N = changed this step, !N = wrong prediction")

    def visualize_sample(self, split: str = "val", sample_idx: int = 0, show_confidence: bool = True):
        """
        Convenience method to visualize thinking on a specific sample.
        
        Args:
            split: Which split to use ('train', 'val', 'test')
            sample_idx: Index of sample in the split
            show_confidence: Whether to show confidence values
        """
        if split == "train":
            dataloader = self.datamodule.train_dataloader()
        elif split == "val":
            dataloader = self.datamodule.val_dataloader()
        elif split == "test":
            dataloader = self.datamodule.test_dataloader()
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Find the right batch
        batch_idx = sample_idx // self.batch_size
        within_batch_idx = sample_idx % self.batch_size
        
        for i, batch in enumerate(dataloader):
            if i == batch_idx:
                return self.visualize_thinking(batch, within_batch_idx, show_confidence=show_confidence)
        
        raise ValueError(f"sample_idx {sample_idx} out of range for {split} split")

    def evaluate_batch(self, batch: Dict[str, torch.Tensor], print_examples: bool = False) -> Dict[str, Any]:
        """Evaluate a single batch."""
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()
        }

        inputs = batch["input"]
        targets = batch["output"]
        batch_size = len(inputs)

        with torch.no_grad():
            carry = self.model.initial_carry(batch)

            all_outputs = []
            all_halted = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            while not all_halted.all():
                carry, outputs = self.model.forward(carry, batch)

                newly_halted = carry.halted & ~all_halted
                if newly_halted.any():
                    all_outputs.append((newly_halted, outputs["logits"]))

                all_halted = all_halted | carry.halted

                if carry.steps.max() > self.model.hparams.N_supervision_val:
                    break

            # Combine outputs
            final_logits = torch.zeros_like(all_outputs[0][1]) if all_outputs else None
            for halted_mask, logits in all_outputs:
                final_logits = torch.where(
                    halted_mask.unsqueeze(-1).unsqueeze(-1), logits, final_logits
                )

            predictions = final_logits.argmax(dim=-1)

            if print_examples:
                self.print_examples(batch, predictions, num_examples=5)

            # Mask for cells to predict (encoded empty cells = 2)
            mask = (inputs == 2).float()

            correct_cells = (predictions == targets).float()

            if mask.sum() > 0:
                cell_accuracy = (correct_cells * mask).sum() / mask.sum()
            else:
                cell_accuracy = torch.tensor(1.0)

            puzzle_correct = []
            valid_puzzles = []
            steps_taken = []

            for i in range(batch_size):
                pred_flat = predictions[i]
                target_flat = targets[i]

                pred_grid = pred_flat.reshape(self.max_grid_size, self.max_grid_size)
                target_grid = target_flat.reshape(self.max_grid_size, self.max_grid_size)

                # Extract actual Sudoku grid (use eval grid size)
                pred_sudoku = pred_grid[: self.grid_size, : self.grid_size]
                target_sudoku = target_grid[: self.grid_size, : self.grid_size]

                exact_match = torch.all(pred_sudoku == target_sudoku).item()
                puzzle_correct.append(exact_match)

                # Decode predictions
                pred_decoded = pred_sudoku.clone()
                valid_mask = pred_sudoku >= 3
                pred_decoded[valid_mask] = pred_sudoku[valid_mask] - 2
                pred_decoded[~valid_mask] = 0

                is_valid = self.check_sudoku_validity(pred_decoded)
                valid_puzzles.append(is_valid)
                
                steps_taken.append(carry.steps[i].item())

        return {
            "cell_accuracy": cell_accuracy.item(),
            "puzzle_correct": puzzle_correct,
            "valid_puzzles": valid_puzzles,
            "steps_taken": steps_taken,
            "batch_size": batch_size,
        }

    def evaluate(self, split: str = None, print_examples: bool = False) -> Dict[str, Any]:
        """Run evaluation on specified split."""
        if split is None:
            split = self.eval_split
            
        print(f"\nEvaluating on {split} split...")
        
        if self.cross_size:
            print(f"  (Model trained on {self.train_grid_size}x{self.train_grid_size}, "
                  f"testing on {self.eval_grid_size}x{self.eval_grid_size})")

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
        total_steps = 0

        all_results = []

        first_batch = True

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split}")):
            results = self.evaluate_batch(batch, print_examples=(first_batch and print_examples))
            first_batch = False

            batch_size = results["batch_size"]
            total_cell_correct += results["cell_accuracy"] * batch_size
            total_cells += batch_size
            total_puzzles_correct += sum(results["puzzle_correct"])
            total_valid_puzzles += sum(results["valid_puzzles"])
            total_puzzles += len(results["puzzle_correct"])
            total_steps += sum(results["steps_taken"])

            for i in range(batch_size):
                all_results.append({
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "exact_match": results["puzzle_correct"][i],
                    "valid_sudoku": results["valid_puzzles"][i],
                    "steps": results["steps_taken"][i],
                })

        overall_cell_accuracy = total_cell_correct / total_cells if total_cells > 0 else 0
        overall_puzzle_accuracy = total_puzzles_correct / total_puzzles if total_puzzles > 0 else 0
        overall_validity_rate = total_valid_puzzles / total_puzzles if total_puzzles > 0 else 0
        avg_steps = total_steps / total_puzzles if total_puzzles > 0 else 0

        return {
            "split": split,
            "cell_accuracy": overall_cell_accuracy,
            "puzzle_accuracy": overall_puzzle_accuracy,
            "validity_rate": overall_validity_rate,
            "puzzles_correct": total_puzzles_correct,
            "valid_puzzles": total_valid_puzzles,
            "total_puzzles": total_puzzles,
            "avg_steps": avg_steps,
            "cross_size": self.cross_size,
            "train_grid_size": self.train_grid_size,
            "eval_grid_size": self.eval_grid_size,
            "detailed_results": all_results,
        }

    def evaluate_full(self, split: str = None, print_examples: bool = False) -> Dict[str, Any]:
        """Evaluate with backward-compatible naming."""
        results = self.evaluate(split, print_examples=print_examples)

        return {
            "pixel_accuracy": results["cell_accuracy"],
            "task_accuracy": results["puzzle_accuracy"],
            "validity_rate": results["validity_rate"],
            "tasks_correct": results["puzzles_correct"],
            "valid_solutions": results["valid_puzzles"],
            "total_tasks": results["total_puzzles"],
            "avg_steps": results["avg_steps"],
            "cross_size": results["cross_size"],
            "train_grid_size": results["train_grid_size"],
            "eval_grid_size": results["eval_grid_size"],
            "detailed_results": results.get("detailed_results", []),
        }

    def evaluate_all_splits(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate on all available splits."""
        results = {}

        for split in ["train", "val", "test"]:
            try:
                results[split] = self.evaluate(split, print_examples=False)
            except Exception as e:
                print(f"Could not evaluate {split} split: {e}")

        return results

    def save_results(self, results: Dict[str, Any], output_dir: str = "evaluation_results"):
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary = {
            "checkpoint": str(self.checkpoint_path),
            "timestamp": timestamp,
            "grid_size": self.grid_size,
            "cross_size": self.cross_size,
            "train_grid_size": self.train_grid_size,
            "eval_grid_size": self.eval_grid_size,
        }

        # Handle different result formats
        if "pixel_accuracy" in results:
            summary["results"] = {
                "pixel_accuracy": results["pixel_accuracy"],
                "task_accuracy": results["task_accuracy"],
                "validity_rate": results.get("validity_rate", 0),
                "tasks_correct": results.get("tasks_correct", 0),
                "total_tasks": results.get("total_tasks", 0),
                "avg_steps": results.get("avg_steps", 0),
            }
        elif "cell_accuracy" in results:
            summary["results"] = {
                "cell_accuracy": results["cell_accuracy"],
                "puzzle_accuracy": results["puzzle_accuracy"],
                "validity_rate": results.get("validity_rate", 0),
                "puzzles_correct": results.get("puzzles_correct", 0),
                "total_puzzles": results.get("total_puzzles", 0),
                "avg_steps": results.get("avg_steps", 0),
            }
        else:
            summary["results"] = {}
            for split, split_results in results.items():
                if isinstance(split_results, dict):
                    summary["results"][split] = {
                        "cell_accuracy": split_results.get("cell_accuracy", 0),
                        "puzzle_accuracy": split_results.get("puzzle_accuracy", 0),
                        "validity_rate": split_results.get("validity_rate", 0),
                        "puzzles_correct": split_results.get("puzzles_correct", 0),
                        "total_puzzles": split_results.get("total_puzzles", 0),
                        "avg_steps": split_results.get("avg_steps", 0),
                    }

        summary_file = output_path / f"sudoku_eval_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {summary_file}")

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
        if self.cross_size:
            print(f"(Cross-size: {self.train_grid_size}x{self.train_grid_size} → "
                  f"{self.eval_grid_size}x{self.eval_grid_size})")
        print("=" * 60)

        if "split" in results:
            self._print_split_results(results["split"], results)
        else:
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
        print(f"  Average steps: {results.get('avg_steps', 0):.1f}")

    def print_examples(self, batch: Dict[str, torch.Tensor], predictions: torch.Tensor, 
                   num_examples: int = 3):
        """Print a few input/output/prediction examples for debugging."""
        inputs = batch["input"]
        targets = batch["output"]
        
        num_examples = min(num_examples, len(inputs))
        
        print("\n" + "=" * 70)
        print("EXAMPLE PREDICTIONS")
        print("=" * 70)
        
        for i in range(num_examples):
            inp = inputs[i].reshape(self.max_grid_size, self.max_grid_size)
            tgt = targets[i].reshape(self.max_grid_size, self.max_grid_size)
            pred = predictions[i].reshape(self.max_grid_size, self.max_grid_size)
            
            # Extract actual grid
            inp_grid = inp[:self.grid_size, :self.grid_size]
            tgt_grid = tgt[:self.grid_size, :self.grid_size]
            pred_grid = pred[:self.grid_size, :self.grid_size]
            
            # Decode: 0=PAD, 1=EOS, 2=empty, 3+=values
            def decode_cell(val, is_input=False):
                val = val.item()
                if val == 0:
                    return "."  # PAD
                elif val == 1:
                    return "X"  # EOS
                elif val == 2:
                    return "_" if is_input else "?"  # Empty
                else:
                    return str(val - 2)  # Actual value
            
            def grid_to_str(grid, is_input=False):
                lines = []
                box_rows, box_cols = self._get_box_dims(self.grid_size)
                
                for r in range(self.grid_size):
                    if r > 0 and r % box_rows == 0:
                        # Add horizontal separator
                        sep = "+".join(["-" * (box_cols * 2 - 1)] * (self.grid_size // box_cols))
                        lines.append(sep)
                    
                    row_str = ""
                    for c in range(self.grid_size):
                        if c > 0 and c % box_cols == 0:
                            row_str += "|"
                        row_str += decode_cell(grid[r, c], is_input) + " "
                    lines.append(row_str.rstrip())
                
                return "\n".join(lines)
            
            # Check correctness
            is_correct = torch.all(pred_grid == tgt_grid).item()
            
            # Decode for validity check
            pred_decoded = pred_grid.clone()
            valid_mask = pred_grid >= 3
            pred_decoded[valid_mask] = pred_grid[valid_mask] - 2
            pred_decoded[~valid_mask] = 0
            is_valid = self.check_sudoku_validity(pred_decoded)
            
            # Count errors
            errors = (pred_grid != tgt_grid).sum().item()
            
            print(f"\n--- Example {i + 1} ---")
            print(f"Status: {'✓ CORRECT' if is_correct else f'✗ WRONG ({errors} errors)'} | "
                f"Valid Sudoku: {'✓' if is_valid else '✗'}")
            
            # Print side by side
            inp_lines = grid_to_str(inp_grid, is_input=True).split("\n")
            tgt_lines = grid_to_str(tgt_grid).split("\n")
            pred_lines = grid_to_str(pred_grid).split("\n")
            
            # Calculate width for alignment
            width = max(len(line) for line in inp_lines) + 4
            
            print(f"\n{'INPUT':<{width}}{'TARGET':<{width}}{'PREDICTION'}")
            print(f"{'-' * (width - 2):<{width}}{'-' * (width - 2):<{width}}{'-' * (width - 2)}")
            
            for inp_line, tgt_line, pred_line in zip(inp_lines, tgt_lines, pred_lines):
                print(f"{inp_line:<{width}}{tgt_line:<{width}}{pred_line}")
            
            # Highlight differences
            if not is_correct:
                print("\nDifferences (row, col): ", end="")
                diffs = []
                for r in range(self.grid_size):
                    for c in range(self.grid_size):
                        if pred_grid[r, c] != tgt_grid[r, c]:
                            pred_val = decode_cell(pred_grid[r, c])
                            tgt_val = decode_cell(tgt_grid[r, c])
                            diffs.append(f"({r},{c}): pred={pred_val} vs target={tgt_val}")
                print(", ".join(diffs[:10]))
                if len(diffs) > 10:
                    print(f"  ... and {len(diffs) - 10} more")
        
        print("\n" + "=" * 70)