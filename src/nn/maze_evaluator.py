"""
Evaluation script for Maze models
Adapted from SudokuEvaluator for maze-solving tasks
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from src.nn.data.sudoku_datamodule import SudokuDataModule
from src.nn.models.trm import TRMModule


class MazeEvaluator:
    """Evaluator for Maze models using SudokuDataModule."""

    # Token encoding from CHARSET = "# SGo" with char2id[ord(c)] = i + 1
    # 0 = PAD, then 1-5 map to "#", " ", "S", "G", "o"
    PAD = 0
    WALL = 1       # '#' - wall
    SPACE = 2      # ' ' - open corridor (not part of solution)
    START = 3      # 'S' - start position
    GOAL = 4       # 'G' - goal/end position
    PATH = 5       # 'o' - solution path

    def __init__(
        self,
        checkpoint_path: str,
        data_dir: str,
        batch_size: int = 64,  # Smaller default for larger grids
        device: str = "auto",
        num_workers: int = 0,
        eval_split: str = "val",
        grid_size: int = 30,  # Default maze size
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
            grid_size: Size of the maze grid
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size
        self.eval_split = eval_split
        self.grid_size = grid_size

        if data_dir is None:
            raise ValueError("data_dir is required for evaluation.")
    
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

        # Get config from datamodule - prefer datamodule values over constructor arg
        self.max_grid_size = getattr(self.datamodule, 'max_grid_size', grid_size)
        self.vocab_size = self.datamodule.vocab_size
        
        # Use grid_size from datamodule if available, otherwise use constructor arg
        datamodule_grid_size = getattr(self.datamodule, 'grid_size', None)
        if datamodule_grid_size is not None:
            self.grid_size = datamodule_grid_size
        else:
            self.grid_size = grid_size

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

    def find_position(self, grid: torch.Tensor, token: int) -> Optional[Tuple[int, int]]:
        """Find the position of a token in the grid."""
        positions = (grid == token).nonzero(as_tuple=False)
        if len(positions) > 0:
            return (positions[0, 0].item(), positions[0, 1].item())
        return None

    def check_maze_validity(self, pred_grid: torch.Tensor, input_grid: torch.Tensor) -> Dict[str, Any]:
        """
        Check if a maze solution is valid.
        
        Returns dict with:
            - valid: bool - overall validity
            - has_path: bool - solution forms connected path
            - reaches_end: bool - path reaches from start to end
            - no_wall_crossing: bool - path doesn't go through walls
            - details: str - human-readable explanation
        """
        n = self.grid_size
        
        # Find start and end positions from input
        start_pos = self.find_position(input_grid, self.START)
        end_pos = self.find_position(input_grid, self.GOAL)
        
        if start_pos is None or end_pos is None:
            return {
                "valid": False,
                "has_path": False,
                "reaches_end": False,
                "no_wall_crossing": True,
                "details": "Missing start or end position"
            }
        
        # Find all solution cells in prediction (PATH = 'o' markers)
        solution_cells = set()
        for r in range(n):
            for c in range(n):
                if pred_grid[r, c] == self.PATH:
                    solution_cells.add((r, c))
        
        # Also include start and goal positions
        solution_cells.add(start_pos)
        solution_cells.add(end_pos)
        
        if len(solution_cells) < 2:
            return {
                "valid": False,
                "has_path": False,
                "reaches_end": False,
                "no_wall_crossing": True,
                "details": "No solution path marked"
            }
        
        # Check for wall crossing
        wall_crossings = []
        for r, c in solution_cells:
            if input_grid[r, c] == self.WALL:
                wall_crossings.append((r, c))
        
        no_wall_crossing = len(wall_crossings) == 0
        
        # Check path connectivity using BFS from start
        visited = set()
        queue = [start_pos]
        visited.add(start_pos)
        
        while queue:
            r, c = queue.pop(0)
            
            # Check 4-connected neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    if (nr, nc) not in visited and (nr, nc) in solution_cells:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        has_path = len(visited) == len(solution_cells)
        reaches_end = end_pos in visited
        
        valid = has_path and reaches_end and no_wall_crossing
        
        details = []
        if not has_path:
            details.append(f"Disconnected path ({len(visited)}/{len(solution_cells)} cells reachable)")
        if not reaches_end:
            details.append("Path doesn't reach end")
        if not no_wall_crossing:
            details.append(f"Crosses {len(wall_crossings)} walls")
        
        return {
            "valid": valid,
            "has_path": has_path,
            "reaches_end": reaches_end,
            "no_wall_crossing": no_wall_crossing,
            "details": "; ".join(details) if details else "Valid solution"
        }

    def visualize_thinking(
        self, 
        batch: Dict[str, torch.Tensor], 
        sample_idx: int = 0,
        max_steps: int = None,
        show_confidence: bool = True,
        save_gif: bool = False,
        gif_path: str = None,
        gif_size: int = 600,  # Larger default for maze
        gif_duration: int = 500,
        save_pngs: bool = True,
    ) -> Dict[str, Any]:
        """
        Visualize the TRM's thinking process for a single maze sample.
        
        Args:
            batch: Batch of samples
            sample_idx: Which sample in the batch to visualize
            max_steps: Maximum steps to run (default: N_supervision_val)
            show_confidence: Whether to show confidence values
            save_gif: Whether to save an animated GIF
            gif_path: Path to save GIF (default: maze_thinking.gif)
            gif_size: Width/height of the GIF in pixels
            gif_duration: Duration of each frame in milliseconds
            save_pngs: Whether to also save individual PNG frames
            
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
                
                logits = outputs["logits"]
                q_halt = outputs["q_halt_logits"]
                
                # Get predictions and confidence for this sample
                sample_logits = logits[sample_idx]
                probs = torch.softmax(sample_logits, dim=-1)
                confidence, preds = probs.max(dim=-1)
                
                # Reshape to grid
                pred_grid = preds.reshape(self.max_grid_size, self.max_grid_size)
                conf_grid = confidence.reshape(self.max_grid_size, self.max_grid_size)
                
                # Extract actual maze region
                pred_maze = pred_grid[:self.grid_size, :self.grid_size].clone()
                conf_maze = conf_grid[:self.grid_size, :self.grid_size].clone()
                
                step_predictions.append(pred_maze.cpu())
                step_confidences.append(conf_maze.cpu())
                step_q_halt.append(q_halt[sample_idx].item())
                step_halted.append(carry.halted[sample_idx].item())
                
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
        
        # Generate GIF if requested
        if save_gif:
            gif_file = gif_path or "maze_thinking.gif"
            self._generate_thinking_gif(
                inp_grid.cpu(),
                tgt_grid.cpu(),
                step_predictions,
                gif_path=gif_file,
                size=gif_size,
                duration=gif_duration,
                save_pngs=save_pngs,
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
    
    def _generate_thinking_gif(
        self,
        inp_grid: torch.Tensor,
        tgt_grid: torch.Tensor,
        step_predictions: list,
        gif_path: str = "maze_thinking.gif",
        size: int = 600,
        duration: int = 500,
        save_pngs: bool = True,
    ):
        """Generate an animated GIF showing the thinking process."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            print("PIL not installed. Run: pip install Pillow")
            return
        
        frames = []
        num_steps = len(step_predictions)
        
        # Mask for cells to predict (SPACE cells that could become PATH in solution)
        prediction_mask = (inp_grid == self.SPACE)
        
        # Try to get a nice font
        try:
            font_size = max(8, size // 40)
            label_font_size = size // 30
            for font_name in ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf", 
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    label_font = ImageFont.truetype(font_name, label_font_size)
                    break
                except (OSError, IOError):
                    continue
            else:
                font = ImageFont.load_default()
                label_font = font
        except Exception:
            font = ImageFont.load_default()
            label_font = font
        
        # Calculate grid dimensions
        padding = size // 15
        grid_area = size - 2 * padding
        cell_size = grid_area // self.grid_size
        grid_size_px = cell_size * self.grid_size
        
        offset = (size - grid_size_px) // 2
        
        # Create initial frame (input maze)
        frames.append(self._create_maze_frame(
            inp_grid, tgt_grid, None, prediction_mask,
            size, cell_size, offset,
            font, label_font, 0, num_steps, is_input=True
        ))
        
        # Create frames for each step
        prev_pred = inp_grid
        for step, pred in enumerate(step_predictions):
            frame = self._create_maze_frame(
                pred, tgt_grid, prev_pred, prediction_mask,
                size, cell_size, offset,
                font, label_font, step + 1, num_steps, is_input=False,
                input_grid=inp_grid
            )
            frames.append(frame)
            prev_pred = pred
        
        # Save individual PNG files
        if save_pngs:
            import os
            png_dir = gif_path.rsplit('.', 1)[0] + "_frames"
            os.makedirs(png_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                png_path = os.path.join(png_dir, f"frame_{i:02d}.png")
                frame.save(png_path)
            print(f"✓ Saved {len(frames)} PNG frames to {png_dir}/")
        
        # Convert frames to palette mode
        if len(frames) > 1:
            total_width = sum(f.width for f in frames)
            combined = Image.new('RGB', (total_width, frames[0].height))
            x_offset = 0
            for frame in frames:
                combined.paste(frame, (x_offset, 0))
                x_offset += frame.width
            
            combined_quantized = combined.quantize(colors=256)
            
            palette_frames = []
            for frame in frames:
                p_frame = frame.quantize(palette=combined_quantized)
                palette_frames.append(p_frame)
        else:
            palette_frames = [frames[0].quantize(colors=256)]
        
        # Save GIF
        palette_frames[0].save(
            gif_path,
            save_all=True,
            append_images=palette_frames[1:] if len(palette_frames) > 1 else [],
            duration=duration,
            loop=0,
        )
        print(f"\n✓ Saved GIF to {gif_path}")
    
    def _create_maze_frame(
        self,
        grid: torch.Tensor,
        target: torch.Tensor,
        prev_grid: torch.Tensor,
        prediction_mask: torch.Tensor,
        size: int,
        cell_size: int,
        offset: int,
        font,
        label_font,
        step: int,
        total_steps: int,
        is_input: bool = False,
        input_grid: torch.Tensor = None,
    ):
        """Create a single frame for the maze GIF."""
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', (size, size), 'white')
        draw = ImageDraw.Draw(img)
        
        # Color scheme
        WALL_COLOR = '#4a4a4a'      # Dark gray for walls
        SPACE_COLOR = '#FFFFFF'     # White for open corridors
        PATH_CORRECT = '#22c55e'    # Green for correct path
        PATH_INCORRECT = '#f97316'  # Orange for incorrect path
        START_END_COLOR = '#ef4444' # Red for S and G labels
        
        # Use input_grid for determining walls/start/goal if provided
        base_grid = input_grid if input_grid is not None else grid
        
        # Draw cells
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x = offset + c * cell_size
                y = offset + r * cell_size
                
                base_val = base_grid[r, c].item()
                cell_val = grid[r, c].item()
                target_val = target[r, c].item()
                
                # Determine cell background color
                if base_val == self.WALL:
                    color = WALL_COLOR
                else:
                    color = SPACE_COLOR
                
                draw.rectangle(
                    [x, y, x + cell_size - 1, y + cell_size - 1],
                    fill=color,
                    outline=None
                )
                
                # Draw cell content
                text = None
                text_color = 'black'
                
                if base_val == self.START:
                    text = "S"
                    text_color = START_END_COLOR
                elif base_val == self.GOAL:
                    text = "E"
                    text_color = START_END_COLOR
                elif cell_val == self.PATH and base_val != self.WALL:
                    # This is a solution path cell - draw a filled circle or marker
                    # Check if correct
                    if cell_val == target_val:
                        marker_color = PATH_CORRECT
                    else:
                        marker_color = PATH_INCORRECT
                    
                    # Draw a filled circle for path
                    margin = max(2, cell_size // 6)
                    draw.ellipse(
                        [x + margin, y + margin, x + cell_size - margin - 1, y + cell_size - margin - 1],
                        fill=marker_color
                    )
                elif not is_input and base_val == self.SPACE:
                    # Check for incorrect prediction: predicted SPACE but should be PATH
                    if target_val == self.PATH and cell_val != self.PATH:
                        # Missing path - show as faint orange outline
                        margin = max(2, cell_size // 6)
                        draw.ellipse(
                            [x + margin, y + margin, x + cell_size - margin - 1, y + cell_size - margin - 1],
                            fill=None,
                            outline=PATH_INCORRECT,
                            width=max(1, cell_size // 8)
                        )
                
                # Draw text (S or E)
                if text and cell_size >= 8:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    text_x = x + (cell_size - text_w) // 2
                    text_y = y + (cell_size - text_h) // 2 - bbox[1]
                    draw.text((text_x, text_y), text, fill=text_color, font=font)
        
        # Draw grid lines (thin, for visibility on larger mazes)
        if cell_size >= 6:
            grid_size_px = cell_size * self.grid_size
            line_color = '#E0E0E0'
            for i in range(self.grid_size + 1):
                y = offset + i * cell_size
                draw.line([(offset, y), (offset + grid_size_px, y)], fill=line_color, width=1)
                x = offset + i * cell_size
                draw.line([(x, offset), (x, offset + grid_size_px)], fill=line_color, width=1)
        
        # Draw step label
        if is_input:
            label = "Input Maze"
        else:
            label = f"Step {step}/{total_steps}"
        
        bbox = draw.textbbox((0, 0), label, font=label_font)
        label_w = bbox[2] - bbox[0]
        label_x = size - label_w - 10
        label_y = 10
        draw.text((label_x, label_y), label, fill='black', font=label_font)
        
        return img
    
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
        """Print a text visualization of the thinking process."""
        
        def decode_cell(val, is_input=False):
            val = val.item() if hasattr(val, 'item') else val
            if val == self.PAD:
                return " "
            elif val == self.WALL:
                return "█"
            elif val == self.SPACE:
                return "·"
            elif val == self.START:
                return "S"
            elif val == self.GOAL:
                return "G"
            elif val == self.PATH:
                return "o"
            else:
                return "?"
        
        def grid_to_str(grid):
            lines = []
            for r in range(self.grid_size):
                row_str = ""
                for c in range(self.grid_size):
                    row_str += decode_cell(grid[r, c])
                lines.append(row_str)
            return "\n".join(lines)
        
        def compute_metrics(pred, target, inp):
            """Compute accuracy metrics for maze."""
            # Mask: cells that are SPACE in input (could become PATH in solution)
            mask = (inp == self.SPACE)
            if mask.sum() == 0:
                return 1.0, 0
            correct = (pred == target) & mask
            total = mask.sum().item()
            return correct.sum().item() / total, (mask & ~(pred == target)).sum().item()
        
        # Mask for cells to predict
        prediction_mask = (inp_grid == self.SPACE)
        
        print("\n" + "=" * 80)
        print("TRM MAZE THINKING VISUALIZATION")
        print(f"H_cycles={self.model.hparams.H_cycles}, L_cycles={self.model.hparams.L_cycles}")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        print("=" * 80)
        
        # For large mazes, just show summary info
        if self.grid_size > 15:
            print(f"\n(Maze too large for text display, showing metrics only)")
            print(f"Cells to predict: {prediction_mask.sum().item()}")
            
            print("\n" + "-" * 80)
            print("STEP-BY-STEP REASONING")
            print("-" * 80)
            
            prev_pred = None
            for step, (pred, conf, q_halt, halted) in enumerate(
                zip(step_predictions, step_confidences, step_q_halt, step_halted)
            ):
                acc, errors = compute_metrics(pred, tgt_grid, inp_grid)
                avg_conf = conf[prediction_mask].mean().item() if prediction_mask.sum() > 0 else 1.0
                
                if prev_pred is not None:
                    changes = ((pred != prev_pred) & prediction_mask).sum().item()
                else:
                    changes = "-"
                
                q_halt_str = f"q={q_halt:+.2f}"
                if q_halt > 0:
                    q_halt_str += " (HALT)"
                
                halt_marker = " ← STOPPED" if halted else ""
                
                print(f"Step {step + 1}: Acc={acc:.1%} ({errors} errors) | Changes: {changes} | "
                      f"Conf: {avg_conf:.2f} | {q_halt_str}{halt_marker}")
                
                prev_pred = pred
        else:
            # Show full grids for smaller mazes
            print(f"\nINPUT MAZE:")
            print(grid_to_str(inp_grid))
            print(f"\nTARGET SOLUTION:")
            print(grid_to_str(tgt_grid))
            print(f"\nCells to predict: {prediction_mask.sum().item()}")
            
            print("\n" + "-" * 80)
            print("STEP-BY-STEP REASONING")
            print("-" * 80)
            
            prev_pred = None
            for step, (pred, conf, q_halt, halted) in enumerate(
                zip(step_predictions, step_confidences, step_q_halt, step_halted)
            ):
                acc, errors = compute_metrics(pred, tgt_grid, inp_grid)
                avg_conf = conf[prediction_mask].mean().item() if prediction_mask.sum() > 0 else 1.0
                
                if prev_pred is not None:
                    changes = ((pred != prev_pred) & prediction_mask).sum().item()
                else:
                    changes = "-"
                
                q_halt_str = f"q={q_halt:+.2f}"
                halt_marker = " ← STOPPED" if halted else ""
                
                print(f"\n┌─ Step {step + 1} ─────────────────────────────")
                print(f"│ Accuracy: {acc:.1%} ({errors} errors) | Changes: {changes} | {q_halt_str}{halt_marker}")
                print("└" + "─" * 40)
                print(grid_to_str(pred))
                
                prev_pred = pred
        
        # Final summary
        final_pred = step_predictions[-1]
        final_acc, final_errors = compute_metrics(final_pred, tgt_grid, inp_grid)
        final_correct = final_errors == 0
        
        validity = self.check_maze_validity(final_pred, inp_grid)
        
        print("\n" + "=" * 80)
        status = "✓ SOLVED" if final_correct else "✗ FAILED"
        valid_str = "✓ Valid path" if validity["valid"] else f"✗ Invalid: {validity['details']}"
        print(f"RESULT: {status} | {valid_str} | {len(step_predictions)} steps")
        print("=" * 80)

    def visualize_sample(
        self, 
        split: str = "val", 
        sample_idx: int = 0, 
        show_confidence: bool = True,
        min_steps: int = None,
        max_search: int = 100,
        save_gif: bool = False,
        gif_path: str = None,
        gif_size: int = 600,
        gif_duration: int = 500,
        save_pngs: bool = True,
    ):
        """
        Convenience method to visualize thinking on a specific sample.
        """
        if split == "train":
            dataloader = self.datamodule.train_dataloader()
        elif split == "val":
            dataloader = self.datamodule.val_dataloader()
        elif split == "test":
            dataloader = self.datamodule.test_dataloader()
        else:
            raise ValueError(f"Invalid split: {split}")
        
        gif_kwargs = dict(
            save_gif=save_gif,
            gif_path=gif_path,
            gif_size=gif_size,
            gif_duration=gif_duration,
            save_pngs=save_pngs,
        )
        
        if min_steps is None:
            batch_idx = sample_idx // self.batch_size
            within_batch_idx = sample_idx % self.batch_size
            
            for i, batch in enumerate(dataloader):
                if i == batch_idx:
                    return self.visualize_thinking(
                        batch, within_batch_idx, 
                        show_confidence=show_confidence,
                        **gif_kwargs
                    )
            
            raise ValueError(f"sample_idx {sample_idx} out of range for {split} split")
        
        # Search for a sample that takes at least min_steps to solve
        # Use efficient batch-wise search
        print(f"Searching for a sample that takes at least {min_steps} steps to solve...")
        
        max_model_steps = self.model.hparams.N_supervision_val
        samples_checked = 0
        start_batch_idx = sample_idx // self.batch_size
        start_within_batch = sample_idx % self.batch_size
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx < start_batch_idx:
                continue
                
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()
            }
            
            batch_size = batch["input"].shape[0]
            start_idx = start_within_batch if batch_idx == start_batch_idx else 0
            
            # Compute steps for ALL samples in this batch at once (efficient!)
            steps_for_batch = self._count_steps_for_batch(batch)
            
            for within_batch_idx in range(start_idx, batch_size):
                if samples_checked >= max_search:
                    print(f"Searched {max_search} samples, none took >= {min_steps} steps.")
                    # Just show the last one we checked
                    return self.visualize_thinking(
                        batch, within_batch_idx, 
                        show_confidence=show_confidence,
                        **gif_kwargs
                    )
                
                steps_to_correct = steps_for_batch[within_batch_idx]
                samples_checked += 1
                
                if steps_to_correct >= min_steps and steps_to_correct <= max_model_steps:
                    actual_idx = batch_idx * self.batch_size + within_batch_idx
                    print(f"Found sample {actual_idx} that takes {steps_to_correct} steps (checked {samples_checked} samples)")
                    return self.visualize_thinking(
                        batch, within_batch_idx, 
                        show_confidence=show_confidence,
                        **gif_kwargs
                    )
            
            # Reset start_within_batch after first batch
            start_within_batch = 0
        
        raise ValueError(f"No sample found that takes >= {min_steps} steps in first {samples_checked} samples")
    
    def _count_steps_for_batch(self, batch: Dict[str, torch.Tensor]) -> list:
        """
        Count how many steps until each sample in the batch gets correct.
        
        Returns a list of step counts (one per sample in batch).
        Values > max_steps mean the sample never got correct.
        """
        max_steps = self.model.hparams.N_supervision_val
        
        targets = batch["output"]
        inputs = batch["input"]
        batch_size = inputs.shape[0]
        
        # Reshape to grids
        targets_grid = targets.reshape(batch_size, self.max_grid_size, self.max_grid_size)
        inputs_grid = inputs.reshape(batch_size, self.max_grid_size, self.max_grid_size)
        
        # Extract actual maze region
        targets_maze = targets_grid[:, :self.grid_size, :self.grid_size]
        inputs_maze = inputs_grid[:, :self.grid_size, :self.grid_size]
        
        # Mask for cells to predict (SPACE cells in input)
        prediction_mask = (inputs_maze == self.SPACE)  # [batch, grid, grid]
        
        # Track when each sample first got correct (-1 means not yet)
        steps_to_correct = [max_steps + 1] * batch_size
        found_correct = [False] * batch_size
        
        with torch.no_grad():
            carry = self.model.initial_carry(batch)
            
            for step in range(max_steps):
                carry, outputs = self.model.forward(carry, batch)
                
                logits = outputs["logits"]  # [batch, seq, vocab]
                preds = logits.argmax(dim=-1)  # [batch, seq]
                preds_grid = preds.reshape(batch_size, self.max_grid_size, self.max_grid_size)
                preds_maze = preds_grid[:, :self.grid_size, :self.grid_size]
                
                # Check correctness for each sample
                for i in range(batch_size):
                    if found_correct[i]:
                        continue
                    
                    mask_i = prediction_mask[i]
                    is_correct = torch.all(preds_maze[i][mask_i] == targets_maze[i][mask_i]).item()
                    
                    if is_correct:
                        steps_to_correct[i] = step + 1
                        found_correct[i] = True
                    elif carry.halted[i]:
                        # Halted but wrong - mark as failed
                        found_correct[i] = True  # Stop checking this sample
                
                # Early exit if all samples resolved
                if all(found_correct):
                    break
        
        return steps_to_correct

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
                self.print_examples(batch, predictions, num_examples=3)

            # Mask for cells to predict (SPACE cells in input)
            mask = (inputs == self.SPACE).float()

            correct_cells = (predictions == targets).float()

            if mask.sum() > 0:
                cell_accuracy = (correct_cells * mask).sum() / mask.sum()
            else:
                cell_accuracy = torch.tensor(1.0)

            maze_correct = []
            valid_mazes = []
            steps_taken = []

            for i in range(batch_size):
                pred_flat = predictions[i]
                target_flat = targets[i]
                input_flat = inputs[i]

                pred_grid = pred_flat.reshape(self.max_grid_size, self.max_grid_size)
                target_grid = target_flat.reshape(self.max_grid_size, self.max_grid_size)
                input_grid = input_flat.reshape(self.max_grid_size, self.max_grid_size)

                pred_maze = pred_grid[:self.grid_size, :self.grid_size]
                target_maze = target_grid[:self.grid_size, :self.grid_size]
                input_maze = input_grid[:self.grid_size, :self.grid_size]

                # Check exact match on predicted cells only
                prediction_mask = (input_maze == self.SPACE)
                exact_match = torch.all(pred_maze[prediction_mask] == target_maze[prediction_mask]).item()
                maze_correct.append(exact_match)

                # Check validity
                validity = self.check_maze_validity(pred_maze, input_maze)
                valid_mazes.append(validity["valid"])
                
                steps_taken.append(carry.steps[i].item())

        return {
            "cell_accuracy": cell_accuracy.item(),
            "maze_correct": maze_correct,
            "valid_mazes": valid_mazes,
            "steps_taken": steps_taken,
            "batch_size": batch_size,
        }

    def evaluate(self, split: str = None, print_examples: bool = False) -> Dict[str, Any]:
        """Run evaluation on specified split."""
        if split is None:
            split = self.eval_split
            
        print(f"\nEvaluating on {split} split...")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")

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
        total_mazes_correct = 0
        total_valid_mazes = 0
        total_mazes = 0
        total_steps = 0

        all_results = []

        first_batch = True

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split}")):
            results = self.evaluate_batch(batch, print_examples=(first_batch and print_examples))
            first_batch = False

            batch_size = results["batch_size"]
            total_cell_correct += results["cell_accuracy"] * batch_size
            total_cells += batch_size
            total_mazes_correct += sum(results["maze_correct"])
            total_valid_mazes += sum(results["valid_mazes"])
            total_mazes += len(results["maze_correct"])
            total_steps += sum(results["steps_taken"])

            for i in range(batch_size):
                all_results.append({
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "exact_match": results["maze_correct"][i],
                    "valid_maze": results["valid_mazes"][i],
                    "steps": results["steps_taken"][i],
                })

        overall_cell_accuracy = total_cell_correct / total_cells if total_cells > 0 else 0
        overall_maze_accuracy = total_mazes_correct / total_mazes if total_mazes > 0 else 0
        overall_validity_rate = total_valid_mazes / total_mazes if total_mazes > 0 else 0
        avg_steps = total_steps / total_mazes if total_mazes > 0 else 0

        return {
            "split": split,
            "cell_accuracy": overall_cell_accuracy,
            "maze_accuracy": overall_maze_accuracy,
            "validity_rate": overall_validity_rate,
            "mazes_correct": total_mazes_correct,
            "valid_mazes": total_valid_mazes,
            "total_mazes": total_mazes,
            "avg_steps": avg_steps,
            "grid_size": self.grid_size,
            "detailed_results": all_results,
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
        }

        if "cell_accuracy" in results:
            summary["results"] = {
                "cell_accuracy": results["cell_accuracy"],
                "maze_accuracy": results["maze_accuracy"],
                "validity_rate": results.get("validity_rate", 0),
                "mazes_correct": results.get("mazes_correct", 0),
                "total_mazes": results.get("total_mazes", 0),
                "avg_steps": results.get("avg_steps", 0),
            }
        else:
            summary["results"] = {}
            for split, split_results in results.items():
                if isinstance(split_results, dict):
                    summary["results"][split] = {
                        "cell_accuracy": split_results.get("cell_accuracy", 0),
                        "maze_accuracy": split_results.get("maze_accuracy", 0),
                        "validity_rate": split_results.get("validity_rate", 0),
                        "mazes_correct": split_results.get("mazes_correct", 0),
                        "total_mazes": split_results.get("total_mazes", 0),
                        "avg_steps": split_results.get("avg_steps", 0),
                    }

        summary_file = output_path / f"maze_eval_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {summary_file}")

        if "detailed_results" in results and results["detailed_results"]:
            detailed_file = output_path / f"maze_eval_detailed_{timestamp}.csv"
            df = pd.DataFrame(results["detailed_results"])
            df.to_csv(detailed_file, index=False)
            print(f"Detailed results saved to {detailed_file}")

        return summary_file

    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("MAZE EVALUATION RESULTS")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
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
        print(f"  Total mazes: {results['total_mazes']}")
        print(f"  Cell accuracy: {results['cell_accuracy']:.2%}")
        print(f"  Exact match accuracy: {results['maze_accuracy']:.2%}")
        print(f"  Valid path rate: {results['validity_rate']:.2%}")
        print(f"  Mazes solved: {results['mazes_correct']}/{results['total_mazes']}")
        print(f"  Valid solutions: {results['valid_mazes']}/{results['total_mazes']}")
        print(f"  Average steps: {results.get('avg_steps', 0):.1f}")

    def print_examples(self, batch: Dict[str, torch.Tensor], predictions: torch.Tensor, 
                       num_examples: int = 3):
        """Print example predictions for debugging."""
        inputs = batch["input"]
        targets = batch["output"]
        
        num_examples = min(num_examples, len(inputs))
        
        print("\n" + "=" * 70)
        print("EXAMPLE PREDICTIONS")
        print("=" * 70)
        
        def decode_cell(val):
            val = val.item()
            if val == self.PAD:
                return " "
            elif val == self.WALL:
                return "█"
            elif val == self.SPACE:
                return "·"
            elif val == self.START:
                return "S"
            elif val == self.GOAL:
                return "G"
            elif val == self.PATH:
                return "o"
            else:
                return "?"
        
        for i in range(num_examples):
            inp = inputs[i].reshape(self.max_grid_size, self.max_grid_size)
            tgt = targets[i].reshape(self.max_grid_size, self.max_grid_size)
            pred = predictions[i].reshape(self.max_grid_size, self.max_grid_size)
            
            inp_grid = inp[:self.grid_size, :self.grid_size]
            tgt_grid = tgt[:self.grid_size, :self.grid_size]
            pred_grid = pred[:self.grid_size, :self.grid_size]
            
            # Check correctness
            prediction_mask = (inp_grid == self.SPACE)
            is_correct = torch.all(pred_grid[prediction_mask] == tgt_grid[prediction_mask]).item()
            
            validity = self.check_maze_validity(pred_grid, inp_grid)
            
            errors = (pred_grid != tgt_grid).sum().item()
            
            print(f"\n--- Example {i + 1} ---")
            print(f"Status: {'✓ CORRECT' if is_correct else f'✗ WRONG ({errors} errors)'} | "
                  f"Valid: {'✓' if validity['valid'] else '✗ ' + validity['details']}")
            
            # For small grids, show side by side
            if self.grid_size <= 15:
                def grid_to_lines(grid):
                    lines = []
                    for r in range(self.grid_size):
                        row_str = ""
                        for c in range(self.grid_size):
                            row_str += decode_cell(grid[r, c])
                        lines.append(row_str)
                    return lines
                
                inp_lines = grid_to_lines(inp_grid)
                tgt_lines = grid_to_lines(tgt_grid)
                pred_lines = grid_to_lines(pred_grid)
                
                width = self.grid_size + 4
                
                print(f"\n{'INPUT':<{width}}{'TARGET':<{width}}{'PREDICTION'}")
                print(f"{'-' * (width - 2):<{width}}{'-' * (width - 2):<{width}}{'-' * (width - 2)}")
                
                for inp_line, tgt_line, pred_line in zip(inp_lines, tgt_lines, pred_lines):
                    print(f"{inp_line:<{width}}{tgt_line:<{width}}{pred_line}")
            else:
                print(f"(Grid too large for display: {self.grid_size}x{self.grid_size})")
        
        print("\n" + "=" * 70)