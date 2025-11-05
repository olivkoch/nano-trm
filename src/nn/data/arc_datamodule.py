"""
TRM DataModule with Puzzle ID support for PyTorch Lightning
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class TRMTransform:
    """Transform class for preprocessing ARC tasks with puzzle ID support."""

    def __init__(
        self,
        max_grid_size: int = 30,
        num_colors: int = 10,
        pad_value: int = 10,
        augment: bool = False,
        puzzle_id_map: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the transform.

        Args:
            max_grid_size: Maximum grid size for padding
            num_colors: Number of colors in ARC
            pad_value: Value to use for padding
            augment: Whether to apply data augmentation
            puzzle_id_map: Mapping from task_id to puzzle_identifier
        """
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        self.pad_value = pad_value
        self.augment = augment
        self.puzzle_id_map = puzzle_id_map or {}

    def pad_grid(self, grid: np.ndarray) -> np.ndarray:
        """Pad a grid to fixed size."""
        padded = np.full((self.max_grid_size, self.max_grid_size), self.pad_value, dtype=np.int64)
        h, w = min(grid.shape[0], self.max_grid_size), min(grid.shape[1], self.max_grid_size)
        padded[:h, :w] = grid[:h, :w]
        return padded

    def apply_augmentation(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation (rotation, flip, color permutation)."""
        if not self.augment:
            return input_grid, output_grid

        # Random rotation (0, 90, 180, 270 degrees)
        k = np.random.randint(0, 4)
        if k > 0:
            input_grid = np.rot90(input_grid, k)
            output_grid = np.rot90(output_grid, k)

        # Random flip
        if np.random.random() > 0.5:
            input_grid = np.fliplr(input_grid)
            output_grid = np.fliplr(output_grid)

        if np.random.random() > 0.5:
            input_grid = np.flipud(input_grid)
            output_grid = np.flipud(output_grid)

        # Color permutation (keeping 0 as background)
        if np.random.random() > 0.5:  # Only apply 50% of the time
            perm = np.random.permutation(self.num_colors - 1) + 1
            perm = np.concatenate([[0], perm])  # Keep 0 as 0

            input_grid = perm[input_grid]
            output_grid = perm[output_grid]

        return input_grid, output_grid

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Transform a single sample with puzzle identifier."""
        # Get input grid
        input_grid = np.array(sample["input"])

        # Get output grid if available (for training)
        output_grid = None
        if "output" in sample:
            output_grid = np.array(sample["output"])

            # Apply augmentation if enabled
            if self.augment:
                input_grid, output_grid = self.apply_augmentation(input_grid, output_grid)

        # Pad grids
        input_padded = self.pad_grid(input_grid)

        result = {
            "input": torch.from_numpy(input_padded).long(),
            "original_shape": torch.tensor(input_grid.shape),
        }

        if output_grid is not None:
            output_padded = self.pad_grid(output_grid)
            result["output"] = torch.from_numpy(output_padded).long()
            result["output_shape"] = torch.tensor(output_grid.shape)

        # Extract base task ID and get puzzle identifier
        if "task_id" in sample:
            result["task_id"] = sample["task_id"]
            # Extract base task_id (remove _train_0, _test_0 suffixes)
            base_task_id = sample["task_id"].split("_train_")[0].split("_test_")[0]
            result["puzzle_identifier"] = self.puzzle_id_map.get(base_task_id, 0)
        else:
            result["puzzle_identifier"] = sample.get("puzzle_identifier", 0)

        return result


class ARCTaskDataset(Dataset):
    """Dataset for ARC tasks with puzzle ID support."""

    def __init__(
        self,
        tasks: Dict[str, Dict],
        solutions: Optional[Dict[str, List]] = None,
        transform: Optional[TRMTransform] = None,
        samples_per_task: int = 1,
    ):
        """
        Initialize dataset.

        Args:
            tasks: Dictionary of task_id -> task_data
            solutions: Optional dictionary of task_id -> solutions
            transform: Transform to apply to samples
            samples_per_task: Number of samples per task (for augmentation)
        """
        self.tasks = tasks
        self.solutions = solutions
        self.transform = transform or TRMTransform()
        self.samples_per_task = samples_per_task

        # Create flat list of samples
        self.samples = []
        for task_id, task_data in tasks.items():
            # Add training examples
            for i, train_ex in enumerate(task_data.get("train", [])):
                for _ in range(samples_per_task):
                    sample = {
                        "task_id": f"{task_id}_train_{i}",
                        "input": train_ex["input"],
                        "output": train_ex["output"],
                    }
                    self.samples.append(sample)

            # Add test examples
            for i, test_ex in enumerate(task_data.get("test", [])):
                sample = {"task_id": f"{task_id}_test_{i}", "input": test_ex["input"]}

                # Add solution if available
                if solutions and task_id in solutions:
                    if i < len(solutions[task_id]):
                        sol = solutions[task_id][i]
                        if isinstance(sol, dict) and "output" in sol:
                            sample["output"] = sol["output"]
                        else:
                            sample["output"] = sol

                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return self.transform(sample)


def collate_fn_with_puzzles(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function that handles puzzle identifiers."""
    # Stack inputs
    inputs = torch.stack([sample["input"] for sample in batch])

    # Stack puzzle identifiers
    puzzle_ids = torch.tensor(
        [sample.get("puzzle_identifier", 0) for sample in batch], dtype=torch.long
    )

    # Create batch dict - using names that match reference
    batch_dict = {
        "input": inputs,  # Note: plural to match reference
        "puzzle_identifiers": puzzle_ids,  # Note: plural to match reference
    }

    # Stack outputs if all samples have them
    if all("output" in sample for sample in batch):
        outputs = torch.stack([sample["output"] for sample in batch])
        batch_dict["output"] = outputs  # Keep as 'output' for your existing code

    # Add metadata
    if "task_id" in batch[0]:
        batch_dict["task_ids"] = [sample["task_id"] for sample in batch]

    if "original_shape" in batch[0]:
        batch_dict["original_shapes"] = torch.stack([sample["original_shape"] for sample in batch])

    return batch_dict


class ARCDataModuleWithPuzzles(LightningDataModule):
    """PyTorch Lightning DataModule with puzzle ID support."""

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 32,
        num_workers: int = 4,
        max_grid_size: int = 30,
        augment_train: bool = True,
        samples_per_task: int = 100,
        use_concept_data: bool = False,
        pad_value: int = 10,
        concept_data_dir: str = "data/concept",
    ):
        """
        Initialize DataModule.

        Args:
            data_dir: Directory containing ARC JSON files
            batch_size: Batch size for training
            num_workers: Number of workers for DataLoader
            max_grid_size: Maximum grid size for padding
            augment_train: Whether to augment training data
            samples_per_task: Number of augmented samples per task
            use_concept_data: Whether to include concept ARC data
            concept_data_dir: Directory containing concept ARC files
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_grid_size = max_grid_size
        self.augment_train = augment_train
        self.samples_per_task = samples_per_task
        self.use_concept_data = use_concept_data
        self.concept_data_dir = concept_data_dir

        # Will be populated in setup()
        self.puzzle_id_map = {}
        self.num_puzzles = 0
        self.pad_value = pad_value

    def setup(self, stage: Optional[str] = None):
        """Load and setup datasets."""
        import json
        from pathlib import Path

        data_path = Path(self.data_dir)

        # Load TRAINING data (for training)
        train_challenges_path = data_path / "arc-agi_training_challenges.json"
        train_solutions_path = data_path / "arc-agi_training_solutions.json"

        # Load EVALUATION data (for validation)
        eval_challenges_path = data_path / "arc-agi_evaluation_challenges.json"
        eval_solutions_path = data_path / "arc-agi_evaluation_solutions.json"

        # Check if files exist
        if not train_challenges_path.exists():
            raise FileNotFoundError(f"Training challenges not found: {train_challenges_path}")
        if not train_solutions_path.exists():
            raise FileNotFoundError(f"Training solutions not found: {train_solutions_path}")
        if not eval_challenges_path.exists():
            raise FileNotFoundError(f"Evaluation challenges not found: {eval_challenges_path}")
        if not eval_solutions_path.exists():
            raise FileNotFoundError(f"Evaluation solutions not found: {eval_solutions_path}")

        # Load training data
        with open(train_challenges_path) as f:
            train_challenges = json.load(f)

        with open(train_solutions_path) as f:
            train_solutions = json.load(f)

        # Load evaluation data
        with open(eval_challenges_path) as f:
            eval_challenges = json.load(f)

        with open(eval_solutions_path) as f:
            eval_solutions = json.load(f)

        if self.use_concept_data:
            concept_path = Path(self.concept_data_dir)

            # Load TRAINING data (for training)
            concept_train_challenges_path = concept_path / "concept_training_challenges.json"
            concept_train_solutions_path = concept_path / "concept_training_solutions.json"

            # Load EVALUATION data (for validation)
            concept_eval_challenges_path = concept_path / "concept_evaluation_challenges.json"
            concept_eval_solutions_path = concept_path / "concept_evaluation_solutions.json"

            # Check if files exist
            if not concept_train_challenges_path.exists():
                raise FileNotFoundError(
                    f"Concept training challenges not found: {concept_train_challenges_path}"
                )
            if not concept_train_solutions_path.exists():
                raise FileNotFoundError(
                    f"Concept training solutions not found: {concept_train_solutions_path}"
                )
            if not concept_eval_challenges_path.exists():
                raise FileNotFoundError(
                    f"Concept evaluation challenges not found: {concept_eval_challenges_path}"
                )
            if not concept_eval_solutions_path.exists():
                raise FileNotFoundError(
                    f"Concept evaluation solutions not found: {concept_eval_solutions_path}"
                )

            # Load training data
            with open(concept_train_challenges_path) as f:
                train_challenges.update(json.load(f))

            with open(concept_train_solutions_path) as f:
                train_solutions.update(json.load(f))

            # Load evaluation data
            with open(concept_eval_challenges_path) as f:
                eval_challenges.update(json.load(f))

            with open(concept_eval_solutions_path) as f:
                eval_solutions.update(json.load(f))

        # Create puzzle ID mapping (1-indexed, 0 reserved for unknown/padding)
        all_task_ids = sorted(set(train_challenges.keys()) | set(eval_challenges.keys()))
        self.puzzle_id_map = {task_id: i + 1 for i, task_id in enumerate(all_task_ids)}
        self.num_puzzles = len(self.puzzle_id_map) + 1  # +1 for reserved 0

        print(f"✓ Created puzzle ID mapping for {len(self.puzzle_id_map)} unique puzzles")

        # Create transforms with puzzle ID mapping
        self.train_transform = TRMTransform(
            max_grid_size=self.max_grid_size,
            augment=self.augment_train,
            puzzle_id_map=self.puzzle_id_map,
            pad_value=self.pad_value,
        )

        self.val_transform = TRMTransform(
            max_grid_size=self.max_grid_size,
            augment=False,
            puzzle_id_map=self.puzzle_id_map,
            pad_value=self.pad_value,
        )

        # Create datasets
        self.train_dataset = ARCTaskDataset(
            train_challenges,
            train_solutions,
            transform=self.train_transform,
            samples_per_task=self.samples_per_task if self.augment_train else 1,
        )

        self.val_dataset = ARCTaskDataset(
            eval_challenges,
            eval_solutions,
            transform=self.val_transform,
            samples_per_task=1,  # No augmentation for validation
        )

        print(f"✓ Loaded training set: {len(train_challenges)} tasks")
        print(f"  → {len(self.train_dataset)} samples (with augmentation)")
        print(f"✓ Loaded evaluation set: {len(eval_challenges)} tasks")
        print(f"  → {len(self.val_dataset)} samples")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_with_puzzles,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True,  # Add this to ensure constant batch size
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_with_puzzles,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True,  # Add this to ensure constant batch size
        )

    def test_dataloader(self):
        # For testing, use evaluation dataset
        return self.val_dataloader()
