"""
TRM Transforms for PyTorch Lightning DataModule
Preprocessing transforms for ARC-AGI tasks
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from src.nn.utils.constants import PAD_VALUE

class TRMTransform:
    """
    Transform class for preprocessing ARC tasks for TRM model.
    Can be used as a transform in a Dataset or DataModule.
    """
    
    def __init__(self, 
                 max_grid_size: int = 30,
                 num_colors: int = 10,
                 pad_value: int = PAD_VALUE,
                 augment: bool = False):
        """
        Initialize the transform.
        
        Args:
            max_grid_size: Maximum grid size for padding
            num_colors: Number of colors in ARC
            pad_value: Value to use for padding
            augment: Whether to apply data augmentation
        """
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        self.pad_value = pad_value
        self.augment = augment
    
    def pad_grid(self, grid: np.ndarray) -> np.ndarray:
        """Pad a grid to fixed size."""
        padded = np.full((self.max_grid_size, self.max_grid_size), 
                         self.pad_value, dtype=np.int64)
        h, w = min(grid.shape[0], self.max_grid_size), min(grid.shape[1], self.max_grid_size)
        padded[:h, :w] = grid[:h, :w]
        return padded
    
    def apply_augmentation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        """
        Transform a single sample.
        
        Args:
            sample: Dictionary with 'input' and optionally 'output' grids
            
        Returns:
            Dictionary with tensors ready for TRM
        """
        # Get input grid
        input_grid = np.array(sample['input'])
        
        # Get output grid if available (for training)
        output_grid = None
        if 'output' in sample:
            output_grid = np.array(sample['output'])
            
            # Apply augmentation if enabled
            if self.augment:
                input_grid, output_grid = self.apply_augmentation(input_grid, output_grid)
        
        # Pad grids
        input_padded = self.pad_grid(input_grid)
        
        result = {
            'input': torch.from_numpy(input_padded).long(),
            'original_shape': torch.tensor(input_grid.shape)
        }
        
        if output_grid is not None:
            output_padded = self.pad_grid(output_grid)
            result['output'] = torch.from_numpy(output_padded).long()
            result['output_shape'] = torch.tensor(output_grid.shape)
        
        # Add metadata if available
        if 'task_id' in sample:
            result['task_id'] = sample['task_id']
        
        return result


class ARCTaskDataset(Dataset):
    """
    Dataset for ARC tasks that works with TRM transform.
    """
    
    def __init__(self, 
                 tasks: Dict[str, Dict],
                 solutions: Optional[Dict[str, List]] = None,
                 transform: Optional[TRMTransform] = None,
                 samples_per_task: int = 1):
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
            for i, train_ex in enumerate(task_data.get('train', [])):
                for _ in range(samples_per_task):
                    sample = {
                        'task_id': f"{task_id}_train_{i}",
                        'input': train_ex['input'],
                        'output': train_ex['output']
                    }
                    self.samples.append(sample)
            
            # Add test examples
            for i, test_ex in enumerate(task_data.get('test', [])):
                sample = {
                    'task_id': f"{task_id}_test_{i}",
                    'input': test_ex['input']
                }
                
                # Add solution if available
                if solutions and task_id in solutions:
                    if i < len(solutions[task_id]):
                        sol = solutions[task_id][i]
                        if isinstance(sol, dict) and 'output' in sol:
                            sample['output'] = sol['output']
                        else:
                            sample['output'] = sol
                
                self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return self.transform(sample)


class ARCDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for training on ARC tasks.
    Uses full training set for training and evaluation set for validation.
    """
    
    def __init__(self,
                 data_dir: str = "data",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 max_grid_size: int = 30,
                 augment_train: bool = True,
                 samples_per_task: int = 100, # For augmentation
                 use_concept_data: bool = False,
                 concept_data_dir: str = "data/concept"
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

        # Create transforms
        self.train_transform = TRMTransform(
            max_grid_size=max_grid_size,
            augment=augment_train
        )
        
        self.val_transform = TRMTransform(
            max_grid_size=max_grid_size,
            augment=False  # No augmentation for validation
        )
    
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
        with open(train_challenges_path, 'r') as f:
            train_challenges = json.load(f)
        
        with open(train_solutions_path, 'r') as f:
            train_solutions = json.load(f)
        
        # Load evaluation data
        with open(eval_challenges_path, 'r') as f:
            eval_challenges = json.load(f)
        
        with open(eval_solutions_path, 'r') as f:
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
                raise FileNotFoundError(f"Concept training challenges not found: {concept_train_challenges_path}")
            if not concept_train_solutions_path.exists():
                raise FileNotFoundError(f"Concept training solutions not found: {concept_train_solutions_path}")
            if not concept_eval_challenges_path.exists():
                raise FileNotFoundError(f"Concept evaluation challenges not found: {concept_eval_challenges_path}")
            if not concept_eval_solutions_path.exists():
                raise FileNotFoundError(f"Concept evaluation solutions not found: {concept_eval_solutions_path}")
            
            # Load training data
            with open(concept_train_challenges_path, 'r') as f:
                train_challenges.update(json.load(f))
            
            with open(concept_train_solutions_path, 'r') as f:
                train_solutions.update(json.load(f))
            
            # Load evaluation data
            with open(concept_eval_challenges_path, 'r') as f:
                eval_challenges.update(json.load(f))
            
            with open(concept_eval_solutions_path, 'r') as f:
                eval_solutions.update(json.load(f))

        # Create datasets
        self.train_dataset = ARCTaskDataset(
            train_challenges,
            train_solutions,
            transform=self.train_transform,
            samples_per_task=self.samples_per_task if self.augment_train else 1
        )
        
        self.val_dataset = ARCTaskDataset(
            eval_challenges,
            eval_solutions,
            transform=self.val_transform,
            samples_per_task=1  # No augmentation for validation
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
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        # For testing, use evaluation dataset
        return self.val_dataloader()


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching ARC samples.
    Handles variable-size outputs and missing values.
    """
    # Stack inputs
    inputs = torch.stack([sample['input'] for sample in batch])
    
    # Create batch dict
    batch_dict = {'input': inputs}
    
    # Stack outputs if all samples have them
    if all('output' in sample for sample in batch):
        outputs = torch.stack([sample['output'] for sample in batch])
        batch_dict['output'] = outputs
    
    # Add other fields
    if 'task_id' in batch[0]:
        batch_dict['task_ids'] = [sample['task_id'] for sample in batch]
    
    if 'original_shape' in batch[0]:
        batch_dict['original_shapes'] = torch.stack([sample['original_shape'] for sample in batch])
    
    return batch_dict


# Example usage with custom collate
class ARCDataModuleWithMetadata(ARCDataModule):
    """Extended DataModule with custom collate function."""
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )