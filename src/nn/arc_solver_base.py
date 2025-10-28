"""
ARC-AGI Solver Interface
Base classes and interfaces for implementing different solving methods
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SolverResult:
    """Result from solving a task."""

    predictions: List[List[List[int]]]  # The predicted outputs
    metadata: Dict[str, Any] = None  # Additional info (confidence, pattern type, etc.)
    solve_time: float = 0.0  # Time taken to solve in seconds

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ARCSolver(ABC):
    """Abstract base class for ARC solvers."""

    def __init__(self, name: str = "UnnamedSolver"):
        """Initialize the solver.

        Args:
            name: Name of the solver for identification
        """
        self.name = name
        self.is_trained = False

    @abstractmethod
    def solve_task(self, task_data: Dict) -> SolverResult:
        """Solve a single ARC task.

        Args:
            task_data: Dictionary with 'train' and 'test' keys

        Returns:
            SolverResult with predictions and metadata
        """
        pass

    def train(
        self,
        training_data: Dict[str, Dict],
        validation_data: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the solver (for ML-based methods).

        Args:
            training_data: Dictionary of task_id -> task_data
            validation_data: Optional validation set
            **kwargs: Additional training parameters

        Returns:
            Training metrics/history
        """
        # Default implementation for non-ML methods
        self.is_trained = True
        return {"message": f"{self.name} does not require training"}

    def save(self, path: str) -> None:
        """Save solver state/model.

        Args:
            path: Path to save the solver
        """
        # Default implementation
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        config = {
            "solver_name": self.name,
            "is_trained": self.is_trained,
            "solver_type": self.__class__.__name__,
        }

        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load(self, path: str) -> None:
        """Load solver state/model.

        Args:
            path: Path to load the solver from
        """
        # Default implementation
        load_path = Path(path)
        config_file = load_path / "config.json"

        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                self.is_trained = config.get("is_trained", False)

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class BaselineSolver(ARCSolver):
    """Wrapper for the existing baseline solver."""

    def __init__(self, name: str = "BaselinePatternMatcher"):
        super().__init__(name)
        # Import here to avoid circular dependency
        from arc_baseline import ARCBaseline, TransformationPattern

        self.baseline = ARCBaseline()
        self.TransformationPattern = TransformationPattern

    def solve_task(self, task_data: Dict) -> SolverResult:
        """Solve using the baseline pattern matching approach."""
        start_time = time.time()

        # Get predictions using baseline method
        predictions = self.baseline.solve_task(task_data)

        # Get pattern information for metadata
        train_examples = task_data.get("train", [])
        pattern = self.baseline.analyze_transformation(train_examples)

        metadata = {
            "pattern_type": pattern.pattern_type,
            "confidence": pattern.confidence,
            "pattern_params": pattern.params,
        }

        solve_time = time.time() - start_time

        return SolverResult(predictions=predictions, metadata=metadata, solve_time=solve_time)


class MLSolver(ARCSolver):
    """Base class for ML-based solvers."""

    def __init__(self, name: str = "MLSolver", model_type: str = "neural"):
        super().__init__(name)
        self.model_type = model_type
        self.model = None
        self.preprocessing_params = {}

    def preprocess_task(self, task_data: Dict) -> Any:
        """Preprocess task data for ML model.

        Override this method in subclasses.

        Args:
            task_data: Raw task dictionary

        Returns:
            Preprocessed data suitable for the model
        """
        # Default: convert to numpy arrays
        train_examples = task_data.get("train", [])
        test_examples = task_data.get("test", [])

        train_inputs = [np.array(ex["input"]) for ex in train_examples]
        train_outputs = [np.array(ex["output"]) for ex in train_examples]
        test_inputs = [np.array(ex["input"]) for ex in test_examples]

        return {
            "train_inputs": train_inputs,
            "train_outputs": train_outputs,
            "test_inputs": test_inputs,
        }

    def postprocess_predictions(self, raw_predictions: Any) -> List[List[List[int]]]:
        """Convert model output to ARC format.

        Override this method in subclasses.

        Args:
            raw_predictions: Raw model output

        Returns:
            List of 2D grids (lists of lists of ints)
        """
        # Default: assume predictions are numpy arrays
        predictions = []
        for pred in raw_predictions:
            if isinstance(pred, np.ndarray):
                predictions.append(pred.astype(int).tolist())
            else:
                predictions.append(pred)
        return predictions

    @abstractmethod
    def predict(self, preprocessed_data: Any) -> Any:
        """Make predictions using the model.

        Args:
            preprocessed_data: Output from preprocess_task

        Returns:
            Raw model predictions
        """
        pass

    def solve_task(self, task_data: Dict) -> SolverResult:
        """Solve using ML approach."""
        start_time = time.time()

        # Preprocess
        preprocessed = self.preprocess_task(task_data)

        # Predict
        raw_predictions = self.predict(preprocessed)

        # Postprocess
        predictions = self.postprocess_predictions(raw_predictions)

        solve_time = time.time() - start_time

        metadata = {"model_type": self.model_type, "is_trained": self.is_trained}

        return SolverResult(predictions=predictions, metadata=metadata, solve_time=solve_time)


class EnsembleSolver(ARCSolver):
    """Combine multiple solvers using ensemble methods."""

    def __init__(self, solvers: List[ARCSolver], name: str = "Ensemble", strategy: str = "voting"):
        """Initialize ensemble solver.

        Args:
            solvers: List of solvers to combine
            name: Name of the ensemble
            strategy: How to combine predictions ('voting', 'weighted', 'selection')
        """
        super().__init__(name)
        self.solvers = solvers
        self.strategy = strategy
        self.weights = [1.0] * len(solvers)  # Equal weights by default

    def solve_task(self, task_data: Dict) -> SolverResult:
        """Solve using ensemble of solvers."""
        start_time = time.time()

        # Get predictions from all solvers
        all_results = []
        for solver in self.solvers:
            try:
                result = solver.solve_task(task_data)
                all_results.append(result)
            except Exception as e:
                print(f"Warning: {solver.name} failed: {e}")
                continue

        if not all_results:
            # No solver succeeded
            test_examples = task_data.get("test", [])
            predictions = [[[0]] for _ in test_examples]  # Dummy predictions
            return SolverResult(predictions=predictions)

        # Combine predictions based on strategy
        if self.strategy == "voting":
            predictions = self._voting_combine(all_results)
        elif self.strategy == "weighted":
            predictions = self._weighted_combine(all_results)
        elif self.strategy == "selection":
            predictions = self._selection_combine(all_results, task_data)
        else:
            predictions = all_results[0].predictions  # Default to first solver

        solve_time = time.time() - start_time

        metadata = {
            "strategy": self.strategy,
            "num_solvers": len(self.solvers),
            "solver_names": [s.name for s in self.solvers],
        }

        return SolverResult(predictions=predictions, metadata=metadata, solve_time=solve_time)

    def _voting_combine(self, results: List[SolverResult]) -> List[List[List[int]]]:
        """Combine by pixel-wise majority voting."""
        if len(results) == 1:
            return results[0].predictions

        # Get all predictions
        all_predictions = [r.predictions for r in results]
        num_tests = len(all_predictions[0])

        combined_predictions = []
        for test_idx in range(num_tests):
            # Get predictions for this test from all solvers
            test_preds = []
            for solver_preds in all_predictions:
                if test_idx < len(solver_preds):
                    test_preds.append(np.array(solver_preds[test_idx]))

            if not test_preds:
                combined_predictions.append([[0]])
                continue

            # Find most common shape
            shapes = [p.shape for p in test_preds]
            from collections import Counter

            most_common_shape = Counter(shapes).most_common(1)[0][0]

            # Filter predictions with matching shape
            matching_preds = [p for p in test_preds if p.shape == most_common_shape]

            if matching_preds:
                # Pixel-wise voting
                stacked = np.stack(matching_preds)
                # Use mode for each pixel
                from scipy import stats

                mode_result = stats.mode(stacked, axis=0)
                combined = mode_result.mode[0]
            else:
                combined = test_preds[0]  # Fallback to first prediction

            combined_predictions.append(combined.astype(int).tolist())

        return combined_predictions

    def _weighted_combine(self, results: List[SolverResult]) -> List[List[List[int]]]:
        """Combine using weighted average (for continuous predictions)."""
        # For discrete ARC tasks, this defaults to voting with weights
        return self._voting_combine(results)

    def _selection_combine(
        self, results: List[SolverResult], task_data: Dict
    ) -> List[List[List[int]]]:
        """Select best solver based on confidence or pattern matching."""
        # Select solver with highest confidence
        best_idx = 0
        best_confidence = 0.0

        for i, result in enumerate(results):
            confidence = result.metadata.get("confidence", 0.0)
            if confidence > best_confidence:
                best_confidence = confidence
                best_idx = i

        return results[best_idx].predictions

    def train(
        self,
        training_data: Dict[str, Dict],
        validation_data: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train all component solvers."""
        training_results = {}

        for solver in self.solvers:
            print(f"Training {solver.name}...")
            result = solver.train(training_data, validation_data, **kwargs)
            training_results[solver.name] = result

        self.is_trained = all(s.is_trained for s in self.solvers)

        return training_results


# Example custom ML solver implementation
class SimpleNeuralSolver(MLSolver):
    """Example neural network solver for ARC tasks."""

    def __init__(self, name: str = "SimpleNeuralNet"):
        super().__init__(name, model_type="neural")
        self.max_grid_size = 30
        self.num_colors = 10

    def predict(self, preprocessed_data: Any) -> Any:
        """Make predictions using a neural network."""
        # This is a placeholder - implement your actual model here
        test_inputs = preprocessed_data["test_inputs"]

        # For demo: just return the input as output
        # Replace with actual model predictions
        predictions = []
        for test_input in test_inputs:
            # Your ML model would go here
            # prediction = self.model.predict(test_input)
            prediction = test_input  # Placeholder
            predictions.append(prediction)

        return predictions

    def train(
        self,
        training_data: Dict[str, Dict],
        validation_data: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the neural network."""
        # Implement your training logic here
        print(f"Training {self.name}...")

        # Placeholder training loop
        epochs = kwargs.get("epochs", 10)
        batch_size = kwargs.get("batch_size", 32)

        history = {
            "loss": [0.5 - i * 0.05 for i in range(epochs)],
            "val_loss": [0.6 - i * 0.04 for i in range(epochs)],
        }

        self.is_trained = True
        return history
