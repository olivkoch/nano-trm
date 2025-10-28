"""
ARC-AGI Solver Evaluator
Framework for evaluating and comparing different solving methods
"""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.arc_solver_base import ARCSolver


@dataclass
class EvaluationResult:
    """Results from evaluating a solver on a dataset."""

    solver_name: str
    dataset_name: str
    total_tasks: int
    total_tests: int
    correct_tests: int
    perfect_tasks: int  # Tasks with all test cases correct
    test_accuracy: float  # Percentage of correct test cases
    task_accuracy: float  # Percentage of perfectly solved tasks
    avg_solve_time: float  # Average time per task in seconds
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        return (
            f"Solver: {self.solver_name}\n"
            f"Dataset: {self.dataset_name}\n"
            f"Tasks: {self.perfect_tasks}/{self.total_tasks} perfect ({self.task_accuracy:.1f}%)\n"
            f"Tests: {self.correct_tests}/{self.total_tests} correct ({self.test_accuracy:.1f}%)\n"
            f"Avg time: {self.avg_solve_time:.3f}s"
        )


class ARCEvaluator:
    """Evaluate and compare ARC solvers."""

    def __init__(self, data_dir: str = "data"):
        """Initialize evaluator.

        Args:
            data_dir: Directory containing ARC data files
        """
        self.data_dir = Path(data_dir)
        self.datasets = {}
        self.results = []
        self.load_datasets()

    def load_datasets(self) -> None:
        """Load available datasets."""
        dataset_files = {
            "training": ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json"),
            "evaluation": (
                "arc-agi_evaluation_challenges.json",
                "arc-agi_evaluation_solutions.json",
            ),
        }

        for dataset_name, (challenges_file, solutions_file) in dataset_files.items():
            challenges_path = self.data_dir / challenges_file
            solutions_path = self.data_dir / solutions_file

            if challenges_path.exists() and solutions_path.exists():
                with open(challenges_path) as f:
                    challenges = json.load(f)
                with open(solutions_path) as f:
                    solutions = json.load(f)

                self.datasets[dataset_name] = {"challenges": challenges, "solutions": solutions}
                print(f"✓ Loaded {dataset_name}: {len(challenges)} tasks")
            else:
                print(f"✗ Dataset {dataset_name} not found")

    def evaluate_solver(
        self,
        solver: ARCSolver,
        dataset: str = "training",
        max_tasks: Optional[int] = None,
        verbose: bool = True,
    ) -> EvaluationResult:
        """Evaluate a solver on a dataset.

        Args:
            solver: Solver to evaluate
            dataset: Dataset name ('training' or 'evaluation')
            max_tasks: Maximum number of tasks to evaluate
            verbose: Whether to print progress

        Returns:
            EvaluationResult with performance metrics
        """
        if dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not loaded")

        challenges = self.datasets[dataset]["challenges"]
        solutions = self.datasets[dataset]["solutions"]

        task_ids = list(challenges.keys())
        if max_tasks:
            task_ids = task_ids[:max_tasks]

        if verbose:
            print(f"\n🔍 Evaluating {solver.name} on {dataset} ({len(task_ids)} tasks)...")

        total_tasks = 0
        total_tests = 0
        correct_tests = 0
        perfect_tasks = 0
        solve_times = []

        task_results = []

        for i, task_id in enumerate(task_ids):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(task_ids)} tasks...")

            task_data = challenges[task_id]
            true_solutions = solutions[task_id]

            # Solve the task
            try:
                result = solver.solve_task(task_data)
                predictions = result.predictions
                solve_times.append(result.solve_time)
            except Exception as e:
                print(f"  Error on task {task_id}: {e}")
                predictions = []
                solve_times.append(0)

            # Evaluate predictions
            task_correct = 0
            task_total = len(true_solutions)

            for j, true_sol in enumerate(true_solutions):
                if j < len(predictions):
                    # Handle solution format
                    if isinstance(true_sol, dict) and "output" in true_sol:
                        true_output = true_sol["output"]
                    else:
                        true_output = true_sol

                    # Compare prediction with ground truth
                    if np.array_equal(predictions[j], true_output):
                        task_correct += 1
                        correct_tests += 1

                total_tests += 1

            # Track task-level performance
            total_tasks += 1
            if task_correct == task_total and task_total > 0:
                perfect_tasks += 1

            task_results.append(
                {
                    "task_id": task_id,
                    "correct": task_correct,
                    "total": task_total,
                    "is_perfect": task_correct == task_total,
                }
            )

        # Calculate metrics
        test_accuracy = (correct_tests / total_tests * 100) if total_tests > 0 else 0
        task_accuracy = (perfect_tasks / total_tasks * 100) if total_tasks > 0 else 0
        avg_solve_time = np.mean(solve_times) if solve_times else 0

        result = EvaluationResult(
            solver_name=solver.name,
            dataset_name=dataset,
            total_tasks=total_tasks,
            total_tests=total_tests,
            correct_tests=correct_tests,
            perfect_tasks=perfect_tasks,
            test_accuracy=test_accuracy,
            task_accuracy=task_accuracy,
            avg_solve_time=avg_solve_time,
            metadata={"task_results": task_results},
        )

        self.results.append(result)

        if verbose:
            print(f"\n{result}")

        return result

    def compare_solvers(
        self, solvers: List[ARCSolver], dataset: str = "training", max_tasks: Optional[int] = None
    ) -> pd.DataFrame:
        """Compare multiple solvers on the same dataset.

        Args:
            solvers: List of solvers to compare
            dataset: Dataset to evaluate on
            max_tasks: Maximum number of tasks

        Returns:
            DataFrame with comparison results
        """
        print(f"\n📊 Comparing {len(solvers)} solvers on {dataset} dataset...")

        comparison_results = []

        for solver in solvers:
            result = self.evaluate_solver(solver, dataset, max_tasks, verbose=False)
            comparison_results.append(
                {
                    "Solver": solver.name,
                    "Task Accuracy (%)": f"{result.task_accuracy:.1f}",
                    "Test Accuracy (%)": f"{result.test_accuracy:.1f}",
                    "Perfect Tasks": f"{result.perfect_tasks}/{result.total_tasks}",
                    "Correct Tests": f"{result.correct_tests}/{result.total_tests}",
                    "Avg Time (s)": f"{result.avg_solve_time:.3f}",
                }
            )
            print(
                f"  ✓ {solver.name}: {result.task_accuracy:.1f}% tasks, {result.test_accuracy:.1f}% tests"
            )

        df = pd.DataFrame(comparison_results)
        return df

    def plot_comparison(self, results: Optional[List[EvaluationResult]] = None) -> None:
        """Create visualization comparing solver performance.

        Args:
            results: List of evaluation results (uses self.results if None)
        """
        if results is None:
            results = self.results

        if not results:
            print("No results to plot!")
            return

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Prepare data
        solver_names = [r.solver_name for r in results]
        task_accs = [r.task_accuracy for r in results]
        test_accs = [r.test_accuracy for r in results]
        solve_times = [r.avg_solve_time for r in results]

        # 1. Task Accuracy Bar Chart
        ax = axes[0, 0]
        bars = ax.bar(solver_names, task_accs, color="steelblue")
        ax.set_ylabel("Task Accuracy (%)")
        ax.set_title("Perfect Task Solve Rate")
        ax.set_ylim(0, max(100, max(task_accs) * 1.1))
        for bar, val in zip(bars, task_accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.1f}%", ha="center"
            )

        # 2. Test Accuracy Bar Chart
        ax = axes[0, 1]
        bars = ax.bar(solver_names, test_accs, color="coral")
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title("Individual Test Case Accuracy")
        ax.set_ylim(0, max(100, max(test_accs) * 1.1))
        for bar, val in zip(bars, test_accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.1f}%", ha="center"
            )

        # 3. Solve Time Comparison
        ax = axes[1, 0]
        bars = ax.bar(solver_names, solve_times, color="green")
        ax.set_ylabel("Average Time (seconds)")
        ax.set_title("Solving Speed")
        for bar, val in zip(bars, solve_times):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}s",
                ha="center",
            )

        # 4. Combined Metric (Efficiency Score)
        ax = axes[1, 1]
        # Efficiency = accuracy / log(time + 1)
        efficiency = [acc / (np.log(t + 1) + 1) for acc, t in zip(task_accs, solve_times)]
        bars = ax.bar(solver_names, efficiency, color="purple")
        ax.set_ylabel("Efficiency Score")
        ax.set_title("Accuracy / Log(Time) Trade-off")
        for bar, val in zip(bars, efficiency):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}", ha="center"
            )

        plt.suptitle("Solver Performance Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def analyze_errors(
        self, solver: ARCSolver, dataset: str = "training", n_samples: int = 5
    ) -> Dict[str, Any]:
        """Analyze error patterns for a solver.

        Args:
            solver: Solver to analyze
            dataset: Dataset to use
            n_samples: Number of error examples to return

        Returns:
            Dictionary with error analysis
        """
        if dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not loaded")

        challenges = self.datasets[dataset]["challenges"]
        solutions = self.datasets[dataset]["solutions"]

        errors = []
        error_types = defaultdict(int)

        for task_id in list(challenges.keys())[:100]:  # Analyze first 100 tasks
            task_data = challenges[task_id]
            true_solutions = solutions[task_id]

            try:
                result = solver.solve_task(task_data)
                predictions = result.predictions
            except Exception:
                continue

            for i, true_sol in enumerate(true_solutions):
                if i < len(predictions):
                    if isinstance(true_sol, dict) and "output" in true_sol:
                        true_output = np.array(true_sol["output"])
                    else:
                        true_output = np.array(true_sol)

                    pred_output = np.array(predictions[i])

                    if not np.array_equal(pred_output, true_output):
                        # Analyze error type
                        error_info = {
                            "task_id": task_id,
                            "test_idx": i,
                            "pred_shape": pred_output.shape,
                            "true_shape": true_output.shape,
                            "shape_mismatch": pred_output.shape != true_output.shape,
                        }

                        if pred_output.shape == true_output.shape:
                            diff_pixels = np.sum(pred_output != true_output)
                            total_pixels = pred_output.size
                            error_info["pixel_error_rate"] = diff_pixels / total_pixels
                            error_types["pixel_errors"] += 1
                        else:
                            error_types["shape_errors"] += 1

                        errors.append(error_info)

        # Get sample errors
        sample_errors = errors[:n_samples]

        return {
            "total_errors": len(errors),
            "error_types": dict(error_types),
            "sample_errors": sample_errors,
        }

    def save_results(self, filepath: str = "evaluation_results.json") -> None:
        """Save evaluation results to file.

        Args:
            filepath: Path to save results
        """
        results_data = [r.to_dict() for r in self.results]

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"✓ Saved {len(results_data)} results to {filepath}")

    def load_results(self, filepath: str = "evaluation_results.json") -> None:
        """Load evaluation results from file.

        Args:
            filepath: Path to load results from
        """
        with open(filepath) as f:
            results_data = json.load(f)

        self.results = [EvaluationResult(**r) for r in results_data]
        print(f"✓ Loaded {len(self.results)} results from {filepath}")
