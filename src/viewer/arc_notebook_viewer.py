"""
ARC-AGI Notebook Viewer for Neural Network Models
Interactive visualization for Jupyter notebooks with PyTorch model support
"""

"""
ARC-AGI Notebook Viewer for Neural Network Models
Interactive visualization for Jupyter notebooks with PyTorch model support
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output
from matplotlib.colors import ListedColormap, Normalize

# Import the DataModule
from src.nn.data.arc_datamodule import ARCDataModule

# ARC-AGI color palette (0-9)
ARC_COLORS = [
    "#000000",  # 0: Black
    "#0074D9",  # 1: Blue
    "#FF4136",  # 2: Red
    "#2ECC40",  # 3: Green
    "#FFDC00",  # 4: Yellow
    "#AAAAAA",  # 5: Gray
    "#F012BE",  # 6: Magenta
    "#FF851B",  # 7: Orange
    "#7FDBFF",  # 8: Sky blue
    "#870C25",  # 9: Maroon
]


class ARCNotebookViewer:
    """Interactive viewer for ARC-AGI tasks in Jupyter notebooks."""

    def __init__(
        self,
        data_dir: str = "data",
        use_concept_data: bool = False,
        concept_data_dir: str = "data/concept",
    ):
        """Initialize the viewer.

        Args:
            data_dir: Directory containing the JSON files
            use_concept_data: If True, also load concept dataset
            concept_data_dir: Directory containing concept dataset
        """
        self.data_dir = Path(data_dir)
        self.use_concept_data = use_concept_data
        self.concept_data_dir = concept_data_dir

        # Data storage
        self.training_challenges = None
        self.training_solutions = None
        self.evaluation_challenges = None
        self.evaluation_solutions = None

        # DataModule
        self.datamodule = ARCDataModule(
            data_dir=str(data_dir),
            batch_size=1,  # Not used for visualization
            use_concept_data=use_concept_data,
            concept_data_dir=concept_data_dir,
        )

        # Create colormap
        self.cmap = ListedColormap(ARC_COLORS)
        self.norm = Normalize(vmin=0, vmax=9)

        # Widget references
        self.task_dropdown = None
        self.dataset_dropdown = None
        self.output_area = None

    def load_data(self, verbose: bool = True) -> Dict:
        """Load all available data files using ARCDataModule.

        Args:
            verbose: Whether to print loading information

        Returns:
            Dictionary with loaded dataset statistics
        """
        if verbose:
            print("Loading data via ARCDataModule...")

        # Setup the datamodule (this loads the JSON files)
        self.datamodule.setup()

        # Extract the raw challenge/solution dicts from the datamodule
        train_dataset = self.datamodule.train_dataset
        val_dataset = self.datamodule.val_dataset

        # Get the original tasks and solutions
        self.training_challenges = train_dataset.tasks
        self.training_solutions = train_dataset.solutions
        self.evaluation_challenges = val_dataset.tasks
        self.evaluation_solutions = val_dataset.solutions

        loaded = {
            "training_challenges": len(self.training_challenges),
            "training_solutions": len(self.training_solutions) if self.training_solutions else 0,
            "evaluation_challenges": len(self.evaluation_challenges),
            "evaluation_solutions": len(self.evaluation_solutions)
            if self.evaluation_solutions
            else 0,
        }

        if verbose:
            print(f"✓ Loaded training set: {loaded['training_challenges']} tasks")
            if self.training_solutions:
                print(f"✓ Loaded training solutions: {loaded['training_solutions']} tasks")
            print(f"✓ Loaded evaluation set: {loaded['evaluation_challenges']} tasks")
            if self.evaluation_solutions:
                print(f"✓ Loaded evaluation solutions: {loaded['evaluation_solutions']} tasks")

        return loaded

    def plot_grid(
        self, grid: Union[List[List[int]], np.ndarray], title: str = "", cell_size: float = 0.3
    ) -> plt.Figure:
        """Plot a single grid as a matplotlib figure.

        Args:
            grid: 2D array of integers (0-9)
            title: Title for the plot
            cell_size: Size of each cell in inches

        Returns:
            Matplotlib figure
        """
        grid_array = np.array(grid)
        height, width = grid_array.shape

        fig, ax = plt.subplots(figsize=(width * cell_size, height * cell_size))

        # Plot the grid
        im = ax.imshow(grid_array, cmap=self.cmap, norm=self.norm, interpolation="nearest")

        # Add grid lines
        for i in range(height + 1):
            ax.axhline(i - 0.5, color="white", linewidth=0.5)
        for j in range(width + 1):
            ax.axvline(j - 0.5, color="white", linewidth=0.5)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)

        plt.tight_layout()
        return fig

    def plot_task(
        self,
        task_id: str,
        task_data: Dict,
        solution: Optional[Union[List, Dict]] = None,
        predictions: Optional[List[Union[List[List[int]], np.ndarray]]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> plt.Figure:
        """Create a comprehensive visualization of a task.

        Args:
            task_id: Task identifier
            task_data: Task dictionary with 'train' and 'test' keys
            solution: Ground truth solution (if available)
            predictions: Model predictions as list of 2D arrays (if available)
            figsize: Figure size (width, height)

        Returns:
            Matplotlib figure
        """
        train_examples = task_data.get("train", [])
        test_examples = task_data.get("test", [])

        n_train = len(train_examples)
        n_test = len(test_examples)

        # Determine number of columns based on what we're showing
        n_cols = 2  # Input and Output
        if predictions is not None and solution is not None:
            n_cols = 4  # Input, Prediction, Ground Truth, Correct?
        elif predictions is not None or solution is not None:
            n_cols = 3  # Input, Output/Prediction, (Ground Truth or Prediction)

        # Calculate figure size if not provided
        if figsize is None:
            figsize = (4 * n_cols, 3 * (n_train + n_test))

        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"Task: {task_id}", fontsize=14, fontweight="bold")

        # Create grid layout
        total_rows = n_train + n_test

        # Plot training examples
        for i, example in enumerate(train_examples):
            # Input
            ax_input = plt.subplot(total_rows, n_cols, i * n_cols + 1)
            self._plot_grid_on_axes(example["input"], ax_input, f"Train {i + 1} - Input")

            # Output
            ax_output = plt.subplot(total_rows, n_cols, i * n_cols + 2)
            self._plot_grid_on_axes(example["output"], ax_output, f"Train {i + 1} - Output")

            # Fill remaining columns if needed
            if n_cols > 2:
                for col in range(3, n_cols + 1):
                    ax = plt.subplot(total_rows, n_cols, i * n_cols + col)
                    ax.axis("off")

        # Plot test examples
        for i, example in enumerate(test_examples):
            row_idx = n_train + i

            # Test input
            ax_input = plt.subplot(total_rows, n_cols, row_idx * n_cols + 1)
            self._plot_grid_on_axes(example["input"], ax_input, f"Test {i + 1} - Input")

            # Handle predictions and solutions
            if predictions is not None and solution is not None:
                # Show both prediction and ground truth

                # Prediction
                ax_pred = plt.subplot(total_rows, n_cols, row_idx * n_cols + 2)
                if i < len(predictions):
                    pred = np.array(predictions[i])
                    self._plot_grid_on_axes(pred.tolist(), ax_pred, f"Test {i + 1} - Prediction")
                else:
                    self._plot_placeholder(ax_pred, f"Test {i + 1} - No Prediction")

                # Ground Truth
                ax_truth = plt.subplot(total_rows, n_cols, row_idx * n_cols + 3)
                try:
                    if isinstance(solution, list) and i < len(solution):
                        if isinstance(solution[i], dict) and "output" in solution[i]:
                            truth = solution[i]["output"]
                        else:
                            truth = solution[i]
                        self._plot_grid_on_axes(truth, ax_truth, f"Test {i + 1} - Ground Truth")
                    else:
                        self._plot_placeholder(ax_truth, f"Test {i + 1} - Unknown")
                except:
                    self._plot_placeholder(ax_truth, f"Test {i + 1} - Unknown")

                # Correctness indicator
                ax_check = plt.subplot(total_rows, n_cols, row_idx * n_cols + 4)
                try:
                    if isinstance(solution, list) and i < len(solution) and i < len(predictions):
                        if isinstance(solution[i], dict) and "output" in solution[i]:
                            truth = solution[i]["output"]
                        else:
                            truth = solution[i]

                        pred = np.array(predictions[i])
                        truth_arr = np.array(truth)
                        is_correct = np.array_equal(pred, truth_arr)
                        color = "#2ECC40" if is_correct else "#FF4136"
                        symbol = "✓" if is_correct else "✗"

                        ax_check.text(
                            0.5, 0.5, symbol, fontsize=40, ha="center", va="center", color=color
                        )
                        ax_check.set_xlim(0, 1)
                        ax_check.set_ylim(0, 1)
                        ax_check.axis("off")
                        ax_check.set_title(
                            "Correct" if is_correct else "Wrong", color=color, fontweight="bold"
                        )
                    else:
                        ax_check.axis("off")
                except:
                    ax_check.axis("off")

            elif predictions is not None:
                # Show only predictions
                ax_pred = plt.subplot(total_rows, n_cols, row_idx * n_cols + 2)
                if i < len(predictions):
                    pred = np.array(predictions[i])
                    self._plot_grid_on_axes(pred.tolist(), ax_pred, f"Test {i + 1} - Prediction")
                else:
                    self._plot_placeholder(ax_pred, f"Test {i + 1} - No Prediction")

                if n_cols > 2:
                    for col in range(3, n_cols + 1):
                        ax = plt.subplot(total_rows, n_cols, row_idx * n_cols + col)
                        ax.axis("off")

            elif solution is not None:
                # Show only ground truth
                ax_truth = plt.subplot(total_rows, n_cols, row_idx * n_cols + 2)
                try:
                    if isinstance(solution, list) and i < len(solution):
                        if isinstance(solution[i], dict) and "output" in solution[i]:
                            truth = solution[i]["output"]
                        else:
                            truth = solution[i]
                        self._plot_grid_on_axes(truth, ax_truth, f"Test {i + 1} - Ground Truth")
                    else:
                        self._plot_placeholder(ax_truth, f"Test {i + 1} - Unknown")
                except:
                    self._plot_placeholder(ax_truth, f"Test {i + 1} - Unknown")

                if n_cols > 2:
                    for col in range(3, n_cols + 1):
                        ax = plt.subplot(total_rows, n_cols, row_idx * n_cols + col)
                        ax.axis("off")
            else:
                # No predictions or solutions
                ax_output = plt.subplot(total_rows, n_cols, row_idx * n_cols + 2)
                self._plot_placeholder(ax_output, f"Test {i + 1} - To Predict")

        plt.tight_layout()
        return fig

    def _plot_grid_on_axes(self, grid: List[List[int]], ax: plt.Axes, title: str):
        """Helper to plot grid on existing axes."""
        grid_array = np.array(grid)

        im = ax.imshow(grid_array, cmap=self.cmap, norm=self.norm, interpolation="nearest")

        # Add grid lines
        height, width = grid_array.shape
        for i in range(height + 1):
            ax.axhline(i - 0.5, color="white", linewidth=0.5)
        for j in range(width + 1):
            ax.axvline(j - 0.5, color="white", linewidth=0.5)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")

    def _plot_placeholder(self, ax: plt.Axes, title: str):
        """Plot a placeholder for unknown outputs."""
        placeholder = np.full((3, 3), 5)  # Gray 3x3 grid
        self._plot_grid_on_axes(placeholder.tolist(), ax, title)

    def show_task(self, task_id: Optional[str] = None, dataset: str = "training") -> None:
        """Display a specific task or random task if none specified.

        Args:
            task_id: Task ID to display (None for random)
            dataset: Which dataset ('training' or 'evaluation')
        """
        # Get challenges and solutions
        if dataset == "training":
            challenges = self.training_challenges
            solutions = self.training_solutions
        else:
            challenges = self.evaluation_challenges
            solutions = self.evaluation_solutions

        if challenges is None:
            print(f"No {dataset} data loaded! Call load_data() first.")
            return

        # Get task ID
        if task_id is None:
            task_id = random.choice(list(challenges.keys()))
        elif task_id not in challenges:
            print(f"Task {task_id} not found in {dataset} set!")
            return

        # Get data
        task_data = challenges[task_id]
        solution = solutions.get(task_id) if solutions else None

        # Create and display plot
        fig = self.plot_task(task_id, task_data, solution)
        plt.show()

        # Print statistics
        self._print_task_stats(task_id, task_data, solution)

    def _print_task_stats(
        self, task_id: str, task_data: Dict, solution: Optional[Union[List, Dict]]
    ):
        """Print statistics about a task."""
        print(f"\n📊 Task Statistics for {task_id}:")
        print(f"  • Training examples: {len(task_data.get('train', []))}")
        print(f"  • Test examples: {len(task_data.get('test', []))}")

        if task_data.get("train"):
            train_ex = task_data["train"][0]
            print(f"  • First training input shape: {np.array(train_ex['input']).shape}")
            print(f"  • First training output shape: {np.array(train_ex['output']).shape}")

            # Color analysis
            all_colors_in = set()
            all_colors_out = set()
            for ex in task_data["train"]:
                all_colors_in.update(np.unique(ex["input"]).tolist())
                all_colors_out.update(np.unique(ex["output"]).tolist())
            print(f"  • Colors used in inputs: {sorted(all_colors_in)}")
            print(f"  • Colors used in outputs: {sorted(all_colors_out)}")

    def predict_with_model(
        self, model: torch.nn.Module, task_data: Dict, device: str = "cpu"
    ) -> List[np.ndarray]:
        """Generate predictions for a task using a PyTorch model.

        Args:
            model: PyTorch model (e.g., TRMModule)
            task_data: Task dictionary with 'test' key
            device: Device to run on

        Returns:
            List of predicted grids as numpy arrays
        """
        model.eval()
        model.to(device)

        predictions = []

        with torch.no_grad():
            for test_example in task_data.get("test", []):
                # Get input grid
                input_grid = np.array(test_example["input"])

                # Convert to tensor
                input_tensor = torch.from_numpy(input_grid).long().unsqueeze(0).to(device)

                # Get logits from model
                logits = model(input_tensor)  # [1, H, W, num_colors]

                # Get predicted classes
                pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # [H, W]

                predictions.append(pred)

        return predictions

    def _evaluate_task_correctness(
        self, task_data: Dict, predictions: List[np.ndarray], solution: Optional[Union[List, Dict]]
    ) -> Dict:
        """Evaluate if predictions for a task are correct.

        Args:
            task_data: Task dictionary
            predictions: Model predictions
            solution: Ground truth solution

        Returns:
            Dictionary with correctness info
        """
        if not solution:
            return {
                "all_correct": False,
                "num_correct": 0,
                "num_total": len(predictions),
                "has_solution": False,
            }

        num_correct = 0
        num_total = len(predictions)

        for i, pred in enumerate(predictions):
            if i < len(solution):
                if isinstance(solution[i], dict) and "output" in solution[i]:
                    truth = solution[i]["output"]
                else:
                    truth = solution[i]

                truth_arr = np.array(truth)
                if np.array_equal(pred, truth_arr):
                    num_correct += 1

        return {
            "all_correct": num_correct == num_total,
            "num_correct": num_correct,
            "num_total": num_total,
            "has_solution": True,
        }

    def _filter_tasks_by_correctness(
        self,
        task_ids: List[str],
        challenges: Dict,
        solutions: Dict,
        model: torch.nn.Module,
        device: str,
        show_only_correct: bool,
        show_only_errors: bool,
        n_tasks: int,
    ) -> List[str]:
        """Filter tasks based on model correctness.

        Args:
            task_ids: List of task IDs to check
            challenges: Challenge dictionary
            solutions: Solutions dictionary
            model: PyTorch model
            device: Device to run on
            show_only_correct: Filter for correct predictions only
            show_only_errors: Filter for incorrect predictions only
            n_tasks: Target number of tasks to return

        Returns:
            Filtered list of task IDs
        """
        filtered_ids = []

        for task_id in task_ids:
            if task_id not in challenges:
                continue

            task_data = challenges[task_id]
            solution = solutions.get(task_id)

            if not solution:
                continue

            try:
                # Get predictions
                predictions = self.predict_with_model(model, task_data, device)

                # Evaluate correctness
                eval_result = self._evaluate_task_correctness(task_data, predictions, solution)

                # Add to filtered list based on criteria
                if show_only_correct and eval_result["all_correct"]:
                    filtered_ids.append(task_id)
                elif show_only_errors and not eval_result["all_correct"]:
                    filtered_ids.append(task_id)

                # Stop if we have enough
                if len(filtered_ids) >= n_tasks:
                    break

            except Exception:
                # Skip tasks that cause errors
                continue

        return filtered_ids

    def _select_tasks_to_visualize(
        self,
        task_ids: Optional[List[str]],
        challenges: Dict,
        solutions: Dict,
        model: Optional[torch.nn.Module],
        device: str,
        dataset: str,
        n_tasks: int,
        show_only_correct: bool,
        show_only_errors: bool,
    ) -> List[str]:
        """Select which tasks to visualize based on criteria.

        Args:
            task_ids: Specific task IDs or None for random selection
            challenges: Challenge dictionary
            solutions: Solutions dictionary
            model: PyTorch model (required if filtering by correctness)
            device: Device to run on
            dataset: Dataset name for messages
            n_tasks: Number of tasks to select
            show_only_correct: Only select tasks with perfect predictions
            show_only_errors: Only select tasks with errors

        Returns:
            List of task IDs to visualize
        """
        # If specific task IDs provided, return them
        if task_ids is not None:
            return task_ids

        available_ids = list(challenges.keys())

        # If no filtering, just sample randomly
        if not (show_only_correct or show_only_errors):
            return random.sample(available_ids, min(n_tasks, len(available_ids)))

        # Filtering requires solutions and model
        if not solutions:
            print(f"⚠️  No {dataset} solutions loaded! Cannot filter by correctness.")
            return random.sample(available_ids, min(n_tasks, len(available_ids)))

        if model is None:
            print("⚠️  Model required for filtering! Showing random tasks instead.")
            return random.sample(available_ids, min(n_tasks, len(available_ids)))

        # Sample more tasks to check for filtering
        check_size = min(len(available_ids), n_tasks * 10)
        task_ids_to_check = random.sample(available_ids, check_size)

        # Filter tasks
        filtered_ids = self._filter_tasks_by_correctness(
            task_ids_to_check,
            challenges,
            solutions,
            model,
            device,
            show_only_correct,
            show_only_errors,
            n_tasks,
        )

        # Handle case where no matching tasks found
        if not filtered_ids:
            filter_type = "correct" if show_only_correct else "with errors"
            print(f"⚠️  No tasks found {filter_type} in sample of {check_size} tasks!")
            print("Showing random tasks instead...")
            return random.sample(available_ids, min(n_tasks, len(available_ids)))

        # Report what we found
        if show_only_correct:
            print(f"✓ Found {len(filtered_ids)} tasks where model got all test examples correct")
        else:
            print(f"✗ Found {len(filtered_ids)} tasks where model made errors")

        return filtered_ids[:n_tasks]

    def _visualize_single_task(
        self,
        task_id: str,
        task_data: Dict,
        solution: Optional[Union[List, Dict]],
        predictions: List[np.ndarray],
        model_name: str,
    ) -> Dict:
        """Visualize a single task and return its metrics.

        Args:
            task_id: Task identifier
            task_data: Task data dictionary
            solution: Ground truth solution
            predictions: Model predictions
            model_name: Name of the model for display

        Returns:
            Dictionary with task metrics
        """
        # Create visualization
        fig = self.plot_task(task_id, task_data, solution, predictions)
        plt.show()

        # Evaluate and print accuracy
        eval_result = self._evaluate_task_correctness(task_data, predictions, solution)

        if eval_result["has_solution"]:
            num_correct = eval_result["num_correct"]
            num_total = eval_result["num_total"]
            accuracy = num_correct / num_total if num_total > 0 else 0

            status = "✓ PERFECT" if eval_result["all_correct"] else f"✗ {num_correct}/{num_total}"
            print(f"\n📊 Task {task_id} - {model_name}: {status} ({accuracy:.0%})")
        else:
            print(f"\n📊 Task {task_id} - {model_name}: No solution available")

        print("-" * 60)

        return eval_result

    def visualize_model_predictions(
        self,
        model: torch.nn.Module,
        task_ids: Optional[List[str]] = None,
        dataset: str = "training",
        n_tasks: int = 3,
        device: str = "cpu",
        show_only_correct: bool = False,
        show_only_errors: bool = False,
    ) -> Dict:
        """Visualize model predictions compared to ground truth.

        Args:
            model: PyTorch model (e.g., TRMModule)
            task_ids: Specific task IDs to visualize (None for random)
            dataset: Which dataset to use ('training' or 'evaluation')
            n_tasks: Number of tasks to visualize if task_ids is None
            device: Device to run model on
            show_only_correct: If True, only show tasks where model got ALL test examples correct
            show_only_errors: If True, only show tasks where model got at least one test example wrong

        Returns:
            Dictionary with summary statistics
        """
        # Ensure data is loaded
        if self.training_challenges is None:
            print("Loading data...")
            self.load_data()

        # Get data
        if dataset == "training":
            challenges = self.training_challenges
            solutions = self.training_solutions
        else:
            challenges = self.evaluation_challenges
            solutions = self.evaluation_solutions

        if not challenges:
            print(f"No {dataset} data loaded!")
            return {}

        # Select tasks to visualize
        selected_task_ids = self._select_tasks_to_visualize(
            task_ids,
            challenges,
            solutions,
            model,
            device,
            dataset,
            n_tasks,
            show_only_correct,
            show_only_errors,
        )

        # Visualize each task and collect metrics
        model_name = model.__class__.__name__
        total_shown = 0
        total_correct = 0
        task_results = []

        for task_id in selected_task_ids:
            if task_id not in challenges:
                print(f"⚠️  Task {task_id} not found in {dataset} set!")
                continue

            task_data = challenges[task_id]
            solution = solutions.get(task_id) if solutions else None

            # Get predictions from model
            try:
                predictions = self.predict_with_model(model, task_data, device)
            except Exception as e:
                print(f"❌ Error predicting task {task_id}: {e}")
                import traceback

                traceback.print_exc()
                continue

            # Visualize and get metrics
            eval_result = self._visualize_single_task(
                task_id, task_data, solution, predictions, model_name
            )

            # Track statistics
            if eval_result["has_solution"]:
                total_shown += 1
                if eval_result["all_correct"]:
                    total_correct += 1

                task_results.append(
                    {
                        "task_id": task_id,
                        "correct": eval_result["all_correct"],
                        "num_correct": eval_result["num_correct"],
                        "num_total": eval_result["num_total"],
                    }
                )

        # Print summary
        if total_shown > 0:
            print(f"\n{'=' * 60}")
            print(
                f"📊 Summary: {total_correct}/{total_shown} tasks completely correct "
                f"({total_correct / total_shown * 100:.1f}%)"
            )
            print(f"{'=' * 60}")

        return {
            "total_shown": total_shown,
            "total_correct": total_correct,
            "accuracy": total_correct / total_shown if total_shown > 0 else 0,
            "task_results": task_results,
        }

    def compare_predictions(
        self, model: torch.nn.Module, task_id: str, dataset: str = "training", device: str = "cpu"
    ) -> None:
        """Show detailed comparison of predictions vs ground truth for a single task.

        Args:
            model: PyTorch model
            task_id: Task ID to analyze
            dataset: Which dataset to use
            device: Device to run model on
        """
        # Ensure data is loaded
        if self.training_challenges is None:
            print("Loading data...")
            self.load_data()

        # Get data
        if dataset == "training":
            challenges = self.training_challenges
            solutions = self.training_solutions
        else:
            challenges = self.evaluation_challenges
            solutions = self.evaluation_solutions

        if not challenges or task_id not in challenges:
            print(f"Task {task_id} not found in {dataset} set!")
            return

        task_data = challenges[task_id]
        solution = solutions.get(task_id) if solutions else None

        # Get predictions
        predictions = self.predict_with_model(model, task_data, device)

        print(f"🔍 Analysis for Task {task_id}")
        print(f"   Model: {model.__class__.__name__}")

        # Create detailed visualization
        fig = self.plot_task(task_id, task_data, solution, predictions)
        plt.show()

        # Detailed comparison
        if solution:
            print("\n📊 Detailed Comparison:")
            for i, pred in enumerate(predictions):
                if i < len(solution):
                    if isinstance(solution[i], dict) and "output" in solution[i]:
                        truth = solution[i]["output"]
                    else:
                        truth = solution[i]

                    pred_arr = np.array(pred)
                    truth_arr = np.array(truth)

                    is_correct = np.array_equal(pred_arr, truth_arr)

                    print(f"\n   Test {i + 1}:")
                    print(f"     Prediction shape: {pred_arr.shape}")
                    print(f"     Ground truth shape: {truth_arr.shape}")
                    print(f"     Correct: {'✓' if is_correct else '✗'}")

                    if not is_correct:
                        # Show what's different
                        if pred_arr.shape == truth_arr.shape:
                            diff_count = np.sum(pred_arr != truth_arr)
                            total_pixels = pred_arr.size
                            print(
                                f"     Pixels different: {diff_count}/{total_pixels} ({diff_count / total_pixels * 100:.1f}%)"
                            )
                        else:
                            print("     Shape mismatch!")

    def evaluate_and_visualize(
        self,
        model: torch.nn.Module,
        dataset: str = "training",
        n_samples: int = 5,
        show_only_errors: bool = False,
        device: str = "cpu",
    ) -> Dict:
        """Evaluate model and visualize results.

        Args:
            model: PyTorch model
            dataset: Which dataset to use
            n_samples: Number of examples to visualize
            show_only_errors: If True, only show incorrect predictions
            device: Device to run model on

        Returns:
            Dictionary with evaluation metrics
        """
        if dataset == "training":
            challenges = self.training_challenges
            solutions = self.training_solutions
        else:
            challenges = self.evaluation_challenges
            solutions = self.evaluation_solutions

        if not challenges or not solutions:
            print(f"No {dataset} data with solutions loaded!")
            return {}

        print(f"🔍 Evaluating on {dataset} set...")

        # Collect results
        results = []
        for task_id in list(challenges.keys())[: n_samples * 3]:  # Check more to find errors
            task_data = challenges[task_id]
            predictions = self.predict_with_model(model, task_data, device)
            solution = solutions[task_id]

            # Check correctness
            all_correct = True
            for i, pred in enumerate(predictions):
                if i < len(solution):
                    if isinstance(solution[i], dict) and "output" in solution[i]:
                        truth = solution[i]["output"]
                    else:
                        truth = solution[i]

                    if not np.array_equal(pred, np.array(truth)):
                        all_correct = False
                        break

            results.append(
                {
                    "task_id": task_id,
                    "correct": all_correct,
                    "predictions": predictions,
                    "solution": solution,
                }
            )

        # Filter based on show_only_errors
        if show_only_errors:
            results_to_show = [r for r in results if not r["correct"]][:n_samples]
            if not results_to_show:
                print("No errors found in the sample!")
                results_to_show = results[:n_samples]
        else:
            results_to_show = results[:n_samples]

        # Summary statistics
        total_tasks = len(results)
        correct_tasks = sum(1 for r in results if r["correct"])
        print(
            f"\n📊 Overall: {correct_tasks}/{total_tasks} tasks fully correct ({correct_tasks / total_tasks * 100:.1f}%)"
        )

        # Visualize selected results
        print(f"\n📈 Showing {len(results_to_show)} examples:")
        for result in results_to_show:
            task_id = result["task_id"]
            task_data = challenges[task_id]

            status = "✓ Correct" if result["correct"] else "✗ Wrong"
            print(f"\nTask {task_id}: {status}")

            fig = self.plot_task(task_id, task_data, result["solution"], result["predictions"])
            plt.show()

        return {
            "total_tasks": total_tasks,
            "correct_tasks": correct_tasks,
            "accuracy": correct_tasks / total_tasks if total_tasks > 0 else 0,
        }

    def create_interactive_widget(self):
        """Create an interactive widget for exploring tasks.

        Returns:
            IPython widget box containing the interface
        """
        # Dataset selector
        self.dataset_dropdown = widgets.Dropdown(
            options=["training", "evaluation"],
            value="training",
            description="Dataset:",
            style={"description_width": "initial"},
        )

        # Task selector (will be populated based on dataset)
        self.task_dropdown = widgets.Dropdown(
            options=[], description="Task ID:", style={"description_width": "initial"}
        )

        # Buttons
        random_button = widgets.Button(
            description="🎲 Random Task", button_style="primary", tooltip="Show a random task"
        )

        refresh_button = widgets.Button(
            description="🔄 Refresh", button_style="info", tooltip="Refresh current task"
        )

        # Output area
        self.output_area = widgets.Output()

        # Event handlers
        def on_dataset_change(change):
            self._update_task_list()

        def on_task_change(change):
            if change["new"]:
                self._display_task()

        def on_random_click(b):
            tasks = list(self.task_dropdown.options)
            if tasks:
                self.task_dropdown.value = random.choice(tasks)

        def on_refresh_click(b):
            self._display_task()

        # Connect event handlers
        self.dataset_dropdown.observe(on_dataset_change, names="value")
        self.task_dropdown.observe(on_task_change, names="value")
        random_button.on_click(on_random_click)
        refresh_button.on_click(on_refresh_click)

        # Initial population
        self._update_task_list()

        # Layout
        controls = widgets.HBox(
            [self.dataset_dropdown, self.task_dropdown, random_button, refresh_button]
        )

        return widgets.VBox(
            [widgets.HTML("<h2>🧩 ARC-AGI Task Explorer</h2>"), controls, self.output_area]
        )

    def _update_task_list(self):
        """Update task dropdown based on selected dataset."""
        dataset = self.dataset_dropdown.value

        if dataset == "training" and self.training_challenges:
            tasks = sorted(self.training_challenges.keys())
        elif dataset == "evaluation" and self.evaluation_challenges:
            tasks = sorted(self.evaluation_challenges.keys())
        else:
            tasks = []

        self.task_dropdown.options = tasks
        if tasks:
            self.task_dropdown.value = tasks[0]

    def _display_task(self):
        """Display the selected task in the output area."""
        with self.output_area:
            clear_output(wait=True)

            dataset = self.dataset_dropdown.value
            task_id = self.task_dropdown.value

            if not task_id:
                print("No task selected")
                return

            # Get data
            if dataset == "training":
                challenges = self.training_challenges
                solutions = self.training_solutions
            else:
                challenges = self.evaluation_challenges
                solutions = self.evaluation_solutions

            if not challenges or task_id not in challenges:
                print(f"Task {task_id} not found")
                return

            task_data = challenges[task_id]
            solution = solutions.get(task_id) if solutions else None
