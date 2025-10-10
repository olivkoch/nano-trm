"""
ARC-AGI Notebook Viewer
Interactive visualization for Jupyter notebooks with widget support
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, Normalize
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed


# ARC-AGI color palette (0-9)
ARC_COLORS = [
    '#000000',  # 0: Black
    '#0074D9',  # 1: Blue
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Gray
    '#F012BE',  # 6: Magenta
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Sky blue
    '#870C25',  # 9: Maroon
]


class ARCNotebookViewer:
    """Interactive viewer for ARC-AGI tasks in Jupyter notebooks."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the viewer.
        
        Args:
            data_dir: Directory containing the JSON files
        """
        self.data_dir = Path(data_dir)
        self.training_challenges = None
        self.training_solutions = None
        self.evaluation_challenges = None
        self.evaluation_solutions = None
        self.test_challenges = None
        
        # Create colormap
        self.cmap = ListedColormap(ARC_COLORS)
        self.norm = Normalize(vmin=0, vmax=9)
        
        # Widget references
        self.task_dropdown = None
        self.dataset_dropdown = None
        self.output_area = None
        
    def load_data(self, verbose: bool = True) -> Dict:
        """Load all available data files.
        
        Args:
            verbose: Whether to print loading information
            
        Returns:
            Dictionary with loaded dataset statistics
        """
        files = {
            'training_challenges': 'arc-agi_training_challenges.json',
            'training_solutions': 'arc-agi_training_solutions.json',
            'evaluation_challenges': 'arc-agi_evaluation_challenges.json',
            'evaluation_solutions': 'arc-agi_evaluation_solutions.json',
            'test_challenges': 'arc-agi_test_challenges.json'
        }
        
        loaded = {}
        for key, filename in files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    setattr(self, key, data)
                    loaded[key] = len(data) if isinstance(data, dict) else 0
                    if verbose:
                        print(f"✓ Loaded {filename}: {loaded[key]} tasks")
            else:
                if verbose:
                    print(f"✗ Not found: {filename}")
        
        return loaded
    
    def plot_grid(self, grid: Union[List[List[int]], np.ndarray], 
                  title: str = "", cell_size: float = 0.3) -> plt.Figure:
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
        im = ax.imshow(grid_array, cmap=self.cmap, norm=self.norm, 
                      interpolation='nearest')
        
        # Add grid lines
        for i in range(height + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.5)
        for j in range(width + 1):
            ax.axvline(j - 0.5, color='white', linewidth=0.5)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_task(self, task_id: str, task_data: Dict, 
                  solution: Optional[Union[List, Dict]] = None,
                  predictions: Optional[List[List[List[int]]]] = None,
                  figsize: Optional[Tuple[float, float]] = None) -> plt.Figure:
        """Create a comprehensive visualization of a task.
        
        Args:
            task_id: Task identifier
            task_data: Task dictionary with 'train' and 'test' keys
            solution: Ground truth solution (if available)
            predictions: Model predictions (if available)
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        train_examples = task_data.get('train', [])
        test_examples = task_data.get('test', [])
        
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
        fig.suptitle(f'Task: {task_id}', fontsize=14, fontweight='bold')
        
        # Create grid layout
        total_rows = n_train + n_test
        
        # Plot training examples
        for i, example in enumerate(train_examples):
            # Input
            ax_input = plt.subplot(total_rows, n_cols, i * n_cols + 1)
            self._plot_grid_on_axes(example['input'], ax_input, 
                                   f'Train {i+1} - Input')
            
            # Output
            ax_output = plt.subplot(total_rows, n_cols, i * n_cols + 2)
            self._plot_grid_on_axes(example['output'], ax_output, 
                                   f'Train {i+1} - Output')
            
            # Fill remaining columns if needed
            if n_cols > 2:
                for col in range(3, n_cols + 1):
                    ax = plt.subplot(total_rows, n_cols, i * n_cols + col)
                    ax.axis('off')
        
        # Plot test examples
        for i, example in enumerate(test_examples):
            row_idx = n_train + i
            
            # Test input
            ax_input = plt.subplot(total_rows, n_cols, row_idx * n_cols + 1)
            self._plot_grid_on_axes(example['input'], ax_input, 
                                   f'Test {i+1} - Input')
            
            # Handle predictions and solutions
            if predictions is not None and solution is not None:
                # Show both prediction and ground truth
                
                # Prediction
                ax_pred = plt.subplot(total_rows, n_cols, row_idx * n_cols + 2)
                if i < len(predictions):
                    self._plot_grid_on_axes(predictions[i], ax_pred,
                                          f'Test {i+1} - Prediction')
                else:
                    self._plot_placeholder(ax_pred, f'Test {i+1} - No Prediction')
                
                # Ground Truth
                ax_truth = plt.subplot(total_rows, n_cols, row_idx * n_cols + 3)
                try:
                    if isinstance(solution, list) and i < len(solution):
                        if isinstance(solution[i], dict) and 'output' in solution[i]:
                            truth = solution[i]['output']
                        else:
                            truth = solution[i]
                        self._plot_grid_on_axes(truth, ax_truth,
                                              f'Test {i+1} - Ground Truth')
                    else:
                        self._plot_placeholder(ax_truth, f'Test {i+1} - Unknown')
                except:
                    self._plot_placeholder(ax_truth, f'Test {i+1} - Unknown')
                
                # Correctness indicator
                ax_check = plt.subplot(total_rows, n_cols, row_idx * n_cols + 4)
                try:
                    if isinstance(solution, list) and i < len(solution) and i < len(predictions):
                        if isinstance(solution[i], dict) and 'output' in solution[i]:
                            truth = solution[i]['output']
                        else:
                            truth = solution[i]
                        
                        is_correct = np.array_equal(predictions[i], truth)
                        color = '#2ECC40' if is_correct else '#FF4136'
                        symbol = '✓' if is_correct else '✗'
                        
                        ax_check.text(0.5, 0.5, symbol, fontsize=40, 
                                    ha='center', va='center', color=color)
                        ax_check.set_xlim(0, 1)
                        ax_check.set_ylim(0, 1)
                        ax_check.axis('off')
                        ax_check.set_title('Correct' if is_correct else 'Wrong', 
                                         color=color, fontweight='bold')
                    else:
                        ax_check.axis('off')
                except:
                    ax_check.axis('off')
                    
            elif predictions is not None:
                # Show only predictions
                ax_pred = plt.subplot(total_rows, n_cols, row_idx * n_cols + 2)
                if i < len(predictions):
                    self._plot_grid_on_axes(predictions[i], ax_pred,
                                          f'Test {i+1} - Prediction')
                else:
                    self._plot_placeholder(ax_pred, f'Test {i+1} - No Prediction')
                    
                if n_cols > 2:
                    for col in range(3, n_cols + 1):
                        ax = plt.subplot(total_rows, n_cols, row_idx * n_cols + col)
                        ax.axis('off')
                        
            elif solution is not None:
                # Show only ground truth
                ax_truth = plt.subplot(total_rows, n_cols, row_idx * n_cols + 2)
                try:
                    if isinstance(solution, list) and i < len(solution):
                        if isinstance(solution[i], dict) and 'output' in solution[i]:
                            truth = solution[i]['output']
                        else:
                            truth = solution[i]
                        self._plot_grid_on_axes(truth, ax_truth,
                                              f'Test {i+1} - Ground Truth')
                    else:
                        self._plot_placeholder(ax_truth, f'Test {i+1} - Unknown')
                except:
                    self._plot_placeholder(ax_truth, f'Test {i+1} - Unknown')
                    
                if n_cols > 2:
                    for col in range(3, n_cols + 1):
                        ax = plt.subplot(total_rows, n_cols, row_idx * n_cols + col)
                        ax.axis('off')
            else:
                # No predictions or solutions
                ax_output = plt.subplot(total_rows, n_cols, row_idx * n_cols + 2)
                self._plot_placeholder(ax_output, f'Test {i+1} - To Predict')
        
        plt.tight_layout()
        return fig
    
    def _plot_grid_on_axes(self, grid: List[List[int]], ax: plt.Axes, title: str):
        """Helper to plot grid on existing axes."""
        grid_array = np.array(grid)
        
        im = ax.imshow(grid_array, cmap=self.cmap, norm=self.norm, 
                      interpolation='nearest')
        
        # Add grid lines
        height, width = grid_array.shape
        for i in range(height + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.5)
        for j in range(width + 1):
            ax.axvline(j - 0.5, color='white', linewidth=0.5)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
    
    def _plot_placeholder(self, ax: plt.Axes, title: str):
        """Plot a placeholder for unknown outputs."""
        placeholder = np.full((3, 3), 5)  # Gray 3x3 grid
        self._plot_grid_on_axes(placeholder.tolist(), ax, title)
    
    def show_task(self, task_id: Optional[str] = None, 
                  dataset: str = 'training') -> None:
        """Display a specific task or random task if none specified.
        
        Args:
            task_id: Task ID to display (None for random)
            dataset: Which dataset ('training' or 'evaluation')
        """
        # Get challenges and solutions
        if dataset == 'training':
            challenges = self.training_challenges
            solutions = self.training_solutions
        else:
            challenges = self.evaluation_challenges
            solutions = self.evaluation_solutions
        
        if challenges is None:
            print(f"No {dataset} data loaded!")
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
    
    def _print_task_stats(self, task_id: str, task_data: Dict, 
                          solution: Optional[Union[List, Dict]]):
        """Print statistics about a task."""
        print(f"\n📊 Task Statistics for {task_id}:")
        print(f"  • Training examples: {len(task_data.get('train', []))}")
        print(f"  • Test examples: {len(task_data.get('test', []))}")
        
        if task_data.get('train'):
            train_ex = task_data['train'][0]
            print(f"  • First training input shape: {np.array(train_ex['input']).shape}")
            print(f"  • First training output shape: {np.array(train_ex['output']).shape}")
            
            # Color analysis
            all_colors_in = set()
            all_colors_out = set()
            for ex in task_data['train']:
                all_colors_in.update(np.unique(ex['input']).tolist())
                all_colors_out.update(np.unique(ex['output']).tolist())
            print(f"  • Colors used in inputs: {sorted(all_colors_in)}")
            print(f"  • Colors used in outputs: {sorted(all_colors_out)}")
    
    def create_interactive_widget(self):
        """Create an interactive widget for exploring tasks.
        
        Returns:
            IPython widget box containing the interface
        """
        # Dataset selector
        self.dataset_dropdown = widgets.Dropdown(
            options=['training', 'evaluation'],
            value='training',
            description='Dataset:',
            style={'description_width': 'initial'}
        )
        
        # Task selector (will be populated based on dataset)
        self.task_dropdown = widgets.Dropdown(
            options=[],
            description='Task ID:',
            style={'description_width': 'initial'}
        )
        
        # Buttons
        random_button = widgets.Button(
            description='🎲 Random Task',
            button_style='primary',
            tooltip='Show a random task'
        )
        
        refresh_button = widgets.Button(
            description='🔄 Refresh',
            button_style='info',
            tooltip='Refresh current task'
        )
        
        # Output area
        self.output_area = widgets.Output()
        
        # Event handlers
        def on_dataset_change(change):
            self._update_task_list()
        
        def on_task_change(change):
            if change['new']:
                self._display_task()
        
        def on_random_click(b):
            tasks = list(self.task_dropdown.options)
            if tasks:
                self.task_dropdown.value = random.choice(tasks)
        
        def on_refresh_click(b):
            self._display_task()
        
        # Connect event handlers
        self.dataset_dropdown.observe(on_dataset_change, names='value')
        self.task_dropdown.observe(on_task_change, names='value')
        random_button.on_click(on_random_click)
        refresh_button.on_click(on_refresh_click)
        
        # Initial population
        self._update_task_list()
        
        # Layout
        controls = widgets.HBox([
            self.dataset_dropdown,
            self.task_dropdown,
            random_button,
            refresh_button
        ])
        
        return widgets.VBox([
            widgets.HTML("<h2>🧩 ARC-AGI Task Explorer</h2>"),
            controls,
            self.output_area
        ])
    
    def _update_task_list(self):
        """Update task dropdown based on selected dataset."""
        dataset = self.dataset_dropdown.value
        
        if dataset == 'training' and self.training_challenges:
            tasks = sorted(self.training_challenges.keys())
        elif dataset == 'evaluation' and self.evaluation_challenges:
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
            if dataset == 'training':
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
            
            # Create and display plot
            fig = self.plot_task(task_id, task_data, solution)
            plt.show()
            
            # Print statistics
            self._print_task_stats(task_id, task_data, solution)
    
    def explore(self):
        """Launch the interactive explorer widget."""
        widget = self.create_interactive_widget()
        display(widget)
    
    def batch_analyze(self, n_samples: int = 5, dataset: str = 'training'):
        """Analyze multiple random tasks at once.
        
        Args:
            n_samples: Number of tasks to analyze
            dataset: Which dataset to use
        """
        if dataset == 'training':
            challenges = self.training_challenges
        else:
            challenges = self.evaluation_challenges
        
        if not challenges:
            print(f"No {dataset} data loaded!")
            return
        
        task_ids = random.sample(list(challenges.keys()), 
                                min(n_samples, len(challenges)))
        
        for task_id in task_ids:
            print(f"\n{'='*60}")
            print(f"Task: {task_id}")
            print('='*60)
            self.show_task(task_id, dataset)


    def visualize_predictions(self, baseline, task_ids: Optional[List[str]] = None, 
                             dataset: str = 'training', n_tasks: int = 3) -> None:
        """Visualize baseline predictions compared to ground truth.
        
        Args:
            baseline: ARCBaseline instance with solve_task method
            task_ids: Specific task IDs to visualize (None for random)
            dataset: Which dataset to use ('training' or 'evaluation')
            n_tasks: Number of tasks to visualize if task_ids is None
        """
        # Get data
        if dataset == 'training':
            challenges = self.training_challenges
            solutions = self.training_solutions
        else:
            challenges = self.evaluation_challenges
            solutions = self.evaluation_solutions
        
        if not challenges:
            print(f"No {dataset} data loaded!")
            return
        
        # Select tasks
        if task_ids is None:
            available_ids = list(challenges.keys())
            task_ids = random.sample(available_ids, min(n_tasks, len(available_ids)))
        
        # Visualize each task
        for task_id in task_ids:
            if task_id not in challenges:
                print(f"Task {task_id} not found in {dataset} set!")
                continue
            
            task_data = challenges[task_id]
            solution = solutions.get(task_id) if solutions else None
            
            # Get predictions from baseline
            predictions = baseline.solve_task(task_data)
            
            # Create visualization
            fig = self.plot_task(task_id, task_data, solution, predictions)
            plt.show()
            
            # Print accuracy for this task
            if solution:
                correct = 0
                total = len(predictions)
                
                for i, pred in enumerate(predictions):
                    if i < len(solution):
                        if isinstance(solution[i], dict) and 'output' in solution[i]:
                            truth = solution[i]['output']
                        else:
                            truth = solution[i]
                        
                        if np.array_equal(pred, truth):
                            correct += 1
                
                accuracy = correct / total if total > 0 else 0
                print(f"\n📊 Task {task_id}: {correct}/{total} correct ({accuracy:.0%})")
                
                # Analyze the pattern detected
                train_examples = task_data.get('train', [])
                if hasattr(baseline, 'analyze_transformation'):
                    pattern = baseline.analyze_transformation(train_examples)
                    print(f"   Pattern detected: {pattern.pattern_type} (confidence: {pattern.confidence:.2f})")
            print("-" * 60)
    
    def compare_predictions(self, baseline, task_id: str, dataset: str = 'training') -> None:
        """Show detailed comparison of predictions vs ground truth for a single task.
        
        Args:
            baseline: ARCBaseline instance
            task_id: Task ID to analyze
            dataset: Which dataset to use
        """
        # Get data
        if dataset == 'training':
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
        predictions = baseline.solve_task(task_data)
        
        # Analyze transformation
        train_examples = task_data.get('train', [])
        if hasattr(baseline, 'analyze_transformation'):
            pattern = baseline.analyze_transformation(train_examples)
            
            print(f"🔍 Analysis for Task {task_id}")
            print(f"   Pattern type: {pattern.pattern_type}")
            print(f"   Confidence: {pattern.confidence:.2f}")
            if pattern.params:
                print(f"   Parameters: {pattern.params}")
        
        # Create detailed visualization
        fig = self.plot_task(task_id, task_data, solution, predictions)
        plt.show()
        
        # Detailed comparison
        if solution:
            print(f"\n📊 Detailed Comparison:")
            for i, pred in enumerate(predictions):
                if i < len(solution):
                    if isinstance(solution[i], dict) and 'output' in solution[i]:
                        truth = solution[i]['output']
                    else:
                        truth = solution[i]
                    
                    pred_arr = np.array(pred)
                    truth_arr = np.array(truth)
                    
                    is_correct = np.array_equal(pred_arr, truth_arr)
                    
                    print(f"\n   Test {i+1}:")
                    print(f"     Prediction shape: {pred_arr.shape}")
                    print(f"     Ground truth shape: {truth_arr.shape}")
                    print(f"     Correct: {'✓' if is_correct else '✗'}")
                    
                    if not is_correct:
                        # Show what's different
                        if pred_arr.shape == truth_arr.shape:
                            diff_count = np.sum(pred_arr != truth_arr)
                            total_pixels = pred_arr.size
                            print(f"     Pixels different: {diff_count}/{total_pixels} ({diff_count/total_pixels*100:.1f}%)")
                        else:
                            print(f"     Shape mismatch!")
    
    def evaluate_and_visualize(self, baseline, dataset: str = 'training', 
                               n_samples: int = 5, show_only_errors: bool = False) -> None:
        """Evaluate baseline and visualize results.
        
        Args:
            baseline: ARCBaseline instance
            dataset: Which dataset to use
            n_samples: Number of examples to visualize
            show_only_errors: If True, only show incorrect predictions
        """
        if dataset == 'training':
            challenges = self.training_challenges
            solutions = self.training_solutions
        else:
            challenges = self.evaluation_challenges
            solutions = self.evaluation_solutions
        
        if not challenges or not solutions:
            print(f"No {dataset} data with solutions loaded!")
            return
        
        print(f"🔍 Evaluating on {dataset} set...")
        
        # Collect results
        results = []
        for task_id in list(challenges.keys())[:n_samples * 3]:  # Check more to find errors
            task_data = challenges[task_id]
            predictions = baseline.solve_task(task_data)
            solution = solutions[task_id]
            
            # Check correctness
            all_correct = True
            for i, pred in enumerate(predictions):
                if i < len(solution):
                    if isinstance(solution[i], dict) and 'output' in solution[i]:
                        truth = solution[i]['output']
                    else:
                        truth = solution[i]
                    
                    if not np.array_equal(pred, truth):
                        all_correct = False
                        break
            
            results.append({
                'task_id': task_id,
                'correct': all_correct,
                'predictions': predictions,
                'solution': solution
            })
        
        # Filter based on show_only_errors
        if show_only_errors:
            results_to_show = [r for r in results if not r['correct']][:n_samples]
            if not results_to_show:
                print("No errors found in the sample!")
                results_to_show = results[:n_samples]
        else:
            results_to_show = results[:n_samples]
        
        # Summary statistics
        total_tasks = len(results)
        correct_tasks = sum(1 for r in results if r['correct'])
        print(f"\n📊 Overall: {correct_tasks}/{total_tasks} tasks fully correct ({correct_tasks/total_tasks*100:.1f}%)")
        
        # Visualize selected results
        print(f"\n📈 Showing {len(results_to_show)} examples:")
        for result in results_to_show:
            task_id = result['task_id']
            task_data = challenges[task_id]
            
            status = "✓ Correct" if result['correct'] else "✗ Wrong"
            print(f"\nTask {task_id}: {status}")
            
            fig = self.plot_task(task_id, task_data, 
                               result['solution'], result['predictions'])
            plt.show()


# Enhanced convenience function
def create_viewer(data_dir: str = "data", auto_load: bool = True) -> ARCNotebookViewer:
    """Create and optionally load an ARC viewer.
    
    Args:
        data_dir: Directory containing the data files
        auto_load: Whether to automatically load data
        
    Returns:
        ARCNotebookViewer instance
    """
    viewer = ARCNotebookViewer(data_dir=data_dir)
    if auto_load:
        viewer.load_data()
    return viewer