"""
Example: How to use different solvers with the ARC framework
Shows baseline, ML, and ensemble approaches
"""

# %% Import required modules
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any
import json

from arc_solver_base import ARCSolver, MLSolver, BaselineSolver, EnsembleSolver, SolverResult
from arc_evaluator import ARCEvaluator
from viewer.arc_notebook_viewer import create_viewer

# %% Example 1: Using the Baseline Solver
print("=" * 60)
print("Example 1: Baseline Pattern Matching Solver")
print("=" * 60)

# Create baseline solver
baseline_solver = BaselineSolver(name="BaselinePatternMatcher")

# Load and evaluate
evaluator = ARCEvaluator(data_dir="data")
result = evaluator.evaluate_solver(baseline_solver, dataset='training', max_tasks=50)

# Visualize predictions
viewer = create_viewer(data_dir="data")
viewer.visualize_predictions(baseline_solver, n_tasks=2)


# %% Example 2: Custom Rule-Based Solver
class CustomRuleSolver(ARCSolver):
    """Example of a custom rule-based solver."""
    
    def __init__(self, name: str = "CustomRules"):
        super().__init__(name)
        
    def solve_task(self, task_data: Dict) -> SolverResult:
        """Apply custom rules to solve the task."""
        import time
        start_time = time.time()
        
        train_examples = task_data.get('train', [])
        test_examples = task_data.get('test', [])
        
        predictions = []
        
        for test_case in test_examples:
            test_input = np.array(test_case['input'])
            
            # Custom rule 1: If all outputs in training are 3x3, predict 3x3
            all_3x3 = all(
                np.array(ex['output']).shape == (3, 3) 
                for ex in train_examples
            )
            
            if all_3x3:
                # Extract most common 3x3 pattern
                output = np.zeros((3, 3), dtype=int)
                # Fill with most common color from training outputs
                colors = []
                for ex in train_examples:
                    colors.extend(np.array(ex['output']).flatten())
                if colors:
                    from collections import Counter
                    most_common = Counter(colors).most_common(1)[0][0]
                    output.fill(most_common)
            else:
                # Default: copy input
                output = test_input
            
            predictions.append(output.tolist())
        
        metadata = {'rule_applied': 'fixed_3x3' if all_3x3 else 'copy'}
        
        return SolverResult(
            predictions=predictions,
            metadata=metadata,
            solve_time=time.time() - start_time
        )


print("\n" + "=" * 60)
print("Example 2: Custom Rule-Based Solver")
print("=" * 60)

custom_solver = CustomRuleSolver()
result = evaluator.evaluate_solver(custom_solver, dataset='training', max_tasks=50)


# %% Example 3: PyTorch Neural Network Solver
class NeuralARCSolver(MLSolver):
    """Example neural network solver using PyTorch."""
    
    def __init__(self, name: str = "NeuralNet", input_size: int = 30, hidden_size: int = 128):
        super().__init__(name, model_type="pytorch")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_output_size = 30
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create a simple model
        self.model = self._build_model()
        
    def _build_model(self):
        """Build a simple neural network model."""
        class SimpleARCNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                flat_size = input_size * input_size * 10  # 10 colors
                self.encoder = nn.Sequential(
                    nn.Linear(flat_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size * output_size * 10),
                )
                self.output_size = output_size
                
            def forward(self, x):
                # x shape: (batch, height, width, colors)
                batch_size = x.shape[0]
                x = x.view(batch_size, -1)
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                # Reshape to (batch, height, width, colors)
                decoded = decoded.view(batch_size, self.output_size, self.output_size, 10)
                return decoded
        
        model = SimpleARCNet(self.input_size, self.hidden_size, self.max_output_size)
        return model.to(self.device)
    
    def preprocess_task(self, task_data: Dict) -> Any:
        """Convert task to tensors for the model."""
        test_examples = task_data.get('test', [])
        
        # For simplicity, just process test inputs
        test_tensors = []
        for test_case in test_examples:
            grid = np.array(test_case['input'])
            # Pad to fixed size
            padded = np.zeros((self.input_size, self.input_size), dtype=int)
            h, w = min(grid.shape[0], self.input_size), min(grid.shape[1], self.input_size)
            padded[:h, :w] = grid[:h, :w]
            
            # One-hot encode colors (0-9)
            one_hot = np.zeros((self.input_size, self.input_size, 10))
            for i in range(self.input_size):
                for j in range(self.input_size):
                    one_hot[i, j, padded[i, j]] = 1
            
            test_tensors.append(torch.FloatTensor(one_hot))
        
        return {'test_inputs': test_tensors}
    
    def predict(self, preprocessed_data: Any) -> Any:
        """Make predictions using the model."""
        test_inputs = preprocessed_data['test_inputs']
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for test_input in test_inputs:
                # Add batch dimension
                x = test_input.unsqueeze(0).to(self.device)
                
                # Forward pass
                output = self.model(x)
                
                # Convert to color indices
                output = output.squeeze(0).cpu().numpy()
                output = np.argmax(output, axis=-1)
                
                # Find actual size (remove padding)
                non_zero_mask = output != 0
                if non_zero_mask.any():
                    rows = np.any(non_zero_mask, axis=1)
                    cols = np.any(non_zero_mask, axis=0)
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    output = output[rmin:rmax+1, cmin:cmax+1]
                else:
                    output = output[:3, :3]  # Default small output
                
                predictions.append(output)
        
        return predictions
    
    def train(self, training_data: Dict[str, Dict], 
              validation_data: Optional[Dict[str, Dict]] = None,
              epochs: int = 10, batch_size: int = 16, lr: float = 0.001) -> Dict[str, Any]:
        """Train the neural network."""
        print(f"Training {self.name}...")
        
        # For demo purposes, we'll just show the structure
        # In practice, you would:
        # 1. Convert all training tasks to input/output pairs
        # 2. Create a DataLoader
        # 3. Train with backpropagation
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        history = {'loss': [], 'val_loss': []}
        
        # Placeholder training loop
        for epoch in range(epochs):
            # This would contain actual training logic
            fake_loss = 0.5 - epoch * 0.02
            history['loss'].append(fake_loss)
            
            if validation_data:
                fake_val_loss = 0.6 - epoch * 0.015
                history['val_loss'].append(fake_val_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={fake_loss:.3f}")
        
        self.is_trained = True
        return history


print("\n" + "=" * 60)
print("Example 3: Neural Network Solver (PyTorch)")
print("=" * 60)

# Create and train neural solver
neural_solver = NeuralARCSolver(name="SimpleNeuralNet")

# Train on some data (placeholder training)
training_data = evaluator.datasets['training']['challenges']
history = neural_solver.train(training_data, epochs=10)
print(f"Training complete. Final loss: {history['loss'][-1]:.3f}")

# Evaluate
result = evaluator.evaluate_solver(neural_solver, dataset='training', max_tasks=20)


# %% Example 4: Ensemble of Multiple Solvers
print("\n" + "=" * 60)
print("Example 4: Ensemble Solver")
print("=" * 60)

# Create ensemble combining different approaches
ensemble = EnsembleSolver(
    solvers=[
        baseline_solver,
        custom_solver,
        neural_solver
    ],
    name="EnsembleVoting",
    strategy="voting"
)

# Evaluate ensemble
result = evaluator.evaluate_solver(ensemble, dataset='training', max_tasks=50)

# Visualize ensemble predictions
viewer.visualize_predictions(ensemble, n_tasks=2)


# %% Example 5: Compare All Solvers
print("\n" + "=" * 60)
print("Example 5: Solver Comparison")
print("=" * 60)

# Compare all solvers
all_solvers = [baseline_solver, custom_solver, neural_solver, ensemble]
comparison_df = evaluator.compare_solvers(all_solvers, dataset='training', max_tasks=50)

print("\n📊 Comparison Table:")
print(comparison_df.to_string(index=False))

# Plot comparison
evaluator.plot_comparison()


# %% Example 6: Custom ML Solver Template
class YourMLSolver(MLSolver):
    """Template for your own ML solver."""
    
    def __init__(self, name: str = "YourModel"):
        super().__init__(name, model_type="custom")
        # Initialize your model here
        # self.model = load_your_model()
        
    def preprocess_task(self, task_data: Dict) -> Any:
        """Preprocess task for your model."""
        # Convert task data to format your model expects
        # Return preprocessed data
        pass
    
    def predict(self, preprocessed_data: Any) -> Any:
        """Use your model to make predictions."""
        # predictions = self.model.predict(preprocessed_data)
        # return predictions
        pass
    
    def postprocess_predictions(self, raw_predictions: Any) -> List[List[List[int]]]:
        """Convert model output to ARC format."""
        # Convert your model's output to list of 2D grids
        pass
    
    def train(self, training_data: Dict[str, Dict],
              validation_data: Optional[Dict[str, Dict]] = None,
              **kwargs) -> Dict[str, Any]:
        """Train your model."""
        # Implement your training logic
        # Return training history/metrics
        pass


# %% Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
The framework now supports:

1. **Baseline Solver**: Pattern matching approach
2. **Custom Rule Solvers**: Implement your own logic
3. **ML Solvers**: Neural networks, transformers, etc.
4. **Ensemble Solvers**: Combine multiple approaches

To add your own solver:
1. Inherit from ARCSolver (or MLSolver for ML methods)
2. Implement solve_task() method
3. Optionally implement train() for learnable methods
4. Use the evaluator to test performance
5. Visualize with the notebook viewer

The same evaluation and visualization tools work with any solver!
""")