from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.arc_solver_base import MLSolver


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
        test_examples = task_data.get("test", [])

        # For simplicity, just process test inputs
        test_tensors = []
        for test_case in test_examples:
            grid = np.array(test_case["input"])
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

        return {"test_inputs": test_tensors}

    def predict(self, preprocessed_data: Any) -> Any:
        """Make predictions using the model."""
        test_inputs = preprocessed_data["test_inputs"]
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
                    output = output[rmin : rmax + 1, cmin : cmax + 1]
                else:
                    output = output[:3, :3]  # Default small output

                predictions.append(output)

        return predictions

    def train(
        self,
        training_data: Dict[str, Dict],
        validation_data: Optional[Dict[str, Dict]] = None,
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 0.001,
    ) -> Dict[str, Any]:
        """Train the neural network."""
        print(f"Training {self.name}...")

        # For demo purposes, we'll just show the structure
        # In practice, you would:
        # 1. Convert all training tasks to input/output pairs
        # 2. Create a DataLoader
        # 3. Train with backpropagation

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        history = {"loss": [], "val_loss": []}

        # Placeholder training loop
        for epoch in range(epochs):
            # This would contain actual training logic
            fake_loss = 0.5 - epoch * 0.02
            history["loss"].append(fake_loss)

            if validation_data:
                fake_val_loss = 0.6 - epoch * 0.015
                history["val_loss"].append(fake_val_loss)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: loss={fake_loss:.3f}")

        self.is_trained = True
        return history
