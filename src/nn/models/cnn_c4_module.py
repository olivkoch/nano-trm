"""
Connect Four Baseline Models - CNN and MLP variants
Matches the TRM interface for direct comparison
"""

import math
from typing import Dict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from src.nn.modules.utils import compute_lr

# You'll need to import these from your codebase
from src.nn.utils.constants import C4_EMPTY_CELL
from src.nn.modules.utils import compute_lr
from src.nn.modules.minimax import ConnectFourMinimax
from src.nn.environments.vectorized_c4_env import VectorizedConnectFour

class ConvBlock(nn.Module):
    """Basic convolutional block with normalization and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Simple residual block for deeper networks"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class C4BaselineModule(LightningModule):
    """
    Baseline model for Connect Four - supports both CNN and MLP architectures
    """
    
    def __init__(
        self,
        model_type: str = "cnn",  # "cnn" or "mlp"
        hidden_size: int = 512,
        num_layers: int = 4,  # For MLP or number of residual blocks for CNN
        cnn_channels: int = 128,  # Number of channels for CNN
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        lr_min_ratio: float = 0.1,
        steps_per_epoch: int = 1000,
        batch_size: int = 256,
        max_epochs: int = 300,
        eval_minimax_depth: int = 4,
        eval_minimax_temperature: float = 0.5,
        eval_games_vs_minimax: int = 100,
        dropout: float = 0.1,
        use_residual: bool = False,  # For CNN - use residual blocks
        output_dir: str = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Board dimensions
        self.board_rows = 6
        self.board_cols = 7
        self.board_size = self.board_rows * self.board_cols
        
        # Build model based on type
        if model_type == "cnn":
            self._build_cnn()
        elif model_type == "mlp":
            self._build_mlp()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Manual optimization to match TRM
        self.automatic_optimization = False
        self.manual_step = 0
        self.games_played = 0
        self.replay_buffer = []
        self.max_steps = self.hparams.steps_per_epoch * self.hparams.max_epochs
        
        print(f"Created {model_type.upper()} baseline with {self._count_parameters():.2f}M parameters")
        
    def _count_parameters(self):
        """Count trainable parameters in millions"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
    
    def _build_cnn(self):
        """Build CNN architecture"""
        # Input processing - board is 6x7, we'll use 3 channels:
        # Channel 0: Current player's pieces (1s)
        # Channel 1: Opponent's pieces (-1s)  
        # Channel 2: Empty cells (0s)
        
        cnn_channels = self.hparams.cnn_channels
        
        # Feature extraction
        if self.hparams.use_residual:
            # Residual network version
            self.input_conv = ConvBlock(3, cnn_channels, kernel_size=3, padding=1)
            
            # Residual blocks
            self.res_blocks = nn.ModuleList([
                ResidualBlock(cnn_channels) for _ in range(self.hparams.num_layers)
            ])
            
            feature_size = cnn_channels * self.board_rows * self.board_cols
        else:
            # Simple CNN version
            layers = []
            in_channels = 3
            
            # First layer
            layers.append(ConvBlock(in_channels, cnn_channels // 2, kernel_size=3, padding=1))
            
            # Middle layers - maintain spatial dimensions
            for _ in range(self.hparams.num_layers - 2):
                layers.append(ConvBlock(cnn_channels // 2, cnn_channels // 2, kernel_size=3, padding=1))
            
            # Last conv layer
            layers.append(ConvBlock(cnn_channels // 2, cnn_channels, kernel_size=3, padding=1))
            
            self.conv_layers = nn.Sequential(*layers)
            feature_size = cnn_channels * self.board_rows * self.board_cols
        
        # Global pooling and heads
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cnn_channels, self.hparams.hidden_size)
        self.dropout = nn.Dropout(self.hparams.dropout)
        
        # Output heads matching TRM
        self.policy_head = nn.Linear(self.hparams.hidden_size, self.board_cols)
        self.value_head = nn.Linear(self.hparams.hidden_size, 1)
        
    def _build_mlp(self):
        """Build MLP architecture"""
        # Simple MLP - takes flattened board as input
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.board_size, self.hparams.hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.hparams.dropout))
        
        # Hidden layers
        for _ in range(self.hparams.num_layers - 1):
            layers.append(nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.hparams.dropout))
        
        self.mlp = nn.Sequential(*layers)
        
        # Output heads
        self.policy_head = nn.Linear(self.hparams.hidden_size, self.board_cols)
        self.value_head = nn.Linear(self.hparams.hidden_size, 1)
    
    def _board_to_channels(self, boards: torch.Tensor) -> torch.Tensor:
        """
        Convert flat board representation to 3-channel image for CNN
        boards: [batch_size, 42] with values in {0, 1, 2}
        Returns: [batch_size, 3, 6, 7]
        """
        batch_size = boards.shape[0]
        # Ensure boards are float
        boards = boards.float()
        boards_2d = boards.view(batch_size, self.board_rows, self.board_cols)
        
        # Create 3-channel representation - always use float32
        channels = torch.zeros(batch_size, 3, self.board_rows, self.board_cols, 
                              device=boards.device, dtype=torch.float32)
        
        # Channel 0: Player 1 pieces
        channels[:, 0] = (boards_2d == 1).float()
        # Channel 1: Player 2 pieces
        channels[:, 1] = (boards_2d == 2).float()
        # Channel 2: Empty cells
        channels[:, 2] = (boards_2d == 0).float()
        
        return channels
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass matching TRM interface
        """
        boards = batch["boards"].float()
        batch_size = boards.shape[0]
        
        if self.hparams.model_type == "cnn":
            # Convert to channel format
            x = self._board_to_channels(boards)
            
            if self.hparams.use_residual:
                # Residual network forward
                x = self.input_conv(x)
                for block in self.res_blocks:
                    x = block(x)
                x = self.global_pool(x).flatten(1)
            else:
                # Simple CNN forward
                x = self.conv_layers(x)
                x = self.global_pool(x).flatten(1)
            
            x = self.dropout(self.fc(x))
            
        else:  # MLP
            # Direct forward through MLP
            x = self.mlp(boards)
        
        # Output heads
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x).squeeze(-1))
        
        # Apply legal moves mask
        boards_reshaped = boards.view(batch_size, self.board_rows, self.board_cols)
        valid_actions = boards_reshaped[:, 0, :] == C4_EMPTY_CELL
        policy_logits = policy_logits.masked_fill(~valid_actions, -1e9)
        
        # Compute policy probabilities
        policy = F.softmax(policy_logits, dim=-1)
        
        return {
            "policy_logits": policy_logits,
            "policy": policy,
            "value": value,
        }
    
    def compute_loss_and_metrics(self, batch: Dict[str, torch.Tensor]):
        """Compute loss and metrics"""
        # Get model outputs
        outputs = self.forward(batch)
        
        # Get legal moves
        boards = batch["boards"].float()
        boards_reshaped = boards.view(-1, self.board_rows, self.board_cols)
        legal_moves = boards_reshaped[:, 0, :] == C4_EMPTY_CELL
        
        # Process targets
        target_policy = batch["policies"]
        target_policy = target_policy * legal_moves.float()
        target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        
        target_values = batch["values"]
        
        # Compute losses
        policy_loss = -torch.sum(target_policy * torch.log(outputs["policy"] + 1e-8), dim=-1).mean()
        value_loss = F.mse_loss(outputs["value"], target_values)
        
        # Combined loss (matching TRM weights)
        total_loss = policy_loss + 0.5 * value_loss
        
        # Compute metrics
        with torch.no_grad():
            pred_actions = outputs["policy"].argmax(dim=-1)
            target_actions = target_policy.argmax(dim=-1)
            policy_accuracy = (pred_actions == target_actions).float().mean()
            
            metrics = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "policy_accuracy": policy_accuracy.item(),
            }
        
        return total_loss, metrics
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step matching TRM interface"""
        
        # Get optimizer
        try:
            opt = self.optimizers()
        except RuntimeError:
            if not hasattr(self, "_optimizer"):
                raise RuntimeError("No optimizer available. Set model._optimizer for testing.")
            opt = self._optimizer
        
        # Forward and compute loss
        loss, metrics = self.compute_loss_and_metrics(batch)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Learning rate scheduling with warmup
        current_step = self.manual_step
        total_steps = self.max_steps
        base_lr = self.hparams.learning_rate
        if current_step < self.hparams.warmup_steps:
            lr = compute_lr(
                    base_lr=base_lr,
                    lr_warmup_steps=self.hparams.warmup_steps,
                    lr_min_ratio=self.hparams.lr_min_ratio,
                    current_step=current_step,
                    total_steps=total_steps,
                )
        else:
            lr = base_lr
        
        # Update learning rate
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        
        # Optimizer step
        opt.step()
        opt.zero_grad()
        
        # Log metrics
        self.log("train/loss", loss.item(), on_step=True, prog_bar=True)
        self.log("train/policy_loss", metrics["policy_loss"], on_step=True)
        self.log("train/value_loss", metrics["value_loss"], on_step=True)
        self.log("train/policy_accuracy", metrics["policy_accuracy"], on_step=True, prog_bar=True)
        self.log("train/lr", lr, on_step=True)
        
        self.manual_step += 1
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer"""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95)
        )
    
    def on_train_epoch_start(self):
        if len(self.replay_buffer) == 0:
                self.load_games_from_file(f"minimax_games_.pkl")

    def on_train_epoch_end(self):
        """Evaluate against minimax at epoch end"""
        if hasattr(self, 'evaluate_vs_minimax_fast'):
            self.evaluate_vs_minimax_fast()
    
    def evaluate_vs_minimax_fast(self):
        """Fast evaluation matching TRM interface"""
        
        print(f"Fast eval vs Minimax (depth={self.hparams.eval_minimax_depth})...")
        
        minimax = ConnectFourMinimax(depth=self.hparams.eval_minimax_depth)
        n_games = self.hparams.eval_games_vs_minimax
        n_parallel = min(32, n_games)  # Smaller batches for baseline
        n_batches = (n_games + n_parallel - 1) // n_parallel
        
        wins = 0
        draws = 0
        
        for batch_idx in range(n_batches):
            games_in_batch = min(n_parallel, n_games - batch_idx * n_parallel)
            vec_env = VectorizedConnectFour(n_envs=games_in_batch, device=self.device)
            states = vec_env.reset()
            
            # Alternate who goes first
            model_players = [1 if (batch_idx * n_parallel + i) % 2 == 0 else 2 
                            for i in range(games_in_batch)]
            
            while not states.is_terminal.all():
                actions = torch.zeros(games_in_batch, dtype=torch.long, device=self.device)
                
                for i in range(games_in_batch):
                    if states.is_terminal[i]:
                        continue
                    
                    current_player = states.current_players[i].item()
                    
                    if current_player == model_players[i]:
                        # Model move
                        with torch.no_grad():
                            batch = {
                                'boards': states.boards[i].unsqueeze(0).flatten(1),
                                'puzzle_identifiers': torch.zeros(1, dtype=torch.long, device=self.device)
                            }
                            outputs = self.forward(batch)
                            
                            # Apply legal moves mask and sample
                            legal = states.legal_moves[i]
                            masked_policy = outputs['policy'][0] * legal.float()
                            if masked_policy.sum() > 0:
                                actions[i] = masked_policy.argmax().item()
                            else:
                                # Fallback to random
                                legal_actions = legal.nonzero(as_tuple=True)[0]
                                actions[i] = legal_actions[torch.randint(len(legal_actions), (1,))].item()
                    else:
                        # Minimax move
                        board_np = states.boards[i].cpu().numpy()
                        actions[i] = minimax.get_best_move(
                            board_np, current_player,
                            temperature=self.hparams.eval_minimax_temperature
                        )
                
                states = vec_env.step(actions)
            
            # Count results
            for i in range(games_in_batch):
                winner = states.winners[i].item()
                if winner == model_players[i]:
                    wins += 1
                elif winner == 0:
                    draws += 1
        
        # Log results
        win_rate = wins / n_games
        draw_rate = draws / n_games
        
        self.log('eval/win_rate_vs_minimax', win_rate, prog_bar=True)
        self.log('eval/draw_rate_vs_minimax', draw_rate)
        
        print(f"Baseline eval: W={win_rate:.1%}, D={draw_rate:.1%}, L={(1-win_rate-draw_rate):.1%}")
    
    def load_games_from_file(self, input_file: str):
        """Load games from file - matching TRM interface"""
        print(f"Loading games from {input_file}...")
        import pickle
        
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        
        positions = data['positions']
        
        for pos in positions:
            board = torch.tensor(pos['board'], dtype=torch.float32, device=self.device)
            policy = torch.tensor(pos['policy'], dtype=torch.float32, device=self.device)
            value = pos['value']
            
            self.replay_buffer.append({
                'board': board,
                'policy': policy,
                'value': value
            })
        
        print(f"Loaded {len(positions)} positions from {data['num_games']} games")
        print(f"Replay buffer now contains {len(self.replay_buffer)} positions")
    
    def train_dataloader(self):
        """Create dataloader matching TRM interface"""
        from torch.utils.data import DataLoader, Dataset
        import random
        
        class SelfPlayDataset(Dataset):
            def __init__(self, module, steps_per_epoch):
                self.module = module
                self.steps_per_epoch = steps_per_epoch
                
            def __len__(self):
                return self.steps_per_epoch
            
            def __getitem__(self, idx):
                # Sample batch
                samples = random.sample(
                    self.module.replay_buffer,
                    self.module.hparams.batch_size
                )
                
                boards = torch.stack([s['board'] for s in samples])
                policies = torch.stack([s['policy'] for s in samples])
                values = torch.tensor([s['value'] for s in samples], device=self.module.device)
                puzzle_identifiers = torch.zeros(
                    self.module.hparams.batch_size, dtype=torch.long, device=self.module.device
                )
                
                return {
                    'boards': boards,
                    'policies': policies,
                    'values': values,
                    'puzzle_identifiers': puzzle_identifiers
                }
        
        dataset = SelfPlayDataset(self, self.hparams.steps_per_epoch)
        return DataLoader(dataset, batch_size=None, num_workers=0, shuffle=False)


def test_baseline():
    """Test the baseline model"""
    
    # Test CNN version
    print("Testing CNN baseline...")
    cnn_model = C4BaselineModule(
        model_type="cnn",
        hidden_size=256,
        num_layers=3,
        cnn_channels=64,
        batch_size=32,
        use_residual=True
    )
    
    # Test forward pass
    batch = {
        'boards': torch.randn(32, 42),
        'puzzle_identifiers': torch.zeros(32, dtype=torch.long)
    }
    
    outputs = cnn_model.forward(batch)
    print(f"CNN outputs: policy shape={outputs['policy'].shape}, value shape={outputs['value'].shape}")
    print(f"CNN parameters: {sum(p.numel() for p in cnn_model.parameters())/1e6:.2f}M")
    
    # Test MLP version
    print("\nTesting MLP baseline...")
    mlp_model = C4BaselineModule(
        model_type="mlp",
        hidden_size=512,
        num_layers=4,
        batch_size=32
    )
    
    outputs = mlp_model.forward(batch)
    print(f"MLP outputs: policy shape={outputs['policy'].shape}, value shape={outputs['value'].shape}")
    print(f"MLP parameters: {sum(p.numel() for p in mlp_model.parameters())/1e6:.2f}M")
    
    print("\nBaseline models ready!")


if __name__ == "__main__":
    test_baseline()