"""
Connect Four CNN Baseline Model - Refactored to use base class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from src.nn.models.c4_base import C4BaseModule
from src.nn.modules.utils import compute_lr, robust_kl_div
from src.nn.utils.constants import C4_EMPTY_CELL

from src.nn.utils import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)


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


class C4CNNModule(C4BaseModule):
    """
    CNN baseline model for Connect Four - inherits self-play from base
    """
    
    def __init__(
        self,
        # Model architecture
        model_type: str = "cnn",  # "cnn" or "mlp"
        hidden_size: int = 512,
        num_layers: int = 4,  # For MLP or number of residual blocks for CNN
        cnn_channels: int = 128,  # Number of channels for CNN
        dropout: float = 0.1,
        use_residual: bool = False,  # For CNN - use residual blocks
        # Training hyperparameters
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        lr_min_ratio: float = 0.1,
        steps_per_epoch: int = 1000,
        batch_size: int = 256,
        max_epochs: int = 300,
        
        # Self-play parameters (set enable_selfplay=True to activate)
        enable_selfplay: bool = False,
        selfplay_buffer_size: int = 100000,
        selfplay_games_per_iteration: int = 50,
        selfplay_mcts_simulations: int = 30,
        selfplay_eval_mcts_simulations: int = 100,
        selfplay_parallel_simulations: int = 8, # for debugging, should be much higher on gpu
        selfplay_temperature_moves: int = 15,
        selfplay_update_interval: int = 10,  # Update "previous model" every N epochs
        selfplay_bootstrap_weight: float = 0.3,  # 0 = pure outcome, 1 = pure MCTS value
        selfplay_temporal_decay: float = 0.95,   # Decay bootstrap for later moves
        curriculum_data_path: str = None,

        # Evaluation parameters
        eval_minimax_depth: int = 4,
        eval_minimax_temperature: float = 0.5,
        eval_games_vs_minimax: int = 100,
        eval_games_vs_random: int = 100,
        eval_interval: int = 5,
        eval_use_mcts: bool = True,

        output_dir: str = None,
        **kwargs
    ):
        # Initialize base class
        super().__init__(
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            lr_min_ratio=lr_min_ratio,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            max_epochs=max_epochs,
            enable_selfplay=enable_selfplay,
            selfplay_buffer_size=selfplay_buffer_size,
            selfplay_games_per_iteration=selfplay_games_per_iteration,
            selfplay_mcts_simulations=selfplay_mcts_simulations,
            selfplay_eval_mcts_simulations=selfplay_eval_mcts_simulations,
            selfplay_parallel_simulations=selfplay_parallel_simulations,
            selfplay_temperature_moves=selfplay_temperature_moves,
            selfplay_update_interval=selfplay_update_interval,
            selfplay_bootstrap_weight=selfplay_bootstrap_weight,
            selfplay_temporal_decay=selfplay_temporal_decay,
            eval_minimax_depth=eval_minimax_depth,
            eval_minimax_temperature=eval_minimax_temperature,
            eval_games_vs_minimax=eval_games_vs_minimax,
            eval_games_vs_random=eval_games_vs_random,
            eval_interval=eval_interval,
            eval_use_mcts=eval_use_mcts,
            output_dir=output_dir,
            model_type=model_type,
            cnn_channels=cnn_channels,
            dropout=dropout,
            use_residual=use_residual,
            **kwargs
        )
        
        # Build model based on type
        if model_type == "cnn":
            self._build_cnn()
        elif model_type == "mlp":
            self._build_mlp()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        print(f"Created {model_type.upper()} baseline with {self._count_parameters():.2f}M parameters")
        if self.hparams.enable_selfplay:
            print(f"Self-play enabled: {self.hparams.selfplay_games_per_iteration} games/iter, "
                  f"{self.hparams.selfplay_mcts_simulations} MCTS sims")
    
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
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass matching TRM interface
        """
        boards = batch["boards"].float()
        current_player = batch["current_player"]
        batch_size = boards.shape[0]
        
        if self.hparams.model_type == "cnn":
            # Convert to channel format
            x = self._board_to_channels(boards, current_player)
            
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
            boards_canonical = self._canonicalize_board(boards, current_player)
            x = self.mlp(boards_canonical)
        
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
    
    def forward_for_mcts(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for MCTS - just calls regular forward"""
        return self.forward(batch)
    
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
        total_loss = policy_loss + value_loss
        
        # Compute metrics
        with torch.no_grad():
            pred_actions = outputs["policy"].argmax(dim=-1)
            target_actions = target_policy.argmax(dim=-1)
            policy_accuracy = (pred_actions == target_actions).float().mean()

            if self.hparams.enable_selfplay:
                # full distribution match for self-play
                kl_div = robust_kl_div(
                    pred_probs=outputs["policy"], 
                    target_probs=target_policy
                    )
            else:
                kl_div = torch.tensor(0.0)

            metrics = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "policy_kl_div": kl_div.item(),
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
        
        self.log_grad_norms()

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
        if self.hparams.enable_selfplay:
            self.log("train/policy_kl_div", metrics["policy_kl_div"], on_step=True)
        
        self.manual_step += 1
        
        return loss
    
    def log_grad_norms(self):
        # gradient monitoring
        total_grad_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        self.log("grad/grad_norm", total_grad_norm, on_step=True)

    def configure_optimizers(self):
        """Configure optimizer"""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95)
        )


def test_baseline():
    """Test the baseline model"""
    
    # Test CNN version
    print("Testing CNN baseline...")
    cnn_model = C4CNNModule(
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
        'current_player': torch.randint(1, 3, (32,)),
        'puzzle_identifiers': torch.zeros(32, dtype=torch.long)
    }
    
    outputs = cnn_model.forward(batch)
    print(f"CNN outputs: policy shape={outputs['policy'].shape}, value shape={outputs['value'].shape}")
    print(f"CNN parameters: {sum(p.numel() for p in cnn_model.parameters())/1e6:.2f}M")
    
    # Test MLP version
    print("\nTesting MLP baseline...")
    mlp_model = C4CNNModule(
        model_type="mlp",
        hidden_size=512,
        num_layers=4,
        batch_size=32
    )
    
    outputs = mlp_model.forward(batch)
    print(f"MLP outputs: policy shape={outputs['policy'].shape}, value shape={outputs['value'].shape}")
    print(f"MLP parameters: {sum(p.numel() for p in mlp_model.parameters())/1e6:.2f}M")
    
    # Test canonicalization
    print("\nTesting canonicalization...")
    board = torch.zeros(2, 42)
    board[0, 0] = 1  # Player 1 piece
    board[0, 1] = 2  # Player 2 piece
    board[1, 0] = 1
    board[1, 1] = 2
    
    current_player = torch.tensor([1, 2])  # First is P1's turn, second is P2's turn
    
    canonical = mlp_model._canonicalize_board(board, current_player)
    print(f"Original board[0]: {board[0, :3].tolist()} (P1's turn)")
    print(f"Canonical board[0]: {canonical[0, :3].tolist()} (should be same)")
    print(f"Original board[1]: {board[1, :3].tolist()} (P2's turn)")
    print(f"Canonical board[1]: {canonical[1, :3].tolist()} (should be swapped)")
    
    assert canonical[0, 0] == 1 and canonical[0, 1] == 2, "P1's turn should not swap"
    assert canonical[1, 0] == 2 and canonical[1, 1] == 1, "P2's turn should swap"
    print("Canonicalization: ✓")
    
    print("\nBaseline models ready!")



if __name__ == "__main__":
    test_baseline()