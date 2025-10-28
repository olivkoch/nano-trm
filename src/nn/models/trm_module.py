"""
HRM PyTorch Lightning Module - Following Figure 2 pseudocode exactly
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW

from src.nn.modules.hybrid_vision_embeddings import HybridDINOEmbedding
from src.nn.modules.sparse_embeddings import CastedSparseEmbedding
from src.nn.modules.trm_block import TransformerBlock
from src.nn.modules.utils import trunc_normal_init
from src.nn.utils import RankedLogger
from src.nn.utils.constants import PAD_VALUE

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class TRMCarry:
    """Carry structure for maintaining state across steps."""

    z_H: torch.Tensor  # High-level state (y in your code)
    z_L: torch.Tensor  # Low-level state (z in your code)
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]  # Stores current batch data


class TRMModule(LightningModule):
    """
    HRM implementation following Figure 2 pseudocode exactly.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_layers: int = 4,  # HRM uses 4 layers
        max_grid_size: int = 30,
        num_colors: int = 10,
        n_latent_recursions: int = 2,  # n in HRM
        T_deep_recursions: int = 2,  # T in HRM
        N_supervision: int = 16,
        ffn_expansion: int = 2,
        learning_rate: float = 1e-4,
        learning_rate_emb: float = 1e-2,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        use_dino_embeddings: bool = True,
        num_puzzles: int = 1000,  # Should be set from datamodule
        puzzle_emb_dim: int = 512,  # Puzzle embedding dimension
        puzzle_emb_len: int = 16,  # How many tokens for puzzle embedding
        output_dir: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # CRITICAL: Manual optimization
        self.automatic_optimization = False

        # Model components
        if use_dino_embeddings:
            self.input_embedding = HybridDINOEmbedding(
                num_colors=num_colors + 1, hidden_size=hidden_size, freeze_dino=True
            )
        else:
            self.input_embedding = nn.Embedding(
                num_colors + 1, hidden_size, padding_idx=PAD_VALUE
            )  # 0 (padding) + 10 colors

        # TODO: replace with RoPE or Casted Embeddings
        self.pos_embedding = nn.Embedding(
            max_grid_size * max_grid_size + puzzle_emb_len, 
            hidden_size
        )
                
        # a single network (not two separate networks)
        self.lenet = self._build_transformer(
            hidden_size, num_layers, num_heads=8, ffn_expansion=ffn_expansion, dropout=0.1
        )

        # Output heads
        self.output_head = nn.Linear(hidden_size, num_colors)

        self.embed_scale = math.sqrt(self.hparams.hidden_size)

        # Halting head for adaptive computation
        # Only learn a halting probability through a Binary-Cross-Entropy loss of having
        # reached the correct solution
        self.Q_head = nn.Linear(hidden_size, 1)  # Q_head returns 1 value: q[0]

        # State for carry (persisted across training steps)
        self.carry = None

        self.register_buffer("z_H_init", trunc_normal_init((hidden_size,), std=0.02))
        self.register_buffer("z_L_init", trunc_normal_init((hidden_size,), std=0.02))

        # Initialize weights
        self.apply(self._init_weights)

        # Add puzzle embeddings
        if puzzle_emb_dim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                num_puzzles, 
                puzzle_emb_dim,
                init_std=0.0,  # Reference uses 0 init
                cast_to=torch.bfloat16
            )
            self.puzzle_emb_len = puzzle_emb_len
        else:
            self.puzzle_emb = None
            self.puzzle_emb_len = 0

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TRMCarry:
        """Create initial carry state for a batch."""
        batch_size = batch["input"].shape[0]
        height, width = batch["input"].shape[1], batch["input"].shape[2]
        seq_len = height * width

        # Account for puzzle embedding positions
        total_seq_len = seq_len + self.puzzle_emb_len

        # Use the actual buffer names
        z_H_init = self.z_H_init.view(1, 1, -1)
        z_L_init = self.z_L_init.view(1, 1, -1)

        current_data = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                current_data[k] = torch.empty_like(v)
            else:
                current_data[k] = v

        return TRMCarry(
            z_H=z_H_init.expand(batch_size, total_seq_len, -1).clone(),
            z_L=z_L_init.expand(batch_size, total_seq_len, -1).clone(),
            steps=torch.zeros(batch_size, dtype=torch.int32, device=self.device),
            halted=torch.ones(batch_size, dtype=torch.bool, device=self.device),
            current_data=current_data,
        )

    def _init_weights(self, module):
        """Initialize weights with small values for stability."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _net_forward(self, net: nn.ModuleList, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward through Transformer network.
        Sum inputs, then pass through all Transformer blocks.
        """
        x = sum(inputs)

        for block in net:
            x = block(x)

        return x

    def _build_transformer(
        self, hidden_size: int, num_layers: int, num_heads: int, ffn_expansion: int, dropout: float
    ) -> nn.ModuleList:
        """Build Transformer as specified in paper."""
        return nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, ffn_expansion, dropout)
                for _ in range(num_layers)
            ]
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step that implements supervision through multiple forward passes.
        Each sequence can run up to N_supervision (halt_max_steps) times.
        """

        # Handle case when not attached to trainer (for testing)
        try:
            opts = self.optimizers()
            if not isinstance(opts, list):
                opts = [opts]
        except RuntimeError:
            # For testing without trainer
            if not hasattr(self, '_optimizers'):
                raise RuntimeError("No optimizer available.")
            opts = self._optimizers if isinstance(self._optimizers, list) else [self._optimizers]


        # Initialize carry if first batch
        if self.carry is None:
            self.carry = self.initial_carry(batch)

        total_loss = 0.0
        n_active_steps = 0

        # Run up to N_supervision steps for each sequence
        for _ in range(self.hparams.N_supervision):
            # Forward pass
            self.carry, outputs = self.forward(self.carry, batch)

            # Compute loss only for non-halted sequences
            active_mask = ~self.carry.halted
            if active_mask.any():
                y_true = batch["output"].flatten(start_dim=1)
                logits = outputs["logits"]

                # Compute loss
                loss = (
                    F.cross_entropy(
                        logits.view(-1, self.hparams.num_colors),
                        y_true.view(-1),
                        ignore_index=PAD_VALUE,
                        reduction="none",
                    )
                    .view(logits.shape[0], -1)
                    .mean(dim=1)
                )  # Per-sequence loss

                # Only backprop for active sequences
                masked_loss = (loss * active_mask.float()).sum() / active_mask.float().sum()
                masked_loss.backward()

                total_loss += masked_loss.item()
                n_active_steps += 1

            # Check if all sequences have halted
            if self.carry.halted.all():
                break

        # Single optimizer step after all supervision steps
        # Only clip non-sparse gradients
        dense_params = [p for p in self.parameters() if p.grad is not None and not p.grad.is_sparse]
        if dense_params:
            torch.nn.utils.clip_grad_norm_(dense_params, max_norm=1.0)

            # Step all optimizers
        for opt in opts:
            opt.step()
            opt.zero_grad()

        avg_loss = total_loss / max(1, n_active_steps)

        self.log("train/loss", avg_loss, prog_bar=True)
        self.log("train/avg_steps", self.carry.steps.float().mean(), prog_bar=True)

        return torch.tensor(avg_loss)

    def forward(self, carry: TRMCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TRMCarry, Dict[str, torch.Tensor]]:
        """Forward pass matching reference implementation with puzzle embeddings."""
        batch_size = batch['input'].shape[0]
        height, width = batch['input'].shape[1], batch['input'].shape[2]
        seq_len = height * width
        
        # Reset states for halted sequences
        reset_mask = carry.halted.view(-1, 1, 1)
        
        z_H_init = self.z_H_init.view(1, 1, -1)
        z_L_init = self.z_L_init.view(1, 1, -1)
        
        # Account for puzzle embedding positions in sequence length
        total_seq_len = seq_len + self.puzzle_emb_len
        
        # Expand init states to total sequence length
        new_z_H = torch.where(reset_mask, 
                            z_H_init.expand(batch_size, total_seq_len, -1), 
                            carry.z_H)
        new_z_L = torch.where(reset_mask, 
                            z_L_init.expand(batch_size, total_seq_len, -1), 
                            carry.z_L)
        
        # Reset steps for halted sequences
        new_steps = torch.where(carry.halted, 
                                torch.zeros_like(carry.steps), 
                                carry.steps)
        
        # Update current_data for halted sequences
        new_current_data = {}
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                mask = carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1))
                new_current_data[k] = torch.where(mask, batch[k], carry.current_data[k])
            else:
                new_current_data[k] = batch[k]
        
        # Get token embeddings from current data
        x_input = new_current_data['input']
        token_emb = self.embed_scale * self.input_embedding(x_input)
        token_emb = token_emb.view(batch_size, seq_len, self.hparams.hidden_size)
        
        # Get and process puzzle embeddings
        if self.puzzle_emb is not None and self.puzzle_emb_len > 0:
            puzzle_ids = new_current_data['puzzle_identifiers']
            puzzle_embedding = self.puzzle_emb(puzzle_ids)  # [B, puzzle_emb_dim]
            
            # Reshape puzzle embedding to sequence positions
            # If puzzle_emb_dim doesn't divide evenly into hidden_size, pad
            pad_count = self.puzzle_emb_len * self.hparams.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            
            # Reshape to [B, puzzle_emb_len, hidden_size]
            puzzle_embedding = puzzle_embedding.view(batch_size, self.puzzle_emb_len, self.hparams.hidden_size)
            
            # Concatenate puzzle embeddings with token embeddings
            x_emb = torch.cat([puzzle_embedding, token_emb], dim=1)  # [B, puzzle_len + seq_len, D]
        else:
            x_emb = token_emb
            total_seq_len = seq_len
        
        # Apply position embeddings 
        # Position embeddings should cover total_seq_len now
        if hasattr(self, 'pos_embedding'):
            positions = torch.arange(total_seq_len, device=x_emb.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embedding(positions)
            x_emb = x_emb + pos_emb

        # Deep recursion with updated states
        z_H, z_L = new_z_H, new_z_L
        
        # H_cycles-1 without gradient
        with torch.no_grad():
            for _ in range(self.hparams.T_deep_recursions - 1):
                for _ in range(self.hparams.n_latent_recursions):
                    z_L = self._net_forward(self.lenet, z_L, z_H + x_emb)
                z_H = self._net_forward(self.lenet, z_H, z_L)
        
        # Last H_cycle WITH gradient
        for _ in range(self.hparams.n_latent_recursions):
            z_L = self._net_forward(self.lenet, z_L, z_H + x_emb)
        z_H = self._net_forward(self.lenet, z_H, z_L)
        
        # Compute outputs - only from grid positions (skip puzzle positions)
        if self.puzzle_emb_len > 0:
            # Skip puzzle embedding positions when computing output
            grid_z_H = z_H[:, self.puzzle_emb_len:, :]  # [B, seq_len, D]
            logits = self.output_head(grid_z_H)
            # Use first puzzle position for Q-head (like reference)
            q_halt_logits = self.Q_head(z_H[:, 0, :])
        else:
            logits = self.output_head(z_H)
            q_halt_logits = self.Q_head(z_H[:, 0, :])
        
        outputs = {
            'logits': logits,  # [B, seq_len, num_colors]
            'q_halt_logits': q_halt_logits  # [B, 1]
        }
        
        # Update carry for next iteration
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.hparams.N_supervision
            
            # Halting logic
            if self.training:
                halted = is_last_step | (q_halt_logits.squeeze() > 0)
            else:
                halted = is_last_step
        
        new_carry = TRMCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
            steps=new_steps,
            halted=halted,
            current_data=new_current_data
        )
        
        return new_carry, outputs

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Validation step following reference approach.
        Since we need complete solutions for validation, we run up to halt_max_steps
        iterations, similar to how the model would process multiple batches during training.
        """

        x_input = batch["input"]
        y_true = batch["output"]
        batch_size, height, width = x_input.shape

        with torch.no_grad():
            # Create fresh carry for validation (don't use self.carry from training)
            carry = self.initial_carry(batch)

            # Run up to halt_max_steps (N_supervision) iterations
            # This simulates what would happen across multiple training batches
            final_logits = None
            steps_taken = []

            for step in range(self.hparams.N_supervision):
                # Forward pass (single iteration)
                carry, outputs = self.forward(carry, batch)

                # Keep track of latest predictions
                final_logits = outputs["logits"]

                # Track when each sequence halts
                if step == 0:
                    steps_taken = torch.ones_like(carry.steps)
                else:
                    # Only update steps for sequences that haven't halted yet
                    steps_taken = torch.where(carry.halted, steps_taken, carry.steps)

                # If all sequences have halted, we can stop early
                if carry.halted.all():
                    break

            # Compute metrics using final predictions
            y_pred = final_logits.view(batch_size, height, width, self.hparams.num_colors)

            # Loss
            loss = F.cross_entropy(
                y_pred.flatten(start_dim=0, end_dim=2),  # [B*H*W, num_colors]
                y_true.flatten(),  # [B*H*W]
                ignore_index=PAD_VALUE,
            )

            # Accuracy metrics
            pred_classes = y_pred.argmax(dim=-1)
            mask = (y_true != PAD_VALUE).float()
            correct = (pred_classes == y_true).float()

            # Per-element accuracy
            accuracy = (correct * mask).sum() / mask.sum() if mask.sum() > 0 else correct.mean()

            # Exact accuracy (entire grid correct)
            pred_flat = pred_classes.view(batch_size, -1)
            true_flat = y_true.view(batch_size, -1)
            mask_flat = mask.view(batch_size, -1)

            # Check if all non-masked elements are correct for each sample
            exact_correct = ((pred_flat == true_flat) | (mask_flat == 0)).all(dim=1)
            exact_accuracy = exact_correct.float().mean()

            # Log metrics
            self.log(
                "val/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
            )
            self.log(
                "val/accuracy",
                accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "val/exact_accuracy",
                exact_accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "val/avg_steps",
                steps_taken.float().mean(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

            # Optional: Log which tasks were solved correctly
            if exact_correct.any() and batch_idx % 100 == 0:
                batch_ids = batch.get("task_ids", torch.arange(batch_size))
                correct_indices = torch.nonzero(exact_correct).squeeze(-1)
                for idx in correct_indices[:5]:  # Log first 5 correct solutions
                    log.info(f"Task {batch_ids[idx]} solved correctly in {steps_taken[idx]} steps")

        return {
            "loss": loss,
            "accuracy": accuracy,
            "exact_accuracy": exact_accuracy,
            "avg_steps": steps_taken.float().mean(),
        }

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step - same as validation."""
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_start(self):
        """Don't interfere with training carry during validation."""
        # Note: We DON'T reset self.carry here because that's for training
        pass

    def on_train_epoch_start(self):
        """Reset carry at the beginning of each training epoch."""
        self.carry = None

    def configure_optimizers(self):
        """Configure optimizer with different learning rates and handle sparse gradients."""
        
        base_lr = self.hparams.learning_rate / self.hparams.N_supervision
        embedding_lr = self.hparams.learning_rate_emb / self.hparams.N_supervision
        
        # Separate parameters into groups
        embedding_params = []
        sparse_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if 'puzzle_emb' in name:
                # Sparse embeddings need SparseAdam
                sparse_params.append(param)
            elif 'input_embedding' in name or 'pos_embedding' in name:
                # Regular embeddings
                embedding_params.append(param)
            else:
                # Everything else (transformer, heads, etc.)
                other_params.append(param)
        
        # Create optimizers list
        optimizers = []
        
        # Main optimizer for dense parameters
        dense_params = []
        if other_params:
            dense_params.append({'params': other_params, 'lr': base_lr})
        if embedding_params:
            dense_params.append({'params': embedding_params, 'lr': embedding_lr})
        
        if dense_params:
            main_optimizer = torch.optim.AdamW(
                dense_params,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.95)
            )
            optimizers.append(main_optimizer)
        
        # Separate optimizer for sparse embeddings
        if sparse_params:
            sparse_optimizer = torch.optim.SparseAdam(
                sparse_params,
                lr=embedding_lr,
                betas=(0.9, 0.95)
            )
            optimizers.append(sparse_optimizer)
        
        return optimizers

        # Just linear warmup, no decay after
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / max(1, self.hparams.warmup_steps)
            else:
                return 1.0  # Constant LR after warmup

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
