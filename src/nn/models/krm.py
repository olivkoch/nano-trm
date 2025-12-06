"""
KME-TRM PyTorch Lightning Module.
Full integration with training infrastructure.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from src.nn.modules.krm_block import (
    KMEState,
    KMEReasoningModule,
    KMEOutputHead,
    KMEQHead,
    KMECategoricalEmbedding,
    KernelKMEReasoningModule,
)
from src.nn.modules.kme_embeddings import GeneralKMEEmbedding
from src.nn.modules.kme_attention import (
    GeneralKMEReasoningModule,
)

from src.nn.modules.trm_block import RotaryEmbedding, RotaryEmbedding2D
from src.nn.modules.sparse_embeddings import (
    CastedSparseEmbedding,
    CastedSparseEmbeddingSignSGD_Distributed,
)
from src.nn.modules.utils import compute_lr, stablemax_cross_entropy, trunc_normal_init_
from src.nn.utils import RankedLogger
from src.nn.utils.constants import IGNORE_LABEL_ID

try:
    from adam_atan2 import AdamATan2
except ImportError:
    AdamATan2 = None

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class GeneralKMETRMInnerCarry:
    """KME-based carry for H and L states."""
    z_H: KMEState
    z_L: KMEState


@dataclass
class GeneralKMETRMCarry:
    """Full carry structure for KME-TRM."""
    inner_carry: GeneralKMETRMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class KMETRMModule(LightningModule):
    """
    KME-based Transformer Reasoning Model.
    
    All internal representations are distributions in base space,
    encoded via Random Fourier Features for attention operations.
    """
    
    def __init__(
        self,
        # Architecture
        d_model : int = 512,
        d_base: int = 64,
        num_atoms: int = 8,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        hidden_size: int = 512,  # For output projection
        max_grid_size: int = 9,
        # Position encoding
        position_encoder: str = 'fourier',  # 'fourier' or 'discrete'
        max_coord: int = 16,
        num_frequencies: int = 8,
        init_bandwidth: float = 1.0,
        # TRM cycles
        H_cycles: int = 3,
        L_cycles: int = 6,
        N_supervision: int = 16,
        N_supervision_val: int = 16,
        # Training
        learning_rate: float = 1e-4,
        learning_rate_emb: float = 1e-2,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        halt_exploration_prob: float = 0.1,
        lr_min_ratio: float = 1.0,
        use_constant_lr: bool = False,
        # Puzzle embeddings
        puzzle_emb_dim: int = 512,
        puzzle_emb_len: int = 16,
        # Position encoding
        rope_theta: int = 10000,
        use_2d_rope: bool = False,
        use_kernel_attention: bool = False,
        # Data config (from datamodule)
        vocab_size: int = 0,
        num_puzzles: int = 0,
        batch_size: int = 0,
        pad_value: int = -1,
        seq_len: int = 0,
        output_dir: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.automatic_optimization = False
        self.forward_dtype = torch.float32
        
        log.info(
            f"Creating KME-TRM: d_base={d_base}, num_atoms={num_atoms}, "
            f"num_layers={num_layers}, num_heads={num_heads}, "
            f"vocab_size={vocab_size}, seq_len={seq_len}"
        )
        
        # # Token embedding: tokens → KME
        # self.token_emb = KMETokenEmbedding(
        #     vocab_size=vocab_size,
        #     d_base=d_base,
        #     num_atoms=num_atoms,
        # )
        # self.token_emb = nn.Embedding(vocab_size, d_model)

        # Project to base space for injection into KME states
        # self.input_proj = nn.Linear(d_model, d_base, bias=False)

        # Categorical KME embedding for digits
        # self.token_emb = KMECategoricalEmbedding(
        #     d_base=d_base,
        #     num_atoms=num_atoms,
        #     vocab_size=vocab_size,
        #     pad_token=0,
        #     eos_token=1,
        #     empty_token=2,
        #     digit_token_start=3,
        # )
        
        # # RoPE - operates on head_dim = d_base (after RFF encoding)
        # total_seq_len = seq_len + puzzle_emb_len
        # if use_2d_rope:
        #     self.pos_embedding = RotaryEmbedding2D(
        #         dim=d_base,  # RFF output dim = d_base
        #         prefix_len=puzzle_emb_len,
        #         max_grid_size=int(math.sqrt(seq_len)),
        #         base=rope_theta,
        #     )
        # else:
        #     self.pos_embedding = RotaryEmbedding(
        #         dim=d_base,
        #         max_position_embeddings=total_seq_len,
        #         base=rope_theta,
        #     )
        # General embedding with coordinates
        self.token_emb = GeneralKMEEmbedding(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
            position_encoder=position_encoder,
            max_coord=max_coord,
            num_frequencies=num_frequencies,
            special_tokens={'pad': 0, 'eos': 1, 'empty': 2},
        )
        
        # Reasoning module - NO RoPE
        self.lenet = GeneralKMEReasoningModule(
            num_layers=num_layers,
            d_base=d_base,
            num_atoms=num_atoms,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
            init_bandwidth=init_bandwidth,
        )

        # # Reasoning module
        # if use_kernel_attention:
        #     self.lenet = KernelKMEReasoningModule(
        #     num_layers=num_layers,
        #     d_base=d_base,
        #     num_atoms=num_atoms,
        #     num_heads=num_heads,
        #     ffn_expansion=ffn_expansion,
        #     init_bandwidth=init_bandwidth,
        # )
        # else:
        #     self.lenet = KMEReasoningModule(
        #         num_layers=num_layers,
        #         d_base=d_base,
        #         num_atoms=num_atoms,
        #         num_heads=num_heads,
        #         ffn_expansion=ffn_expansion,
        #         bandwidth=math.sqrt(d_base),
        #     )
        
        # Output heads
        self.lm_head = KMEOutputHead(
            d_base=d_base,
            num_atoms=num_atoms,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            bandwidth=math.sqrt(d_base),
        )
        
        self.q_head = KMEQHead(
            d_base=d_base,
            num_atoms=num_atoms,
            hidden_size=hidden_size,
            bandwidth=math.sqrt(d_base),
        )
        
        # Learnable initial states
        self.init_atoms = nn.Parameter(
            trunc_normal_init_(torch.empty(num_atoms, d_base), std=0.5)
        )
        self.init_log_weights = nn.Parameter(torch.zeros(num_atoms))
        
        # Puzzle embeddings (optional, for per-puzzle learning)
        if puzzle_emb_dim > 0 and num_puzzles > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                num_embeddings=num_puzzles,
                embedding_dim=puzzle_emb_dim,
                batch_size=batch_size,
                init_std=0.0,
                cast_to=self.forward_dtype,
            )
            self.puzzle_emb_len = puzzle_emb_len
            
            # Project puzzle embedding to d_base for input injection
            self.puzzle_to_base = nn.Linear(puzzle_emb_dim, d_base, bias=False)

            log.info(f"Created puzzle embeddings: num_puzzles={num_puzzles}")
        else:
            self.puzzle_emb = None
            self.puzzle_emb_len = 0
        
        self.carry = None
        self.last_step_time = None
        self.manual_step = 0
    
    def setup(self, stage: str):
        """Called by Lightning when setting up."""
        if stage == "fit":
            if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
                train_loader = self.trainer.datamodule.train_dataloader()
                steps_per_epoch = len(train_loader)
            else:
                steps_per_epoch = self.trainer.num_training_batches
            
            if self.trainer.max_epochs > 0:
                computed_total_steps = steps_per_epoch * self.trainer.max_epochs
            else:
                computed_total_steps = float("inf")
            
            if self.trainer.max_steps > 0:
                self.total_steps = min(self.trainer.max_steps, computed_total_steps)
            else:
                self.total_steps = computed_total_steps
            
            log.info(f"Training: {steps_per_epoch} steps/epoch, {self.total_steps} total steps")
    
    def initial_state(self, batch_size: int, seq_len: int, device: torch.device) -> KMEState:
        """Create initial KME state."""
        atoms = self.init_atoms.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, -1, -1
        ).clone().to(device)
        
        log_weights = self.init_log_weights.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, -1
        ).clone().to(device)
        
        return KMEState(atoms, log_weights)
    
    def inject_input(self, state: KMEState, input_state: KMEState) -> KMEState:
        """Additive injection of input into reasoning state."""
        return KMEState(
            atoms=state.atoms + input_state.atoms,
            log_weights=state.log_weights,
        )

    def embed_input(
        self,
        input_ids: torch.Tensor,
        coords: torch.Tensor,
        grid_size: Tuple[int, int],
        puzzle_identifiers: Optional[torch.Tensor] = None,
    ) -> KMEState:
        """
        Embed input with coordinates.
        
        Args:
            input_ids: [B, S] token ids
            coords: [B, S, 2] (row, col) coordinates
            grid_size: (H, W) for normalization
            puzzle_identifiers: [B] puzzle ids (optional)
        
        Returns:
            KMEState with atoms [B, S (+ prefix), num_atoms, d_base]
        """
        input_state = self.token_emb(input_ids, coords, grid_size)
        
        if self.puzzle_emb is not None and self.puzzle_emb_len > 0 and puzzle_identifiers is not None:
            B = input_ids.shape[0]
            device = input_ids.device
            
            # Puzzle embedding → project to d_base
            puzzle_emb = self.puzzle_emb(puzzle_identifiers)
            puzzle_proj = self.puzzle_to_base(puzzle_emb)
            
            # Create prefix state
            puzzle_atoms = puzzle_proj.unsqueeze(1).unsqueeze(2).expand(
                -1, self.puzzle_emb_len, self.hparams.num_atoms, -1
            ).clone()
            puzzle_log_weights = torch.zeros(
                B, self.puzzle_emb_len, self.hparams.num_atoms, device=device
            )
            
            return KMEState(
                atoms=torch.cat([puzzle_atoms, input_state.atoms], dim=1),
                log_weights=torch.cat([puzzle_log_weights, input_state.log_weights], dim=1),
            )
        
        return input_state
    
    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> GeneralKMETRMCarry:
        """Create initial carry for a batch."""
        batch_size = batch["input"].shape[0]
        seq_len = batch["input"].shape[1]
        device = batch["input"].device
        
        total_seq_len = seq_len + self.puzzle_emb_len
        
        return GeneralKMETRMCarry(
            inner_carry=GeneralKMETRMInnerCarry(
                z_H=self.initial_state(batch_size, total_seq_len, device),
                z_L=self.initial_state(batch_size, total_seq_len, device),
            ),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )
    
    def reset_carry(
        self,
        reset_flag: torch.Tensor,
        carry: GeneralKMETRMInnerCarry,
        seq_len: int,
    ) -> GeneralKMETRMInnerCarry:
        """Reset carry for halted sequences."""
        batch_size = reset_flag.shape[0]
        device = reset_flag.device
        
        init_state = self.initial_state(batch_size, seq_len + self.puzzle_emb_len, device)
        
        return GeneralKMETRMInnerCarry(
            z_H=KMEState.where(reset_flag, init_state, carry.z_H),
            z_L=KMEState.where(reset_flag, init_state, carry.z_L),
        )
    
    def _get_coords(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Generate coords from input shape, assuming square grid."""
        B, S = input_ids.shape
        device = input_ids.device
        
        grid_dim = int(math.sqrt(S))
        assert grid_dim * grid_dim == S, f"Sequence length {S} is not a perfect square"
        
        rows = torch.arange(grid_dim, device=device).unsqueeze(1).expand(grid_dim, grid_dim).reshape(-1)
        cols = torch.arange(grid_dim, device=device).unsqueeze(0).expand(grid_dim, grid_dim).reshape(-1)
        coords = torch.stack([rows, cols], dim=-1).unsqueeze(0).expand(B, -1, -1)
        
        return coords, (grid_dim, grid_dim)

    def inner_forward(
        self,
        carry: GeneralKMETRMInnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[GeneralKMETRMInnerCarry, torch.Tensor, torch.Tensor, Dict]:
        """Core reasoning forward pass."""
        input_ids = batch["input"]
        puzzle_identifiers = batch.get("puzzle_identifiers")        

        coords, grid_size = self._get_coords(input_ids)
        
        input_emb = self.embed_input(input_ids, coords, grid_size, puzzle_identifiers)
        
        z_H, z_L = carry.z_H, carry.z_L
        
        # H_cycles - 1 without grad
        with torch.no_grad():
            for _ in range(self.hparams.H_cycles - 1):
                for _ in range(self.hparams.L_cycles):
                    z_H_with_input = self.inject_input(z_H, input_emb)
                    z_L = self.lenet(z_L, z_H_with_input)
                
                z_H = self.lenet(z_H, z_L)
                z_H = self.inject_input(z_H, input_emb)
        
        # Final H-cycle with grad
        for _ in range(self.hparams.L_cycles):
            z_H_with_input = self.inject_input(z_H, input_emb)
            z_L = self.lenet(z_L, z_H_with_input)
        
        z_H = self.lenet(z_H, z_L)
        z_H = self.inject_input(z_H, input_emb)
        
        # Output heads - skip prefix
        output_state = KMEState(
            atoms=z_H.atoms[:, self.puzzle_emb_len:],
            log_weights=z_H.log_weights[:, self.puzzle_emb_len:],
        )
        
        logits = self.lm_head(output_state)
        q_logits = self.q_head(z_H, position=0).to(torch.float32)
        
        intermediates = {"z_H": z_H, "z_L": z_L}
        new_carry = GeneralKMETRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        
        return new_carry, logits, q_logits[..., 0], intermediates
    
    def forward(
        self,
        carry: GeneralKMETRMCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[GeneralKMETRMCarry, Dict[str, torch.Tensor], Dict]:
        """Full forward pass with carry management."""
        seq_len = batch["input"].shape[1]
        
        new_inner_carry = self.reset_carry(carry.halted, carry.inner_carry, seq_len)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v if v.shape == batch[k].shape else batch[k]
            )
            for k, v in carry.current_data.items()
        }
        
        new_inner_carry, logits, q_halt_logits, intermediates = self.inner_forward(
            new_inner_carry, new_current_data
        )
        
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
        }
        
        with torch.no_grad():
            new_steps = new_steps + 1
            n_supervision_steps = (
                self.hparams.N_supervision if self.training else self.hparams.N_supervision_val
            )
            
            is_last_step = new_steps >= n_supervision_steps
            halted = is_last_step
            
            if self.training and (self.hparams.N_supervision > 1):
                halted = halted | (q_halt_logits > 0)
                
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.hparams.halt_exploration_prob
                ) * torch.randint_like(new_steps, low=2, high=self.hparams.N_supervision + 1)
                halted = halted & (new_steps >= min_halt_steps)
        
        return GeneralKMETRMCarry(new_inner_carry, new_steps, halted, new_current_data), outputs, intermediates
    
    def atom_diversity_loss(self, state: KMEState) -> torch.Tensor:
        """Prevent atoms from collapsing to same values."""
        atoms = state.atoms  # [B, S, M, d_base]
        B, S, M, D = atoms.shape
        
        # Compute pairwise squared distances between atoms at each position
        # atoms: [B, S, M, D] -> [B, S, M, 1, D] - [B, S, 1, M, D] -> [B, S, M, M]
        diff = atoms.unsqueeze(3) - atoms.unsqueeze(2)  # [B, S, M, M, D]
        sq_dist = (diff ** 2).sum(dim=-1)  # [B, S, M, M]
        
        # Mask diagonal (self-distance = 0)
        mask = ~torch.eye(M, dtype=torch.bool, device=atoms.device)
        
        # Mean pairwise distance - we want this to be LARGE
        mean_dist = sq_dist[..., mask].mean()
        
        # Return negative (minimize this = maximize distance)
        return -mean_dist

    def compute_loss_and_metrics(
        self,
        carry: GeneralKMETRMCarry,
        batch: Dict[str, torch.Tensor],
    ):
        """Compute loss and metrics."""
        new_carry, outputs, intermediates = self.forward(carry, batch)
        labels = new_carry.current_data["output"]
        
        with torch.no_grad():
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)
            
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            
            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            valid_metrics = new_carry.halted & (loss_counts > 0)
            
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics, (is_correct.float() / loss_divisor).sum(-1), 0
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"].squeeze() >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }
        
        lm_loss = (
            stablemax_cross_entropy(
                outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask
            )
            / loss_divisor
        ).sum()
        
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        
        total_loss = lm_loss + 0.5 * q_halt_loss
        
        # Add diversity loss to prevent atom collapse
        if self.training:
            # Use NON-DETACHED states for diversity loss
            z_H = intermediates["z_H"]
            z_L = intermediates["z_L"]
            
            diversity_loss_H = self.atom_diversity_loss(z_H)
            diversity_loss_L = self.atom_diversity_loss(z_L)
            diversity_loss = diversity_loss_H + diversity_loss_L
            
            # Scale factor - tune this (start with 0.1)
            total_loss = total_loss + 0.1 * diversity_loss
            
            metrics["diversity_loss"] = diversity_loss.detach()
            
        self.log_debugging_metrics(new_carry, outputs, labels, metrics)

        return new_carry, total_loss, metrics, new_carry.halted.all()
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""

        batch_size = batch["input"].shape[0]

        try:
            opts = self.optimizers()
            if not isinstance(opts, list):
                opts = [opts]
        except RuntimeError:
            if not hasattr(self, "_optimizers"):
                raise RuntimeError("No optimizer available.")
            opts = self._optimizers
        
        if self.carry is None or batch["input"].shape[1] != self.carry.current_data.get("input", torch.empty(0)).shape[-1]:
            self.carry = self.initial_carry(batch)
        
        self.carry, loss, metrics, all_halted = self.compute_loss_and_metrics(self.carry, batch)

        scaled_loss = loss / batch_size
        scaled_loss.backward()

        # Gradient monitoring
        with torch.no_grad():
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=float('inf')
            ).item()
            self.log('grad/total_norm', total_grad_norm, on_step=True, prog_bar=True)
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Optimizer step with LR scheduling
        current_step = self.manual_step
        total_steps = getattr(self, "total_steps", self.hparams.max_steps)
        
        base_lrs = [self.hparams.learning_rate]
        if len(opts) > 1:
            base_lrs.append(self.hparams.learning_rate_emb)
        
        for opt, base_lr in zip(opts, base_lrs):
            if current_step < self.hparams.warmup_steps or not self.hparams.use_constant_lr:
                lr = compute_lr(
                    base_lr=base_lr,
                    lr_warmup_steps=self.hparams.warmup_steps,
                    lr_min_ratio=self.hparams.lr_min_ratio,
                    current_step=current_step,
                    total_steps=total_steps,
                )
            else:
                lr = base_lr
            
            if hasattr(opt, "_optimizer"):
                for pg in opt._optimizer.param_groups:
                    pg["lr"] = lr
                opt._optimizer.step()
                opt._optimizer.zero_grad()
            else:
                for pg in opt.param_groups:
                    pg["lr"] = lr
                opt.step()
                opt.zero_grad()
        
        self.log("train/lr", lr, on_step=True)
        
        # Log metrics
        if metrics.get("count", 0) > 0:
            with torch.no_grad():
                count = metrics["count"]
                self.log("train/accuracy", metrics["accuracy"] / count, on_step=True)
                self.log("train/exact_accuracy", metrics["exact_accuracy"] / count, prog_bar=True, on_step=True)
                self.log("train/q_halt_accuracy", metrics["q_halt_accuracy"] / count, on_step=True)
                self.log("train/steps", metrics["steps"] / count, prog_bar=True, on_step=True)
                self.log("train/lm_loss", metrics["lm_loss"] / batch_size, on_step=True)
                self.log("train/q_halt_loss", metrics["q_halt_loss"] / batch_size, on_step=True)
        
        assert not torch.isnan(metrics.get("lm_loss")), f"LM loss is NaN at step {self.manual_step}"

        self.manual_step += 1
        
        return loss
    
    def log_debugging_metrics(self, carry: GeneralKMETRMCarry, outputs, labels, metrics):
        with torch.no_grad():
            z_H = carry.inner_carry.z_H
            z_L = carry.inner_carry.z_L
            
            # 1. Atom diversity: are atoms collapsed or spread?
            atom_std_H = z_H.atoms.std(dim=2).mean()  # Std across atoms
            atom_std_L = z_L.atoms.std(dim=2).mean()
            self.log("diag/atom_std_H", atom_std_H, on_step=True)
            self.log("diag/atom_std_L", atom_std_L, on_step=True)
            
            # 2. Weight entropy: are weights uniform or peaked?
            weights_H = z_H.weights
            entropy_H = -(weights_H * torch.log(weights_H + 1e-10)).sum(dim=-1).mean()
            max_entropy = math.log(self.hparams.num_atoms)
            self.log("diag/weight_entropy_H", entropy_H / max_entropy, on_step=True)  # Normalized 0-1
            
            # 3. Atom norm: are atoms exploding or vanishing?
            atom_norm_H = z_H.atoms.norm(dim=-1).mean()
            self.log("diag/atom_norm_H", atom_norm_H, on_step=True)
            
            # 4. Per-position prediction confidence
            logits = outputs["logits"]
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values.mean()
            self.log("diag/pred_confidence", confidence, on_step=True)
            
            # 5. How many positions are correct per puzzle?
            correct_per_puzzle = (outputs["preds"] == labels).float().sum(dim=-1)
            self.log("diag/correct_positions_mean", correct_per_puzzle.mean(), on_step=True)
            self.log("diag/correct_positions_max", correct_per_puzzle.max(), on_step=True)

            if "diversity_loss" in metrics:
                self.log("diag/diversity_loss", metrics["diversity_loss"], on_step=True)

        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step."""
        batch_size = batch["input"].shape[0]
        
        with torch.no_grad():
            carry = self.initial_carry(batch)
            
            accumulated_metrics = {}
            total_loss = 0.0
            n_steps = 0
            
            while True:
                carry, loss, metrics, all_halted = self.compute_loss_and_metrics(carry, batch)
                
                for k, v in metrics.items():
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0) + v.item()
                
                total_loss += loss.item()
                n_steps += 1
                
                if all_halted:
                    break
            
            count = accumulated_metrics.get("count", batch_size)
            if count > 0:
                avg_metrics = {
                    "val/loss": total_loss / (n_steps * batch_size),
                    "val/accuracy": accumulated_metrics.get("accuracy", 0) / count,
                    "val/exact_accuracy": accumulated_metrics.get("exact_accuracy", 0) / count,
                    "val/q_halt_accuracy": accumulated_metrics.get("q_halt_accuracy", 0) / count,
                    "val/steps": accumulated_metrics.get("steps", 0) / count,
                    "val/lm_loss": accumulated_metrics.get("lm_loss", 0) / (n_steps * batch_size),
                    "val/q_halt_loss": accumulated_metrics.get("q_halt_loss", 0) / (n_steps * batch_size),
                }
            else:
                avg_metrics = {f"val/{k}": 0.0 for k in ["loss", "accuracy", "exact_accuracy", "q_halt_accuracy", "steps", "lm_loss", "q_halt_loss"]}
            
            for name, value in avg_metrics.items():
                self.log(name, value, on_step=False, on_epoch=True, prog_bar=(name in ["val/loss", "val/exact_accuracy"]), sync_dist=True)
            
            return avg_metrics
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """Configure optimizers."""
        optimizers = []
        
        if AdamATan2 is not None:
            main_opt = AdamATan2(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.95),
            )
        else:
            main_opt = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.95),
            )
        optimizers.append(main_opt)
        
        if hasattr(self, "puzzle_emb") and self.puzzle_emb is not None:
            self.puzzle_emb.local_weights = self.puzzle_emb.local_weights.detach().requires_grad_(True)
            sparse_opt = CastedSparseEmbeddingSignSGD_Distributed(
                self.puzzle_emb.buffers(),
                lr=self.hparams.learning_rate_emb,
                weight_decay=self.hparams.weight_decay,
                world_size=1,
            )
            optimizers.append(sparse_opt)
        
        return optimizers