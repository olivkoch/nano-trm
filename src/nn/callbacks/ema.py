# src/nn/callbacks/ema.py
import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from lightning import Callback, LightningModule, Trainer


class EMACallback(Callback):
    """
    Exponential Moving Average callback for PyTorch Lightning.
    
    Maintains shadow weights and optionally uses them for validation/testing.
    """
    
    def __init__(
        self, 
        decay: float = 0.999,
        use_ema_for_validation: bool = True,
        use_ema_for_test: bool = True,
    ):
        super().__init__()
        self.decay = decay
        self.use_ema_for_validation = use_ema_for_validation
        self.use_ema_for_test = use_ema_for_test
        
        self.shadow: Dict[str, torch.Tensor] = {}
        self.original_weights: Dict[str, torch.Tensor] = {}
        self._ema_initialized = False
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        """Register all trainable parameters."""
        self._register(pl_module)
    
    def _register(self, module: nn.Module):
        """Initialize shadow weights from current model weights."""
        self.shadow = {}
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
        self._ema_initialized = True
    
    def on_train_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Update EMA weights after each training step."""
        if not self._ema_initialized:
            return
            
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # EMA update: shadow = decay * shadow + (1 - decay) * current
                    self.shadow[name].mul_(self.decay).add_(
                        param.data, alpha=1.0 - self.decay
                    )
    
    def _swap_to_ema(self, module: nn.Module):
        """Swap model weights with EMA weights, saving originals."""
        self.original_weights = {}
        for name, param in module.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.original_weights[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def _restore_original(self, module: nn.Module):
        """Restore original weights after validation."""
        for name, param in module.named_parameters():
            if name in self.original_weights:
                param.data.copy_(self.original_weights[name])
        self.original_weights = {}
    
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        if self.use_ema_for_validation and self._ema_initialized:
            self._swap_to_ema(pl_module)
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.use_ema_for_validation and self._ema_initialized:
            self._restore_original(pl_module)
    
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        if self.use_ema_for_test and self._ema_initialized:
            self._swap_to_ema(pl_module)
    
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.use_ema_for_test and self._ema_initialized:
            self._restore_original(pl_module)
    
    def state_dict(self) -> Dict[str, Any]:
        """For checkpointing."""
        return {
            "shadow": self.shadow,
            "decay": self.decay,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load from checkpoint."""
        self.shadow = state_dict["shadow"]
        self.decay = state_dict.get("decay", self.decay)
        self._ema_initialized = True
    
    def get_ema_model(self, module: nn.Module) -> nn.Module:
        """Return a deep copy with EMA weights (useful for saving)."""
        module_copy = copy.deepcopy(module)
        for name, param in module_copy.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
        return module_copy