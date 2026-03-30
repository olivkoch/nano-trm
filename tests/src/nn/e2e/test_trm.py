"""End-to-end tests for TRM model with synthetic data."""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from adam_atan2 import AdamATan2
    HAS_ADAM_ATAN2 = True
except ImportError:
    HAS_ADAM_ATAN2 = False

from src.nn.models.trm import TRMModule
from src.nn.modules.sparse_embeddings import (
    CastedSparseEmbedding,
    CastedSparseEmbeddingSignSGD_Distributed,
)


class SyntheticDataModule:
    """Mock datamodule with synthetic data for functional testing."""
    
    def __init__(
        self,
        batch_size: int = 8,
        num_puzzles: int = 5,
        grid_size: int = 10,
        num_colors: int = 10,
        pad_value: int = 10,
    ):
        self.batch_size = batch_size
        self.num_puzzles = num_puzzles
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.pad_value = pad_value
        
    def setup(self):
        """Setup method to match ARCDataModuleWithPuzzles interface."""
        pass
    
    def train_dataloader(self):
        """Create a synthetic train dataloader."""
        # Create random input grids (batch_size, grid_size, grid_size)
        inputs = torch.randint(
            0, self.num_colors, (self.batch_size, self.grid_size, self.grid_size)
        )
        
        # Create random output grids
        outputs = torch.randint(
            0, self.num_colors, (self.batch_size, self.grid_size, self.grid_size)
        )
        
        # Flatten the grids to sequences
        inputs_flat = inputs.view(self.batch_size, -1)
        outputs_flat = outputs.view(self.batch_size, -1)
        
        # Create random puzzle identifiers
        puzzle_ids = torch.randint(0, self.num_puzzles, (self.batch_size,))
        
        # Create dataset and loader
        dataset = TensorDataset(inputs_flat, outputs_flat, puzzle_ids)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        return loader
    
    def get_batch(self):
        """Get a single batch formatted like ARCDataModuleWithPuzzles."""
        loader = self.train_dataloader()
        inputs, outputs, puzzle_ids = next(iter(loader))
        
        return {
            'input': inputs,
            'output': outputs,
            'puzzle_identifiers': puzzle_ids,
        }


@pytest.fixture
def synthetic_datamodule():
    """Fixture providing a synthetic datamodule."""
    dm = SyntheticDataModule(
        batch_size=8,
        num_puzzles=5,
        grid_size=10,
        num_colors=10,
        pad_value=10,
    )
    dm.setup()
    return dm


@pytest.fixture
def synthetic_batch(synthetic_datamodule):
    """Fixture providing a synthetic batch."""
    return synthetic_datamodule.get_batch()


@pytest.fixture
def trm_model(synthetic_datamodule):
    """Fixture providing a TRM model."""
    seq_len = synthetic_datamodule.grid_size * synthetic_datamodule.grid_size
    model = TRMModule(
        hidden_size=64,
        num_layers=2,
        puzzle_emb_dim=64,
        puzzle_emb_len=4,
        N_supervision=3,
        H_cycles=2,
        L_cycles=3,
        num_puzzles=synthetic_datamodule.num_puzzles,
        batch_size=synthetic_datamodule.batch_size,
        pad_value=synthetic_datamodule.pad_value,
        vocab_size=synthetic_datamodule.num_colors + 1,  # colors + pad
        seq_len=seq_len,
        max_grid_size=synthetic_datamodule.grid_size,
    )
    return model


def test_synthetic_data_loading(synthetic_datamodule, synthetic_batch):
    """Test that synthetic data loads correctly with puzzle IDs."""
    assert 'input' in synthetic_batch
    assert 'output' in synthetic_batch
    assert 'puzzle_identifiers' in synthetic_batch
    
    # Check shapes - inputs are flattened
    seq_len = synthetic_datamodule.grid_size * synthetic_datamodule.grid_size
    assert synthetic_batch['input'].shape == (8, seq_len)
    assert synthetic_batch['output'].shape == (8, seq_len)
    assert synthetic_batch['puzzle_identifiers'].shape == (8,)
    
    # Check puzzle IDs are in valid range
    assert synthetic_batch['puzzle_identifiers'].min() >= 0
    assert synthetic_batch['puzzle_identifiers'].max() < synthetic_datamodule.num_puzzles
    
    # Check color values are in valid range
    assert synthetic_batch['input'].min() >= 0
    assert synthetic_batch['input'].max() < synthetic_datamodule.num_colors


def test_model_initial_carry(trm_model, synthetic_batch):
    """Test model initial carry creation."""
    carry = trm_model.initial_carry(synthetic_batch)
    
    assert hasattr(carry, 'inner_carry')
    assert hasattr(carry.inner_carry, 'z_H')
    assert hasattr(carry.inner_carry, 'z_L')
    assert carry.inner_carry.z_H.shape[0] == synthetic_batch['input'].shape[0]
    assert carry.inner_carry.z_L.shape[0] == synthetic_batch['input'].shape[0]


def test_model_forward_pass(trm_model, synthetic_batch):
    """Test model forward pass with carry."""
    trm_model.eval()
    
    # Test initial carry
    carry = trm_model.initial_carry(synthetic_batch)
    
    # Test forward pass
    with torch.no_grad():
        new_carry, outputs = trm_model.forward(carry, synthetic_batch)
    
    assert 'logits' in outputs
    assert 'q_halt_logits' in outputs
    
    # Check output shapes
    batch_size = synthetic_batch['input'].shape[0]
    assert outputs['logits'].shape[0] == batch_size
    assert outputs['q_halt_logits'].shape[0] == batch_size


def test_training_step(trm_model, synthetic_batch):
    """Test a single training step."""
    # Setup optimizers
    base_lr = 1e-4 / trm_model.hparams.N_supervision
    embedding_lr = 1e-3 / trm_model.hparams.N_supervision
    
    optimizers = []
    
    # Collect parameters for main optimizer (excluding sparse embeddings)
    main_params = []
    for name, param in trm_model.named_parameters():
        if "puzzle_emb" not in name:
            main_params.append(param)
    
    # Main optimizer
    if main_params:
        if HAS_ADAM_ATAN2:
            main_opt = AdamATan2(
                main_params, lr=base_lr, weight_decay=0.01, betas=(0.9, 0.95)
            )
        else:
            main_opt = torch.optim.AdamW(
                main_params, lr=base_lr, weight_decay=0.01, betas=(0.9, 0.95)
            )
        optimizers.append(main_opt)
    
    # Sparse embedding optimizer
    for _, module in trm_model.named_modules():
        if isinstance(module, CastedSparseEmbedding):
            sparse_opt = CastedSparseEmbeddingSignSGD_Distributed(
                [module.local_weights, module.local_ids, module.weights],
                world_size=1,
                lr=embedding_lr,
                weight_decay=0.01,
            )
            optimizers.append(sparse_opt)
            break
    
    # Store optimizers for testing
    trm_model._optimizers = optimizers
    
    # Initialize total_steps (normally set by Lightning's setup method)
    trm_model.total_steps = 1000
    
    # Simulate training step
    trm_model.train()
    trm_model.carry = None
    
    # Run one training step
    loss = trm_model.training_step(synthetic_batch, 0)
    
    # Assertions
    assert loss is not None
    assert torch.isfinite(loss)
    assert loss.requires_grad
    
    # Check carry state was updated
    assert trm_model.carry is not None
    assert hasattr(trm_model.carry, 'steps')
    assert hasattr(trm_model.carry, 'halted')


def test_multiple_training_steps(trm_model, synthetic_datamodule):
    """Test multiple consecutive training steps."""
    # Setup optimizers
    base_lr = 1e-4 / trm_model.hparams.N_supervision
    embedding_lr = 1e-3 / trm_model.hparams.N_supervision
    
    optimizers = []
    main_params = []
    
    for name, param in trm_model.named_parameters():
        if "puzzle_emb" not in name:
            main_params.append(param)
    
    if main_params:
        if HAS_ADAM_ATAN2:
            main_opt = AdamATan2(
                main_params, lr=base_lr, weight_decay=0.01, betas=(0.9, 0.95)
            )
        else:
            main_opt = torch.optim.AdamW(
                main_params, lr=base_lr, weight_decay=0.01, betas=(0.9, 0.95)
            )
        optimizers.append(main_opt)
    
    for _, module in trm_model.named_modules():
        if isinstance(module, CastedSparseEmbedding):
            sparse_opt = CastedSparseEmbeddingSignSGD_Distributed(
                [module.local_weights, module.local_ids, module.weights],
                world_size=1,
                lr=embedding_lr,
                weight_decay=0.01,
            )
            optimizers.append(sparse_opt)
            break
    
    trm_model._optimizers = optimizers
    trm_model.train()
    trm_model.carry = None
    
    # Initialize total_steps (normally set by Lightning's setup method)
    trm_model.total_steps = 1000
    
    losses = []
    for _ in range(3):
        batch = synthetic_datamodule.get_batch()
        loss = trm_model.training_step(batch, 0)
        losses.append(loss.item())
        
        assert torch.isfinite(loss)
    
    # Check that we got valid losses for all steps
    assert len(losses) == 3
    assert all(loss > 0 for loss in losses)
