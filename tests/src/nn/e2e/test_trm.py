# debug_trm.py
import torch
from src.nn.models.trm_module import TRMModule
from src.nn.data.arc_datamodule import ARCDataModuleWithPuzzles

def test_data_loading():
    """Test that data loads correctly with puzzle IDs."""
    print("Testing data loading...")
    dm = ARCDataModuleWithPuzzles(
        data_dir="data",
        batch_size=2,
        samples_per_task=1,
        num_workers=0
    )
    dm.setup()
    
    print(f"Num puzzles: {dm.num_puzzles}")
    
    # Get a batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Inputs shape: {batch['input'].shape}")
    print(f"Puzzle IDs shape: {batch['puzzle_identifiers'].shape}")
    print(f"Sample puzzle IDs: {batch['puzzle_identifiers'][:5]}")
    
    if 'output' in batch:
        print(f"Output shape: {batch['output'].shape}")
    
    return dm, batch

def test_model_forward():
    """Test model forward pass with carry."""
    print("\nTesting model forward...")
    
    # Get data
    dm, batch = test_data_loading()
    
    # Create model
    model = TRMModule(
        hidden_size=128,
        num_layers=2,
        num_puzzles=dm.num_puzzles,
        puzzle_emb_dim=128,
        puzzle_emb_len=4,
        N_supervision=3,
        n_latent_recursions=1,
        T_deep_recursions=1
    )
    
    # Test initial carry
    carry = model.initial_carry(batch)
    print(f"Initial carry z_H shape: {carry.z_H.shape}")
    print(f"Initial carry z_L shape: {carry.z_L.shape}")
    
    # Test forward pass
    new_carry, outputs = model.forward(carry, batch)
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Q halt logits shape: {outputs['q_halt_logits'].shape}")
    
    return model, batch

def test_training_step():
    """Test a single training step."""
    print("\nTesting training step...")
    
    model, batch = test_model_forward()
    
    # Manual optimization setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model._optimizer = optimizer  # Store optimizer directly for testing
    
    # Simulate training step
    model.train()
    model.carry = None  # Start fresh
    
    # Run one step
    loss = model.training_step(batch, 0)
    print(f"Training loss: {loss.item():.4f}")
    
    # Check carry state
    print(f"Carry steps: {model.carry.steps}")
    print(f"Carry halted: {model.carry.halted}")

if __name__ == "__main__":
    test_data_loading()
    test_model_forward()
    test_training_step()
    print("\n✅ All tests passed!")