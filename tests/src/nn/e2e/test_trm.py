# debug_trm.py
import torch
from adam_atan2 import AdamATan2

from src.nn.data.arc_datamodule import ARCDataModuleWithPuzzles
from src.nn.models.trm_module import TRMModule
from src.nn.modules.sparse_embeddings import (
    CastedSparseEmbedding,
    CastedSparseEmbeddingSignSGD_Distributed,
)


def test_data_loading():
    """Test that data loads correctly with puzzle IDs."""
    print("Testing data loading...")
    dm = ARCDataModuleWithPuzzles(data_dir="data", batch_size=2, samples_per_task=1, num_workers=0)
    dm.setup()

    print(f"Num puzzles: {dm.num_puzzles}")

    # Get a batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    print(f"Batch keys: {batch.keys()}")
    print(f"Inputs shape: {batch['input'].shape}")
    print(f"Puzzle IDs shape: {batch['puzzle_identifiers'].shape}")
    print(f"Sample puzzle IDs: {batch['puzzle_identifiers'][:5]}")

    if "output" in batch:
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
        T_deep_recursions=1,
        batch_size=dm.batch_size,
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

    # Manual optimization setup with proper optimizers
    base_lr = 1e-4 / model.hparams.N_supervision
    embedding_lr = 1e-3 / model.hparams.N_supervision

    optimizers = []

    # Collect parameters for main optimizer (excluding sparse embeddings)
    main_params = []

    # Collect other parameters
    for name, param in model.named_parameters():
        if "puzzle_emb" not in name:  # Exclude puzzle_emb as it's handled separately
            main_params.append(param)

    # Main optimizer (use AdamATan2 if available, else AdamW)
    if main_params:
        try:
            main_opt = AdamATan2(main_params, lr=base_lr, weight_decay=0.01, betas=(0.9, 0.95))
        except:
            # Fallback to AdamW if AdamATan2 not available
            main_opt = torch.optim.AdamW(
                main_params, lr=base_lr, weight_decay=0.01, betas=(0.9, 0.95)
            )
        optimizers.append(main_opt)

    # Sparse embedding optimizer
    for _, module in model.named_modules():
        if isinstance(module, CastedSparseEmbedding):
            # Pass the three components directly as a list, NOT a list of lists
            sparse_opt = CastedSparseEmbeddingSignSGD_Distributed(
                [module.local_weights, module.local_ids, module.weights],  # Single list
                world_size=1,
                lr=embedding_lr,
                weight_decay=0.01,
            )
            optimizers.append(sparse_opt)
            break  # Only one sparse embedding module

    # Store optimizers for testing
    model._optimizers = optimizers

    # Simulate training step
    model.train()
    model.carry = None  # Start fresh

    # Run one step
    loss = model.training_step(batch, 0)
    print(f"Training loss: {loss.item():.4f}")

    # Check carry state
    print(f"Carry steps: {model.carry.steps}")
    print(f"Carry halted: {model.carry.halted}")

    # Check if parameters are updating
    print(f"Number of optimizers: {len(optimizers)}")
    for i, opt in enumerate(optimizers):
        print(f"Optimizer {i}: {type(opt).__name__}")


if __name__ == "__main__":
    # test_data_loading()
    # test_model_forward()
    test_training_step()
    print("\n✅ All tests passed!")
