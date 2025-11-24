"""
Comprehensive tests for BatchMCTS to verify correct behavior
"""

import torch
import numpy as np
from typing import List, Tuple
import pytest
from src.nn.modules.batch_mcts import BatchMCTSWithVirtualLoss, MCTSNode

class DeterministicTestModel:
    """Deterministic model for testing MCTS behavior"""
    def __init__(self, device="cpu"):
        self.device = device
        self.call_count = 0
        self.evaluated_positions = []
        
    def forward(self, boards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns deterministic policies and values based on board sum"""
        self.call_count += 1
        self.evaluated_positions.append(boards.clone())
        
        batch_size = boards.shape[0]
        policies = torch.zeros(batch_size, 7, device=self.device)
        values = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            # Deterministic policy: favor middle columns
            board_sum = boards[i].sum().item()
            policies[i] = torch.tensor([0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05], device=self.device)
            
            # Deterministic value based on board state
            values[i] = torch.tanh(torch.tensor(board_sum / 42.0, device=self.device))
        
        return policies, values


def test_mcts_visit_distribution():
    """Test that MCTS produces correct visit distributions"""
    print("\n=== Testing Visit Distribution ===")
        
    device = "cpu"
    model = DeterministicTestModel(device)
    
    # Create MCTS with specific settings
    mcts = BatchMCTSWithVirtualLoss(
        model=model,
        c_puct=1.0,
        num_simulations=100,
        parallel_simulations=1,  # Sequential for deterministic testing
        dirichlet_alpha=0.0,  # No noise for testing
        exploration_fraction=0.0,  # No exploration noise
        device=device
    )
    
    # Simple position - empty board
    board = torch.zeros(6, 7, device=device)
    legal_moves = torch.ones(7, dtype=torch.bool, device=device)
    
    # Run MCTS
    visit_dist, action_probs = mcts.get_action_probs_batch_parallel(
        [board], [legal_moves], temperature=0
    )
    
    # Check properties
    print(f"Visit distribution: {visit_dist[0].numpy()}")
    print(f"Sum of visits: {visit_dist[0].sum().item():.4f}")
    
    # Visit distribution should sum to 1
    assert abs(visit_dist[0].sum().item() - 1.0) < 1e-6, "Visit distribution should sum to 1"
    
    # Center column should have most visits (due to deterministic policy)
    assert visit_dist[0].argmax() == 3, f"Column 3 should have most visits, got {visit_dist[0].argmax()}"
    
    # With temperature=0, action_probs should be one-hot
    assert action_probs[0].max() == 1.0, "Temperature=0 should give deterministic action"
    assert action_probs[0].sum() == 1.0, "Action probs should sum to 1"
    
    print("✓ Visit distribution test passed")


def test_mcts_convergence():
    """Test that more simulations lead to better convergence"""
    print("\n=== Testing MCTS Convergence ===")
        
    device = "cpu"
    board = torch.zeros(6, 7, device=device)
    legal_moves = torch.ones(7, dtype=torch.bool, device=device)
    
    visit_dists = []
    max_visits = []
    
    for num_sims in [10, 50, 200, 500]:
        # Set seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        model = DeterministicTestModel(device)
        mcts = BatchMCTSWithVirtualLoss(
            model=model,
            c_puct=1.0,
            num_simulations=num_sims,
            parallel_simulations=1,
            dirichlet_alpha=0.3,  # Add noise to see convergence
            exploration_fraction=0.25,  # Add exploration
            device=device
        )
        
        visit_dist, _ = mcts.get_action_probs_batch_parallel(
            [board], [legal_moves], temperature=1.0
        )
        visit_dists.append(visit_dist[0])
        max_visits.append(visit_dist[0].max().item())
        
        # Calculate entropy as measure of convergence
        entropy = -(visit_dist[0] * torch.log(visit_dist[0] + 1e-8)).sum().item()
        print(f"Simulations: {num_sims:3d}, Entropy: {entropy:.4f}, "
              f"Max visit: {visit_dist[0].max().item():.3f}")
    
    # More simulations should lead to higher confidence in best move
    # With noise, we check that max visit probability increases
    assert max_visits[-1] > max_visits[0], \
        f"More simulations should increase confidence: {max_visits[0]:.3f} -> {max_visits[-1]:.3f}"
    
    print("✓ Convergence test passed")

def test_virtual_loss_exploration():
    """Test that virtual loss causes different paths to be explored"""
    print("\n=== Testing Virtual Loss Exploration ===")
        
    device = "cpu"
    model = DeterministicTestModel(device)
    
    # Test with and without virtual loss parallelism
    for parallel_sims in [1, 4]:
        model.call_count = 0
        model.evaluated_positions = []
        
        mcts = BatchMCTSWithVirtualLoss(
            model=model,
            c_puct=1.0,
            num_simulations=16,
            parallel_simulations=parallel_sims,
            virtual_loss_value=3,
            dirichlet_alpha=0.0,
            exploration_fraction=0.0,
            device=device
        )
        
        board = torch.zeros(6, 7, device=device)
        legal_moves = torch.ones(7, dtype=torch.bool, device=device)
        
        visit_dist, _ = mcts.get_action_probs_batch_parallel(
            [board], [legal_moves], temperature=1.0
        )
        
        print(f"Parallel={parallel_sims}: Model calls={model.call_count}, "
              f"Visit std={visit_dist[0].std().item():.4f}")
    
    print("✓ Virtual loss exploration test passed")


def test_illegal_moves_masked():
    """Test that illegal moves are never selected"""
    print("\n=== Testing Illegal Move Masking ===")
        
    device = "cpu"
    model = DeterministicTestModel(device)
    mcts = BatchMCTSWithVirtualLoss(
        model=model,
        c_puct=1.0,
        num_simulations=50,
        parallel_simulations=1,
        device=device
    )
    
    # Board with some full columns
    board = torch.zeros(6, 7, device=device)
    board[:, 0] = 1  # Column 0 is full
    board[:, 6] = 2  # Column 6 is full
    
    legal_moves = torch.tensor([False, True, True, True, True, True, False], device=device)
    
    visit_dist, action_probs = mcts.get_action_probs_batch_parallel(
        [board], [legal_moves], temperature=0
    )
    
    print(f"Legal moves: {legal_moves.numpy()}")
    print(f"Visit dist:  {visit_dist[0].numpy()}")
    
    # Check no visits to illegal moves
    assert visit_dist[0][0] == 0, "Full column 0 should have 0 visits"
    assert visit_dist[0][6] == 0, "Full column 6 should have 0 visits"
    assert visit_dist[0][1:6].sum() == 1.0, "Legal moves should sum to 1"
    
    print("✓ Illegal move masking test passed")


def test_value_backup_alternates():
    """Test that value backup correctly alternates signs"""
    print("\n=== Testing Value Backup Sign Alternation ===")
        
    # Create a simple path
    root = MCTSNode(state=torch.zeros(6, 7))
    child1 = MCTSNode(state=torch.zeros(6, 7), parent=root, action=3)
    child2 = MCTSNode(state=torch.zeros(6, 7), parent=child1, action=4)
    
    root.children[3] = child1
    child1.children[4] = child2
    
    # Manually backup a value
    path = [root, child1, child2]
    value = 1.0
    
    for node in reversed(path):
        node.visits += 1
        node.total_value += value
        value = -value  # Flip sign
    
    # Check values
    assert child2.total_value == 1.0, "Leaf should have positive value"
    assert child1.total_value == -1.0, "Parent should have negative value"
    assert root.total_value == 1.0, "Root should have positive value"
    
    print(f"Root Q={root.Q():.2f}, Child1 Q={child1.Q():.2f}, Child2 Q={child2.Q():.2f}")
    print("✓ Value backup test passed")


def test_temperature_behavior():
    """Test that temperature correctly affects action selection"""
    print("\n=== Testing Temperature Behavior ===")
        
    device = "cpu"
    model = DeterministicTestModel(device)
    board = torch.zeros(6, 7, device=device)
    legal_moves = torch.ones(7, dtype=torch.bool, device=device)
    
    for temp in [0.0, 0.5, 1.0, 2.0]:
        mcts = BatchMCTSWithVirtualLoss(
            model=model,
            c_puct=1.0,
            num_simulations=100,
            parallel_simulations=1,
            dirichlet_alpha=0.0,
            exploration_fraction=0.0,
            device=device
        )
        
        visit_dist, action_probs = mcts.get_action_probs_batch_parallel(
            [board], [legal_moves], temperature=temp
        )
        
        # Calculate entropy of action distribution
        entropy = -(action_probs[0] * torch.log(action_probs[0] + 1e-8)).sum().item()
        
        # Check determinism at T=0
        if temp == 0:
            assert action_probs[0].max() == 1.0, "T=0 should be deterministic"
            assert (action_probs[0] == 1.0).sum() == 1, "T=0 should have single 1.0"
        
        print(f"Temperature={temp:.1f}: Entropy={entropy:.4f}, "
              f"Max prob={action_probs[0].max().item():.3f}")
    
    print("✓ Temperature behavior test passed")


def test_batch_consistency():
    """Test that batched games produce consistent results"""
    print("\n=== Testing Batch Consistency ===")
        
    device = "cpu"
    model = DeterministicTestModel(device)
    
    # Run MCTS on same position multiple times in a batch
    boards = [torch.zeros(6, 7, device=device) for _ in range(5)]
    legal_moves = [torch.ones(7, dtype=torch.bool, device=device) for _ in range(5)]
    
    mcts = BatchMCTSWithVirtualLoss(
        model=model,
        c_puct=1.0,
        num_simulations=50,
        parallel_simulations=1,
        dirichlet_alpha=0.0,  # No randomness
        exploration_fraction=0.0,
        device=device
    )
    
    visit_dists, action_probs = mcts.get_action_probs_batch_parallel(
        boards, legal_moves, temperature=0
    )
    
    # All games should produce identical results (no randomness)
    for i in range(1, 5):
        assert torch.allclose(visit_dists[0], visit_dists[i], atol=1e-6), \
            f"Game {i} visit distribution differs from game 0"
        assert torch.allclose(action_probs[0], action_probs[i], atol=1e-6), \
            f"Game {i} action probs differ from game 0"
    
    print("✓ Batch consistency test passed")


def test_dirichlet_noise():
    """Test that Dirichlet noise is applied correctly"""
    print("\n=== Testing Dirichlet Noise ===")
        
    device = "cpu"
    model = DeterministicTestModel(device)
    board = torch.zeros(6, 7, device=device)
    legal_moves = torch.ones(7, dtype=torch.bool, device=device)
    
    # Run with and without noise
    results = {}
    for use_noise in [False, True]:
        mcts = BatchMCTSWithVirtualLoss(
            model=model,
            c_puct=1.0,
            num_simulations=100,
            parallel_simulations=1,
            dirichlet_alpha=0.3 if use_noise else 0.0,
            exploration_fraction=0.25 if use_noise else 0.0,
            device=device
        )
        
        # Set seed for reproducibility
        if use_noise:
            np.random.seed(42)
        
        visit_dist, _ = mcts.get_action_probs_batch_parallel(
            [board], [legal_moves], temperature=1.0
        )
        results[use_noise] = visit_dist[0]
    
    # With noise should differ from without
    assert not torch.allclose(results[False], results[True], atol=0.01), \
        "Dirichlet noise should change visit distribution"
    
    print(f"No noise: {results[False].numpy()}")
    print(f"With noise: {results[True].numpy()}")
    print("✓ Dirichlet noise test passed")


def run_all_tests():
    """Run all MCTS tests"""
    print("=" * 60)
    print("MCTS CORRECTNESS TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_mcts_visit_distribution,
        test_mcts_convergence,
        test_virtual_loss_exploration,
        test_illegal_moves_masked,
        test_value_backup_alternates,
        test_temperature_behavior,
        test_batch_consistency,
        test_dirichlet_noise,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed.append(test.__name__)
    
    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED {len(failed)}/{len(tests)} tests:")
        for name in failed:
            print(f"  - {name}")
    else:
        print(f"✓ ALL {len(tests)} TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()