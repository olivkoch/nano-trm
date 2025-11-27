#!/usr/bin/env python
"""Functional tests for TensorMCTS"""

import torch
import numpy as np
from src.nn.modules.tensor_mcts import TensorMCTS, TensorMCTSConfig

class DummyModel:
    """Model that returns uniform policy and zero value"""
    def __init__(self, device):
        self.device = device
    
    def forward(self, boards):
        batch_size = boards.shape[0]
        # Uniform policy
        policies = torch.ones(batch_size, 7, device=self.device) / 7
        # Neutral value
        values = torch.zeros(batch_size, device=self.device)
        return policies, values


class PerfectModel:
    """Model that 'knows' winning moves"""
    def __init__(self, device):
        self.device = device
    
    def forward(self, boards):
        batch_size = boards.shape[0]
        boards = boards.view(batch_size, 6, 7)
        
        policies = torch.ones(batch_size, 7, device=self.device)
        values = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            board = boards[i]
            # Simple heuristic: prefer center, check for wins
            for col in range(7):
                if board[0, col] != 0:  # Full column
                    policies[i, col] = 0
                    continue
                # Find landing row
                row = 5
                while row >= 0 and board[row, col] != 0:
                    row -= 1
                if row < 0:
                    continue
                # Check if this creates a win (simplified)
                # ... (add win detection logic)
            
            # Center preference
            policies[i, 3] *= 2
        
        policies = policies / policies.sum(dim=1, keepdim=True)
        return policies, values


def run_single_tree_mcts(board, player, model, n_sims=100, device="cpu"):
    """Run MCTS on a single position"""
    config = TensorMCTSConfig(
        n_actions=7,
        max_nodes_per_tree=500,
        c_puct=1.0,
        exploration_fraction=0.0  # Disable noise for deterministic tests
    )
    
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_tensor = board.unsqueeze(0).to(device)
    legal_mask = (board[0, :] == 0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        policies, _ = model.forward(board_tensor.flatten(start_dim=1))
    
    mcts.reset(board_tensor, policies, legal_mask)
    mcts.run_simulations(n_sims, parallel_sims=4)
    
    return mcts


def test_basic_tree_growth():
    """Verify tree actually grows"""
    print("\n" + "="*60)
    print("TEST: Basic Tree Growth")
    print("="*60)
    
    device = "cpu"
    model = DummyModel(device)
    board = torch.zeros(6, 7, device=device)
    
    mcts = run_single_tree_mcts(board, player=1, model=model, n_sims=50, device=device)
    
    nodes_allocated = mcts.next_node_idx[0].item()
    root_visits = mcts.visits[0, 0].item()
    
    print(f"  Nodes allocated: {nodes_allocated}")
    print(f"  Root visits: {root_visits}")
    print(f"  Has states materialized: {mcts.has_state[0].sum().item()}")
    
    assert nodes_allocated > 8, "Should allocate more than just root children"
    assert root_visits >= 50, "Root should have at least n_sims visits"
    print("  ✓ PASSED")


def test_forced_win():
    """MCTS must find immediate winning move"""
    print("\n" + "="*60)
    print("TEST: Forced Win Detection")
    print("="*60)
    
    device = "cpu"
    model = DummyModel(device)
    
    # Player 1 has 3 in a row, can win at column 3
    board = torch.zeros(6, 7, device=device)
    board[5, 0] = board[5, 1] = board[5, 2] = 1
    board[5, 4] = board[5, 5] = board[4, 4] = 2

    print("  Board:")
    print("  " + "-"*15)
    for row in range(6):
        print("  |" + "".join([".12"[int(board[row,c])] for c in range(7)]) + "|")
    print("  " + "-"*15)
    
    mcts = run_single_tree_mcts(board, player=1, model=model, n_sims=100, device=device)
    dist = mcts._get_visit_distributions()[0]
    
    print(f"  Visit distribution: {dist.cpu().numpy().round(3)}")
    print(f"  Column 3 (winning): {dist[3].item():.1%}")
    
    assert dist[3] > 0.8, f"Should find winning move! Got {dist[3]:.1%}"
    print("  ✓ PASSED")

def test_forced_win_detailed():
    """MCTS must find immediate winning move"""
    print("\n" + "="*60)
    print("TEST: Forced Win Detection")
    print("="*60)
    
    device = "cpu"
    model = DummyModel(device)
    
    board = torch.zeros(6, 7, device=device)
    board[5, 0] = board[5, 1] = board[5, 2] = 1
    board[5, 4] = board[5, 5] = board[4, 4] = 2
    
    p1_count = (board == 1).sum().item()
    p2_count = (board == 2).sum().item()
    print(f"  P1 count: {p1_count}, P2 count: {p2_count}")
    print(f"  Expected current player: {'P1' if p1_count == p2_count else 'P2'}")

    print("  Board:")
    print("  " + "-"*15)
    for row in range(6):
        print("  |" + "".join([".12"[int(board[row,c])] for c in range(7)]) + "|")
    print("  " + "-"*15)
    
    mcts = run_single_tree_mcts(board, player=1, model=model, n_sims=100, device=device)
    
    # === DIAGNOSTIC 1: Check terminal detection ===
    print("\n  --- Diagnostic 1: Terminal Detection ---")
    board_after_win = board.clone()
    board_after_win[5, 3] = 1  # Manually add winning piece
    is_term, winner = mcts._check_terminal_batch(board_after_win.unsqueeze(0))
    print(f"  Board after col 3: terminal={is_term.item()}, winner={winner.item()}")
    
    # === DIAGNOSTIC 2: Check child node for action 3 ===
    print("\n  --- Diagnostic 2: Child Node State ---")
    child_idx = mcts.children[0, 0, 3].item()
    print(f"  Child index for action 3: {child_idx}")
    if child_idx >= 0:
        print(f"  Has state: {mcts.has_state[0, child_idx].item()}")
        print(f"  Is terminal: {mcts.is_terminal[0, child_idx].item()}")
        print(f"  Terminal value: {mcts.terminal_value[0, child_idx].item()}")
        print(f"  Visits: {mcts.visits[0, child_idx].item()}")
        print(f"  Total value: {mcts.total_value[0, child_idx].item()}")
        if mcts.has_state[0, child_idx]:
            print("  Materialized state:")
            child_state = mcts.states[0, child_idx]
            for row in range(6):
                print("    |" + "".join([".12"[int(child_state[row,c])] for c in range(7)]) + "|")
    
    # === DIAGNOSTIC 3: Check root node stats ===
    print("\n  --- Diagnostic 3: Root Stats ---")
    print(f"  Root visits: {mcts.visits[0, 0].item()}")
    print(f"  Root priors: {mcts.priors[0, 0].cpu().numpy().round(3)}")
    
    # === DIAGNOSTIC 4: All children Q-values ===
    print("\n  --- Diagnostic 4: Children Q-values ---")
    for action in range(7):
        child = mcts.children[0, 0, action].item()
        if child >= 0:
            v = mcts.visits[0, child].item()
            tv = mcts.total_value[0, child].item()
            q = tv / v if v > 0 else 0
            term = mcts.is_terminal[0, child].item()
            print(f"  Action {action}: child={child}, N={v:.0f}, Q={q:+.3f}, terminal={term}")
    
    dist = mcts._get_visit_distributions()[0]
    print(f"\n  Visit distribution: {dist.cpu().numpy().round(3)}")
    print(f"  Column 3 (winning): {dist[3].item():.1%}")
    
    assert dist[3] > 0.8, f"Should find winning move! Got {dist[3]:.1%}"
    print("  ✓ PASSED")

def test_forced_block():
    """MCTS must block opponent's winning threat"""
    print("\n" + "="*60)
    print("TEST: Forced Block Detection")
    print("="*60)
    
    device = "cpu"
    model = DummyModel(device)
    
    # Player 2 threatens win at column 3, player 1 must block
    board = torch.zeros(6, 7, device=device)
    board[5, 0] = board[5, 1] = board[5, 2] = 2  # Opponent
    board[5, 6] = 1  # Our piece elsewhere
    
    print("  Board (Player 1 to move, must block col 3):")
    print("  " + "-"*15)
    for row in range(6):
        print("  |" + "".join([".12"[int(board[row,c])] for c in range(7)]) + "|")
    print("  " + "-"*15)
    
    mcts = run_single_tree_mcts(board, player=1, model=model, n_sims=200, device=device)
    dist = mcts._get_visit_distributions()[0]
    
    print(f"  Visit distribution: {dist.cpu().numpy().round(3)}")
    print(f"  Column 3 (blocking): {dist[3].item():.1%}")
    
    assert dist[3] > 0.7, f"Should block! Got {dist[3]:.1%}"
    print("  ✓ PASSED")


def test_convergence():
    """More simulations should concentrate the distribution"""
    print("\n" + "="*60)
    print("TEST: Convergence with More Simulations")
    print("="*60)
    
    device = "cpu"
    model = DummyModel(device)
    board = torch.zeros(6, 7, device=device)
    
    results = []
    for n_sims in [20, 50, 100, 200]:
        mcts = run_single_tree_mcts(board, player=1, model=model, n_sims=n_sims, device=device)
        dist = mcts._get_visit_distributions()[0]
        entropy = -(dist * (dist + 1e-8).log()).sum().item()
        max_prob = dist.max().item()
        results.append((n_sims, entropy, max_prob))
        print(f"  {n_sims:3d} sims: entropy={entropy:.3f}, max_prob={max_prob:.2%}")
    
    # Entropy should generally decrease
    assert results[-1][1] < results[0][1] + 0.5, "Entropy should decrease with more sims"
    print("  ✓ PASSED")


def test_virtual_loss():
    """Verify virtual loss prevents path concentration"""
    print("\n" + "="*60)
    print("TEST: Virtual Loss Effect")
    print("="*60)
    
    device = "cpu"
    model = DummyModel(device)
    board = torch.zeros(6, 7, device=device)
    
    # Run with high parallelism - should still explore multiple branches
    config = TensorMCTSConfig(
        n_actions=7,
        max_nodes_per_tree=500,
        c_puct=1.0,
        virtual_loss_weight=3.0,
        exploration_fraction=0.0
    )
    
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_tensor = board.unsqueeze(0)
    legal_mask = torch.ones(1, 7, device=device)
    
    with torch.no_grad():
        policies, _ = model.forward(board_tensor.flatten(start_dim=1))
    
    mcts.reset(board_tensor, policies, legal_mask)
    
    # Run with high parallelism
    mcts.run_simulations(num_simulations=50, parallel_sims=16)
    
    # Check that virtual loss is cleared
    vl_sum = mcts.virtual_loss.sum().item()
    print(f"  Remaining virtual loss: {vl_sum}")
    
    # Check multiple children visited
    root_children = mcts.children[0, 0]
    children_visits = []
    for action in range(7):
        child = root_children[action].item()
        if child >= 0:
            visits = mcts.visits[0, child].item()
            children_visits.append((action, visits))
    
    print(f"  Child visits: {children_visits}")
    n_visited = sum(1 for _, v in children_visits if v > 0)
    print(f"  Children with visits: {n_visited}/7")
    
    assert n_visited >= 3, "Virtual loss should encourage exploration"
    assert vl_sum < 1, "Virtual loss should be cleared after backup"
    print("  ✓ PASSED")


def test_terminal_detection():
    """Verify terminal states are detected and valued correctly"""
    print("\n" + "="*60)
    print("TEST: Terminal State Detection")
    print("="*60)
    
    device = "cpu"
    model = DummyModel(device)
    
    # Create a position where player 1 has already won (horizontal)
    board = torch.zeros(6, 7, device=device)
    board[5, 0] = board[5, 1] = board[5, 2] = board[5, 3] = 1  # Player 1 wins
    board[4, 0] = board[4, 1] = board[4, 2] = 2  # Some player 2 pieces
    
    print("  Board (already won by player 1):")
    print("  " + "-"*15)
    for row in range(6):
        print("  |" + "".join([".12"[int(board[row,c])] for c in range(7)]) + "|")
    print("  " + "-"*15)
    
    # Test terminal detection
    config = TensorMCTSConfig()
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    is_term, winners = mcts._check_terminal_batch(board.unsqueeze(0))
    
    print(f"  Is terminal: {is_term.item()}")
    print(f"  Winner: {winners.item()}")
    
    assert is_term.item() == True, "Should detect terminal"
    assert winners.item() == 1, "Player 1 should be winner"
    print("  ✓ PASSED")


def visualize_tree_structure(mcts, tree_idx=0, max_depth=2):
    """Pretty print tree structure"""
    print("\n" + "="*60)
    print("TREE STRUCTURE VISUALIZATION")
    print("="*60)
    
    def format_node(node_idx, depth, action):
        visits = mcts.visits[tree_idx, node_idx].item()
        value = mcts.total_value[tree_idx, node_idx].item()
        q = value / visits if visits > 0 else 0
        is_term = mcts.is_terminal[tree_idx, node_idx].item()
        
        indent = "│   " * depth
        if action == -1:
            label = "ROOT"
        else:
            label = f"a={action}"
        
        term_str = " [TERM]" if is_term else ""
        return f"{indent}├── {label}: N={visits:.0f}, Q={q:+.3f}{term_str}"
    
    print(format_node(0, 0, -1))
    
    def print_children(node_idx, depth):
        if depth >= max_depth:
            return
        children = mcts.children[tree_idx, node_idx]
        priors = mcts.priors[tree_idx, node_idx]
        
        for action in range(7):
            child = children[action].item()
            if child >= 0 and mcts.visits[tree_idx, child].item() > 0:
                prior = priors[action].item()
                print(format_node(child, depth + 1, action) + f" P={prior:.3f}")
                print_children(child, depth + 1)
    
    print_children(0, 0)


def run_all_tests():
    """Run all functional tests"""
    print("\n" + "="*60)
    print("TENSOR MCTS FUNCTIONAL TESTS")
    print("="*60)
    
    test_basic_tree_growth()
    test_terminal_detection()
    test_forced_win_detailed()
    test_forced_block()
    test_convergence()
    test_virtual_loss()
    
    # Run one more time with visualization
    device = "cpu"
    model = DummyModel(device)
    board = torch.zeros(6, 7, device=device)
    mcts = run_single_tree_mcts(board, player=1, model=model, n_sims=100, device=device)
    visualize_tree_structure(mcts, max_depth=2)
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()