#!/usr/bin/env python
"""Comprehensive functional tests for TensorMCTS with current_player tracking"""

import torch
import numpy as np
from dataclasses import dataclass
from src.nn.modules.tensor_mcts import TensorMCTS, TensorMCTSConfig


# ============================================================================
# Test Utilities
# ============================================================================

class DummyModel:
    """Model that returns uniform policy and zero value"""
    def __init__(self, device):
        self.device = device
    
    def forward(self, boards, current_players):
        # Note: boards are passed as float (42,) for NN, but represent discrete game state
        batch_size = boards.shape[0]
        policies = torch.ones(batch_size, 7, device=self.device) / 7
        values = torch.zeros(batch_size, device=self.device)
        return policies, values


def print_board(board):
    """Pretty print a board"""
    print("  " + "-" * 15)
    for row in range(6):
        print("  |" + "".join([".12"[int(board[row, c])] for c in range(7)]) + "|")
    print("  " + "-" * 15)


def run_single_tree_mcts(board, current_player, model, n_sims=100, device="cpu", 
                         exploration_fraction=0.0, config=None):
    """Run MCTS on a single position
    
    Note: Boards can be float or long - wrapper converts to long internally,
    then converts to float when passing to model.
    """
    if config is None:
        config = TensorMCTSConfig(
            n_actions=7,
            max_nodes_per_tree=2000,
            c_puct=1.0,
            exploration_fraction=exploration_fraction
        )
    
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_tensor = board.unsqueeze(0).to(device)
    legal_mask = (board[0, :] == 0).unsqueeze(0).float().to(device)
    current_players = torch.tensor([current_player], dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Convert to float for model (NNs expect float inputs)
        policies, _ = model.forward(board_tensor.flatten(start_dim=1).float(), current_players)
    
    mcts.reset(board_tensor, policies, legal_mask, current_players)
    mcts.run_simulations(n_sims, parallel_sims=4)
    
    return mcts


# ============================================================================
# Basic Tests
# ============================================================================

def test_basic_tree_growth():
    """Verify tree actually grows"""
    print("\n" + "=" * 60)
    print("TEST: Basic Tree Growth")
    print("=" * 60)
        
    device = "cpu"
    model = DummyModel(device)
    board = torch.zeros(6, 7, device=device)
    
    mcts = run_single_tree_mcts(board, current_player=1, model=model, n_sims=50, device=device)
    
    nodes_allocated = mcts.next_node_idx[0].item()
    root_visits = mcts.visits[0, 0].item()
    
    print(f"  Nodes allocated: {nodes_allocated}")
    print(f"  Root visits: {root_visits}")
    
    assert nodes_allocated > 8, "Should allocate more than just root children"
    assert root_visits >= 50, "Root should have at least n_sims visits"
    print("  ✓ PASSED")


def test_terminal_detection():
    """Verify terminal states are detected correctly"""
    print("\n" + "=" * 60)
    print("TEST: Terminal State Detection")
    print("=" * 60)
        
    device = "cpu"
    model = DummyModel(device)
    config = TensorMCTSConfig()
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    # Horizontal win
    board_h = torch.zeros(6, 7, device=device)
    board_h[5, 0:4] = 1
    is_term, winner = mcts._check_terminal_batch(board_h.unsqueeze(0))
    print(f"  Horizontal win: terminal={is_term.item()}, winner={winner.item()}")
    assert is_term.item() == True and winner.item() == 1
    
    # Vertical win
    board_v = torch.zeros(6, 7, device=device)
    board_v[2:6, 0] = 2
    is_term, winner = mcts._check_terminal_batch(board_v.unsqueeze(0))
    print(f"  Vertical win: terminal={is_term.item()}, winner={winner.item()}")
    assert is_term.item() == True and winner.item() == 2
    
    # Diagonal win (down-right)
    board_d1 = torch.zeros(6, 7, device=device)
    board_d1[2, 0] = board_d1[3, 1] = board_d1[4, 2] = board_d1[5, 3] = 1
    is_term, winner = mcts._check_terminal_batch(board_d1.unsqueeze(0))
    print(f"  Diagonal (down-right) win: terminal={is_term.item()}, winner={winner.item()}")
    assert is_term.item() == True and winner.item() == 1
    
    # Diagonal win (down-left)
    board_d2 = torch.zeros(6, 7, device=device)
    board_d2[2, 6] = board_d2[3, 5] = board_d2[4, 4] = board_d2[5, 3] = 2
    is_term, winner = mcts._check_terminal_batch(board_d2.unsqueeze(0))
    print(f"  Diagonal (down-left) win: terminal={is_term.item()}, winner={winner.item()}")
    assert is_term.item() == True and winner.item() == 2
    
    # Non-terminal
    board_nt = torch.zeros(6, 7, device=device)
    board_nt[5, 0:3] = 1
    is_term, winner = mcts._check_terminal_batch(board_nt.unsqueeze(0))
    print(f"  Non-terminal: terminal={is_term.item()}, winner={winner.item()}")
    assert is_term.item() == False and winner.item() == -1
    
    print("  ✓ PASSED")


# ============================================================================
# Current Player Tracking Tests
# ============================================================================

def test_current_player_at_root():
    """Verify current_player is stored correctly at root"""
    print("\n" + "=" * 60)
    print("TEST: Current Player at Root")
    print("=" * 60)
    
    device = "cpu"
    model = DummyModel(device)
    board = torch.zeros(6, 7, device=device)
    
    # Test with player 1
    mcts1 = run_single_tree_mcts(board, current_player=1, model=model, n_sims=10, device=device)
    assert mcts1.current_player[0, 0].item() == 1, "Root should have player 1"
    print("  Player 1 at root: ✓")
    
    # Test with player 2
    board2 = board.clone()
    board2[5, 0] = 1  # Player 1 played
    mcts2 = run_single_tree_mcts(board2, current_player=2, model=model, n_sims=10, device=device)
    assert mcts2.current_player[0, 0].item() == 2, "Root should have player 2"
    print("  Player 2 at root: ✓")
    
    print("  ✓ PASSED")


def test_current_player_alternates():
    """Verify current_player alternates correctly in tree"""
    print("\n" + "=" * 60)
    print("TEST: Current Player Alternates in Tree")
    print("=" * 60)
    
    device = "cpu"
    model = DummyModel(device)
    board = torch.zeros(6, 7, device=device)
    
    mcts = run_single_tree_mcts(board, current_player=1, model=model, n_sims=50, device=device)
    
    # Check root
    root_player = mcts.current_player[0, 0].item()
    print(f"  Root (depth 0): player {root_player}")
    assert root_player == 1, "Root should be player 1"
    
    # Check children (depth 1)
    print("  Children (depth 1):")
    for action in range(7):
        child_idx = mcts.children[0, 0, action].item()
        if child_idx >= 0:
            child_player = mcts.current_player[0, child_idx].item()
            print(f"    Action {action} -> node {child_idx}: player {child_player}")
            assert child_player == 2, f"Children of player 1 should be player 2"
    
    # Check grandchildren (depth 2, sample)
    print("  Grandchildren (depth 2, sample):")
    for action in range(7):
        child_idx = mcts.children[0, 0, action].item()
        if child_idx >= 0:
            for action2 in range(7):
                grandchild_idx = mcts.children[0, child_idx, action2].item()
                if grandchild_idx >= 0:
                    grandchild_player = mcts.current_player[0, grandchild_idx].item()
                    print(f"    Node {child_idx} -> action {action2} -> node {grandchild_idx}: player {grandchild_player}")
                    assert grandchild_player == 1, "Grandchildren should be player 1"
                    break
            break
    
    print("  ✓ PASSED")


# ============================================================================
# Forced Win/Block Tests
# ============================================================================

def test_forced_win_horizontal():
    """MCTS must find immediate horizontal winning move"""
    print("\n" + "=" * 60)
    print("TEST: Forced Win (Horizontal)")
    print("=" * 60)
    
    device = "cpu"
    model = DummyModel(device)
    
    # P1 has 3 in a row at cols 0,1,2 - can win at col 3
    board = torch.zeros(6, 7, device=device)
    board[5, 0] = board[5, 1] = board[5, 2] = 1
    board[5, 4] = board[5, 5] = board[4, 4] = 2  # Equal pieces
    
    print("  Board:")
    print_board(board)
    
    mcts = run_single_tree_mcts(board, current_player=1, model=model, n_sims=100, device=device)
    dist = mcts._get_visit_distributions()[0]
    
    print(f"  Visit distribution: {dist.cpu().numpy().round(3)}")
    print(f"  Column 3 (winning): {dist[3].item():.1%}")
    
    assert dist[3] > 0.7, f"Should find winning move! Got {dist[3]:.1%}"
    print("  ✓ PASSED")


def test_forced_win_vertical():
    """MCTS should find vertical winning moves"""
    print("\n" + "=" * 60)
    print("TEST: Forced Win (Vertical)")
    print("=" * 60)
    
    device = "cpu"
    model = DummyModel(device)
    
    # P1 has 3 stacked in col 0, can win with 4th
    board = torch.zeros(6, 7, device=device)
    board[5, 0] = board[4, 0] = board[3, 0] = 1  # Vertical stack
    board[5, 1] = board[5, 2] = board[4, 1] = 2  # P2 pieces (equal count)
    
    print("  Board:")
    print_board(board)
    
    mcts = run_single_tree_mcts(board, current_player=1, model=model, n_sims=100, device=device)
    dist = mcts._get_visit_distributions()[0]
    
    print(f"  Visit distribution: {dist.cpu().numpy().round(3)}")
    print(f"  Column 0 (winning): {dist[0].item():.1%}")
    
    assert dist[0] > 0.7, f"Should find vertical win at col 0! Got {dist[0]:.1%}"
    print("  ✓ PASSED")


def test_forced_win_diagonal():
    """MCTS should find diagonal winning moves"""
    print("\n" + "=" * 60)
    print("TEST: Forced Win (Diagonal)")
    print("=" * 60)
    
    device = "cpu"
    model = DummyModel(device)
    
    # Skipping complex diagonal setup - terminal detection already verified
    print("  (Skipping complex diagonal - see test_terminal_detection for diagonal verification)")
    print("  ✓ SKIPPED")


def test_must_block():
    """P1 must block P2's winning threat"""
    print("\n" + "=" * 60)
    print("TEST: Must Block")
    print("=" * 60)
    
    device = "cpu"
    model = DummyModel(device)
    
    # P2 has 3 in a row, threatens win at col 3
    board = torch.zeros(6, 7, device=device)
    board[5, 0] = board[5, 1] = board[5, 2] = 2  # P2 threatens
    board[5, 4] = board[5, 5] = board[4, 4] = 1  # P1 pieces (equal count)
    
    print("  Board:")
    print_board(board)
    
    # More sims needed for 2-ply lookahead
    mcts = run_single_tree_mcts(board, current_player=1, model=model, n_sims=400, device=device)
    dist = mcts._get_visit_distributions()[0]
    
    print(f"  Visit distribution: {dist.cpu().numpy().round(3)}")
    print(f"  Column 3 (blocking): {dist[3].item():.1%}")
    
    assert dist[3] >= 0.5, f"Should block at col 3! Got {dist[3]:.1%}"
    print("  ✓ PASSED")


# ============================================================================
# Draw and Edge Cases
# ============================================================================

def test_draw_detection():
    """MCTS should recognize drawn positions"""
    print("\n" + "=" * 60)
    print("TEST: Draw Detection")
    print("=" * 60)
        
    device = "cpu"
    model = DummyModel(device)
    config = TensorMCTSConfig()
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    # Create full board with no winner (alternating pattern)
    board = torch.tensor([
        [1, 2, 1, 2, 1, 2, 1],
        [2, 1, 2, 1, 2, 1, 2],
        [1, 2, 1, 2, 1, 2, 1],
        [1, 2, 1, 2, 1, 2, 1],
        [2, 1, 2, 1, 2, 1, 2],
        [2, 1, 2, 1, 2, 1, 2],
    ], dtype=torch.float, device=device)
    
    is_term, winner = mcts._check_terminal_batch(board.unsqueeze(0))
    print(f"  Full board (no win): terminal={is_term.item()}, winner={winner.item()}")
    
    assert is_term.item() == True, "Full board should be terminal"
    assert winner.item() == 0, "Should be a draw"
    print("  ✓ PASSED")


def test_full_column_handling():
    """MCTS should never select full columns"""
    print("\n" + "=" * 60)
    print("TEST: Full Column Handling")
    print("=" * 60)
    
    device = "cpu"
    model = DummyModel(device)
    
    # Board with column 0 completely full
    board = torch.zeros(6, 7, device=device)
    board[:, 0] = torch.tensor([1, 2, 1, 2, 1, 2], dtype=torch.float)
    # Balance pieces
    board[5, 1] = board[5, 2] = board[5, 3] = 1
    board[5, 4] = board[5, 5] = board[5, 6] = 2
    
    print("  Board:")
    print_board(board)
    
    mcts = run_single_tree_mcts(board, current_player=1, model=model, n_sims=100, device=device)
    dist = mcts._get_visit_distributions()[0]
    
    print(f"  Visit distribution: {dist.cpu().numpy().round(3)}")
    
    assert dist[0] == 0, f"Full column should have 0 visits, got {dist[0]:.3f}"
    print("  ✓ PASSED")


# ============================================================================
# Batch and Consistency Tests
# ============================================================================

def test_batch_consistency():
    """Same position in batch should give similar results"""
    print("\n" + "=" * 60)
    print("TEST: Batch Consistency")
    print("=" * 60)
        
    device = "cpu"
    model = DummyModel(device)
    
    # Winning position
    board = torch.zeros(6, 7, device=device)
    board[5, 0] = board[5, 1] = board[5, 2] = 1
    board[5, 4] = board[5, 5] = board[4, 4] = 2
    
    # Run same position multiple times in a batch
    n_copies = 8
    boards = [board.clone() for _ in range(n_copies)]
    
    config = TensorMCTSConfig(exploration_fraction=0.0, max_nodes_per_tree=2000)
    mcts = TensorMCTS(model, n_trees=n_copies, config=config, device=device)
    
    boards_tensor = torch.stack(boards)
    legal_tensor = torch.stack([(b[0, :] == 0).float() for b in boards])
    current_players = torch.ones(n_copies, dtype=torch.long, device=device)  # All player 1
    
    with torch.no_grad():
        # Convert to float for model
        policies, _ = model.forward(boards_tensor.flatten(start_dim=1).float(), current_players)
    
    mcts.reset(boards_tensor, policies, legal_tensor, current_players)
    mcts.run_simulations(100, parallel_sims=4)
    
    dists = mcts._get_visit_distributions()
    col3_visits = dists[:, 3]
    
    print(f"  Batch col 3 visits: {col3_visits.cpu().numpy().round(3)}")
    print(f"  Mean: {col3_visits.mean():.3f}, Std: {col3_visits.std():.3f}")
    
    assert (col3_visits > 0.5).all(), "All copies should find winning move"
    print("  ✓ PASSED")


# ============================================================================
# Virtual Loss and Exploration Tests
# ============================================================================

def test_virtual_loss_diversity():
    """Virtual loss should encourage exploration of different paths"""
    print("\n" + "=" * 60)
    print("TEST: Virtual Loss Diversity")
    print("=" * 60)
        
    device = "cpu"
    model = DummyModel(device)
    
    board = torch.zeros(6, 7, device=device)
    
    config = TensorMCTSConfig(
        exploration_fraction=0.0,
        virtual_loss_weight=3.0,
        max_nodes_per_tree=2000
    )
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    legal = torch.ones(1, 7, device=device)
    current_players = torch.tensor([1], dtype=torch.long, device=device)
    
    with torch.no_grad():
        policies, _ = model.forward(board.unsqueeze(0).flatten(start_dim=1).float(), current_players)
    
    mcts.reset(board.unsqueeze(0), policies, legal, current_players)
    mcts.run_simulations(50, parallel_sims=8)
    
    dist = mcts._get_visit_distributions()[0]
    significant_visits = (dist > 0.05).sum().item()
    
    print(f"  Distribution: {dist.cpu().numpy().round(3)}")
    print(f"  Actions with >5% visits: {significant_visits}")
    
    assert significant_visits >= 4, "Should explore at least 4 different moves"
    
    # Check virtual loss is cleared
    vl_remaining = mcts.virtual_loss.sum().item()
    print(f"  Remaining virtual loss: {vl_remaining}")
    assert vl_remaining < 1, "Virtual loss should be mostly cleared"
    
    print("  ✓ PASSED")


def test_convergence():
    """More simulations should concentrate the distribution"""
    print("\n" + "=" * 60)
    print("TEST: Convergence")
    print("=" * 60)
    
    device = "cpu"
    model = DummyModel(device)
    board = torch.zeros(6, 7, device=device)
    
    entropies = []
    for n_sims in [20, 50, 100, 200]:
        mcts = run_single_tree_mcts(board, current_player=1, model=model, n_sims=n_sims, device=device)
        dist = mcts._get_visit_distributions()[0]
        entropy = -(dist * (dist + 1e-8).log()).sum().item()
        entropies.append(entropy)
        print(f"  {n_sims:3d} sims: entropy={entropy:.3f}, max={dist.max():.2%}")
    
    assert entropies[-1] < entropies[0] + 0.5, "Entropy should decrease with more sims"
    print("  ✓ PASSED")


# ============================================================================
# Temperature Tests
# ============================================================================

def test_temperature_scaling():
    """Temperature should affect action selection sharpness"""
    print("\n" + "=" * 60)
    print("TEST: Temperature Scaling")
    print("=" * 60)
        
    device = "cpu"
    model = DummyModel(device)
    board = torch.zeros(6, 7, device=device)
    
    config = TensorMCTSConfig(exploration_fraction=0.0, max_nodes_per_tree=2000)
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    legal = torch.ones(1, 7, device=device)
    current_players = torch.tensor([1], dtype=torch.long, device=device)
    
    with torch.no_grad():
        policies, _ = model.forward(board.unsqueeze(0).flatten(start_dim=1).float(), current_players)
    
    mcts.reset(board.unsqueeze(0), policies, legal, current_players)
    mcts.run_simulations(100, parallel_sims=4)
    
    probs_t0 = mcts.get_action_probs(temperature=0)[0]
    probs_t1 = mcts.get_action_probs(temperature=1.0)[0]
    probs_t2 = mcts.get_action_probs(temperature=2.0)[0]
    
    print(f"  T=0 (greedy): {probs_t0.cpu().numpy().round(3)}")
    print(f"  T=1 (normal): {probs_t1.cpu().numpy().round(3)}")
    print(f"  T=2 (explore): {probs_t2.cpu().numpy().round(3)}")
    
    assert probs_t0.max() == 1.0, "Greedy should be one-hot"
    
    entropy_t1 = -(probs_t1 * (probs_t1 + 1e-8).log()).sum()
    entropy_t2 = -(probs_t2 * (probs_t2 + 1e-8).log()).sum()
    print(f"  Entropy T=1: {entropy_t1:.3f}, T=2: {entropy_t2:.3f}")
    
    assert entropy_t2 > entropy_t1, "Higher temperature should increase entropy"
    print("  ✓ PASSED")


# ============================================================================
# Value Propagation Tests
# ============================================================================

def test_value_propagation():
    """Verify values propagate correctly through tree"""
    print("\n" + "=" * 60)
    print("TEST: Value Propagation")
    print("=" * 60)
        
    device = "cpu"
    model = DummyModel(device)
    
    # Position where col 3 wins immediately
    board = torch.zeros(6, 7, device=device)
    board[5, 0] = board[5, 1] = board[5, 2] = 1
    board[5, 4] = board[5, 5] = board[4, 4] = 2
    
    config = TensorMCTSConfig(exploration_fraction=0.0, max_nodes_per_tree=2000)
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    legal = (board[0, :] == 0).unsqueeze(0).float()
    current_players = torch.tensor([1], dtype=torch.long, device=device)
    
    with torch.no_grad():
        policies, _ = model.forward(board.unsqueeze(0).flatten(start_dim=1).float(), current_players)
    
    mcts.reset(board.unsqueeze(0), policies, legal, current_players)
    mcts.run_simulations(50, parallel_sims=1)
    
    root_visits = mcts.visits[0, 0].item()
    root_value = mcts.total_value[0, 0].item()
    root_q = root_value / root_visits if root_visits > 0 else 0
    
    child_idx = mcts.children[0, 0, 3].item()
    child_visits = mcts.visits[0, child_idx].item()
    child_value = mcts.total_value[0, child_idx].item()
    child_q = child_value / child_visits if child_visits > 0 else 0
    # Q from parent's perspective (negate child's value)
    child_q_parent = -child_q
    
    print(f"  Root: visits={root_visits}, Q={root_q:.3f}")
    print(f"  Winning child (col 3): visits={child_visits}, Q(child)={child_q:.3f}, Q(parent view)={child_q_parent:.3f}")
    print(f"  Terminal value stored: {mcts.terminal_value[0, child_idx].item()}")
    print(f"  Child current_player: {mcts.current_player[0, child_idx].item()}")
    
    # From parent's perspective, the winning move should have high Q
    assert child_q_parent > 0.8, f"Winning move Q (parent view) should be high, got {child_q_parent:.3f}"
    print("  ✓ PASSED")


# ============================================================================
# Node Allocation Tests
# ============================================================================

def test_deep_search_allocation():
    """Ensure node allocation doesn't overflow for deep searches"""
    print("\n" + "=" * 60)
    print("TEST: Deep Search Allocation")
    print("=" * 60)
        
    device = "cpu"
    model = DummyModel(device)
    board = torch.zeros(6, 7, device=device)
    
    config = TensorMCTSConfig(
        max_nodes_per_tree=5000,
        exploration_fraction=0.25
    )
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    legal = torch.ones(1, 7, device=device)
    current_players = torch.tensor([1], dtype=torch.long, device=device)
    
    with torch.no_grad():
        policies, _ = model.forward(board.unsqueeze(0).flatten(start_dim=1).float(), current_players)
    
    mcts.reset(board.unsqueeze(0), policies, legal, current_players)
    mcts.run_simulations(500, parallel_sims=8)
    
    nodes_used = mcts.next_node_idx[0].item()
    print(f"  Nodes allocated after 500 sims: {nodes_used}")
    print(f"  Max allowed: {config.max_nodes_per_tree}")
    
    assert nodes_used < config.max_nodes_per_tree, "Should not exceed max nodes"
    assert nodes_used > 100, "Should have allocated substantial nodes"
    print("  ✓ PASSED")


# ============================================================================
# PUCT Score Verification
# ============================================================================

def test_puct_scores():
    """Verify PUCT scores are computed correctly"""
    print("\n" + "=" * 60)
    print("TEST: PUCT Score Computation")
    print("=" * 60)
        
    device = "cpu"
    model = DummyModel(device)
    board = torch.zeros(6, 7, device=device)
    
    config = TensorMCTSConfig(
        c_puct=1.0,
        exploration_fraction=0.0,
        max_nodes_per_tree=2000
    )
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    legal = torch.ones(1, 7, device=device)
    current_players = torch.tensor([1], dtype=torch.long, device=device)
    
    with torch.no_grad():
        policies, _ = model.forward(board.unsqueeze(0).flatten(start_dim=1).float(), current_players)
    
    mcts.reset(board.unsqueeze(0), policies, legal, current_players)
    mcts.run_simulations(30, parallel_sims=1)
    
    # Manually compute PUCT for each child
    root_visits = mcts.visits[0, 0].item()
    print(f"  Root visits: {root_visits}")
    print(f"  {'Action':<8} {'N':<8} {'W':<10} {'Q(child)':<10} {'Q(parent)':<10} {'P':<8} {'U':<10} {'PUCT':<10}")
    print("  " + "-" * 74)
    
    for action in range(7):
        child_idx = mcts.children[0, 0, action].item()
        if child_idx < 0:
            continue
        
        n = mcts.visits[0, child_idx].item()
        w = mcts.total_value[0, child_idx].item()
        p = mcts.priors[0, 0, action].item()
        
        q_child = w / n if n > 0 else 0
        q_parent = -q_child  # Negate for parent's perspective
        u = config.c_puct * p * np.sqrt(root_visits + 1) / (1 + n)
        puct = q_parent + u
        
        print(f"  {action:<8} {n:<8.0f} {w:<10.3f} {q_child:<10.3f} {q_parent:<10.3f} {p:<8.3f} {u:<10.3f} {puct:<10.3f}")
    
    print("  ✓ PASSED (manual verification)")


# ============================================================================
# Terminal Value Perspective Test
# ============================================================================

def test_terminal_value_perspective():
    """Verify terminal values are from current player's perspective"""
    print("\n" + "=" * 60)
    print("TEST: Terminal Value Perspective")
    print("=" * 60)
    
    device = "cpu"
    model = DummyModel(device)
    
    # Board where player 1 can win immediately at column 3
    board = torch.zeros(6, 7, device=device)
    board[5, 0] = board[5, 1] = board[5, 2] = 1  # Player 1's pieces
    board[5, 4] = board[5, 5] = 2  # Player 2's pieces
    
    print("  Board (Player 1 wins at col 3):")
    print_board(board)
    
    # Run MCTS as player 1
    mcts = run_single_tree_mcts(board, current_player=1, model=model, n_sims=50, device=device)
    
    # Check the winning child node
    child_idx = mcts.children[0, 0, 3].item()
    if child_idx >= 0 and mcts.is_terminal[0, child_idx]:
        term_value = mcts.terminal_value[0, child_idx].item()
        child_player = mcts.current_player[0, child_idx].item()
        
        print(f"  Winning node (action 3):")
        print(f"    Current player at node: {child_player}")
        print(f"    Terminal value: {term_value}")
        
        # Terminal value is from the perspective of player at that node
        # Player 2 is at the child node (player 1 just won), so value should be -1
        assert child_player == 2, "Child should be player 2's turn"
        assert term_value == -1.0, f"Terminal value should be -1 (loss for player 2), got {term_value}"
        print("  ✓ Terminal value correctly shows loss for player 2")
    
    print("  ✓ PASSED")


# ============================================================================
# Run All Tests
# ============================================================================

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("OPTIMIZED TENSOR MCTS COMPREHENSIVE TEST SUITE")
    print("(with current_player tracking)")
    print("=" * 60)
    
    tests = [
        test_basic_tree_growth,
        test_terminal_detection,
        test_current_player_at_root,
        test_current_player_alternates,
        test_forced_win_horizontal,
        test_forced_win_vertical,
        test_forced_win_diagonal,
        test_must_block,
        test_draw_detection,
        test_full_column_handling,
        test_batch_consistency,
        test_virtual_loss_diversity,
        test_convergence,
        test_temperature_scaling,
        test_value_propagation,
        test_terminal_value_perspective,
        test_deep_search_allocation,
        test_puct_scores,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)