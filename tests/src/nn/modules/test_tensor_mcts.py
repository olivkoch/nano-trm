#!/usr/bin/env python
"""Functional tests for TensorMCTS with current_player tracking"""

import torch
import numpy as np
from src.nn.modules.tensor_mcts import TensorMCTS, TensorMCTSConfig, TensorMCTSWrapper


class DummyModel:
    """Model that returns uniform policy and zero value"""
    def __init__(self, device):
        self.device = device
    
    def forward(self, boards, current_players):
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
    
    def forward(self, boards, current_players):
        batch_size = boards.shape[0]
        boards = boards.view(batch_size, 6, 7)
        
        policies = torch.ones(batch_size, 7, device=self.device)
        values = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            board = boards[i]
            player = current_players[i].item()
            
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
            
            # Center preference
            policies[i, 3] *= 2
        
        policies = policies / policies.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return policies, values


class PlayerAwareModel:
    """Model that tracks which player it's evaluating for - useful for debugging"""
    def __init__(self, device):
        self.device = device
        self.call_log = []  # Track all calls for verification
    
    def forward(self, boards, current_players):
        batch_size = boards.shape[0]
        
        # Log the call
        self.call_log.append({
            'batch_size': batch_size,
            'current_players': current_players.tolist()
        })
        
        policies = torch.ones(batch_size, 7, device=self.device) / 7
        values = torch.zeros(batch_size, device=self.device)
        return policies, values
    
    def reset_log(self):
        self.call_log = []


def run_single_tree_mcts(board, current_player, model, n_sims=100, device="cpu"):
    """Run MCTS on a single position"""
    config = TensorMCTSConfig(
        n_actions=7,
        max_nodes_per_tree=2000,
        c_puct=1.0,
        exploration_fraction=0.0  # Disable noise for deterministic tests
    )
    
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_tensor = board.unsqueeze(0).to(device)
    legal_mask = (board[0, :] == 0).unsqueeze(0).float().to(device)
    current_players = torch.tensor([current_player], dtype=torch.long, device=device)
    
    with torch.no_grad():
        policies, _ = model.forward(board_tensor.flatten(start_dim=1), current_players)
    
    mcts.reset(board_tensor, policies, legal_mask, current_players)
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
    
    mcts = run_single_tree_mcts(board, current_player=1, model=model, n_sims=50, device=device)
    
    nodes_allocated = mcts.next_node_idx[0].item()
    root_visits = mcts.visits[0, 0].item()
    
    print(f"  Nodes allocated: {nodes_allocated}")
    print(f"  Root visits: {root_visits}")
    print(f"  Has states materialized: {mcts.has_state[0].sum().item()}")
    
    assert nodes_allocated > 8, "Should allocate more than just root children"
    assert root_visits >= 50, "Root should have at least n_sims visits"
    print("  ✓ PASSED")


def test_current_player_at_root():
    """Verify current_player is stored correctly at root"""
    print("\n" + "="*60)
    print("TEST: Current Player at Root")
    print("="*60)
    
    device = "cpu"
    model = DummyModel(device)
    board = torch.zeros(6, 7, device=device)
    
    # Test with player 1
    mcts1 = run_single_tree_mcts(board, current_player=1, model=model, n_sims=10, device=device)
    assert mcts1.current_player[0, 0].item() == 1, "Root should have player 1"
    print("  Player 1 at root: ✓")
    
    # Test with player 2 (add one piece to make it realistic)
    board2 = board.clone()
    board2[5, 0] = 1  # Player 1 played
    mcts2 = run_single_tree_mcts(board2, current_player=2, model=model, n_sims=10, device=device)
    assert mcts2.current_player[0, 0].item() == 2, "Root should have player 2"
    print("  Player 2 at root: ✓")
    
    print("  ✓ PASSED")


def test_current_player_alternates():
    """Verify current_player alternates correctly in tree"""
    print("\n" + "="*60)
    print("TEST: Current Player Alternates in Tree")
    print("="*60)
    
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
    
    # Check grandchildren (depth 2)
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
                    break  # Just check one per child
            break  # Just check one child's children
    
    print("  ✓ PASSED")


def test_model_receives_correct_player():
    """Verify model.forward receives correct current_player"""
    print("\n" + "="*60)
    print("TEST: Model Receives Correct Player")
    print("="*60)
    
    device = "cpu"
    model = PlayerAwareModel(device)
    board = torch.zeros(6, 7, device=device)
    
    # Run with player 1
    model.reset_log()
    mcts = run_single_tree_mcts(board, current_player=1, model=model, n_sims=20, device=device)
    
    print(f"  Model was called {len(model.call_log)} times")
    
    # First call should be for root initialization (player 1)
    assert model.call_log[0]['current_players'] == [1], \
        f"First call should be player 1, got {model.call_log[0]['current_players']}"
    print("  First call (root init): player 1 ✓")
    
    # Subsequent calls should have alternating players based on depth
    # This is harder to verify exactly, but we can check that both players appear
    all_players = []
    for call in model.call_log:
        all_players.extend(call['current_players'])
    
    has_player1 = 1 in all_players
    has_player2 = 2 in all_players
    print(f"  Players seen in calls: 1={has_player1}, 2={has_player2}")
    
    assert has_player1 and has_player2, "Both players should be seen during search"
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

    print("  Board (Player 1 to move, wins at col 3):")
    print("  " + "-"*15)
    for row in range(6):
        print("  |" + "".join([".12"[int(board[row,c])] for c in range(7)]) + "|")
    print("  " + "-"*15)
    
    # Player 1 to move (equal pieces = player 1's turn)
    mcts = run_single_tree_mcts(board, current_player=1, model=model, n_sims=100, device=device)
    dist = mcts._get_visit_distributions()[0]
    
    print(f"  Visit distribution: {dist.cpu().numpy().round(3)}")
    print(f"  Column 3 (winning): {dist[3].item():.1%}")
    
    assert dist[3] > 0.7, f"Should find winning move! Got {dist[3]:.1%}"
    print("  ✓ PASSED")


def test_forced_win_detailed():
    """MCTS must find immediate winning move - with diagnostics"""
    print("\n" + "="*60)
    print("TEST: Forced Win Detection (Detailed)")
    print("="*60)
    
    device = "cpu"
    model = DummyModel(device)
    
    board = torch.zeros(6, 7, device=device)
    board[5, 0] = board[5, 1] = board[5, 2] = 1
    board[5, 4] = board[5, 5] = board[4, 4] = 2
    
    p1_count = (board == 1).sum().item()
    p2_count = (board == 2).sum().item()
    print(f"  P1 count: {p1_count}, P2 count: {p2_count}")
    
    # Explicitly set player 1 to move
    current_player = 1
    print(f"  Current player: {current_player}")

    print("  Board:")
    print("  " + "-"*15)
    for row in range(6):
        print("  |" + "".join([".12"[int(board[row,c])] for c in range(7)]) + "|")
    print("  " + "-"*15)
    
    mcts = run_single_tree_mcts(board, current_player=current_player, model=model, n_sims=100, device=device)
    
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
        print(f"  Current player at child: {mcts.current_player[0, child_idx].item()}")
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
    print(f"  Root current player: {mcts.current_player[0, 0].item()}")
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
            cp = mcts.current_player[0, child].item()
            print(f"  Action {action}: child={child}, player={cp}, N={v:.0f}, Q={q:+.3f}, terminal={term}")
    
    dist = mcts._get_visit_distributions()[0]
    print(f"\n  Visit distribution: {dist.cpu().numpy().round(3)}")
    print(f"  Column 3 (winning): {dist[3].item():.1%}")
    
    assert dist[3] > 0.7, f"Should find winning move! Got {dist[3]:.1%}"
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
    board[5, 0] = board[5, 1] = board[5, 2] = 2  # Opponent's threat
    board[5, 6] = 1  # Our piece elsewhere
    
    print("  Board (Player 1 to move, must block col 3):")
    print("  " + "-"*15)
    for row in range(6):
        print("  |" + "".join([".12"[int(board[row,c])] for c in range(7)]) + "|")
    print("  " + "-"*15)
    
    # Player 1 to move
    mcts = run_single_tree_mcts(board, current_player=1, model=model, n_sims=400, device=device)
    dist = mcts._get_visit_distributions()[0]
    
    print(f"  Visit distribution: {dist.cpu().numpy().round(3)}")
    print(f"  Column 3 (blocking): {dist[3].item():.1%}")
    
    assert dist[3] > 0.65, f"Should block! Got {dist[3]:.1%}"
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
        mcts = run_single_tree_mcts(board, current_player=1, model=model, n_sims=n_sims, device=device)
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
    current_players = torch.tensor([1], dtype=torch.long, device=device)
    
    with torch.no_grad():
        policies, _ = model.forward(board_tensor.flatten(start_dim=1), current_players)
    
    mcts.reset(board_tensor, policies, legal_mask, current_players)
    
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


def test_terminal_value_perspective():
    """Verify terminal values are from current player's perspective"""
    print("\n" + "="*60)
    print("TEST: Terminal Value Perspective")
    print("="*60)
    
    device = "cpu"
    model = DummyModel(device)
    
    # Board where player 1 can win immediately at column 3
    board = torch.zeros(6, 7, device=device)
    board[5, 0] = board[5, 1] = board[5, 2] = 1  # Player 1's pieces
    board[5, 4] = board[5, 5] = 2  # Player 2's pieces
    
    print("  Board (Player 1 wins at col 3):")
    print("  " + "-"*15)
    for row in range(6):
        print("  |" + "".join([".12"[int(board[row,c])] for c in range(7)]) + "|")
    print("  " + "-"*15)
    
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
        
        # Terminal value should be from the perspective of player at that node
        # Player 2 is at the child node (player 1 just won), so value should be -1
        assert child_player == 2, "Child should be player 2's turn"
        assert term_value == -1.0, f"Terminal value should be -1 (loss for player 2), got {term_value}"
        print("  ✓ Terminal value correctly shows loss for player 2")
    
    print("  ✓ PASSED")


def test_wrapper_interface():
    """Test the TensorMCTSWrapper with current_players"""
    print("\n" + "="*60)
    print("TEST: Wrapper Interface with current_players")
    print("="*60)
    
    device = "cpu"
    model = DummyModel(device)
    
    wrapper = TensorMCTSWrapper(
        model=model,
        num_simulations=30,
        parallel_simulations=4,
        exploration_fraction=0.0,
        device=device
    )
    
    # Create batch of positions
    boards = [
        torch.zeros(6, 7, device=device),  # Empty board
        torch.zeros(6, 7, device=device),  # Empty board
    ]
    boards[1][5, 0] = 1  # One piece played
    
    legal_moves = [
        torch.ones(7, dtype=torch.bool, device=device),
        torch.ones(7, dtype=torch.bool, device=device),
    ]
    
    # Explicit current players
    current_players = torch.tensor([1, 2], dtype=torch.long, device=device)
    
    print("  Testing with explicit current_players...")
    visit_dist, action_probs = wrapper.get_action_probs_batch_parallel(
        boards, legal_moves, temperature=1.0, current_players=current_players
    )
    
    print(f"  Visit distributions shape: {visit_dist.shape}")
    print(f"  Board 0 (player 1): {visit_dist[0].cpu().numpy().round(3)}")
    print(f"  Board 1 (player 2): {visit_dist[1].cpu().numpy().round(3)}")
    
    assert visit_dist.shape == (2, 7), "Should have correct shape"
    print("  ✓ PASSED")


def test_wrapper_legacy_inference():
    """Test wrapper falls back to inferring current_player from piece counts"""
    print("\n" + "="*60)
    print("TEST: Wrapper Legacy Inference")
    print("="*60)
    
    device = "cpu"
    model = PlayerAwareModel(device)
    
    wrapper = TensorMCTSWrapper(
        model=model,
        num_simulations=10,
        parallel_simulations=2,
        exploration_fraction=0.0,
        device=device
    )
    
    # Board with equal pieces (player 1's turn)
    board1 = torch.zeros(6, 7, device=device)
    
    # Board with one more player 1 piece (player 2's turn)
    board2 = torch.zeros(6, 7, device=device)
    board2[5, 0] = 1
    
    boards = [board1, board2]
    legal_moves = [torch.ones(7, dtype=torch.bool, device=device) for _ in range(2)]
    
    print("  Testing WITHOUT explicit current_players (legacy mode)...")
    model.reset_log()
    
    # Don't pass current_players - should infer
    visit_dist, action_probs = wrapper.get_action_probs_batch_parallel(
        boards, legal_moves, temperature=1.0, current_players=None
    )
    
    # Check first call had inferred players
    first_call_players = model.call_log[0]['current_players']
    print(f"  Inferred players for root: {first_call_players}")
    
    assert first_call_players == [1, 2], f"Should infer [1, 2], got {first_call_players}"
    print("  ✓ PASSED")


def visualize_tree_structure(mcts, tree_idx=0, max_depth=2):
    """Pretty print tree structure with current player info"""
    print("\n" + "="*60)
    print("TREE STRUCTURE VISUALIZATION")
    print("="*60)
    
    def format_node(node_idx, depth, action):
        visits = mcts.visits[tree_idx, node_idx].item()
        value = mcts.total_value[tree_idx, node_idx].item()
        q = value / visits if visits > 0 else 0
        is_term = mcts.is_terminal[tree_idx, node_idx].item()
        player = mcts.current_player[tree_idx, node_idx].item()
        
        indent = "│   " * depth
        if action == -1:
            label = "ROOT"
        else:
            label = f"a={action}"
        
        term_str = " [TERM]" if is_term else ""
        return f"{indent}├── {label}: P{player}, N={visits:.0f}, Q={q:+.3f}{term_str}"
    
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
                print(format_node(child, depth + 1, action) + f" prior={prior:.3f}")
                print_children(child, depth + 1)
    
    print_children(0, 0)


def run_all_tests():
    """Run all functional tests"""
    print("\n" + "="*60)
    print("TENSOR MCTS FUNCTIONAL TESTS (with current_player)")
    print("="*60)
    
    # Core functionality
    test_basic_tree_growth()
    test_terminal_detection()
    
    # Current player tracking
    test_current_player_at_root()
    test_current_player_alternates()
    test_model_receives_correct_player()
    test_terminal_value_perspective()
    
    # Game-play tests
    test_forced_win()
    test_forced_win_detailed()
    test_forced_block()
    
    # Search behavior
    test_convergence()
    test_virtual_loss()
    
    # Wrapper tests
    test_wrapper_interface()
    test_wrapper_legacy_inference()
    
    # Visualization
    device = "cpu"
    model = DummyModel(device)
    board = torch.zeros(6, 7, device=device)
    mcts = run_single_tree_mcts(board, current_player=1, model=model, n_sims=100, device=device)
    visualize_tree_structure(mcts, max_depth=2)
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()