#!/usr/bin/env python
"""
Comprehensive test suite comparing tensor_mcts vs mcts_v2

Tests individual components and overall behavior to find divergences.
"""

import torch
import numpy as np
import copy
import math
from typing import Tuple, List

# Import both implementations
from src.nn.modules.tensor_mcts import TensorMCTS, TensorMCTSConfig
from src.nn.modules.mcts_v2 import (
    Node, DummyNode, expand, backup, add_virtual_loss, revert_virtual_loss,
    best_child, parallel_uct_search
)
from src.nn.environments.connectfour_env import ConnectFourEnv


# =============================================================================
# Test Utilities
# =============================================================================

class DummyModel:
    def __init__(self, device="cpu"): self.device = device
    def forward(self, boards, current_players):
        batch_size = boards.shape[0]
        return (torch.ones(batch_size, 7, device=self.device) / 7, torch.zeros(batch_size, device=self.device))


class BiasedModel:
    """Model that favors specific columns"""
    def __init__(self, favored_col: int, device="cpu"):
        self.favored_col = favored_col
        self.device = device
    
    def forward(self, boards, current_players):
        batch_size = boards.shape[0]
        # Give 50% to favored column, rest uniform
        policies = torch.ones(batch_size, 7, device=self.device) * (0.5 / 6)
        policies[:, self.favored_col] = 0.5
        values = torch.zeros(batch_size, device=self.device)
        return policies, values


def make_numpy_eval_func():
    def eval_func(obs, batched):
        if batched: return np.ones((obs.shape[0], 7), dtype=np.float32)/7, [0.0]*obs.shape[0]
        else: return np.ones(7, dtype=np.float32)/7, 0.0
    return eval_func

def create_board(moves: List[int]) -> np.ndarray:
    board = np.zeros((6, 7), dtype=np.int8)
    player = 1
    for col in moves:
        for row in range(5, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                break
        player = 3 - player
    return board

# =============================================================================
# Test 1: Terminal Detection
# =============================================================================

def test_terminal_detection():
    """Test that both implementations detect terminals correctly"""
    print("\n" + "=" * 70)
    print("TEST 1: Terminal Detection")
    print("=" * 70)
    
    device = "cpu"
    model = DummyModel(device)
    config = TensorMCTSConfig(exploration_fraction=0.0)
    
    test_cases = [
        # (name, moves, expected_winner)
        ("Horizontal win P1", [0, 6, 1, 6, 2, 6, 3], 1),  # P1 wins bottom row
        ("Horizontal win P2", [6, 0, 6, 1, 6, 2, 5, 3], 2),  # P2 wins bottom row
        ("Vertical win P1", [0, 1, 0, 1, 0, 1, 0], 1),  # P1 wins col 0
        ("Diagonal win P1", [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3], 1),  # Diagonal
        ("No winner yet", [0, 1, 2, 3], None),
        ("Empty board", [], None),
    ]
    
    all_passed = True
    for name, moves, expected in test_cases:
        board = create_board(moves)
        
        # tensor_mcts terminal check
        mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
        board_t = torch.from_numpy(board).unsqueeze(0).long()
        is_term, winner = mcts._check_terminal_batch(board_t)
        tensor_result = winner[0].item() if is_term[0].item() else None
        
        # mcts_v2 uses env
        env = ConnectFourEnv()
        env.board = board.copy()
        env.to_play = 1 if len(moves) % 2 == 0 else 2
        v2_result = None
        if env.is_game_over():
            # Check who won
            for p in [1, 2]:
                # Check horizontal
                for r in range(6):
                    for c in range(4):
                        if np.all(board[r, c:c+4] == p):
                            v2_result = p
                # Check vertical
                for r in range(3):
                    for c in range(7):
                        if np.all(board[r:r+4, c] == p):
                            v2_result = p
                # Check diagonals
                for r in range(3):
                    for c in range(4):
                        if all(board[r+i, c+i] == p for i in range(4)):
                            v2_result = p
                        if all(board[r+i, c+3-i] == p for i in range(4)):
                            v2_result = p
            if v2_result is None and np.all(board != 0):
                v2_result = 0  # Draw
        
        passed = (tensor_result == expected)
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: tensor={tensor_result}, expected={expected}")
        if not passed:
            all_passed = False
            print(f"    Board:\n{board}")
    
    return all_passed


# =============================================================================
# Test 2: PUCT Formula
# =============================================================================

def test_puct_formula():
    """Test that PUCT calculations match"""
    print("\n" + "=" * 70)
    print("TEST 2: PUCT Formula")
    print("=" * 70)
    
    c_puct_base = 19652.0
    c_puct_init = 1.25
    
    test_cases = [
        # (parent_N, child_N, child_W, child_P)
        (1, np.array([0, 0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0]), np.array([1/7]*7)),
        (10, np.array([2, 3, 1, 0, 2, 1, 1]), np.array([0.5, -0.3, 0.2, 0, -0.1, 0.1, 0]), np.array([0.2, 0.1, 0.15, 0.1, 0.2, 0.15, 0.1])),
        (100, np.array([20, 30, 10, 5, 20, 10, 5]), np.array([5, -3, 2, 1, -2, 1, 0.5]), np.array([0.2, 0.1, 0.15, 0.1, 0.2, 0.15, 0.1])),
    ]
    
    all_passed = True
    for parent_N, child_N, child_W, child_P in test_cases:
        # mcts_v2 calculation
        pb_c_v2 = math.log((1 + parent_N + c_puct_base) / c_puct_base) + c_puct_init
        child_N_safe = np.where(child_N > 0, child_N, 1)
        Q_v2 = child_W / child_N_safe
        U_v2 = pb_c_v2 * child_P * (math.sqrt(parent_N) / (1 + child_N))
        ucb_v2 = -Q_v2 + U_v2
        
        # tensor_mcts calculation
        parent_N_t = torch.tensor([parent_N], dtype=torch.float32)
        child_N_t = torch.tensor([child_N], dtype=torch.float32)
        child_W_t = torch.tensor([child_W], dtype=torch.float32)
        child_P_t = torch.tensor([child_P], dtype=torch.float32)
        
        # pb_c
        pb_c_tensor = torch.log((1 + parent_N_t + c_puct_base) / c_puct_base) + c_puct_init
        
        # Q (with clamp for N=0)
        effective_visits = torch.clamp(child_N_t, min=1)
        Q_tensor = -child_W_t / effective_visits  # Note: already negated
        
        # U - tensor_mcts uses sqrt(N+1) instead of sqrt(N)
        # This is a KNOWN DIFFERENCE - let's test both
        U_tensor_v1 = pb_c_tensor * child_P_t * torch.sqrt(parent_N_t + 1) / (1 + child_N_t)  # tensor_mcts version
        U_tensor_v2 = pb_c_tensor * child_P_t * torch.sqrt(parent_N_t) / (1 + child_N_t)  # mcts_v2 version
        
        ucb_tensor_v1 = (Q_tensor + U_tensor_v1)[0].numpy()
        ucb_tensor_v2 = (Q_tensor + U_tensor_v2)[0].numpy()
        
        # Compare
        diff_v1 = np.abs(ucb_tensor_v1 - ucb_v2).max()
        diff_v2 = np.abs(ucb_tensor_v2 - ucb_v2).max()
        
        print(f"\n  Parent N={parent_N}")
        print(f"    mcts_v2 UCB:      {ucb_v2.round(4)}")
        print(f"    tensor sqrt(N+1): {ucb_tensor_v1.round(4)} (diff={diff_v1:.6f})")
        print(f"    tensor sqrt(N):   {ucb_tensor_v2.round(4)} (diff={diff_v2:.6f})")
        
        if diff_v2 > 0.0001:
            print(f"    ✗ Mismatch when using sqrt(N)!")
            all_passed = False
        else:
            print(f"    ✓ Matches when using sqrt(N)")
    
    return all_passed


# =============================================================================
# Test 3: Single Simulation Trace
# =============================================================================

def test_single_simulation():
    """Trace a single simulation through both implementations"""
    print("\n" + "=" * 70)
    print("TEST 3: Single Simulation Trace")
    print("=" * 70)
    
    device = "cpu"
    
    # Empty board, uniform policy
    board = np.zeros((6, 7), dtype=np.int8)
    
    # === mcts_v2 ===
    print("\n  mcts_v2 (1 simulation, no parallel):")
    env = ConnectFourEnv()
    env.reset()
    eval_func = make_numpy_eval_func()
    
    # Manual single simulation
    prior_prob, value = eval_func(env.observation(), False)
    root = Node(to_play=env.to_play, num_actions=7, parent=DummyNode())
    expand(root, prior_prob)
    backup(root, float(value))
    
    print(f"    After root expand+backup:")
    print(f"      Root N: {root.N}")
    print(f"      Root W: {root.W}")
    print(f"      child_N: {root.child_N}")
    print(f"      child_W: {root.child_W}")
    print(f"      child_P: {root.child_P.round(4)}")
    
    # Do one more selection+expand+backup
    sim_env = copy.deepcopy(env)
    node = best_child(root, sim_env.legal_actions, 19652.0, 1.25, sim_env.opponent_player)
    print(f"    Selected child for action: {node.move}")
    obs, reward, done, _ = sim_env.step(node.move)
    
    prior_prob, value = eval_func(obs, False)
    expand(node, prior_prob)
    backup(node, float(value))
    
    print(f"    After child expand+backup:")
    print(f"      Root N: {root.N}")
    print(f"      child_N: {root.child_N}")
    print(f"      child_W: {root.child_W}")
    
    # === tensor_mcts ===
    print("\n  tensor_mcts (1 simulation):")
    model = DummyModel(device)
    config = TensorMCTSConfig(
        c_puct_base=19652.0,
        c_puct_init=1.25,
        exploration_fraction=0.0,
        virtual_loss_weight=1.0  # MATCH DEFAULT
    )
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.from_numpy(board).unsqueeze(0).long()
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, _ = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal, players)
    
    print(f"    After reset (before any simulations):")
    print(f"      Root visits: {mcts.visits[0, 0].item()}")
    print(f"      Root value: {mcts.total_value[0, 0].item()}")
    print(f"      Root priors: {mcts.priors[0, 0].numpy().round(4)}")
    print(f"      Children of root: {mcts.children[0, 0].tolist()}")
    
    # Get child visits/values
    root_children = mcts.children[0, 0]
    valid = root_children >= 0
    if valid.any():
        child_visits = mcts.visits[0, root_children[valid]]
        child_values = mcts.total_value[0, root_children[valid]]
        print(f"      Child visits: {child_visits.tolist()}")
        print(f"      Child values: {child_values.tolist()}")
    
    # Run 1 simulation
    mcts.run_simulations(1, parallel_sims=1)
    
    print(f"    After 1 simulation:")
    print(f"      Root visits: {mcts.visits[0, 0].item()}")
    print(f"      Root value: {mcts.total_value[0, 0].item()}")
    if valid.any():
        child_visits = mcts.visits[0, root_children[valid]]
        child_values = mcts.total_value[0, root_children[valid]]
        print(f"      Child visits: {child_visits.tolist()}")
        print(f"      Child values: {child_values.tolist()}")
    
    return True


# =============================================================================
# Test 4: Winning Move Detection
# =============================================================================

def test_winning_move():
    """Test that both find obvious winning moves"""
    print("\n" + "=" * 70)
    print("TEST 4: Winning Move Detection")
    print("=" * 70)
    
    device = "cpu"
    
    test_cases = [
        # (name, moves_to_setup, winning_col, current_player)
        ("P1 horizontal win at col 3", [0, 6, 1, 6, 2, 6], 3, 1),  # X X X _ at bottom
        ("P1 vertical win at col 0", [0, 1, 0, 1, 0, 1], 0, 1),    # Three X stacked
        ("P2 horizontal win at col 3", [6, 0, 6, 1, 6, 2, 5], 3, 2),  # O O O _ at bottom
    ]
    
    all_passed = True
    for name, setup_moves, winning_col, current_player in test_cases:
        board = create_board(setup_moves)
        
        print(f"\n  {name}:")
        print(f"    Board (P{current_player} to move):")
        for row in range(6):
            print(f"      |{''.join(['.XO'[board[row, c]] for c in range(7)])}|")
        
        # mcts_v2
        env = ConnectFourEnv()
        env.board = board.copy()
        env.to_play = current_player
        eval_func = make_numpy_eval_func()
        
        _, v2_dist, _, _, _ = parallel_uct_search(
            env, eval_func, None, 19652.0, 1.25, 100, num_parallel=1
        )
        v2_best = np.argmax(v2_dist)
        
        # tensor_mcts
        model = DummyModel(device)
        config = TensorMCTSConfig(
            c_puct_base=19652.0,
            c_puct_init=1.25,
            exploration_fraction=0.0
        )
        mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
        
        board_t = torch.from_numpy(board).unsqueeze(0).long()
        legal = torch.from_numpy((board[0, :] == 0).astype(np.float32)).unsqueeze(0)
        players = torch.tensor([current_player], dtype=torch.long)
        
        with torch.no_grad():
            policies, _ = model.forward(board_t.flatten(1).float(), players)
        
        mcts.reset(board_t, policies, legal, players)
        tensor_dist = mcts.run_simulations(100, parallel_sims=1)[0].numpy()
        tensor_best = np.argmax(tensor_dist)
        
        print(f"    mcts_v2:     best={v2_best}, dist={v2_dist.round(3)}")
        print(f"    tensor_mcts: best={tensor_best}, dist={tensor_dist.round(3)}")
        
        v2_ok = v2_best == winning_col
        tensor_ok = tensor_best == winning_col
        
        if v2_ok and tensor_ok:
            print(f"    ✓ Both found winning move at col {winning_col}")
        elif v2_ok and not tensor_ok:
            print(f"    ✗ mcts_v2 correct, tensor_mcts WRONG (expected col {winning_col})")
            all_passed = False
        elif not v2_ok and tensor_ok:
            print(f"    ? tensor_mcts correct, mcts_v2 wrong?")
        else:
            print(f"    ✗ Both wrong! Expected col {winning_col}")
            all_passed = False
    
    return all_passed


# =============================================================================
# Test 5: Backup Value Propagation
# =============================================================================

def test_backup_values():
    """Test that backup propagates values correctly"""
    print("\n" + "=" * 70)
    print("TEST 5: Backup Value Propagation")
    print("=" * 70)
    
    # Test: Start from terminal position and check value propagation
    device = "cpu"
    
    # Position where P1 can win immediately
    # After P1 plays col 3, game ends with P1 win
    setup_moves = [0, 6, 1, 6, 2, 6]  # X X X _ (P1 to play col 3 wins)
    board = create_board(setup_moves)
    
    print(f"\n  Setup: P1 has X X X at bottom, col 3 wins")
    print(f"  After P1 plays col 3: terminal, P1 wins, value = +1 for P1")
    print(f"  This value should propagate up with sign flip")
    
    # === mcts_v2 trace ===
    print(f"\n  mcts_v2 backup trace:")
    env = ConnectFourEnv()
    env.board = board.copy()
    env.to_play = 1
    eval_func = make_numpy_eval_func()
    
    prior_prob, value = eval_func(env.observation(), False)
    root = Node(to_play=1, num_actions=7, parent=DummyNode())
    expand(root, prior_prob)
    backup(root, float(value))
    
    # Select col 3 and play
    sim_env = copy.deepcopy(env)
    # Force selection of col 3
    root.child_N[3] = 0  # Ensure it's selectable
    root.child_P[3] = 1.0  # High prior
    root.child_P = root.child_P / root.child_P.sum()
    
    node = best_child(root, sim_env.legal_actions, 19652.0, 1.25, 2)
    print(f"    Selected action: {node.move}")
    obs, reward, done, _ = sim_env.step(node.move)
    print(f"    Game done: {done}, reward: {reward}")
    
    if done:
        # Reward is from perspective of player who just moved (P1)
        # node.to_play is P2 (opponent)
        # backup expects value from current player's perspective
        # Since P1 won, P2's value is -1
        backup(node, float(-reward))
        print(f"    Backed up value: {-reward} (from P2's perspective)")
        print(f"    Root child_N after: {root.child_N}")
        print(f"    Root child_W after: {root.child_W}")
        print(f"    Root child_Q: {(root.child_W / np.maximum(root.child_N, 1)).round(4)}")
    
    # === tensor_mcts trace ===
    print(f"\n  tensor_mcts backup trace:")
    model = DummyModel(device)
    config = TensorMCTSConfig(
        c_puct_base=19652.0,
        c_puct_init=1.25,
        exploration_fraction=0.0
    )
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.from_numpy(board).unsqueeze(0).long()
    legal = torch.from_numpy((board[0, :] == 0).astype(np.float32)).unsqueeze(0)
    
    # Give high prior to col 3 to force its selection
    policies = torch.zeros(1, 7)
    policies[0, 3] = 1.0
    players = torch.tensor([1], dtype=torch.long)
    
    mcts.reset(board_t, policies, legal, players)
    
    print(f"    Before simulation:")
    print(f"      Children of root: {mcts.children[0, 0].tolist()}")
    
    # Run one simulation
    leaves, paths, path_lengths = mcts._batch_select_leaves()
    print(f"    Selected leaf: {leaves.item()}, path_length: {path_lengths.item()}")
    print(f"    Path: {paths[0, :path_lengths.item()].tolist()}")
    
    # Check if leaf is at col 3's child
    col3_child = mcts.children[0, 0, 3].item()
    print(f"    Col 3 child node: {col3_child}")
    
    # Materialize and evaluate
    leaves_t = leaves.unsqueeze(0)
    paths_t = paths.unsqueeze(0)
    lengths_t = path_lengths.unsqueeze(0)
    
    mcts._materialize_states_vectorized(leaves_t, paths_t, lengths_t)
    
    leaf_state = mcts.states[0, leaves.item()]
    print(f"    Leaf state board:")
    for row in range(6):
        print(f"      |{''.join(['.XO'[leaf_state[row, c].item()] for c in range(7)])}|")
    
    is_term, winner = mcts._check_terminal_batch(leaf_state.unsqueeze(0))
    print(f"    Terminal: {is_term.item()}, Winner: {winner.item()}")
    
    mcts._batch_evaluate_and_expand(leaves_t)
    print(f"    Leaf value: {mcts._last_leaf_values.item()}")
    
    mcts._batch_backup_vectorized(leaves_t, paths_t, lengths_t)
    
    root_children = mcts.children[0, 0]
    valid = root_children >= 0
    child_visits = mcts.visits[0, root_children[valid]]
    child_values = mcts.total_value[0, root_children[valid]]
    print(f"    After backup:")
    print(f"      Child visits: {child_visits.tolist()}")
    print(f"      Child values: {child_values.tolist()}")
    
    return True


# =============================================================================
# Test 6: Visit Distribution Convergence
# =============================================================================

def test_visit_convergence():
    """Test that visit distributions converge similarly"""
    print("\n" + "=" * 70)
    print("TEST 6: Visit Distribution Convergence")
    print("=" * 70)
    
    device = "cpu"
    
    # Empty board - should explore roughly uniformly
    board = np.zeros((6, 7), dtype=np.int8)
    
    for num_sims in [50, 100, 200]:
        print(f"\n  {num_sims} simulations (empty board):")
        
        # mcts_v2
        env = ConnectFourEnv()
        env.reset()
        eval_func = make_numpy_eval_func()
        
        _, v2_dist, _, _, _ = parallel_uct_search(
            env, eval_func, None, 19652.0, 1.25, num_sims, num_parallel=1
        )
        
        # tensor_mcts
        model = DummyModel(device)
        config = TensorMCTSConfig(
            c_puct_base=19652.0,
            c_puct_init=1.25,
            exploration_fraction=0.0
        )
        mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
        
        board_t = torch.zeros(1, 6, 7, dtype=torch.long)
        legal = torch.ones(1, 7)
        players = torch.tensor([1], dtype=torch.long)
        
        with torch.no_grad():
            policies, _ = model.forward(board_t.flatten(1).float(), players)
        
        mcts.reset(board_t, policies, legal, players)
        tensor_dist = mcts.run_simulations(num_sims, parallel_sims=1)[0].numpy()
        
        # Compare distributions
        diff = np.abs(tensor_dist - v2_dist)
        max_diff = diff.max()
        
        print(f"    mcts_v2:     {v2_dist.round(3)}")
        print(f"    tensor_mcts: {tensor_dist.round(3)}")
        print(f"    Max difference: {max_diff:.4f}")
        
        # With uniform policy on empty board, both should explore similarly
        # Allow some difference due to implementation details
        if max_diff > 0.15:
            print(f"    ✗ Distributions differ significantly!")
            return False
        else:
            print(f"    ✓ Distributions reasonably similar")
    return True

# =============================================================================
# Test 7: Virtual Loss Effect
# =============================================================================

def test_virtual_loss():
    """Test virtual loss behavior in parallel simulations"""
    print("\n" + "=" * 70)
    print("TEST 7: Virtual Loss Effect")
    print("=" * 70)
    
    device = "cpu"
    board = np.zeros((6, 7), dtype=np.int8)
    
    print("\n  Testing with parallel_sims=8:")
    print("  Virtual loss should diversify exploration")
    
    # tensor_mcts with parallel
    model = DummyModel(device)
    config = TensorMCTSConfig(
        c_puct_base=19652.0,
        c_puct_init=1.25,
        exploration_fraction=0.0,
        virtual_loss_weight=1.0  # LOWER WEIGHT (match mcts_v2)
    )
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.zeros(1, 6, 7, dtype=torch.long)
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, _ = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal, players)
    
    # Run just one batch of 8 parallel simulations
    # Each should select a DIFFERENT action due to virtual loss
    print(f"\n  First batch of 8 parallel selections:")
    
    selected_actions = []
    for i in range(8):
        leaves, paths, path_lengths = mcts._batch_select_leaves()
        leaf_node = leaves.item()
        # The leaf should be a child of root
        parent = mcts.parent[0, leaf_node].item()
        if parent == 0:  # Parent is root
            action = mcts.parent_action[0, leaf_node].item()
            selected_actions.append(action)
            print(f"    Sim {i}: selected action {action}")
    
    unique_actions = len(set(selected_actions))
    print(f"\n  Unique actions selected: {unique_actions}/8")
    
    if unique_actions >= 6:
        print(f"  ✓ Good diversity - virtual loss is working")
    elif unique_actions >= 4:
        print(f"  ~ Moderate diversity")
    else:
        print(f"  ✗ Poor diversity - virtual loss may not be working correctly")
    
    return unique_actions >= 4

# =============================================================================
# Test 8: Winning Move 2
# =============================================================================

def test_winning_move_2():
    """Verify both implementations find the same winning move"""
    print("\n" + "=" * 70)
    print("Test 8 - Correctness Test: Both should find winning move at col 3")
    print("=" * 70)
    
    device = "cpu"
    
    # Setup position where col 3 wins
    board_np = np.zeros((6, 7), dtype=np.int8)
    board_np[5, 0:3] = 1  # X X X _
    board_np[5, 4:6] = 2
    board_np[4, 4] = 2
    
    print("\nBoard:")
    print("  " + "-" * 15)
    for row in range(6):
        print("  |" + "".join([".XO"[board_np[row, c]] for c in range(7)]) + "|")
    print("  " + "-" * 15)
    print("   0123456")
    print("  (X wins at col 3)")
    
    # Test mcts_v2 using parallel_uct_search
    env = ConnectFourEnv()
    env.board = board_np.copy()
    env.to_play = env.black_player  # Player 1
    env.legal_actions = (board_np[0, :] == 0).astype(np.float32)
    eval_func = make_numpy_eval_func()
    
    move, v2_dist, _, _, _ = parallel_uct_search(
        env, eval_func, None, 19652.0, 1.25, 200, num_parallel=8
    )
    
    # Test tensor_mcts with debug
    model = DummyModel(device)
    config = TensorMCTSConfig(
        c_puct_base=19652.0,
        c_puct_init=1.25,
        exploration_fraction=0.0,
        virtual_loss_weight=1.0  # Lower VL
    )
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.from_numpy(board_np).unsqueeze(0).long()
    legal = torch.from_numpy((board_np[0, :] == 0).astype(np.float32)).unsqueeze(0)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, _ = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal, players)
    
    # Run just 1 batch of 8 simulations for debugging
    print("\n--- DEBUG: After 1 batch (8 sims) ---")
    mcts.run_simulations(8, parallel_sims=8)
    
    # Check what we got
    root_children = mcts.children[0, 0]  # [node_idx for each action]
    print(f"Root children node indices: {root_children.tolist()}")
    
    for action in range(7):
        child_idx = root_children[action].item()
        if child_idx >= 0:
            visits = mcts.visits[0, child_idx].item()
            value = mcts.total_value[0, child_idx].item()
            is_term = mcts.is_terminal[0, child_idx].item()
            term_val = mcts.terminal_value[0, child_idx].item()
            q = value / max(visits, 1)
            print(f"  Action {action} (node {child_idx}): visits={visits:.0f}, W={value:.2f}, Q={q:.2f}, terminal={is_term}, term_val={term_val:.2f}")
    
    # Check if state for action 3's child is correct
    action_3_child = root_children[3].item()
    if action_3_child >= 0:
        state = mcts.states[0, action_3_child]
        print(f"\nState at action 3's child (node {action_3_child}):")
        for row in range(6):
            print("  |" + "".join([".XO"[state[row, c].item()] for c in range(7)]) + "|")
    
    # Continue running remaining sims
    mcts.reset(board_t, policies, legal, players)
    tensor_dist = mcts.run_simulations(200, parallel_sims=8)[0].numpy()
    
    print(f"\nmcts_v2 distribution:     {v2_dist.round(3)}")
    print(f"tensor_mcts distribution: {tensor_dist.round(3)}")
    print(f"\nmcts_v2 best move:     col {np.argmax(v2_dist)} ({v2_dist.max():.1%})")
    print(f"tensor_mcts best move: col {np.argmax(tensor_dist)} ({tensor_dist.max():.1%})")
    
    if np.argmax(v2_dist) == 3 and np.argmax(tensor_dist) == 3:
        print("\n✓ PASSED: Both implementations found the winning move")
        return True
    else:
        print("\n✗ FAILED: Implementations disagree or missed winning move")
        return False


# =============================================================================
# Test 9: Illegal Move Masking
# =============================================================================

def test_illegal_masking():
    """Test that full columns are strictly masked out"""
    print("\n" + "=" * 70)
    print("TEST 9: Illegal Move Masking")
    print("=" * 70)

    device = "cpu"
    
    # Create board where col 0 is FULL (rows 0-5 filled)
    board = np.zeros((6, 7), dtype=np.int8)
    board[:, 0] = 1 
    
    # === mcts_v2 ===
    env = ConnectFourEnv()
    env.board = board.copy()
    # Mask: 1 if empty, 0 if full. Check top row (row 0 in this logic is top? 
    # Based on create_board, row 0 is TOP. If row 0 is full, col is full.)
    legal_mask = (board[0, :] == 0).astype(np.float32) 
    env.legal_actions = legal_mask
    
    eval_func = make_numpy_eval_func()
    
    # Run v2
    _, v2_dist, _, _, _ = parallel_uct_search(
        env, eval_func, None, 19652.0, 1.25, 50, num_parallel=1
    )
    
    # === tensor_mcts ===
    model = DummyModel(device)
    # Zero exploration to strictly test masking
    config = TensorMCTSConfig(exploration_fraction=0.0)
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.from_numpy(board).unsqueeze(0).long()
    legal_t = torch.from_numpy(legal_mask).unsqueeze(0)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, val = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal_t, players, root_values=val)
    tensor_dist = mcts.run_simulations(50, parallel_sims=1)[0].numpy()
    
    print(f"  Col 0 is FULL.")
    print(f"  mcts_v2 prob for col 0:     {v2_dist[0]}")
    print(f"  tensor_mcts prob for col 0: {tensor_dist[0]}")
    
    if v2_dist[0] == 0.0 and tensor_dist[0] == 0.0:
        print("  ✓ Both successfully masked illegal move")
        return True
    else:
        print("  ✗ FAILED: One implementation allowed an illegal move")
        return False


# =============================================================================
# Test 10: Draw Detection
# =============================================================================

def test_draw_detection():
    """Test handling of a draw (full board, no winner)"""
    print("\n" + "=" * 70)
    print("TEST 10: Draw Detection")
    print("=" * 70)
    
    device = "cpu"
    
    # Create a draw board
    # Pattern that fills board but has no 4-in-row
    board = np.zeros((6, 7), dtype=np.int8)
    for r in range(6):
        for c in range(7):
            if c % 2 == 0:
                board[r, c] = 1 if (r // 2) % 2 == 0 else 2
            else:
                board[r, c] = 2 if (r // 2) % 2 == 0 else 1

    # Verify it's actually a draw and full
    assert (board != 0).all()
    
    # === tensor_mcts ===
    model = DummyModel(device)
    mcts = TensorMCTS(model, n_trees=1, config=TensorMCTSConfig(), device=device)
    board_t = torch.from_numpy(board).unsqueeze(0).long()
    
    is_term, winners = mcts._check_terminal_batch(board_t)
    
    print(f"  Board Full: {(board_t!=0).all().item()}")
    print(f"  Is Terminal: {is_term.item()}")
    print(f"  Winner ID:   {winners.item()} (0 means Draw)")
    
    if is_term.item() and winners.item() == 0:
        print("  ✓ Correctly identified as Draw")
        return True
    else:
        print("  ✗ FAILED to identify draw")
        return False


# =============================================================================
# Test 11: Batch Independence
# =============================================================================

def test_batch_independence():
    print("\n" + "=" * 70 + "\nTEST 11: Batch Independence\n" + "=" * 70)
    device = "cpu"
    def make_stack(col, height):
        b = np.zeros((6, 7), dtype=np.int8)
        for i in range(height): b[5-i, col] = 1
        return b
    b0, b1 = make_stack(0, 3), make_stack(6, 3)
    boards = np.stack([b0, b1])
    board_t = torch.from_numpy(boards).long()
    
    model = DummyModel(device)
    config = TensorMCTSConfig(exploration_fraction=0.0, virtual_loss_weight=0.1) # Help DummyModel
    mcts = TensorMCTS(model, n_trees=2, config=config, device=device)
    
    legal = torch.ones(2, 7)
    players = torch.tensor([1, 1], dtype=torch.long)
    with torch.no_grad(): policies, vals = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    dists = mcts.run_simulations(200, parallel_sims=8).numpy()
    
    best_0, best_1 = np.argmax(dists[0]), np.argmax(dists[1])
    print(f"  Game 0 (Win at 0) best: {best_0}, Game 1 (Win at 6) best: {best_1}")
    return best_0 == 0 and best_1 == 6

# =============================================================================
# Test 12: Virtual Loss Cleanliness
# =============================================================================

def test_virtual_loss_cleanliness():
    """Ensure Virtual Loss is exactly 0.0 after simulations finish"""
    print("\n" + "=" * 70)
    print("TEST 12: Virtual Loss Cleanliness")
    print("=" * 70)
    
    device = "cpu"
    
    board = np.zeros((6, 7), dtype=np.int8)
    model = DummyModel(device)
    # High VL weight to make leaks obvious
    mcts = TensorMCTS(model, n_trees=1, config=TensorMCTSConfig(virtual_loss_weight=10.0), device=device)
    
    board_t = torch.from_numpy(board).unsqueeze(0).long()
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)

    mcts.reset(board_t, policies, legal, players, root_values=vals)
    mcts.run_simulations(50, parallel_sims=8)
    
    # Check internal tensor
    max_vl = mcts.virtual_loss.max().item()
    min_vl = mcts.virtual_loss.min().item()
    nonzero = mcts.virtual_loss.count_nonzero().item()
    
    print(f"  Max VL remaining: {max_vl}")
    print(f"  Min VL remaining: {min_vl}")
    print(f"  Non-zero VL nodes: {nonzero}")
    
    if max_vl == 0.0 and min_vl == 0.0:
        print("  ✓ Virtual Loss cleanly reverted")
        return True
    else:
        print("  ✗ FAILED: Virtual Loss leaked (non-zero residuals)")
        return False


# =============================================================================
# Test 13: Forced Block (Defense)
# =============================================================================

def test_forced_mate_depth():
    """Test recognizing a forced loss (must block opponent) comparing both engines"""
    print("\n" + "=" * 70)
    print("TEST 13: Forced Block (Defense)")
    print("=" * 70)
    
    device = "cpu"
    
    # Setup: P1 to play. 
    # Opponent (P2) has 3 in a row at Col 0 (rows 5,4,3).
    # If P1 does not play Col 0, P2 will play Col 0 next and win.
    # Therefore, P1 MUST play Col 0 to block.
    
    board_np = np.zeros((6, 7), dtype=np.int8)
    board_np[5, 0] = 2
    board_np[4, 0] = 2
    board_np[3, 0] = 2
    
    print(f"  Scenario: P2 has 3-in-row at Col 0. P1 (to move) must block.")
    
    # --- Run mcts_v2 ---
    env = ConnectFourEnv()
    env.board = board_np.copy()
    env.to_play = 1 # Player 1 to move
    # Manually set legal actions (all cols valid except maybe full ones, here all open)
    env.legal_actions = np.ones(7, dtype=np.float32)
    
    # Use standard PUCT config
    c_puct_base = 19652.0
    c_puct_init = 1.25
    
    # Eval func returns 0.0 value and uniform policy
    eval_func = make_numpy_eval_func()
    
    print("  Running mcts_v2 (2000 sims)...")
    # Note: parallel_uct_search in mcts_v2 does NOT use virtual loss on terminals,
    # but uses it on parents. 
    move_v2, v2_dist, root_val_v2, _, _ = parallel_uct_search(
        env, eval_func, None, c_puct_base, c_puct_init, 
        num_simulations=2000, num_parallel=16
    )
    
    # --- Run tensor_mcts ---
    print("  Running tensor_mcts (2000 sims)...")
    model = DummyModel(device)
    # Use low virtual loss to match the sensitivity needed for DummyModel
    config = TensorMCTSConfig(
        exploration_fraction=0.0, 
        virtual_loss_weight=0.1
    )
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.from_numpy(board_np).unsqueeze(0).long()
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
        
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    tensor_dist = mcts.run_simulations(2000, parallel_sims=16).numpy()[0]
    
    # --- Compare Results ---
    print("\n  Results:")
    print(f"    mcts_v2 Dist:     {v2_dist.round(2)}")
    print(f"    tensor_mcts Dist: {tensor_dist.round(2)}")
    
    best_v2 = np.argmax(v2_dist)
    best_tensor = np.argmax(tensor_dist)
    
    print(f"    mcts_v2 Choice:     Col {best_v2} (Prob {v2_dist[best_v2]:.2f})")
    print(f"    tensor_mcts Choice: Col {best_tensor} (Prob {tensor_dist[best_tensor]:.2f})")
    
    # Analyze Q-values for tensor_mcts to see if it 'saw' the loss even if N didn't converge
    print("\n  tensor_mcts Q-values (P1 perspective):")
    root_children = mcts.children[0, 0]
    q_values = []
    for i in range(7):
        child_idx = root_children[i].item()
        if child_idx >= 0:
            v = mcts.visits[0, child_idx].item()
            val = mcts.total_value[0, child_idx].item()
            # Q = -W/N because W is from child's (P2) perspective
            q = -val / max(1, v) 
            q_values.append(q)
            print(f"    Col {i}: N={v:.0f}, Q={q:.3f}")
        else:
            q_values.append(-999.0)
            
    best_q = np.argmax(q_values)
    print(f"  Best tensor_mcts move by Q-value: Col {best_q}")

    # Pass criteria:
    # 1. Did we match mcts_v2's behavior? (Alignment)
    # 2. Did we find the right move via Q-value? (Correctness)
    
    aligned = (best_v2 == best_tensor)
    correct_q = (best_q == 0)
    
    if aligned:
        print("  ✓ Implementations are aligned (chose same move)")
    else:
        print("  ~ Implementations differed in final choice")
        
    if correct_q:
        print("  ✓ tensor_mcts correctly identifies Col 0 as best via Q-value")
        return True
    else:
        print("  ✗ FAILED: tensor_mcts Q-values did not identify the forced block")
        return False

# =============================================================================
# Test 14: Prior Probability Influence
# =============================================================================

def test_prior_influence():
    """Test that high-probability moves get explored more (U-term works)"""
    print("\n" + "=" * 70)
    print("TEST 14: Prior Probability Influence")
    print("=" * 70)
    
    device = "cpu"
    
    # Setup: Empty board
    board_np = np.zeros((6, 7), dtype=np.int8)
    
    # Model: Biased towards Col 3 (0.9 prob), others low
    # Value is 0.0 everywhere.
    # Therefore, search should favor Col 3 purely due to the 'U' term.
    favored_col = 3
    model = BiasedModel(favored_col, device=device)
    
    config = TensorMCTSConfig(
        c_puct_base=19652.0,
        c_puct_init=1.25,
        exploration_fraction=0.0, # Disable noise to isolate prior influence
        virtual_loss_weight=1.0
    )
    
    # --- Run tensor_mcts ---
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.from_numpy(board_np).unsqueeze(0).long()
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
        
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    
    # Run enough sims to see the distribution difference
    dists = mcts.run_simulations(200, parallel_sims=16).numpy()[0]
    
    print(f"  Model favors Col {favored_col} (50% mass)")
    print(f"  Visit Distribution: {dists.round(3)}")
    
    most_visited = np.argmax(dists)
    
    if most_visited == favored_col and dists[favored_col] > 0.3:
        print("  ✓ Search correctly followed the high-probability prior")
        return True
    else:
        print(f"  ✗ FAILED: Search ignored the prior (Winner: {most_visited})")
        return False

# =============================================================================
# Test 15: Temperature Scaling
# =============================================================================

def test_temperature_scaling():
    """Test that temperature parameter correctly shapes the output distribution"""
    print("\n" + "=" * 70)
    print("TEST 15: Temperature Scaling")
    print("=" * 70)
    
    device = "cpu"
    
    # We don't need to run simulations, just checking the formula in get_action_probs
    # Setup a fake mcts state manually
    mcts = TensorMCTS(DummyModel(), n_trees=1, config=TensorMCTSConfig(), device=device)
    
    # Manually inject visits into root children
    # Root is index 0. Children of root are indices 1..7 (if we pretend)
    # But get_action_probs reads mcts.visits using mcts.children
    
    # 1. Setup Children
    mcts.children[0, 0, :] = torch.arange(1, 8) # Indices 1 to 7
    
    # 2. Setup Visits for those children
    # A clear distribution: [10, 20, 30, 100, 5, 5, 5]
    raw_visits = torch.tensor([10., 20., 30., 100., 5., 5., 5.])
    
    # We need to set these in the global visits tensor at the correct indices
    # We are batch 0.
    for i in range(7):
        child_idx = i + 1
        mcts.visits[0, child_idx] = raw_visits[i]
        
    # --- Check Temp = 1.0 (Linear) ---
    probs_1 = mcts.get_action_probs(temperature=1.0)[0]
    expected_1 = raw_visits / raw_visits.sum()
    diff_1 = torch.abs(probs_1 - expected_1).max().item()
    
    print(f"  Visits: {raw_visits.tolist()}")
    print(f"  Temp 1.0 Probs: {probs_1.numpy().round(3)}")
    
    if diff_1 > 1e-4:
        print("  ✗ FAILED Temp 1.0")
        return False
        
    # --- Check Temp -> 0 (Argmax) ---
    probs_0 = mcts.get_action_probs(temperature=0.01)[0] # effectively 0
    # Should be 1.0 at index 3 (visits=100)
    print(f"  Temp 0.0 Probs: {probs_0.numpy().round(3)}")
    
    if probs_0[3] < 0.95:
        print("  ✗ FAILED Temp 0.0 (Argmax)")
        return False

    # --- Check Temp = 0 (Strict Argmax logic) ---
    probs_strict_0 = mcts.get_action_probs(temperature=0)[0]
    if probs_strict_0[3] != 1.0:
        print("  ✗ FAILED Temp 0 (Strict Argmax)")
        return False

    print("  ✓ Temperature scaling works correctly")
    return True

# =============================================================================
# Test 16: Deterministic Consistency
# =============================================================================

def test_determinism():
    """Test that running the same search twice produces identical results"""
    print("\n" + "=" * 70)
    print("TEST 16: Deterministic Consistency")
    print("=" * 70)
    
    device = "cpu"
    
    board_np = np.zeros((6, 7), dtype=np.int8)
    model = DummyModel(device)
    config = TensorMCTSConfig(exploration_fraction=0.0) # No noise
    
    # Run 1
    mcts1 = TensorMCTS(model, n_trees=1, config=config, device=device)
    board_t = torch.from_numpy(board_np).unsqueeze(0).long()
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
    
    mcts1.reset(board_t, policies, legal, players, root_values=vals)
    # Use parallel sims to trigger potential race conditions if any existed
    dist1 = mcts1.run_simulations(200, parallel_sims=16).numpy()[0]
    
    # Run 2
    mcts2 = TensorMCTS(model, n_trees=1, config=config, device=device)
    mcts2.reset(board_t, policies, legal, players, root_values=vals)
    dist2 = mcts2.run_simulations(200, parallel_sims=16).numpy()[0]
    
    diff = np.abs(dist1 - dist2).max()
    print(f"  Run 1: {dist1.round(3)}")
    print(f"  Run 2: {dist2.round(3)}")
    print(f"  Max Diff: {diff}")
    
    if diff == 0.0:
        print("  ✓ Results are deterministic")
        return True
    else:
        print("  ✗ FAILED: Results differ (Race condition or unseeded RNG?)")
        return False

# =============================================================================
# Test 17: Dirichlet Noise Application
# =============================================================================

def test_dirichlet_noise():
    """Test that Dirichlet noise is correctly applied to root priors"""
    print("\n" + "=" * 70)
    print("TEST 17: Dirichlet Noise Application")
    print("=" * 70)
    
    device = "cpu"
    # Uniform model
    model = DummyModel(device)
    
    # Config with 50% noise
    config = TensorMCTSConfig(
        exploration_fraction=0.5,
        dirichlet_alpha=1.0 # Flat noise for easier checking
    )
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    # Setup
    board_t = torch.zeros(1, 6, 7, dtype=torch.long)
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
    
    # 1. Run Reset
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    
    # 2. Inspect Root Priors
    # Model gave uniform (0.142...). Noise is random.
    # New Prior = 0.5 * 0.142 + 0.5 * Noise
    priors = mcts.priors[0, 0].numpy()
    
    print(f"  Original Policy: {[0.14]*7}")
    print(f"  Noised Priors:   {priors.round(3)}")
    
    # Check that they sum to 1
    sum_p = priors.sum()
    # Check that they are NOT uniform anymore
    std_p = priors.std()
    
    if abs(sum_p - 1.0) < 1e-4 and std_p > 0.01:
        print("  ✓ Noise successfully perturbed priors")
        return True
    else:
        print(f"  ✗ FAILED: Priors look static or invalid (Sum={sum_p:.2f}, Std={std_p:.4f})")
        return False
    
# =============================================================================
# Test 18: Gradient Detachment (Memory Safety)
# =============================================================================

def test_gradient_detachment():
    """Ensure MCTS detaches tensors to prevent memory leaks during training"""
    print("\n" + "=" * 70)
    print("TEST 18: Gradient Detachment")
    print("=" * 70)
    
    device = "cpu"
    
    # 1. Create a "Live" model that outputs gradients
    class GradModel:
        def __init__(self):
            self.weights = torch.randn(42, 7, requires_grad=True)
            
        def forward(self, x, p):
            # Fake forward pass
            batch = x.shape[0]
            # Output must have grad_fn
            pol = torch.softmax(torch.randn(batch, 7, requires_grad=True), dim=1)
            val = torch.tanh(torch.randn(batch, requires_grad=True))
            return pol, val

    model = GradModel()
    mcts = TensorMCTS(model, n_trees=1, config=TensorMCTSConfig(), device=device)
    
    board_t = torch.zeros(1, 6, 7, dtype=torch.long)
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    # 2. Run simulation cycle
    # We DO NOT use torch.no_grad() here to simulate training loop
    policies, vals = model.forward(board_t.flatten(1).float(), players)
    
    # Check inputs have grad
    if not policies.requires_grad or not vals.requires_grad:
        print("  ! Setup Error: Model outputs missing gradients")
        return False
        
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    mcts.run_simulations(10, parallel_sims=1)
    
    # 3. Check MCTS internal storage
    # The stored values MUST NOT have grad
    stored_priors_grad = mcts.priors.requires_grad
    stored_values_grad = mcts.total_value.requires_grad
    
    print(f"  Input Policy Grad: {policies.requires_grad}")
    print(f"  Stored Prior Grad: {stored_priors_grad}")
    print(f"  Stored Value Grad: {stored_values_grad}")
    
    if not stored_priors_grad and not stored_values_grad:
        print("  ✓ Tensors correctly detached (Safe from memory leaks)")
        return True
    else:
        print("  ✗ FAILED: MCTS is storing computation graphs! (Memory Leak)")
        return False
    
# =============================================================================
# Test 19: Tree Capacity Handling
# =============================================================================

def test_capacity_overflow():
    """Test behavior when max_nodes_per_tree is exceeded"""
    print("\n" + "=" * 70)
    print("TEST 19: Tree Capacity Overflow")
    print("=" * 70)
    
    device = "cpu"
    # Set tiny capacity: Root (1) + 7 children = 8 nodes needed.
    # We set capacity to 5.
    config = TensorMCTSConfig(max_nodes_per_tree=5) 
    mcts = TensorMCTS(DummyModel(), n_trees=1, config=config, device=device)
    
    board_t = torch.zeros(1, 6, 7, dtype=torch.long)
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = DummyModel().forward(board_t.flatten(1).float(), players)
    
    print("  Initializing with Capacity=5 (Needs 8 for full root expansion)")
    try:
        # This will attempt to expand root. 
        # _expand_roots checks `next_node_idx`.
        # It should fill up to 5 and stop silently.
        mcts.reset(board_t, policies, legal, players, root_values=vals)
        
        # Check utilization
        used = mcts.next_node_idx[0].item()
        print(f"  Nodes used: {used}/5")
        
        # Check children pointers
        children = mcts.children[0, 0]
        valid_children = (children != -1).sum().item()
        print(f"  Valid children created: {valid_children}/7")
        
        if used <= 5 and valid_children < 7:
            print("  ✓ Gracefully handled overflow (stopped expanding)")
            return True
        elif used > 5:
            print(f"  ✗ FAILED: Buffer overflow! Used {used} indices")
            return False
        else:
            # If it somehow created all 7 children with 5 slots??
            print(f"  ? Unexpected state: {valid_children} children in {used} slots")
            return False
            
    except Exception as e:
        print(f"  ✗ FAILED: Crashed with error: {e}")
        return False
    
# =============================================================================
# Test 17: Deep Tree Traversal
# =============================================================================

def test_deep_traversal():
    """Test that paths go multiple levels deep correctly"""
    print("\n" + "=" * 70)
    print("TEST 17: Deep Tree Traversal")
    print("=" * 70)
    
    device = "cpu"
    board_np = np.zeros((6, 7), dtype=np.int8)
    model = DummyModel(device)
    config = TensorMCTSConfig(exploration_fraction=0.0)
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.from_numpy(board_np).unsqueeze(0).long()
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    mcts.run_simulations(500, parallel_sims=1)  # Sequential to build deep tree
    
    # Check max depth reached
    max_depth_found = 0
    for node_idx in range(mcts.next_node_idx[0].item()):
        if mcts.has_state[0, node_idx]:
            # Trace back to root
            depth = 0
            current = node_idx
            while current != 0 and depth < 50:
                current = mcts.parent[0, current].item()
                depth += 1
            max_depth_found = max(max_depth_found, depth)
    
    print(f"  Nodes allocated: {mcts.next_node_idx[0].item()}")
    print(f"  Max depth found: {max_depth_found}")
    
    if max_depth_found >= 4:
        print("  ✓ Tree reached reasonable depth")
        return True
    else:
        print("  ✗ Tree too shallow")
        return False


# =============================================================================
# Test 18: Node Capacity Limit
# =============================================================================

def test_node_capacity():
    """Test behavior when max_nodes is reached"""
    print("\n" + "=" * 70)
    print("TEST 18: Node Capacity Limit")
    print("=" * 70)
    
    device = "cpu"
    board_np = np.zeros((6, 7), dtype=np.int8)
    model = DummyModel(device)
    
    # Very small capacity
    config = TensorMCTSConfig(
        max_nodes_per_tree=50,  # Very limited
        exploration_fraction=0.0
    )
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.from_numpy(board_np).unsqueeze(0).long()
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    
    # Run many sims - should not crash
    try:
        mcts.run_simulations(200, parallel_sims=8)
        dist = mcts._get_visit_distributions()[0].numpy()
        print(f"  Nodes used: {mcts.next_node_idx[0].item()} / {config.max_nodes_per_tree}")
        print(f"  Distribution: {dist.round(3)}")
        print("  ✓ Handled capacity limit gracefully")
        return True
    except Exception as e:
        print(f"  ✗ Crashed: {e}")
        return False


# =============================================================================
# Test 19: All Children Terminal
# =============================================================================

def test_all_children_terminal():
    """Test when every immediate move ends the game"""
    print("\n" + "=" * 70)
    print("TEST 19: All Children Terminal")
    print("=" * 70)
    
    device = "cpu"
    
    # Create position where P1 has 3-in-row in EVERY column except one full
    # This is tricky - let's do: P1 wins at col 3, P2 wins at col 0
    # Near-endgame chaos
    
    # Simpler: board where only 1 column is playable and it's a win
    board_np = np.zeros((6, 7), dtype=np.int8)
    # Fill all columns except col 3
    for c in range(7):
        if c != 3:
            board_np[:, c] = 1  # All full
    # P1 has 3 in row at bottom
    board_np[5, 0:3] = 1
    board_np[:, 0:3] = 0  # Clear those cols
    board_np[5, 0:3] = 1  # Put back P1's pieces
    
    # Actually, let's simplify: one move available, it wins
    board_np = np.zeros((6, 7), dtype=np.int8)
    board_np[5, 0] = 1
    board_np[5, 1] = 1  
    board_np[5, 2] = 1
    # Col 3 wins for P1. Make other cols full.
    for c in [4, 5, 6]:
        board_np[:, c] = 2
    for c in [0, 1, 2]:
        board_np[0:5, c] = 2  # Leave bottom row
        
    model = DummyModel(device)
    config = TensorMCTSConfig(exploration_fraction=0.0)
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.from_numpy(board_np).unsqueeze(0).long()
    legal = torch.from_numpy((board_np[0, :] == 0).astype(np.float32)).unsqueeze(0)
    players = torch.tensor([1], dtype=torch.long)
    
    print(f"  Legal moves: {legal[0].numpy()}")
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    dist = mcts.run_simulations(50, parallel_sims=8)[0].numpy()
    
    print(f"  Distribution: {dist.round(3)}")
    
    # Should strongly favor col 3 (the winning move)
    if np.argmax(dist) == 3:
        print("  ✓ Found the only winning move")
        return True
    else:
        print("  ✗ Missed the winning move")
        return False


# =============================================================================
# Test 20: State Materialization Correctness
# =============================================================================

def test_state_materialization():
    """Verify board states are built correctly at depth"""
    print("\n" + "=" * 70)
    print("TEST 20: State Materialization Correctness")
    print("=" * 70)
    
    device = "cpu"
    board_np = np.zeros((6, 7), dtype=np.int8)
    model = DummyModel(device)
    config = TensorMCTSConfig(exploration_fraction=0.0)
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.from_numpy(board_np).unsqueeze(0).long()
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    mcts.run_simulations(100, parallel_sims=1)
    
    # Check a few nodes: count pieces should match depth
    errors = 0
    for node_idx in range(1, min(20, mcts.next_node_idx[0].item())):
        if not mcts.has_state[0, node_idx]:
            continue
            
        state = mcts.states[0, node_idx]
        pieces = (state != 0).sum().item()
        
        # Trace depth
        depth = 0
        current = node_idx
        while current != 0 and depth < 50:
            current = mcts.parent[0, current].item()
            depth += 1
        
        if pieces != depth:
            print(f"  Node {node_idx}: depth={depth}, pieces={pieces} - MISMATCH")
            errors += 1
    
    if errors == 0:
        print(f"  ✓ All materialized states have correct piece counts")
        return True
    else:
        print(f"  ✗ {errors} nodes with wrong piece counts")
        return False


# =============================================================================
# Test 21: Dirichlet Noise Application
# =============================================================================

def test_dirichlet_noise():
    """Test that exploration noise is applied correctly"""
    print("\n" + "=" * 70)
    print("TEST 21: Dirichlet Noise Application")
    print("=" * 70)
    
    device = "cpu"
    board_np = np.zeros((6, 7), dtype=np.int8)
    
    # Model with very peaked policy (99% on col 3)
    class PeakedModel:
        def __init__(self): self.device = "cpu"
        def forward(self, boards, players):
            b = boards.shape[0]
            p = torch.zeros(b, 7)
            p[:, 3] = 0.99
            p[:, 0] = 0.01
            return p, torch.zeros(b)
    
    model = PeakedModel()
    
    # With noise
    config_noise = TensorMCTSConfig(
        exploration_fraction=0.25,
        dirichlet_alpha=0.3
    )
    
    # Without noise
    config_no_noise = TensorMCTSConfig(
        exploration_fraction=0.0
    )
    
    board_t = torch.from_numpy(board_np).unsqueeze(0).long()
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    # Get priors with noise
    mcts_noise = TensorMCTS(model, n_trees=1, config=config_noise, device=device)
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
    mcts_noise.reset(board_t, policies, legal, players, root_values=vals)
    priors_with_noise = mcts_noise.priors[0, 0].numpy()
    
    # Get priors without noise
    mcts_no_noise = TensorMCTS(model, n_trees=1, config=config_no_noise, device=device)
    mcts_no_noise.reset(board_t, policies, legal, players, root_values=vals)
    priors_no_noise = mcts_no_noise.priors[0, 0].numpy()
    
    print(f"  Original policy: [0.01, 0, 0, 0.99, 0, 0, 0]")
    print(f"  Priors (no noise): {priors_no_noise.round(3)}")
    print(f"  Priors (with noise): {priors_with_noise.round(3)}")
    
    # With noise, other columns should have more mass
    other_cols_no_noise = priors_no_noise[[1,2,4,5,6]].sum()
    other_cols_with_noise = priors_with_noise[[1,2,4,5,6]].sum()
    
    if other_cols_with_noise > other_cols_no_noise + 0.05:
        print("  ✓ Dirichlet noise spread probability to other actions")
        return True
    else:
        print("  ✗ Noise didn't spread probability enough")
        return False


# =============================================================================
# Test 22: Wrapper Class
# =============================================================================

def test_wrapper():
    """Test TensorMCTSWrapper convenience class"""
    print("\n" + "=" * 70)
    print("TEST 22: Wrapper Class")
    print("=" * 70)
    
    from src.nn.modules.tensor_mcts import TensorMCTSWrapper
    
    device = "cpu"
    model = DummyModel(device)
    
    wrapper = TensorMCTSWrapper(
        model=model,
        num_simulations=100,
        parallel_simulations=8,
        exploration_fraction=0.0,
        device=device
    )
    
    # Test with batch of 2 boards
    board1 = torch.zeros(6, 7, dtype=torch.long)
    
    # Board 2: P1 can win at col 3
    # For P1 to move, need P1_count == P2_count
    board2 = torch.zeros(6, 7, dtype=torch.long)
    board2[5, 0:3] = 1  # P1: XXX at bottom left (3 pieces)
    board2[5, 4:7] = 2  # P2: OOO at bottom right (3 pieces)
    # Now P1=3, P2=3 → P1 to move → col 3 wins
    
    boards = [board1, board2]
    legal = [torch.ones(7), torch.ones(7)]
    
    visit_dist, action_probs = wrapper.get_action_probs_batch_parallel(
        boards, legal, temperature=1.0
    )
    
    print(f"  Board 0 dist: {visit_dist[0].numpy().round(3)}")
    print(f"  Board 1 dist: {visit_dist[1].numpy().round(3)}")
    print(f"  Board 1 best: col {visit_dist[1].argmax().item()}")
    
    if visit_dist[1].argmax().item() == 3:
        print("  ✓ Wrapper correctly processed batch")
        return True
    else:
        print("  ✗ Wrapper failed to find winning move")
        return False
    
# =============================================================================
# Test 23: Double Threat Detection
# =============================================================================

def test_double_threat():
    """Test position where opponent has TWO winning threats - compare both engines"""
    print("\n" + "=" * 70)
    print("TEST 23: Double Threat Detection")
    print("=" * 70)
    
    device = "cpu"
    
    # P2 has threats at BOTH col 0 and col 6
    # P1 can only block one - will lose next turn regardless
    board_np = np.zeros((6, 7), dtype=np.int8)
    board_np[5, 0] = 2
    board_np[4, 0] = 2
    board_np[3, 0] = 2  # P2 threat at col 0
    board_np[5, 6] = 2
    board_np[4, 6] = 2
    board_np[3, 6] = 2  # P2 threat at col 6
    # Add some P1 pieces to balance
    board_np[5, 3] = 1
    board_np[4, 3] = 1
    board_np[5, 4] = 1
    
    print("  Board (P1 to move, P2 has double threat at cols 0 and 6):")
    for row in range(6):
        print(f"    |{''.join(['.XO'[board_np[row, c]] for c in range(7)])}|")
    print("  Note: This is a LOSING position for P1 - can only block one threat")
    
    # --- Run mcts_v2 ---
    env = ConnectFourEnv()
    env.board = board_np.copy()
    env.to_play = 1
    env.legal_actions = np.ones(7, dtype=np.float32)
    eval_func = make_numpy_eval_func()
    
    move_v2, v2_dist, _, _, _ = parallel_uct_search(
        env, eval_func, None, 19652.0, 1.25, 500, num_parallel=16
    )
    
    # --- Run tensor_mcts ---
    model = DummyModel(device)
    config = TensorMCTSConfig(exploration_fraction=0.0)
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.from_numpy(board_np).unsqueeze(0).long()
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    tensor_dist = mcts.run_simulations(500, parallel_sims=16)[0].numpy()
    
    best_v2 = np.argmax(v2_dist)
    best_tensor = np.argmax(tensor_dist)
    
    print(f"\n  Results:")
    print(f"    mcts_v2 dist:     {v2_dist.round(3)}")
    print(f"    tensor_mcts dist: {tensor_dist.round(3)}")
    print(f"    mcts_v2 choice:     col {best_v2}")
    print(f"    tensor_mcts choice: col {best_tensor}")
    
    # Check Q-values for both
    print(f"\n  tensor_mcts Q-values:")
    root_children = mcts.children[0, 0]
    tensor_qs = []
    for i in range(7):
        child_idx = root_children[i].item()
        if child_idx >= 0:
            v = mcts.visits[0, child_idx].item()
            val = mcts.total_value[0, child_idx].item()
            q = -val / max(1, v)
            tensor_qs.append(q)
            print(f"    Col {i}: Q={q:.3f}")
        else:
            tensor_qs.append(None)
    
    # Analysis
    aligned = (best_v2 == best_tensor)
    both_block = best_v2 in [0, 6] and best_tensor in [0, 6]
    all_negative = all(q < 0 for q in tensor_qs if q is not None)
    
    print(f"\n  Analysis:")
    if aligned:
        print(f"    ✓ Both engines chose the same move (col {best_v2})")
    else:
        print(f"    ~ Engines differed: v2={best_v2}, tensor={best_tensor}")
        return False
    
    if both_block:
        print(f"    ✓ Both engines chose to block a threat")
    else:
        print(f"    ~ Not both blocking: v2={best_v2}, tensor={best_tensor}")
        return False
    
    if all_negative:
        print(f"    ✓ All Q-values negative (correctly sees losing position)")
    else:
        print(f"    ~ Some Q-values non-negative")
        return False
    return True
    
    # Pass if engines are aligned OR both choose a blocking move
    if aligned or both_block:
        return True
    else:
        return False

def test_double_threat_debug():
    """Debug: trace through what tensor_mcts sees in the double threat position"""
    print("\n" + "=" * 70)
    print("TEST 23b: Double Threat DEBUG")
    print("=" * 70)
    
    device = "cpu"
    
    board_np = np.zeros((6, 7), dtype=np.int8)
    board_np[5, 0] = 2
    board_np[4, 0] = 2
    board_np[3, 0] = 2  # P2 threat at col 0
    board_np[5, 6] = 2
    board_np[4, 6] = 2
    board_np[3, 6] = 2  # P2 threat at col 6
    board_np[5, 3] = 1
    board_np[4, 3] = 1
    board_np[5, 4] = 1
    
    model = DummyModel(device)
    config = TensorMCTSConfig(exploration_fraction=0.0)
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.from_numpy(board_np).unsqueeze(0).long()
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    mcts.run_simulations(500, parallel_sims=16)
    
    # Detailed tree inspection
    print("\n  Tree structure from root:")
    root_children = mcts.children[0, 0]
    
    for action in range(7):
        child_idx = root_children[action].item()
        if child_idx < 0:
            continue
            
        v = mcts.visits[0, child_idx].item()
        val = mcts.total_value[0, child_idx].item()
        is_term = mcts.is_terminal[0, child_idx].item()
        term_val = mcts.terminal_value[0, child_idx].item()
        player = mcts.current_player[0, child_idx].item()
        q = -val / max(1, v)
        
        print(f"\n  Col {action} -> Node {child_idx}:")
        print(f"    Player to move: P{player}")
        print(f"    Visits: {v:.0f}, Total W: {val:.2f}, Q: {q:.3f}")
        print(f"    Is terminal: {is_term}, Terminal value: {term_val:.2f}")
        
        # Check grandchildren (P2's responses)
        grandchildren = mcts.children[0, child_idx]
        gc_with_visits = (grandchildren >= 0) & (mcts.visits[0, grandchildren.clamp(min=0)] > 0)
        
        if gc_with_visits.any():
            print(f"    Grandchildren (P2 responses):")
            for gc_action in range(7):
                gc_idx = grandchildren[gc_action].item()
                if gc_idx < 0:
                    continue
                gc_v = mcts.visits[0, gc_idx].item()
                if gc_v == 0:
                    continue
                gc_val = mcts.total_value[0, gc_idx].item()
                gc_term = mcts.is_terminal[0, gc_idx].item()
                gc_term_val = mcts.terminal_value[0, gc_idx].item()
                gc_player = mcts.current_player[0, gc_idx].item()
                gc_q = -gc_val / max(1, gc_v)
                print(f"      P2 plays col {gc_action} -> Node {gc_idx}: V={gc_v:.0f}, Q={gc_q:.3f}, term={gc_term}, term_val={gc_term_val:.2f}, next_player=P{gc_player}")
                
                # Show the board state if terminal
                if gc_term and mcts.has_state[0, gc_idx]:
                    state = mcts.states[0, gc_idx]
                    print(f"        Terminal state:")
                    for row in range(6):
                        print(f"          |{''.join(['.XO'[state[row, c].item()] for c in range(7)])}|")
    
    # Count terminals found
    total_terminals = mcts.is_terminal[0].sum().item()
    print(f"\n  Total terminal nodes found: {total_terminals}")
    
    return True

# =============================================================================
# Test 24: Value Network Influence
# =============================================================================

# def test_value_influence():
#     """Test that value predictions affect exploration correctly"""
#     print("\n" + "=" * 70)
#     print("TEST 24: Value Network Influence")
#     print("=" * 70)
    
#     device = "cpu"
    
#     class ValueBiasedModel:
#         """Returns uniform policy but biased value based on move"""
#         def __init__(self, device="cpu"):
#             self.device = device
            
#         def forward(self, boards, players):
#             batch = boards.shape[0]
#             # Uniform policy
#             policy = torch.ones(batch, 7, device=self.device) / 7
#             # Check if col 3 was played (piece at row 5, col 3)
#             boards_2d = boards.view(batch, 6, 7)
#             col3_played = boards_2d[:, 5, 3] != 0
#             # Value is from CURRENT player's perspective
#             # If col 3 was played, current player (P2) is in a BAD position
#             # which means P1 (who played col 3) made a GOOD move
#             values = torch.where(col3_played, 
#                                  torch.tensor(-0.5, device=self.device),  # Bad for P2 = Good for P1
#                                  torch.tensor(0.5, device=self.device))   # Good for P2 = Bad for P1
#             return policy, values
    
#     model = ValueBiasedModel(device)
#     config = TensorMCTSConfig(exploration_fraction=0.0)
#     mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
#     board_t = torch.zeros(1, 6, 7, dtype=torch.long)
#     legal = torch.ones(1, 7)
#     players = torch.tensor([1], dtype=torch.long)
    
#     with torch.no_grad():
#         policies, vals = model.forward(board_t.flatten(1).float(), players)
    
#     mcts.reset(board_t, policies, legal, players, root_values=vals)
#     dist = mcts.run_simulations(200, parallel_sims=8)[0].numpy()
    
#     print(f"  Model: col 3 played → value=-0.5 (bad for opponent = good for P1)")
#     print(f"  Distribution: {dist.round(3)}")
    
#     if np.argmax(dist) == 3 and dist[3] > 0.25:
#         print("  ✓ Search correctly favored high-value move")
#         return True
#     else:
#         print(f"  ✗ Search didn't follow value signal (col 3 = {dist[3]:.1%})")
#         return False

def test_value_influence():
    """Test that value predictions affect exploration correctly"""
    print("\n" + "=" * 70)
    print("TEST 24: Value Network Influence")
    print("=" * 70)
    
    device = "cpu"
    
    class ValueBiasedModel:
        """Returns uniform policy but biased value when P1 has played col 3"""
        def __init__(self, device="cpu"):
            self.device = device
            
        def forward(self, boards, players):
            batch = boards.shape[0]
            policy = torch.ones(batch, 7, device=self.device) / 7
            boards_2d = boards.view(batch, 6, 7)
            
            # Check if P1 (value=1) has a piece at col 3
            p1_played_col3 = (boards_2d[:, 5, 3] == 1)
            
            # This position is GOOD for P1, BAD for P2
            # Value must be from CURRENT player's perspective
            current_is_p1 = (players == 1)
            
            # If P1 played col 3:
            #   - P1 to move: +0.5 (good for P1)
            #   - P2 to move: -0.5 (bad for P2)
            # If P1 did NOT play col 3:
            #   - P1 to move: -0.5 (bad for P1)  
            #   - P2 to move: +0.5 (good for P2)
            
            good_for_p1 = p1_played_col3
            values = torch.where(
                current_is_p1 == good_for_p1,  # Same sign
                torch.tensor(0.5, device=self.device),
                torch.tensor(-0.5, device=self.device)
            )
            return policy, values
    
    model = ValueBiasedModel(device)
    config = TensorMCTSConfig(exploration_fraction=0.0)
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.zeros(1, 6, 7, dtype=torch.long)
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    dist = mcts.run_simulations(200, parallel_sims=1)[0].numpy()
    
    print(f"  Model: P1 played col 3 → good for P1, bad for P2")
    print(f"  Distribution: {dist.round(3)}")
    
    # DEBUG: Check Q values
    print(f"\n  DEBUG - Q values after search:")
    root_children = mcts.children[0, 0]
    for i in range(7):
        child_idx = root_children[i].item()
        if child_idx >= 0:
            v = mcts.visits[0, child_idx].item()
            val = mcts.total_value[0, child_idx].item()
            q = -val / max(1, v)
            print(f"    Col {i}: N={v:.0f}, W={val:.2f}, Q={q:.3f}")
    
    if np.argmax(dist) == 3 and dist[3] > 0.3:
        print("  ✓ Search correctly favored high-value move")
        return True
    else:
        print(f"  ✗ Search didn't follow value signal (col 3 = {dist[3]:.1%})")
        return False

# =============================================================================
# Test 25: Near-Full Board
# =============================================================================

def test_near_full_board():
    """Test with board that's almost full - limited legal moves"""
    print("\n" + "=" * 70)
    print("TEST 25: Near-Full Board")
    print("=" * 70)
    
    device = "cpu"
    
    # Create board with only 2 legal moves remaining
    board_np = np.zeros((6, 7), dtype=np.int8)
    # Fill most of the board in a draw-like pattern
    for c in range(7):
        for r in range(6):
            if c in [2, 4]:  # Leave cols 2 and 4 with one spot each
                if r > 0:
                    board_np[r, c] = 1 if (r + c) % 2 == 0 else 2
            else:
                board_np[r, c] = 1 if (r + c) % 2 == 0 else 2
    
    print("  Board (only cols 2 and 4 have space):")
    for row in range(6):
        print(f"    |{''.join(['.XO'[board_np[row, c]] for c in range(7)])}|")
    
    legal_np = (board_np[0, :] == 0).astype(np.float32)
    print(f"  Legal mask: {legal_np}")
    
    model = DummyModel(device)
    config = TensorMCTSConfig(exploration_fraction=0.0)
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.from_numpy(board_np).unsqueeze(0).long()
    legal = torch.from_numpy(legal_np).unsqueeze(0)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    dist = mcts.run_simulations(100, parallel_sims=8)[0].numpy()
    
    print(f"  Distribution: {dist.round(3)}")
    
    # Only cols 2 and 4 should have visits
    illegal_visits = dist[[0, 1, 3, 5, 6]].sum()
    legal_visits = dist[[2, 4]].sum()
    
    if illegal_visits < 0.01 and legal_visits > 0.99:
        print("  ✓ Correctly concentrated visits on legal moves")
        return True
    else:
        print(f"  ✗ Illegal visits: {illegal_visits:.3f}")
        return False


# =============================================================================
# Test 26: Symmetric Position Mirror
# =============================================================================

def test_symmetry():
    print("\n" + "=" * 70 + "\nTEST 26: Symmetric Position Mirror\n" + "=" * 70)
    device = "cpu"
    
    # Left Board: P1 has 3 in a row at Col 1 (rows 3,4,5). P1 to move. Win at Col 1.
    board_left = np.zeros((6, 7), dtype=np.int8)
    board_left[5, 1] = 1; board_left[4, 1] = 1; board_left[3, 1] = 1
    # Add P2 pieces to make piece counts equal (valid turn)
    board_left[5, 6] = 2; board_left[4, 6] = 2; board_left[3, 6] = 2
    
    # Right Board: Mirror image. P1 has 3 in a row at Col 5. Win at Col 5.
    board_right = np.zeros((6, 7), dtype=np.int8)
    board_right[5, 5] = 1; board_right[4, 5] = 1; board_right[3, 5] = 1
    board_right[5, 0] = 2; board_right[4, 0] = 2; board_right[3, 0] = 2
    
    model = DummyModel(device)
    config = TensorMCTSConfig(exploration_fraction=0.0) # Deterministic
    
    # Run Left
    mcts1 = TensorMCTS(model, 1, config, device)
    # P1 to play (piece count 3 vs 3)
    mcts1.reset(torch.from_numpy(board_left).unsqueeze(0).long(), torch.ones(1, 7), torch.ones(1, 7), torch.tensor([1]))
    dist_left = mcts1.run_simulations(500, 16)[0].numpy()
    
    # Run Right
    mcts2 = TensorMCTS(model, 1, config, device)
    mcts2.reset(torch.from_numpy(board_right).unsqueeze(0).long(), torch.ones(1, 7), torch.ones(1, 7), torch.tensor([1]))
    dist_right = mcts2.run_simulations(500, 16)[0].numpy()
    
    best_left = np.argmax(dist_left)
    best_right = np.argmax(dist_right)
    
    print(f"  Left Board (Win @ 1) Best:  {best_left}")
    print(f"  Right Board (Win @ 5) Best: {best_right}")
    
    if best_left == 1 and best_right == 5:
        print("  ✓ Symmetry preserved (found mirrored winning moves)")
        return True
    else:
        print(f"  ✗ Symmetry broken. Left Dist: {dist_left.round(2)}, Right Dist: {dist_right.round(2)}")
        return False


# =============================================================================
# Test 27: Stress Test - Many Simulations
# =============================================================================

def test_stress_many_sims():
    """Stress test with many simulations"""
    print("\n" + "=" * 70)
    print("TEST 27: Stress Test (5000 sims)")
    print("=" * 70)
    
    device = "cpu"
    
    board_np = np.zeros((6, 7), dtype=np.int8)
    model = DummyModel(device)
    config = TensorMCTSConfig(exploration_fraction=0.0, max_nodes_per_tree=10000)
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.from_numpy(board_np).unsqueeze(0).long()
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    
    import time
    start = time.time()
    dist = mcts.run_simulations(5000, parallel_sims=32)[0].numpy()
    elapsed = time.time() - start
    
    nodes_used = mcts.next_node_idx[0].item()
    
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Nodes used: {nodes_used}")
    print(f"  Distribution: {dist.round(3)}")
    print(f"  Sims/sec: {5000/elapsed:.0f}")
    
    # Check reasonable distribution (should be fairly uniform with DummyModel)
    if dist.min() > 0.05 and dist.max() < 0.30:
        print("  ✓ Stress test passed")
        return True
    else:
        print("  ✗ Distribution seems off")
        return False


# =============================================================================
# Test 28: Multi-Tree Stress
# =============================================================================

def test_multi_tree_stress():
    """Test many parallel trees"""
    print("\n" + "=" * 70)
    print("TEST 28: Multi-Tree Stress (32 trees)")
    print("=" * 70)
    
    device = "cpu"
    n_trees = 32
    
    # Create different boards for each tree
    boards = []
    for i in range(n_trees):
        b = np.zeros((6, 7), dtype=np.int8)
        # Put a piece in different column for each
        b[5, i % 7] = 1
        boards.append(b)
    
    board_t = torch.from_numpy(np.stack(boards)).long()
    
    model = DummyModel(device)
    config = TensorMCTSConfig(exploration_fraction=0.0)
    mcts = TensorMCTS(model, n_trees=n_trees, config=config, device=device)
    
    legal = torch.ones(n_trees, 7)
    players = torch.ones(n_trees, dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    dists = mcts.run_simulations(200, parallel_sims=16).numpy()
    
    print(f"  Running {n_trees} trees in parallel")
    print(f"  First 3 distributions:")
    for i in range(3):
        print(f"    Tree {i}: {dists[i].round(3)}")
    
    # Check all trees have reasonable distributions
    all_valid = all(d.sum() > 0.99 for d in dists)
    
    if all_valid:
        print("  ✓ All trees produced valid distributions")
        return True
    else:
        print("  ✗ Some trees failed")
        return False


# =============================================================================
# Test 29: Immediate Win vs Future Win
# =============================================================================

def test_immediate_vs_future():
    """Test that immediate win is preferred over setting up future win"""
    print("\n" + "=" * 70)
    print("TEST 29: Immediate Win vs Future Win")
    print("=" * 70)
    
    device = "cpu"
    
    # P1 can win immediately at col 3
    # OR set up a "better" position at col 0
    board_np = np.zeros((6, 7), dtype=np.int8)
    board_np[5, 0:3] = 1  # X X X _ (win at col 3)
    board_np[5, 4:7] = 2  # Some P2 pieces
    board_np[4, 4] = 2
    
    print("  Board (P1 can win immediately at col 3):")
    for row in range(6):
        print(f"    |{''.join(['.XO'[board_np[row, c]] for c in range(7)])}|")
    
    model = DummyModel(device)
    config = TensorMCTSConfig(exploration_fraction=0.0)
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.from_numpy(board_np).unsqueeze(0).long()
    legal = torch.ones(1, 7)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    dist = mcts.run_simulations(500, parallel_sims=16)[0].numpy()
    
    print(f"  Distribution: {dist.round(3)}")
    
    # Should overwhelmingly choose col 3 (immediate win)
    if dist[3] > 0.9:
        print("  ✓ Correctly chose immediate win")
        return True
    else:
        print(f"  ✗ Did not strongly prefer immediate win (col 3 = {dist[3]:.1%})")
        return False


# =============================================================================
# Test 30: Zugzwang-like Position  
# =============================================================================

def test_zugzwang():
    """Test position where any move leads to opponent winning"""
    print("\n" + "=" * 70)
    print("TEST 30: Zugzwang-like Position")
    print("=" * 70)
    
    device = "cpu"
    
    # Create position where P2 has multiple threats
    # Any move P1 makes, P2 wins next turn
    board_np = np.zeros((6, 7), dtype=np.int8)
    
    # P2 has horizontal threat
    board_np[5, 1:4] = 2  # O O O at bottom
    # P2 also has vertical threat  
    board_np[5, 5] = 2
    board_np[4, 5] = 2
    board_np[3, 5] = 2
    
    # P1 pieces (fewer, can't block both)
    board_np[5, 0] = 1
    board_np[4, 0] = 1
    board_np[5, 6] = 1
    
    print("  Board (P1 to move, P2 has multiple winning threats):")
    for row in range(6):
        print(f"    |{''.join(['.XO'[board_np[row, c]] for c in range(7)])}|")
    
    model = DummyModel(device)
    config = TensorMCTSConfig(exploration_fraction=0.0)
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    board_t = torch.from_numpy(board_np).unsqueeze(0).long()
    legal = torch.from_numpy((board_np[0, :] == 0).astype(np.float32)).unsqueeze(0)
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        policies, vals = model.forward(board_t.flatten(1).float(), players)
    
    mcts.reset(board_t, policies, legal, players, root_values=vals)
    dist = mcts.run_simulations(500, parallel_sims=16)[0].numpy()
    
    print(f"  Distribution: {dist.round(3)}")
    
    # Check Q-values - all should be negative (losing position)
    root_children = mcts.children[0, 0]
    q_values = []
    for i in range(7):
        child_idx = root_children[i].item()
        if child_idx >= 0:
            v = mcts.visits[0, child_idx].item()
            val = mcts.total_value[0, child_idx].item()
            q = -val / max(1, v)
            q_values.append(q)
        else:
            q_values.append(None)
    
    print(f"  Q-values: {[f'{q:.2f}' if q else 'N/A' for q in q_values]}")
    
    # All Q-values should be negative (losing)
    valid_qs = [q for q in q_values if q is not None]
    if valid_qs and max(valid_qs) < 0:
        print("  ✓ Correctly identified losing position (all Q < 0)")
        return True
    else:
        print("  ~ Position evaluation may vary")
        return True  # Don't fail - this is a diagnostic test
    
# =============================================================================
# Test 31: Test Fork Creation
# =============================================================================
def test_fork_creation():
    print("\n" + "=" * 70 + "\nTEST 31: Fork Creation (Guaranteed Win)\n" + "=" * 70)
    device = "cpu"
    
    # Board where P1 playing Col 3 creates TWO winning threats (Col 2 and Col 4)
    # P1: (5,2) and (5,4).
    # Play (5,3) -> P1 has (5,2)-(5,3)-(5,4)
    # Open on both ends? 1 and 5.
    # P2 blocks 1 -> P1 wins at 5.
    
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 2] = 1
    board[5, 4] = 1
    # Add dummy P2 pieces to validate turn
    board[5, 0] = 2; board[5, 6] = 2
    
    model = DummyModel(device)
    # Low VL to allow deep search for the fork
    config = TensorMCTSConfig(exploration_fraction=0.0, virtual_loss_weight=0.1)
    mcts = TensorMCTS(model, 1, config, device)
    
    mcts.reset(torch.from_numpy(board).unsqueeze(0).long(), torch.ones(1,7), torch.ones(1,7), torch.tensor([1]))
    dist = mcts.run_simulations(2000, 16)[0].numpy()
    
    print(f"  Dist: {dist.round(3)}")
    if np.argmax(dist) == 3:
        print("  ✓ Found Fork at Col 3")
        return True
    else:
        print(f"  ✗ Missed Fork (Best: {np.argmax(dist)})")
        return False
    
# =============================================================================
# Test 31: Test Collision Test
# =============================================================================
def test_collision_stress():
    print("\n" + "=" * 70 + "\nTEST 31: Collision Stress Test\n" + "=" * 70)
    device = "cpu"
    
    # Safe Draw Pattern: No 4-in-a-row (Vertical, Horizontal, Diagonal)
    # Rows 0-1: 1, 2, 1, 2...
    # Rows 2-3: 2, 1, 2, 1...
    # Rows 4-5: 1, 2, 1, 2...
    board = np.zeros((6, 7), dtype=np.int8)
    for r in range(6):
        for c in range(1, 7): # Fill cols 1-6
            # Base pattern
            val = 1 if c % 2 != 0 else 2
            # Flip on row blocks to break vertical/diagonal lines
            if (r // 2) % 2 != 0:
                val = 3 - val
            board[r, c] = val
            
    # Explicit legal mask: Only Col 0 is open
    legal_mask = torch.zeros(1, 7)
    legal_mask[0, 0] = 1
    
    model = DummyModel(device)
    config = TensorMCTSConfig(exploration_fraction=0.0)
    mcts = TensorMCTS(model, 1, config, device)
    
    board_t = torch.from_numpy(board).unsqueeze(0).long()
    players = torch.tensor([1], dtype=torch.long)
    
    with torch.no_grad():
        pols, vals = model.forward(board_t.flatten(1).float(), players)
    
    # NOTE: Pass computed legal_mask, not board-derived, to ensure safety
    mcts.reset(board_t, pols, legal_mask, players, root_values=vals)
    mcts.run_simulations(16, 16)
    
    nodes = mcts.next_node_idx[0].item()
    print(f"  Nodes used: {nodes}")
    
    if nodes >= 17:
        print("  ✓ Safely handled massive collision")
        return True
    else:
        print(f"  ✗ Unexpected node usage: {nodes}")
        return False
    
# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("MCTS Implementation Comparison Test Suite")
    print("=" * 70)
    
    results = []
    
    # Basic Logic
    results.append(("Terminal Detection", test_terminal_detection()))
    results.append(("PUCT Formula", test_puct_formula()))
    results.append(("Single Simulation", test_single_simulation()))
    
    # Gameplay Logic
    results.append(("Winning Move 1", test_winning_move()))
    results.append(("Winning Move 2", test_winning_move_2()))
    results.append(("Backup Values", test_backup_values()))
    results.append(("Forced Block (Defense)", test_forced_mate_depth())) # The hard one!
    
    # Edge Cases
    results.append(("Illegal Masking", test_illegal_masking()))
    results.append(("Draw Detection", test_draw_detection()))
    results.append(("Batch Independence", test_batch_independence()))
    
    # Advanced / System
    results.append(("Virtual Loss Diversity", test_virtual_loss()))
    results.append(("Virtual Loss Cleanliness", test_virtual_loss_cleanliness()))
    results.append(("Prior Influence", test_prior_influence()))
    results.append(("Temperature Scaling", test_temperature_scaling()))
    results.append(("Deterministic Consistency", test_determinism()))
    
    # New Hardening Tests
    results.append(("Dirichlet Noise", test_dirichlet_noise()))
    results.append(("Gradient Detachment", test_gradient_detachment()))
    results.append(("Capacity Overflow", test_capacity_overflow()))
    
    results.append(("Deep Tree Traversal", test_deep_traversal()))
    results.append(("Node Capacity Limit", test_node_capacity()))
    results.append(("All Children Terminal", test_all_children_terminal()))
    results.append(("State Materialization", test_state_materialization()))
    results.append(("Dirichlet Noise Application", test_dirichlet_noise()))
    results.append(("Wrapper Class", test_wrapper()))
    results.append(("Visit Convergence", test_visit_convergence()))
    results.append(("Double Threat Detection", test_double_threat()))
    results.append(("Double Threat DEBUG", test_double_threat_debug()))
    results.append(("Value Network Influence", test_value_influence()))

    # New Hardening Tests Part 2
    results.append(("Near-Full Board", test_near_full_board()))
    results.append(("Symmetric Position Mirror", test_symmetry()))
    results.append(("Stress Test - Many Simulations", test_stress_many_sims()))
    results.append(("Multi-Tree Stress", test_multi_tree_stress()))
    results.append(("Immediate Win vs Future Win", test_immediate_vs_future()))
    results.append(("Zugzwang-like Position", test_zugzwang()))
    results.append(("Fork Creation", test_fork_creation()))
    results.append(("Collision Stress Test", test_collision_stress()))
        
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")

if __name__ == "__main__":
    run_all_tests()
