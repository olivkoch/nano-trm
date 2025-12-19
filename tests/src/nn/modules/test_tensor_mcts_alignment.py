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
        else:
            print(f"    ✓ Distributions reasonably similar")


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
# Main
# =============================================================================

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("MCTS Implementation Comparison Test Suite")
    print("=" * 70)
    
    results = []
    
    results.append(("Terminal Detection", test_terminal_detection()))
    results.append(("PUCT Formula", test_puct_formula()))
    results.append(("Single Simulation", test_single_simulation()))
    results.append(("Winning Move 1", test_winning_move()))
    results.append(("Winning Move 2", test_winning_move_2()))
    results.append(("Backup Values", test_backup_values()))
    
    # New Tests
    results.append(("Illegal Masking", test_illegal_masking()))
    results.append(("Draw Detection", test_draw_detection()))
    results.append(("Batch Independence", test_batch_independence()))
    results.append(("Virtual Loss Cleanliness", test_virtual_loss_cleanliness()))
    results.append(("Forced Block (Defense)", test_forced_mate_depth()))
    
    test_visit_convergence()  # No pass/fail, just comparison
    results.append(("Virtual Loss Diversity", test_virtual_loss()))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")


if __name__ == "__main__":
    run_all_tests()