#!/usr/bin/env python
"""
PUCT and Core Function Tests for mcts_v2.py

Tests the reference MCTS implementation without modification.
Focuses on:
1. PUCT formula verification (child_Q, child_U, best_child)
2. Backup arithmetic
3. Node expansion
4. Virtual loss mechanics
"""

import numpy as np
import math
from typing import Tuple

# Import the mcts_v2 module - adjust path as needed
from src.nn.modules.mcts_v2 import (
    Node, DummyNode, 
    best_child, expand, backup,
    add_virtual_loss, revert_virtual_loss,
    add_dirichlet_noise, generate_search_policy
)


def compute_puct_reference(child_W: np.ndarray, child_N: np.ndarray, child_P: np.ndarray,
                           parent_N: float, c_puct_base: float, c_puct_init: float,
                           legal_actions: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Compute PUCT scores using the exact formula from mcts_v2.py
    
    Returns: (ucb_scores, best_action)
    """
    # child_Q calculation (from Node.child_Q)
    safe_child_N = np.where(child_N > 0, child_N, 1)
    child_Q = child_W / safe_child_N
    
    # child_U calculation (from Node.child_U)
    pb_c = math.log((1 + parent_N + c_puct_base) / c_puct_base) + c_puct_init
    child_U = pb_c * child_P * (math.sqrt(parent_N) / (1 + child_N))
    
    # UCB score with sign flip (from best_child)
    ucb_scores = -child_Q + child_U
    
    # Mask illegal actions
    ucb_scores = np.where(legal_actions == 1, ucb_scores, -9999)
    
    return ucb_scores, np.argmax(ucb_scores)


# =============================================================================
# Test: Basic PUCT Formula
# =============================================================================

def test_child_q_calculation():
    """Test that child_Q correctly computes W/N with division safety."""
    print("\n" + "=" * 60)
    print("TEST: child_Q Calculation")
    print("=" * 60)
    
    num_actions = 7
    node = Node(to_play=1, num_actions=num_actions, parent=DummyNode())
    
    # Set up known values
    node.child_W = np.array([1.0, -0.5, 0.0, 2.0, 0.0, -1.0, 0.5], dtype=np.float32)
    node.child_N = np.array([10.0, 5.0, 0.0, 20.0, 0.0, 10.0, 5.0], dtype=np.float32)
    
    Q = node.child_Q()
    
    print(f"\n  child_W: {node.child_W}")
    print(f"  child_N: {node.child_N}")
    print(f"  child_Q: {Q}")
    
    # Expected Q values
    expected = np.array([0.1, -0.1, 0.0, 0.1, 0.0, -0.1, 0.1], dtype=np.float32)
    
    passed = True
    for i in range(num_actions):
        if node.child_N[i] > 0:
            expected_q = node.child_W[i] / node.child_N[i]
        else:
            expected_q = 0.0  # Division by 1 when N=0
        
        if abs(Q[i] - expected_q) > 1e-6:
            print(f"  ✗ Action {i}: got {Q[i]:.4f}, expected {expected_q:.4f}")
            passed = False
    
    if passed:
        print("\n  ✓ PASSED: child_Q computed correctly")
    return passed


def test_child_u_calculation():
    """Test that child_U correctly computes exploration bonus."""
    print("\n" + "=" * 60)
    print("TEST: child_U Calculation")
    print("=" * 60)
    
    num_actions = 7
    node = Node(to_play=1, num_actions=num_actions, parent=DummyNode())
    
    # Set parent visit count
    node.parent.child_N[None] = 100  # Using None as dummy move
    node.move = None
    
    # Set up known values
    node.child_P = np.array([0.3, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1], dtype=np.float32)
    node.child_N = np.array([10.0, 5.0, 0.0, 20.0, 1.0, 10.0, 5.0], dtype=np.float32)
    
    c_puct_base = 19652.0
    c_puct_init = 1.25
    
    U = node.child_U(c_puct_base, c_puct_init)
    
    print(f"\n  Parent N: {node.N}")
    print(f"  child_P: {node.child_P}")
    print(f"  child_N: {node.child_N}")
    print(f"  child_U: {U}")
    
    # Compute expected U manually
    pb_c = math.log((1 + node.N + c_puct_base) / c_puct_base) + c_puct_init
    expected_U = pb_c * node.child_P * (math.sqrt(node.N) / (1 + node.child_N))
    
    print(f"\n  pb_c coefficient: {pb_c:.4f}")
    print(f"  Expected U: {expected_U}")
    
    if np.allclose(U, expected_U, atol=1e-6):
        print("\n  ✓ PASSED: child_U computed correctly")
        return True
    else:
        print(f"\n  ✗ FAILED: U mismatch")
        print(f"    Max diff: {np.max(np.abs(U - expected_U)):.6f}")
        return False


def test_puct_sign_convention():
    """
    Test that PUCT uses -Q + U (negated Q for opponent perspective).
    
    In a two-player zero-sum game, child values are from opponent's perspective,
    so we negate Q when selecting from parent's perspective.
    """
    print("\n" + "=" * 60)
    print("TEST: PUCT Sign Convention (-Q + U)")
    print("=" * 60)
    
    num_actions = 7
    node = Node(to_play=1, num_actions=num_actions, parent=DummyNode())
    node.move = None
    node.parent.child_N[None] = 50
    node.is_expanded = True
    
    # Child A: High W (good for opponent = bad for us) -> -Q should be negative
    # Child B: Low W (bad for opponent = good for us) -> -Q should be positive
    node.child_W = np.array([5.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    node.child_N = np.array([10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    node.child_P = np.array([0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16], dtype=np.float32)  # Equal priors
    
    legal_actions = np.ones(num_actions, dtype=np.float32)
    
    c_puct_base = 19652.0
    c_puct_init = 1.25
    
    # Compute UCB scores manually
    ucb_scores, best_action = compute_puct_reference(
        node.child_W, node.child_N, node.child_P,
        node.N, c_puct_base, c_puct_init, legal_actions
    )
    
    print(f"\n  child_W: {node.child_W[:2]} (action 0: good for opponent, action 1: bad for opponent)")
    print(f"  child_N: {node.child_N[:2]}")
    
    Q = node.child_Q()
    print(f"\n  child_Q: {Q[:2]}")
    print(f"  -child_Q: {-Q[:2]}")
    
    print(f"\n  UCB scores: {ucb_scores[:2]}")
    print(f"  Best action: {best_action}")
    
    # Action 1 should be preferred (opponent's bad position = our good position)
    # -Q for action 0 = -0.5, -Q for action 1 = +0.5
    if ucb_scores[1] > ucb_scores[0]:
        print("\n  ✓ PASSED: Sign convention correct (prefers opponent's bad position)")
        return True
    else:
        print("\n  ✗ FAILED: Sign convention wrong")
        return False


def test_best_child_selection():
    """Test that best_child selects correctly based on PUCT scores."""
    print("\n" + "=" * 60)
    print("TEST: best_child Selection")
    print("=" * 60)
    
    num_actions = 7
    node = Node(to_play=1, num_actions=num_actions, parent=DummyNode())
    node.move = None
    node.parent.child_N[None] = 100
    node.is_expanded = True
    
    # Set up scenario: action 3 should win
    # - Action 3: Low Q (good for us after negation), high prior
    # - Action 0: High Q (bad for us), low prior
    node.child_W = np.array([8.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0], dtype=np.float32)
    node.child_N = np.array([10.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0], dtype=np.float32)
    node.child_P = np.array([0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1], dtype=np.float32)
    
    legal_actions = np.ones(num_actions, dtype=np.float32)
    
    c_puct_base = 19652.0
    c_puct_init = 1.25
    
    # Call best_child
    child = best_child(node, legal_actions, c_puct_base, c_puct_init, child_to_play=2)
    
    # Compute expected
    ucb_scores, expected_action = compute_puct_reference(
        node.child_W, node.child_N, node.child_P,
        node.N, c_puct_base, c_puct_init, legal_actions
    )
    
    print(f"\n  UCB scores: {ucb_scores.round(4)}")
    print(f"  Expected best action: {expected_action}")
    print(f"  Actual selected action: {child.move}")
    
    if child.move == expected_action:
        print("\n  ✓ PASSED: best_child selected correctly")
        return True
    else:
        print("\n  ✗ FAILED: Wrong action selected")
        return False


def test_best_child_respects_legal_mask():
    """Test that best_child never selects illegal actions."""
    print("\n" + "=" * 60)
    print("TEST: best_child Respects Legal Mask")
    print("=" * 60)
    
    num_actions = 7
    node = Node(to_play=1, num_actions=num_actions, parent=DummyNode())
    node.move = None
    node.parent.child_N[None] = 50
    node.is_expanded = True
    
    # Action 0 would be best, but it's illegal
    node.child_W = np.array([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    node.child_N = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    node.child_P = np.array([0.5, 0.08, 0.08, 0.08, 0.08, 0.08, 0.1], dtype=np.float32)
    
    # Action 0 is illegal!
    legal_actions = np.array([0, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    
    c_puct_base = 19652.0
    c_puct_init = 1.25
    
    child = best_child(node, legal_actions, c_puct_base, c_puct_init, child_to_play=2)
    
    print(f"\n  Illegal action: 0")
    print(f"  Selected action: {child.move}")
    
    if child.move != 0 and legal_actions[child.move] == 1:
        print("\n  ✓ PASSED: Illegal action not selected")
        return True
    else:
        print("\n  ✗ FAILED: Illegal action was selected")
        return False


def test_unvisited_children_exploration():
    """
    Test that unvisited children (N=0) get high exploration bonus.
    With equal priors, an unvisited child should be preferred over visited ones.
    """
    print("\n" + "=" * 60)
    print("TEST: Unvisited Children Get High Exploration Bonus")
    print("=" * 60)
    
    num_actions = 7
    node = Node(to_play=1, num_actions=num_actions, parent=DummyNode())
    node.move = None
    node.parent.child_N[None] = 100
    node.is_expanded = True
    
    # Action 3 is unvisited, others have visits
    node.child_W = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    node.child_N = np.array([10.0, 10.0, 10.0, 0.0, 10.0, 10.0, 10.0], dtype=np.float32)
    node.child_P = np.ones(num_actions, dtype=np.float32) / num_actions  # Equal priors
    
    legal_actions = np.ones(num_actions, dtype=np.float32)
    
    c_puct_base = 19652.0
    c_puct_init = 1.25
    
    child = best_child(node, legal_actions, c_puct_base, c_puct_init, child_to_play=2)
    
    # Compute U values to show the difference
    U = node.child_U(c_puct_base, c_puct_init)
    
    print(f"\n  child_N: {node.child_N}")
    print(f"  child_U: {U.round(4)}")
    print(f"  U[3] (unvisited): {U[3]:.4f}")
    print(f"  U[0] (visited 10x): {U[0]:.4f}")
    print(f"  Ratio: {U[3]/U[0]:.2f}x higher for unvisited")
    print(f"\n  Selected action: {child.move}")
    
    if child.move == 3:
        print("\n  ✓ PASSED: Unvisited child selected (exploration bonus working)")
        return True
    else:
        print("\n  ✗ FAILED: Unvisited child not selected")
        return False


def test_high_visit_q_dominates():
    """
    Test that at very high visit counts, Q value dominates over exploration.
    A clearly worse action should not be selected even with exploration bonus.
    """
    print("\n" + "=" * 60)
    print("TEST: Q Dominates at High Visit Counts")
    print("=" * 60)
    
    num_actions = 7
    node = Node(to_play=1, num_actions=num_actions, parent=DummyNode())
    node.move = None
    node.parent.child_N[None] = 10000  # Very high parent visits
    node.is_expanded = True
    
    # Action 0: Very bad Q (positive W = good for opponent = bad for us)
    # Action 3: Very good Q (negative W = bad for opponent = good for us)
    # Both have high visits, so exploration bonus is small
    node.child_W = np.array([900.0, 0.0, 0.0, -900.0, 0.0, 0.0, 0.0], dtype=np.float32)
    node.child_N = np.array([1000.0, 100.0, 100.0, 1000.0, 100.0, 100.0, 100.0], dtype=np.float32)
    node.child_P = np.ones(num_actions, dtype=np.float32) / num_actions
    
    legal_actions = np.ones(num_actions, dtype=np.float32)
    
    c_puct_base = 19652.0
    c_puct_init = 1.25
    
    child = best_child(node, legal_actions, c_puct_base, c_puct_init, child_to_play=2)
    
    Q = node.child_Q()
    U = node.child_U(c_puct_base, c_puct_init)
    
    print(f"\n  Action 0: Q={Q[0]:.3f}, -Q={-Q[0]:.3f}, U={U[0]:.4f}")
    print(f"  Action 3: Q={Q[3]:.3f}, -Q={-Q[3]:.3f}, U={U[3]:.4f}")
    print(f"  UCB[0] = {-Q[0] + U[0]:.4f}")
    print(f"  UCB[3] = {-Q[3] + U[3]:.4f}")
    print(f"\n  Selected action: {child.move}")
    
    if child.move == 3:
        print("\n  ✓ PASSED: Q value correctly dominates at high N")
        return True
    else:
        print("\n  ✗ FAILED: Exploration overrode clear Q advantage")
        return False


# =============================================================================
# Test: Backup Arithmetic
# =============================================================================

def test_backup_single_node():
    """Test backup with a single node (root)."""
    print("\n" + "=" * 60)
    print("TEST: Backup Single Node")
    print("=" * 60)
    
    node = Node(to_play=1, num_actions=7, move=3, parent=DummyNode())
    
    print(f"\n  Before backup:")
    print(f"    N: {node.N}, W: {node.W}")
    
    backup(node, 0.7)
    
    print(f"\n  After backup(0.7):")
    print(f"    N: {node.N}, W: {node.W}")
    
    if node.N == 1 and abs(node.W - 0.7) < 1e-6:
        print("\n  ✓ PASSED: Single node backup correct")
        return True
    else:
        print("\n  ✗ FAILED: Wrong values after backup")
        return False


def test_backup_sign_alternation():
    """
    Test that backup alternates signs through the path.
    
    Value at leaf propagates as:
    - Leaf: +v
    - Parent: -v
    - Grandparent: +v
    """
    print("\n" + "=" * 60)
    print("TEST: Backup Sign Alternation")
    print("=" * 60)
    
    # Create a 3-node chain: root -> child -> grandchild
    root = Node(to_play=1, num_actions=7, move=None, parent=DummyNode())
    root.parent.child_N[None] = 0
    root.parent.child_W[None] = 0
    
    child = Node(to_play=2, num_actions=7, move=3, parent=root)
    root.children[3] = child
    
    grandchild = Node(to_play=1, num_actions=7, move=5, parent=child)
    child.children[5] = grandchild
    
    print("\n  Tree: Root (P1) -> Child (P2) -> Grandchild (P1)")
    print("  Backup value: +0.8 from grandchild's perspective (P1)")
    
    # Backup from grandchild with value +0.8
    backup(grandchild, 0.8)
    
    print(f"\n  After backup:")
    print(f"    Grandchild: N={grandchild.N}, W={grandchild.W:.4f}, Q={grandchild.Q:.4f}")
    print(f"    Child:      N={child.N}, W={child.W:.4f}, Q={child.Q:.4f}")
    print(f"    Root:       N={root.N}, W={root.W:.4f}, Q={root.Q:.4f}")
    
    passed = True
    
    # Grandchild gets +0.8
    if abs(grandchild.W - 0.8) > 1e-6:
        print(f"  ✗ Grandchild W wrong: expected 0.8, got {grandchild.W}")
        passed = False
    
    # Child gets -0.8 (negated)
    if abs(child.W - (-0.8)) > 1e-6:
        print(f"  ✗ Child W wrong: expected -0.8, got {child.W}")
        passed = False
    
    # Root gets +0.8 (negated again)
    if abs(root.W - 0.8) > 1e-6:
        print(f"  ✗ Root W wrong: expected 0.8, got {root.W}")
        passed = False
    
    if passed:
        print("\n  ✓ PASSED: Sign alternation correct")
    return passed


def test_backup_accumulation():
    """Test that multiple backups accumulate correctly."""
    print("\n" + "=" * 60)
    print("TEST: Backup Accumulation")
    print("=" * 60)
    
    root = Node(to_play=1, num_actions=7, move=None, parent=DummyNode())
    root.parent.child_N[None] = 0
    root.parent.child_W[None] = 0
    
    child = Node(to_play=2, num_actions=7, move=3, parent=root)
    root.children[3] = child
    
    # First backup: +0.5
    backup(child, 0.5)
    print(f"\n  After backup(+0.5):")
    print(f"    Child: N={child.N}, W={child.W:.4f}")
    print(f"    Root:  N={root.N}, W={root.W:.4f}")
    
    # Second backup: -0.3
    backup(child, -0.3)
    print(f"\n  After backup(-0.3):")
    print(f"    Child: N={child.N}, W={child.W:.4f}")
    print(f"    Root:  N={root.N}, W={root.W:.4f}")
    
    # Third backup: +0.2
    backup(child, 0.2)
    print(f"\n  After backup(+0.2):")
    print(f"    Child: N={child.N}, W={child.W:.4f}")
    print(f"    Root:  N={root.N}, W={root.W:.4f}")
    
    # Expected: Child W = 0.5 + (-0.3) + 0.2 = 0.4, N = 3
    # Root W = -0.5 + 0.3 + (-0.2) = -0.4, N = 3
    passed = True
    
    if child.N != 3:
        print(f"  ✗ Child N wrong: expected 3, got {child.N}")
        passed = False
    if abs(child.W - 0.4) > 1e-6:
        print(f"  ✗ Child W wrong: expected 0.4, got {child.W}")
        passed = False
    if root.N != 3:
        print(f"  ✗ Root N wrong: expected 3, got {root.N}")
        passed = False
    if abs(root.W - (-0.4)) > 1e-6:
        print(f"  ✗ Root W wrong: expected -0.4, got {root.W}")
        passed = False
    
    if passed:
        print("\n  ✓ PASSED: Backup accumulation correct")
    return passed


# =============================================================================
# Test: Virtual Loss
# =============================================================================

def test_virtual_loss_discourages_selection():
    """
    Test that virtual loss makes a node less attractive for selection.
    
    VL adds +1 to W, which increases Q. Since PUCT uses -Q, this makes
    the node less attractive.
    """
    print("\n" + "=" * 60)
    print("TEST: Virtual Loss Discourages Selection")
    print("=" * 60)
    
    num_actions = 7
    node = Node(to_play=1, num_actions=num_actions, parent=DummyNode())
    node.move = None
    node.parent.child_N[None] = 100
    node.is_expanded = True
    
    # All children equal
    node.child_W = np.zeros(num_actions, dtype=np.float32)
    node.child_N = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    node.child_P = np.ones(num_actions, dtype=np.float32) / num_actions
    
    # Create child 0
    child0 = Node(to_play=2, num_actions=num_actions, move=0, parent=node)
    node.children[0] = child0
    
    legal_actions = np.ones(num_actions, dtype=np.float32)
    c_puct_base = 19652.0
    c_puct_init = 1.25
    
    # Before VL: compute UCB for action 0
    Q_before = node.child_Q()
    ucb_before = -Q_before + node.child_U(c_puct_base, c_puct_init)
    
    print(f"\n  Before virtual loss on action 0:")
    print(f"    W[0]={node.child_W[0]:.1f}, Q[0]={Q_before[0]:.4f}, UCB[0]={ucb_before[0]:.4f}")
    
    # Apply virtual loss to child 0
    add_virtual_loss(child0)
    
    # After VL: compute UCB for action 0
    Q_after = node.child_Q()
    ucb_after = -Q_after + node.child_U(c_puct_base, c_puct_init)
    
    print(f"\n  After virtual loss on action 0:")
    print(f"    W[0]={node.child_W[0]:.1f}, Q[0]={Q_after[0]:.4f}, UCB[0]={ucb_after[0]:.4f}")
    print(f"    losses_applied: {child0.losses_applied}")
    
    if ucb_after[0] < ucb_before[0]:
        print("\n  ✓ PASSED: Virtual loss reduced UCB score")
        return True
    else:
        print("\n  ✗ FAILED: Virtual loss did not reduce UCB score")
        return False


def test_virtual_loss_reverts_completely():
    """Test that revert_virtual_loss completely undoes the effect."""
    print("\n" + "=" * 60)
    print("TEST: Virtual Loss Reverts Completely")
    print("=" * 60)
    
    node = Node(to_play=1, num_actions=7, move=3, parent=DummyNode())
    node.parent.child_N[3] = 10
    node.parent.child_W[3] = 5.0
    
    W_before = node.W
    N_before = node.N
    
    print(f"\n  Before VL: N={N_before}, W={W_before}")
    
    # Apply VL
    add_virtual_loss(node)
    print(f"  After add_virtual_loss: N={node.N}, W={node.W}, losses_applied={node.losses_applied}")
    
    # Revert VL
    revert_virtual_loss(node)
    print(f"  After revert_virtual_loss: N={node.N}, W={node.W}, losses_applied={node.losses_applied}")
    
    if node.W == W_before and node.losses_applied == 0:
        print("\n  ✓ PASSED: Virtual loss fully reverted")
        return True
    else:
        print("\n  ✗ FAILED: Virtual loss not fully reverted")
        return False


def test_virtual_loss_multiple_applications():
    """Test multiple virtual losses on the same path."""
    print("\n" + "=" * 60)
    print("TEST: Multiple Virtual Losses")
    print("=" * 60)
    
    root = Node(to_play=1, num_actions=7, move=None, parent=DummyNode())
    root.parent.child_N[None] = 0
    root.parent.child_W[None] = 0.0
    
    child = Node(to_play=2, num_actions=7, move=3, parent=root)
    root.children[3] = child
    
    print(f"\n  Initial: root.W={root.W}, child.W={child.W}")
    
    # Apply VL 3 times
    add_virtual_loss(child)
    add_virtual_loss(child)
    add_virtual_loss(child)
    
    print(f"  After 3x add_virtual_loss: root.W={root.W}, child.W={child.W}")
    print(f"    child.losses_applied={child.losses_applied}")
    
    # VL adds +1 each time, so child.W should be +3
    if child.W != 3.0:
        print(f"  ✗ FAILED: Expected child.W=3.0, got {child.W}")
        return False
    
    # Revert all
    revert_virtual_loss(child)
    revert_virtual_loss(child)
    revert_virtual_loss(child)
    
    print(f"  After 3x revert_virtual_loss: root.W={root.W}, child.W={child.W}")
    print(f"    child.losses_applied={child.losses_applied}")
    
    if child.W == 0.0 and child.losses_applied == 0:
        print("\n  ✓ PASSED: Multiple VLs applied and reverted correctly")
        return True
    else:
        print("\n  ✗ FAILED: VL accounting error")
        return False


# =============================================================================
# Test: Expand and Search Policy
# =============================================================================

def test_expand_sets_priors():
    """Test that expand correctly sets prior probabilities."""
    print("\n" + "=" * 60)
    print("TEST: Expand Sets Priors")
    print("=" * 60)
    
    node = Node(to_play=1, num_actions=7, parent=DummyNode())
    
    priors = np.array([0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05], dtype=np.float32)
    
    print(f"\n  Before expand: is_expanded={node.is_expanded}")
    print(f"  Priors to set: {priors}")
    
    expand(node, priors)
    
    print(f"\n  After expand: is_expanded={node.is_expanded}")
    print(f"  child_P: {node.child_P}")
    
    if node.is_expanded and np.allclose(node.child_P, priors):
        print("\n  ✓ PASSED: Expand correctly set priors")
        return True
    else:
        print("\n  ✗ FAILED: Priors not set correctly")
        return False


def test_search_policy_proportional_to_visits():
    """Test that search policy is proportional to visit counts."""
    print("\n" + "=" * 60)
    print("TEST: Search Policy Proportional to Visits")
    print("=" * 60)
    
    child_N = np.array([100.0, 50.0, 25.0, 200.0, 10.0, 5.0, 10.0], dtype=np.float32)
    legal_actions = np.ones(7, dtype=np.float32)
    
    # Temperature 1.0: proportional to N
    policy = generate_search_policy(child_N, 1.0, legal_actions)
    
    print(f"\n  child_N: {child_N}")
    print(f"  Policy (temp=1.0): {policy.round(3)}")
    
    # Check proportionality
    expected = child_N / child_N.sum()
    
    if np.allclose(policy, expected, atol=1e-6):
        print("  ✓ Policy proportional to visits at temp=1.0")
    else:
        print("  ✗ Policy not proportional")
        return False
    
    # Temperature 0.1: should be more peaked (using power)
    policy_cold = generate_search_policy(child_N, 0.1, legal_actions)
    print(f"  Policy (temp=0.1): {policy_cold.round(3)}")
    
    # Most visited (action 3) should have higher probability than at temp=1.0
    if policy_cold[3] > policy[3]:
        print("  ✓ Lower temperature increases peak")
        print("\n  ✓ PASSED: Search policy generation correct")
        return True
    else:
        print("  ✗ Temperature effect wrong")
        return False


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_puct_tests():
    """Run all PUCT tests for mcts_v2"""
    print("\n" + "=" * 60)
    print("MCTS_V2 PUCT TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("child_Q Calculation", test_child_q_calculation),
        ("child_U Calculation", test_child_u_calculation),
        ("PUCT Sign Convention", test_puct_sign_convention),
        ("best_child Selection", test_best_child_selection),
        ("Legal Mask Respected", test_best_child_respects_legal_mask),
        ("Unvisited Exploration", test_unvisited_children_exploration),
        ("Q Dominates at High N", test_high_visit_q_dominates),
        ("Backup Single Node", test_backup_single_node),
        ("Backup Sign Alternation", test_backup_sign_alternation),
        ("Backup Accumulation", test_backup_accumulation),
        ("Virtual Loss Discourages", test_virtual_loss_discourages_selection),
        ("Virtual Loss Reverts", test_virtual_loss_reverts_completely),
        ("Multiple Virtual Losses", test_virtual_loss_multiple_applications),
        ("Expand Sets Priors", test_expand_sets_priors),
        ("Search Policy", test_search_policy_proportional_to_visits),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n  ✗ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"MCTS_V2 PUCT RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_puct_tests()
    exit(0 if success else 1)