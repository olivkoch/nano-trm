#!/usr/bin/env python
"""
Prior-Value Conflict Tests

These tests verify that MCTS can overcome misleading priors through tree search.
This is a critical property: the neural network might be wrong, but MCTS should
still find the correct move through simulation.

Key scenarios:
1. Prior points to neutral move, but another move wins immediately
2. Prior points to losing move, MCTS must find the only non-losing move
3. Prior is completely inverted (best move has lowest prior)
4. Prior is uniform but position has clear best move
5. Strong prior on move that leads to loss in 2 plies
"""

import torch
import numpy as np
from src.nn.modules.tensor_mcts import TensorMCTS, TensorMCTSConfig


def print_board(board):
    """Pretty print a board"""
    if board.dim() == 3:
        board = board[0]
    print("  " + "-" * 15)
    for row in range(6):
        print("  |" + "".join([".XO"[int(board[row, c])] for c in range(7)]) + "|")
    print("  " + "-" * 15)
    print("   0123456")


class MisleadingPriorModel:
    """
    Model that gives WRONG priors - points away from the best move.
    Used to test if MCTS can overcome bad policy through search.
    """
    def __init__(self, device, bad_action, bad_prior=0.9):
        self.device = device
        self.bad_action = bad_action  # Action to give high (misleading) prior
        self.bad_prior = bad_prior
    
    def forward(self, boards, current_players):
        batch_size = boards.shape[0]
        
        # Give high prior to the "bad" action
        remaining = (1.0 - self.bad_prior) / 6
        policies = torch.full((batch_size, 7), remaining, device=self.device)
        policies[:, self.bad_action] = self.bad_prior
        
        # Neutral value prediction (model doesn't know position is won/lost)
        values = torch.zeros(batch_size, device=self.device)
        
        return policies, values


class InvertedPriorModel:
    """
    Model where priors are exactly inverted from optimal.
    Best moves get lowest priors, worst moves get highest.
    """
    def __init__(self, device, action_ranking):
        """
        action_ranking: list of actions from WORST to BEST
        Worst action gets highest prior, best gets lowest.
        """
        self.device = device
        self.action_ranking = action_ranking
    
    def forward(self, boards, current_players):
        batch_size = boards.shape[0]
        
        # Create inverted priors
        n_actions = len(self.action_ranking)
        # Exponentially decreasing priors for better actions
        raw_priors = torch.tensor([2.0 ** (n_actions - i - 1) for i in range(n_actions)], 
                                   device=self.device)
        raw_priors = raw_priors / raw_priors.sum()
        
        policies = torch.zeros(batch_size, 7, device=self.device)
        for rank, action in enumerate(self.action_ranking):
            policies[:, action] = raw_priors[rank]
        
        values = torch.zeros(batch_size, device=self.device)
        return policies, values


class DummyModel:
    def __init__(self, device):
        self.device = device
    
    def forward(self, boards, current_players):
        batch_size = boards.shape[0]
        policies = torch.ones(batch_size, 7, device=self.device) / 7
        values = torch.zeros(batch_size, device=self.device)
        return policies, values


def test_overcome_misleading_prior_immediate_win():
    """
    Setup: P1 wins immediately at column 3.
    Model: 90% prior on column 0 (neutral move).
    
    MCTS should discover the winning move through search despite the prior.
    """
    print("\n" + "=" * 60)
    print("TEST: Overcome Misleading Prior (Immediate Win)")
    print("=" * 60)
    
    device = "cpu"
    
    # Model strongly prefers column 0
    model = MisleadingPriorModel(device, bad_action=0, bad_prior=0.9)
    
    config = TensorMCTSConfig(
        n_actions=7,
        max_nodes_per_tree=2000,
        c_puct_init=1.0,
        exploration_fraction=0.0,  # No noise, pure MCTS
        virtual_loss_weight=3.0
    )
    
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    # P1 wins at column 3
    board = torch.zeros(1, 6, 7, dtype=torch.long, device=device)
    board[0, 5, 0:3] = 1  # P1 has X X X _
    board[0, 5, 4:6] = 2  # P2 pieces for balance
    board[0, 4, 4] = 2
    
    print("\n  Board (X=P1, O=P2, P1 wins at col 3):")
    print_board(board)
    
    legal = (board[0, 0, :] == 0).unsqueeze(0).float()
    players = torch.tensor([1], dtype=torch.long, device=device)
    
    with torch.no_grad():
        policies, _ = model.forward(board.flatten(start_dim=1).float(), players)
    
    print(f"\n  Model's prior: {policies[0].cpu().numpy().round(3)}")
    print(f"  Prior on col 0 (misleading): {policies[0, 0].item():.1%}")
    print(f"  Prior on col 3 (winning): {policies[0, 3].item():.1%}")
    
    mcts.reset(board, policies, legal, players)
    
    # Test with increasing simulations
    print(f"\n  {'Sims':<8} {'Col 0 %':<12} {'Col 3 %':<12} {'Winner':<10}")
    print("  " + "-" * 42)
    
    for n_sims in [50, 100, 200, 400]:
        mcts.reset(board, policies, legal, players)
        mcts.run_simulations(n_sims, parallel_sims=8)
        
        dist = mcts._get_visit_distributions()[0]
        winner = "Col 3 ✓" if dist[3] > dist[0] else "Col 0 ✗"
        print(f"  {n_sims:<8} {dist[0].item():<12.1%} {dist[3].item():<12.1%} {winner}")
    
    # Final check
    if dist[3].item() > 0.7:
        print(f"\n  ✓ PASSED: MCTS overcame misleading prior ({dist[3].item():.1%} on winning move)")
        return True
    else:
        print(f"\n  ✗ FAILED: MCTS did not find winning move (only {dist[3].item():.1%})")
        return False


def test_overcome_misleading_prior_must_block():
    """
    Setup: P2 threatens to win at column 3. P1 must block.
    Model: 90% prior on column 6 (irrelevant move).
    
    MCTS needs 2-ply lookahead to see the threat.
    """
    print("\n" + "=" * 60)
    print("TEST: Overcome Misleading Prior (Must Block)")
    print("=" * 60)
    
    device = "cpu"
    
    # Model strongly prefers column 6 (far from the threat)
    model = MisleadingPriorModel(device, bad_action=6, bad_prior=0.9)
    
    config = TensorMCTSConfig(
        n_actions=7,
        max_nodes_per_tree=2000,
        c_puct_init=1.0,
        exploration_fraction=0.0,
        virtual_loss_weight=3.0
    )
    
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    # P2 threatens at column 3
    board = torch.zeros(1, 6, 7, dtype=torch.long, device=device)
    board[0, 5, 0:3] = 2  # P2 has O O O _
    board[0, 5, 4:6] = 1  # P1 pieces
    board[0, 4, 4] = 1
    
    print("\n  Board (X=P1, O=P2, P1 must block at col 3):")
    print_board(board)
    
    legal = (board[0, 0, :] == 0).unsqueeze(0).float()
    players = torch.tensor([1], dtype=torch.long, device=device)
    
    with torch.no_grad():
        policies, _ = model.forward(board.flatten(start_dim=1).float(), players)
    
    print(f"\n  Model's prior: {policies[0].cpu().numpy().round(3)}")
    print(f"  Prior on col 6 (misleading): {policies[0, 6].item():.1%}")
    print(f"  Prior on col 3 (blocking): {policies[0, 3].item():.1%}")
    
    mcts.reset(board, policies, legal, players)
    
    print(f"\n  {'Sims':<8} {'Col 6 %':<12} {'Col 3 %':<12} {'Winner':<10}")
    print("  " + "-" * 42)
    
    for n_sims in [100, 200, 400, 800]:
        mcts.reset(board, policies, legal, players)
        mcts.run_simulations(n_sims, parallel_sims=8)
        
        dist = mcts._get_visit_distributions()[0]
        winner = "Col 3 ✓" if dist[3] > dist[6] else "Col 6 ✗"
        print(f"  {n_sims:<8} {dist[6].item():<12.1%} {dist[3].item():<12.1%} {winner}")
    
    # Final check - col 3 should beat col 6, not necessarily > 50%
    if dist[3].item() > dist[6].item():
        print(f"\n  ✓ PASSED: MCTS found blocking move despite misleading prior")
        return True
    else:
        print(f"\n  ✗ FAILED: MCTS did not find blocking move")
        return False


def test_inverted_prior_ranking():
    """
    Setup: Position with clear move ranking (col 3 wins, col 0 loses)
    Model: Priors are exactly inverted (losing move has highest prior)
    
    This is the hardest test - MCTS must completely reverse the prior ranking.
    """
    print("\n" + "=" * 60)
    print("TEST: Inverted Prior Ranking")
    print("=" * 60)
    
    device = "cpu"
    
    # Inverted ranking: worst to best = [3, 1, 2, 0, 4, 5, 6]
    # So col 3 (winning) gets LOWEST prior, col 0 gets moderate prior
    # Actually let's make it: col 0 has highest prior, col 3 has lowest
    action_ranking = [3, 6, 5, 4, 2, 1, 0]  # Best to worst gets low to high prior
    model = InvertedPriorModel(device, action_ranking)
    
    config = TensorMCTSConfig(
        n_actions=7,
        max_nodes_per_tree=2000,
        c_puct_init=1.0,
        exploration_fraction=0.0,
        virtual_loss_weight=3.0
    )
    
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    # P1 wins at column 3
    board = torch.zeros(1, 6, 7, dtype=torch.long, device=device)
    board[0, 5, 0:3] = 1
    board[0, 5, 4:6] = 2
    board[0, 4, 4] = 2
    
    print("\n  Board (P1 wins at col 3):")
    print_board(board)
    
    legal = (board[0, 0, :] == 0).unsqueeze(0).float()
    players = torch.tensor([1], dtype=torch.long, device=device)
    
    with torch.no_grad():
        policies, _ = model.forward(board.flatten(start_dim=1).float(), players)
    
    print(f"\n  Inverted priors (winning move has LOWEST prior):")
    print(f"  {policies[0].cpu().numpy().round(3)}")
    print(f"  Col 3 (winning): {policies[0, 3].item():.3f} <- lowest!")
    print(f"  Col 0 (neutral): {policies[0, 0].item():.3f} <- highest!")
    
    mcts.reset(board, policies, legal, players)
    mcts.run_simulations(500, parallel_sims=8)
    
    dist = mcts._get_visit_distributions()[0]
    
    print(f"\n  Final visit distribution:")
    print(f"  {dist.cpu().numpy().round(3)}")
    print(f"  Col 3: {dist[3].item():.1%}")
    
    if dist[3].item() > 0.7:
        print(f"\n  ✓ PASSED: MCTS completely reversed inverted prior")
        return True
    else:
        print(f"\n  ✗ FAILED: Could not overcome inverted prior")
        return False


def test_prior_vs_value_tradeoff():
    """
    Test the tradeoff between prior and value at different visit counts.
    
    At low visits, prior should dominate (exploration).
    At high visits, value should dominate (exploitation).
    """
    print("\n" + "=" * 60)
    print("TEST: Prior vs Value Tradeoff")
    print("=" * 60)
    
    device = "cpu"
    model = MisleadingPriorModel(device, bad_action=0, bad_prior=0.8)
    
    config = TensorMCTSConfig(
        n_actions=7,
        max_nodes_per_tree=2000,
        c_puct_init=1.0,
        exploration_fraction=0.0,
        virtual_loss_weight=3.0
    )
    
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    # P1 wins at col 3
    board = torch.zeros(1, 6, 7, dtype=torch.long, device=device)
    board[0, 5, 0:3] = 1
    board[0, 5, 4:6] = 2
    board[0, 4, 4] = 2
    
    legal = (board[0, 0, :] == 0).unsqueeze(0).float()
    players = torch.tensor([1], dtype=torch.long, device=device)
    
    with torch.no_grad():
        policies, _ = model.forward(board.flatten(start_dim=1).float(), players)
    
    print(f"\n  Prior: col 0 = {policies[0, 0].item():.1%}, col 3 = {policies[0, 3].item():.1%}")
    print(f"\n  Watching prior→value transition:")
    print(f"  {'Sims':<8} {'Col 0':<10} {'Col 3':<10} {'Dominant':<12}")
    print("  " + "-" * 40)
    
    crossover_point = None
    results = []
    
    for n_sims in [10, 20, 30, 50, 75, 100, 150, 200, 300]:
        mcts.reset(board, policies, legal, players)
        mcts.run_simulations(n_sims, parallel_sims=4)
        
        dist = mcts._get_visit_distributions()[0]
        col0, col3 = dist[0].item(), dist[3].item()
        
        dominant = "Prior (col 0)" if col0 > col3 else "Value (col 3)"
        results.append((n_sims, col0, col3))
        
        if crossover_point is None and col3 > col0:
            crossover_point = n_sims
        
        print(f"  {n_sims:<8} {col0:<10.1%} {col3:<10.1%} {dominant}")
    
    if crossover_point:
        print(f"\n  Crossover from prior to value at ~{crossover_point} simulations")
        print("  ✓ PASSED: Observed prior→value transition")
        return True
    else:
        print("\n  ✗ FAILED: Never saw value dominate prior")
        return False


def test_uniform_prior_finds_best():
    """
    With uniform prior, MCTS should find the best move purely through search.
    """
    print("\n" + "=" * 60)
    print("TEST: Uniform Prior Finds Best Move")
    print("=" * 60)
    
    device = "cpu"
    model = DummyModel(device)  # Uniform prior
    
    config = TensorMCTSConfig(
        n_actions=7,
        max_nodes_per_tree=2000,
        c_puct_init=1.0,
        exploration_fraction=0.0,
        virtual_loss_weight=3.0
    )
    
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    # P1 wins at col 3
    board = torch.zeros(1, 6, 7, dtype=torch.long, device=device)
    board[0, 5, 0:3] = 1
    board[0, 5, 4:6] = 2
    board[0, 4, 4] = 2
    
    print("\n  Board:")
    print_board(board)
    
    legal = (board[0, 0, :] == 0).unsqueeze(0).float()
    players = torch.tensor([1], dtype=torch.long, device=device)
    
    with torch.no_grad():
        policies, _ = model.forward(board.flatten(start_dim=1).float(), players)
    
    print(f"\n  Uniform prior: {policies[0].cpu().numpy().round(3)}")
    
    mcts.reset(board, policies, legal, players)
    mcts.run_simulations(200, parallel_sims=8)
    
    dist = mcts._get_visit_distributions()[0]
    
    print(f"\n  Visit distribution: {dist.cpu().numpy().round(3)}")
    print(f"  Winning move (col 3): {dist[3].item():.1%}")
    
    if dist[3].item() > 0.8:
        print("\n  ✓ PASSED: Pure search found winning move")
        return True
    else:
        print("\n  ✗ FAILED: Search alone did not find winning move")
        return False


def test_misleading_prior_losing_trap():
    """
    Model strongly prefers a move that leads to a loss in 2 plies.
    
    Setup:
    - P1 plays col 0 (model's preference)
    - P2 plays col 3 and wins
    
    MCTS must see through the trap.
    """
    print("\n" + "=" * 60)
    print("TEST: Avoid Losing Trap (2-ply lookahead)")
    print("=" * 60)
    
    device = "cpu"
    
    # Model loves col 0, but it's a trap
    model = MisleadingPriorModel(device, bad_action=0, bad_prior=0.9)
    
    config = TensorMCTSConfig(
        n_actions=7,
        max_nodes_per_tree=2000,
        c_puct_init=1.0,
        exploration_fraction=0.0,
        virtual_loss_weight=3.0
    )
    
    mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
    
    # P2 threatens win at col 3. If P1 plays col 0, P2 wins at col 3.
    # P1 MUST play col 3 to block.
    board = torch.zeros(1, 6, 7, dtype=torch.long, device=device)
    board[0, 5, 0:3] = 2  # P2: O O O _
    board[0, 5, 4:6] = 1  # P1 pieces
    board[0, 4, 4] = 1
    
    print("\n  Board (P2 threatens at col 3):")
    print_board(board)
    print("  If P1 plays col 0, P2 plays col 3 and wins!")
    
    legal = (board[0, 0, :] == 0).unsqueeze(0).float()
    players = torch.tensor([1], dtype=torch.long, device=device)
    
    with torch.no_grad():
        policies, _ = model.forward(board.flatten(start_dim=1).float(), players)
    
    print(f"\n  Model's prior (trap!): col 0 = {policies[0, 0].item():.1%}")
    
    # Run with increasing simulations and show Q values
    print(f"\n  {'Sims':<8} {'Col 0 %':<10} {'Col 3 %':<10} {'Q(col0)':<10} {'Q(col3)':<10} {'Winner':<10}")
    print("  " + "-" * 58)
    
    for n_sims in [100, 200, 400, 800, 1600]:
        mcts.reset(board, policies, legal, players)
        mcts.run_simulations(n_sims, parallel_sims=8)
        
        dist = mcts._get_visit_distributions()[0]
        
        # Get Q values for analysis
        child0_idx = mcts.children[0, 0, 0].item()
        child3_idx = mcts.children[0, 0, 3].item()
        
        q0 = "N/A"
        q3 = "N/A"
        if child0_idx >= 0 and mcts.visits[0, child0_idx].item() > 0:
            q0_child = mcts.total_value[0, child0_idx].item() / mcts.visits[0, child0_idx].item()
            q0 = f"{-q0_child:.3f}"  # Negate for parent's perspective
        if child3_idx >= 0 and mcts.visits[0, child3_idx].item() > 0:
            q3_child = mcts.total_value[0, child3_idx].item() / mcts.visits[0, child3_idx].item()
            q3 = f"{-q3_child:.3f}"
        
        winner = "Col 3 ✓" if dist[3] > dist[0] else "Col 0 ✗"
        print(f"  {n_sims:<8} {dist[0].item():<10.1%} {dist[3].item():<10.1%} {q0:<10} {q3:<10} {winner}")
    
    # Use final distribution
    if dist[3].item() > dist[0].item():
        print("\n  ✓ PASSED: MCTS avoided the trap")
        return True
    else:
        print(f"\n  ✗ FAILED: MCTS fell into the trap")
        
        # Additional diagnostics
        print("\n  Diagnostic info:")
        print(f"    Nodes allocated: {mcts.next_node_idx[0].item()}")
        
        # Check all children Q values
        print("\n    All root children:")
        for a in range(7):
            child_idx = mcts.children[0, 0, a].item()
            if child_idx >= 0:
                visits = mcts.visits[0, child_idx].item()
                if visits > 0:
                    q_child = mcts.total_value[0, child_idx].item() / visits
                    q_parent = -q_child
                    prior = mcts.priors[0, 0, a].item()
                    print(f"      Col {a}: visits={visits:.0f}, Q_parent={q_parent:.3f}, prior={prior:.3f}")
        
        # Check if col 0's children include the winning response
        if child0_idx >= 0:
            print(f"\n    Col 0 subtree (node {child0_idx}):")
            print(f"      Visits: {mcts.visits[0, child0_idx].item()}")
            
            # Check col 0's grandchildren (P2's responses)
            for a in range(7):
                gc_idx = mcts.children[0, child0_idx, a].item()
                if gc_idx >= 0:
                    gc_visits = mcts.visits[0, gc_idx].item()
                    gc_term = mcts.is_terminal[0, gc_idx].item()
                    gc_val = mcts.terminal_value[0, gc_idx].item() if gc_term else "N/A"
                    print(f"        -> P2 plays {a}: visits={gc_visits}, term={gc_term}, val={gc_val}")
        
        return False


def test_c_puct_effect_on_prior_override():
    """
    Test how c_puct affects the ability to override priors.
    
    Higher c_puct = more exploration = harder to overcome bad priors initially
    Lower c_puct = more exploitation = faster to find value but might miss it
    """
    print("\n" + "=" * 60)
    print("TEST: c_puct Effect on Prior Override")
    print("=" * 60)
    
    device = "cpu"
    model = MisleadingPriorModel(device, bad_action=0, bad_prior=0.8)
    
    # P1 wins at col 3
    board = torch.zeros(1, 6, 7, dtype=torch.long, device=device)
    board[0, 5, 0:3] = 1
    board[0, 5, 4:6] = 2
    board[0, 4, 4] = 2
    
    legal = (board[0, 0, :] == 0).unsqueeze(0).float()
    players = torch.tensor([1], dtype=torch.long, device=device)
    
    with torch.no_grad():
        policies, _ = model.forward(board.flatten(start_dim=1).float(), players)
    
    print(f"\n  Testing c_puct values with 200 simulations:")
    print(f"  {'c_puct':<10} {'Col 0 %':<12} {'Col 3 %':<12} {'Found Win':<10}")
    print("  " + "-" * 44)
    
    results = []
    for c_puct in [0.5, 1.0, 2.0, 4.0]:
        config = TensorMCTSConfig(
            n_actions=7,
            max_nodes_per_tree=2000,
            c_puct_init=c_puct,
            exploration_fraction=0.0,
            virtual_loss_weight=3.0
        )
        
        mcts = TensorMCTS(model, n_trees=1, config=config, device=device)
        mcts.reset(board, policies, legal, players)
        mcts.run_simulations(200, parallel_sims=8)
        
        dist = mcts._get_visit_distributions()[0]
        found = "Yes ✓" if dist[3] > 0.5 else "No ✗"
        results.append(dist[3].item())
        print(f"  {c_puct:<10} {dist[0].item():<12.1%} {dist[3].item():<12.1%} {found}")
    
    # All should eventually find it, but lower c_puct should find it with higher confidence
    if all(r > 0.4 for r in results):
        print("\n  ✓ PASSED: All c_puct values found the winning move")
        return True
    else:
        print("\n  ✗ FAILED: Some c_puct values failed to find winning move")
        return False


def run_all_prior_conflict_tests():
    """Run all prior-value conflict tests"""
    print("\n" + "=" * 60)
    print("PRIOR-VALUE CONFLICT TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Immediate Win vs Misleading Prior", test_overcome_misleading_prior_immediate_win),
        ("Must Block vs Misleading Prior", test_overcome_misleading_prior_must_block),
        ("Inverted Prior Ranking", test_inverted_prior_ranking),
        ("Prior vs Value Tradeoff", test_prior_vs_value_tradeoff),
        ("Uniform Prior Finds Best", test_uniform_prior_finds_best),
        ("Avoid Losing Trap", test_misleading_prior_losing_trap),
        ("c_puct Effect", test_c_puct_effect_on_prior_override),
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
    print(f"PRIOR-VALUE CONFLICT RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_prior_conflict_tests()
    exit(0 if success else 1)