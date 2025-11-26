"""
Batch MCTS with Virtual Loss for true parallel simulations
"""

import torch
import time
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

@dataclass
class MCTSNode:
    """MCTS node with virtual loss support
    
    Virtual loss discourages multiple parallel simulations from exploring
    the same path. Each ongoing simulation adds virtual_loss_value to the
    visit count and subtracts the same amount from the value sum, effectively
    assuming ongoing simulations will result in losses (value = -1).
    """
    state: torch.Tensor
    parent: Optional['MCTSNode'] = None
    action: Optional[int] = None
    
    # Statistics
    visits: int = 0
    total_value: float = 0.0
    prior: float = 0.0
    
    # Virtual loss for parallel simulations
    virtual_loss: int = 0  # tracks ongoing simulations
    
    # Children
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)
    
    # Terminal state
    is_terminal: bool = False
    terminal_value: Optional[float] = None
    
    def Q(self) -> float:
        """Average value including virtual loss"""
        # Virtual loss assumes ongoing simulations will be losses
        effective_visits = self.visits + self.virtual_loss
        if effective_visits == 0:
            return 0.0
        # Subtract virtual loss from value (assumes -1 value per virtual loss)
        effective_value = self.total_value - self.virtual_loss
        return effective_value / effective_visits
    
    def U(self, c_puct: float) -> float:
        """Upper confidence bound"""
        if self.parent is None:
            return 0.0
        # Include virtual loss in visit count
        effective_visits = self.visits + self.virtual_loss
        return c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + effective_visits)
    
    def puct_score(self, c_puct: float) -> float:
        """PUCT = Q + U (with virtual loss)"""
        return self.Q() + self.U(c_puct)
    
    def add_virtual_loss(self, n: int = 1):
        """Add virtual loss when selecting this node"""
        self.virtual_loss += n
    
    def remove_virtual_loss(self, n: int = 1):
        """Remove virtual loss after backup"""
        self.virtual_loss = max(0, self.virtual_loss - n)
    
    def is_expanded(self) -> bool:
        return len(self.children) > 0


class BatchMCTSWithVirtualLoss:
    """MCTS with virtual loss for parallel simulations within each tree"""
    
    def __init__(
        self,
        model,
        c_puct: float = 1.0,
        num_simulations: int = 800,
        parallel_simulations: int = 8,  # ← NEW: simulations per batch within tree
        dirichlet_alpha: float = 0.3,
        exploration_fraction: float = 0.25,
        virtual_loss_value: int = 3,  # ← NEW: virtual loss weight
        device: str = "cpu"
    ):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.parallel_simulations = parallel_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.virtual_loss_value = virtual_loss_value
        self.device = device
        
    def get_action_probs_batch_parallel(
        self,
        boards: List[torch.Tensor],
        legal_moves_list: List[torch.Tensor],
        temperature: float = 1.0,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run MCTS with both:
        1. Parallel across different games
        2. Parallel simulations within each tree (virtual loss)
        """
        t0 = time.time()

        n_positions = len(boards)
        
        # Create root nodes
        roots = []
        for board, legal_moves in zip(boards, legal_moves_list):
            root = MCTSNode(state=board.clone())
            if self.exploration_fraction > 0:
                self._add_dirichlet_noise(root, legal_moves)
            roots.append(root)
        
        # Run simulations in batches
        num_batches = (self.num_simulations + self.parallel_simulations - 1) // self.parallel_simulations
        
        for batch_idx in range(num_batches):

            t0_collect = time.time()
            sims_in_batch = min(self.parallel_simulations, 
                              self.num_simulations - batch_idx * self.parallel_simulations)
            
            # Collect multiple leaves from EACH tree
            all_leaves = []  # Will contain (tree_idx, node, path) tuples
            
            for tree_idx, root in enumerate(roots):
                # Select multiple leaves from this tree
                for sim in range(sims_in_batch):
                    node, path = self._select_leaf_with_virtual_loss(root)
                    all_leaves.append((tree_idx, node, path))
            
            # Batch evaluate ALL leaves from ALL trees
            leaves_to_eval = []
            terminal_leaves = []
            
            for tree_idx, node, path in all_leaves:
                if node.is_terminal:
                    terminal_leaves.append((tree_idx, node, path))
                else:
                    # Compute state if needed
                    if node.state is None:
                        node.state = self._compute_state_efficient(path)
                    leaves_to_eval.append((tree_idx, node, path))
            
            if verbose:
                t1_collect = time.time()
                print(f"Batch {batch_idx + 1}/{num_batches}: Collected {len(all_leaves)} leaves "
                    f"({len(leaves_to_eval)} to eval, {len(terminal_leaves)} terminal) "
                    f"in {t1_collect - t0_collect:.3f} seconds")

            # Evaluate non-terminal leaves in one batch
            if leaves_to_eval:
                leaf_boards = torch.stack([node.state
                                          for _, node, _ in leaves_to_eval]) # N, rows, cols
                
                with torch.no_grad():
                    t0_forward = time.time()
                    policies, values = self.model.forward(leaf_boards.flatten(start_dim=1))
                    t1_forward = time.time()
                    if verbose:
                        print(f"Batch evaluation took {t1_forward - t0_forward:.3f} seconds for batch shape {leaf_boards.shape}")
                
                t0_terminal = time.time()
                is_terminal_batch, winners_batch = self._is_terminal_and_winner_batch(leaf_boards)
                t1_terminal = time.time()
                if verbose:
                    print(f"Terminal check took {t1_terminal - t0_terminal:.3f} seconds for batch shape {leaf_boards.shape}")

                current_players_batch = self._get_current_player_batch(leaf_boards)

                t0_expand = time.time()
                # Expand and backup
                for i, (tree_idx, node, path) in enumerate(leaves_to_eval):

                    is_terminal = is_terminal_batch[i].item()
                    winner = winners_batch[i].item()
                    current_player = current_players_batch[i].item()

                    if is_terminal:
                        node.is_terminal = True
                        if winner == 0:
                            value = 0.0  # Draw
                        else:
                            # Determine value from perspective
                            value = 1.0 if winner == current_player else -1.0
                        node.terminal_value = value
                    else:
                        # Expand node
                        policy = policies[i]
                        value = values[i].item()
                        legal_moves = node.state[0, :] == 0
                        
                        # Apply legal mask
                        policy = policy * legal_moves.float()
                        if policy.sum() > 0:
                            policy = policy / policy.sum()
                        
                        # Create children
                        if not node.is_expanded():
                            for action in range(7):
                                if legal_moves[action]:
                                    child = MCTSNode(
                                        state=None,
                                        parent=node,
                                        action=action,
                                        prior=policy[action].item()
                                    )
                                    node.children[action] = child
                    
                    # Backup with virtual loss removal
                    self._backup_with_virtual_loss(path, value)
            t1_expand = time.time()
            if verbose:
                print(f"Expansion and backup took {t1_expand - t0_expand:.3f} seconds")

            # Handle terminal leaves
            for tree_idx, node, path in terminal_leaves:
                value = node.terminal_value
                self._backup_with_virtual_loss(path, value)
        
        # Extract final visit distributions
        t0_extract = time.time()
        visit_distributions = torch.zeros(n_positions, 7, device=self.device)
        action_probs = torch.zeros(n_positions, 7, device=self.device)
        
        for i, root in enumerate(roots):
            visits = torch.zeros(7, device=self.device)
            for action, child in root.children.items():
                visits[action] = child.visits
            
            # Visit distribution for training
            if visits.sum() > 0:
                visit_distributions[i] = visits / visits.sum()
            else:
                visit_distributions[i] = legal_moves_list[i].float() / legal_moves_list[i].sum()
            
            # Action probabilities for playing
            if temperature == 0:
                action_probs[i][visits.argmax()] = 1.0
            else:
                visits_temp = visits ** (1.0 / temperature)
                action_probs[i] = visits_temp / visits_temp.sum() if visits.sum() > 0 else visit_distributions[i]
        t1_extract = time.time()
        if verbose:
            print(f"Extracting visit distributions took {t1_extract - t0_extract:.3f} seconds")

        t1 = time.time()
        total = t1 - t0
        if verbose:
            print(f"Total MCTS batch time: {total:.3f} seconds")

        return visit_distributions, action_probs
    
    def _select_leaf_with_virtual_loss(self, root: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        """Select leaf and apply virtual loss along the path"""
        node = root
        path = [node]
        iterations = 0
        max_iterations = 100  # Safety check
        
        while node.is_expanded() and not node.is_terminal:
            iterations += 1
            if iterations > max_iterations:
                print(f"ERROR: Infinite loop detected in _select_leaf_with_virtual_loss!")
                print(f"  Path length: {len(path)}")
                print(f"  Current node: expanded={node.is_expanded()}, terminal={node.is_terminal}")
                print(f"  Children: {len(node.children)}")
                raise RuntimeError("Infinite loop in MCTS selection")
        
            node.add_virtual_loss(self.virtual_loss_value)
            
            # Select child with best PUCT
            best_action = None
            best_score = -float('inf')
            
            for action, child in node.children.items():
                score = child.puct_score(self.c_puct)
                if score > best_score:
                    best_score = score
                    best_action = action
            
            if best_action is None:
                break
            
            node = node.children[best_action]
            path.append(node)
        
        # Materialize state only for the SELECTED leaf (lazy + cache)
        if node.state is None:
            node.state = self._compute_state_efficient(path)  # ← Only compute once!
        
        node.add_virtual_loss(self.virtual_loss_value)
        return node, path
        
    def _backup_with_virtual_loss(self, path: List[MCTSNode], value: float):
        """Backup value and remove virtual loss"""
        for node in reversed(path):
            node.visits += 1
            node.total_value += value
            node.remove_virtual_loss(self.virtual_loss_value)
            value = -value  # Flip for opponent
    
    def _add_dirichlet_noise(self, node: MCTSNode, legal_moves: torch.Tensor):
        """Add Dirichlet noise to root priors"""
        with torch.no_grad():
            policy, _ = self.model.forward(node.state.flatten().unsqueeze(0))
            policy = policy.squeeze(0)
        
        policy = policy * legal_moves.float()
        if policy.sum() > 0:
            policy = policy / policy.sum()
        else:
            policy = legal_moves.float() / legal_moves.sum()
        
        # Add noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * 7)
        noise = torch.tensor(noise, dtype=torch.float32, device=self.device) * legal_moves.float()
        noise = noise / (noise.sum() + 1e-8)
        
        policy = (1 - self.exploration_fraction) * policy + self.exploration_fraction * noise
        
        # Create children with noisy priors
        for action in range(7):
            if legal_moves[action]:
                child = MCTSNode(
                    state=None,
                    parent=node,
                    action=action,
                    prior=policy[action].item()
                )
                node.children[action] = child
    
    def _compute_state_efficient(self, path: List[MCTSNode]) -> torch.Tensor:
        """Efficiently compute state by finding nearest cached state"""
        # Work backwards to find first materialized state
        for i in range(len(path) - 1, -1, -1):
            if path[i].state is not None:
                # Found cached state, replay from here
                state = path[i].state.clone()
                current_player = self._get_current_player(state)
                
                # Only replay actions from i+1 to end
                for j in range(i + 1, len(path)):
                    if path[j].action is not None:
                        state = self._make_move(state, path[j].action, current_player)
                        current_player = 3 - current_player
                        # Cache the state in the node
                        path[j].state = state  # ← Cache for next time!
                    
                return state
    
        raise RuntimeError("No state found in path - root should always have state")
    
    def _get_current_player(self, state: torch.Tensor) -> int:
        """Determine current player from board state"""
        p1_count = (state == 1).sum()
        p2_count = (state == 2).sum()
        return 1 if p1_count == p2_count else 2
    
    def _make_move(self, board: torch.Tensor, action: int, player: int) -> torch.Tensor:
        """Make a move on the board"""
        new_board = board.clone()
        col_values = new_board[:, action]
        empty_rows = (col_values == 0).nonzero(as_tuple=True)[0]
        
        if len(empty_rows) > 0:
            row = empty_rows[-1].item()
            new_board[row, action] = player
        
        return new_board
    
    def _get_current_player_batch(self, boards: torch.Tensor) -> torch.Tensor:
        """Vectorized current player computation
        
        Args:
            boards: (batch, 6, 7) tensor
        
        Returns:
            current_players: (batch,) tensor of 1s and 2s
        """
        p1_count = (boards == 1).sum(dim=(1, 2))  # (batch,)
        p2_count = (boards == 2).sum(dim=(1, 2))  # (batch,)
        return torch.where(p1_count == p2_count, 
                        torch.ones_like(p1_count), 
                        torch.ones_like(p1_count) * 2)

    def _is_terminal_and_winner_batch(self, boards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized terminal check for batch of boards
        
        Args:
            boards: (batch, 6, 7) tensor
        
        Returns:
            is_terminal: (batch,) bool tensor
            winners: (batch,) int tensor (0=draw, 1=player1, 2=player2, -1=ongoing)
        """
        batch_size = boards.shape[0]
        device = boards.device
        
        # Check for full boards (draws)
        board_full = ~(boards[:, 0, :] == 0).any(dim=1)  # (batch,)
        
        # Initialize winners as -1 (ongoing)
        winners = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        
        # Check horizontal wins
        for row in range(6):
            for col in range(4):
                window = boards[:, row, col:col+4]  # (batch, 4)
                # Check if all 4 are same and non-zero
                player = window[:, 0]
                all_same = (window == player.unsqueeze(1)).all(dim=1)
                valid_player = player != 0
                has_winner = all_same & valid_player
                winners = torch.where(has_winner, player, winners)
        
        # Check vertical wins
        for row in range(3):
            for col in range(7):
                window = boards[:, row:row+4, col]  # (batch, 4)
                player = window[:, 0]
                all_same = (window == player.unsqueeze(1)).all(dim=1)
                valid_player = player != 0
                has_winner = all_same & valid_player
                winners = torch.where(has_winner, player, winners)
        
        # Check diagonal wins (down-right)
        for row in range(3):
            for col in range(4):
                cells = torch.stack([
                    boards[:, row, col],
                    boards[:, row+1, col+1],
                    boards[:, row+2, col+2],
                    boards[:, row+3, col+3]
                ], dim=1)  # (batch, 4)
                player = cells[:, 0]
                all_same = (cells == player.unsqueeze(1)).all(dim=1)
                valid_player = player != 0
                has_winner = all_same & valid_player
                winners = torch.where(has_winner, player, winners)
        
        # Check diagonal wins (down-left)
        for row in range(3):
            for col in range(4):
                cells = torch.stack([
                    boards[:, row, col+3],
                    boards[:, row+1, col+2],
                    boards[:, row+2, col+1],
                    boards[:, row+3, col]
                ], dim=1)  # (batch, 4)
                player = cells[:, 0]
                all_same = (cells == player.unsqueeze(1)).all(dim=1)
                valid_player = player != 0
                has_winner = all_same & valid_player
                winners = torch.where(has_winner, player, winners)
        
        # Set draws where board is full but no winner
        winners = torch.where(board_full & (winners == -1), torch.zeros_like(winners), winners)
        
        # is_terminal = has winner OR board full
        is_terminal = (winners >= 0)
        
        return is_terminal, winners

    def _get_legal_moves(self, board: torch.Tensor) -> torch.Tensor:
        """Fast legal moves check"""
        return board[0, :] == 0


def benchmark_virtual_loss():
    """Compare performance with and without virtual loss"""
    import time
    
    # Dummy model for testing
    class RealisticModel:
        """Simulates a real neural network with inference time"""
        def __init__(self, device, inference_ms=2):
            self.device = device
            self.inference_ms = inference_ms
            
        def forward(self, boards):
            import time
            batch_size = boards.shape[0]
            # Simulate model inference time (but with batching efficiency)
            # Real models are more efficient with larger batches
            if batch_size == 1:
                time.sleep(self.inference_ms / 1000)
            else:
                # Batching provides sublinear scaling
                time.sleep(self.inference_ms * (1 + batch_size * 0.2) / 1000)
            
            policies = torch.rand(batch_size, 7, device=self.device)
            policies = torch.softmax(policies, dim=-1)
            values = torch.rand(batch_size, device=self.device) * 2 - 1
            return policies, values
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RealisticModel(device)
    
    # Test configurations
    configs = [
        ("No Virtual Loss (sequential)", 1, 100),
        ("Virtual Loss (4 parallel)", 4, 100),
        ("Virtual Loss (8 parallel)", 8, 100),
    ]
    
    # Random board positions
    boards = [torch.randint(0, 3, (6, 7), device=device) for _ in range(10)]
    legal_moves = [torch.ones(7, device=device, dtype=torch.bool) for _ in range(10)]
    
    print("Benchmarking MCTS with Virtual Loss:\n")
    
    for name, parallel_sims, num_sims in configs:
        mcts = BatchMCTSWithVirtualLoss(
            model=model,
            num_simulations=num_sims,
            parallel_simulations=parallel_sims,
            device=device
        )
        
        start = time.time()
        mcts.get_action_probs_batch_parallel(boards, legal_moves, temperature=1.0)
        elapsed = time.time() - start
        
        print(f"{name:30} Time: {elapsed:.3f}s  Speed: {len(boards)/elapsed:.1f} games/sec")
    
    print("\nNote: Real speedup depends on model inference time.")
    print("Larger models benefit more from batching within trees.")


if __name__ == "__main__":
    benchmark_virtual_loss()