"""
Batch MCTS with Virtual Loss for true parallel simulations
"""

import torch
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

@dataclass
class MCTSNode:
    """MCTS node with virtual loss support"""
    state: torch.Tensor
    parent: Optional['MCTSNode'] = None
    action: Optional[int] = None
    
    # Statistics
    visits: int = 0
    total_value: float = 0.0
    prior: float = 0.0
    
    # Virtual loss for parallel simulations
    virtual_loss: int = 0  # ← NEW: tracks ongoing simulations
    
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
        
        from src.nn.environments.vectorized_c4_env import VectorizedConnectFour
        self.env = VectorizedConnectFour(n_envs=1, device=device)

    def get_action_probs_batch_parallel(
        self,
        boards: List[torch.Tensor],
        legal_moves_list: List[torch.Tensor],
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run MCTS with both:
        1. Parallel across different games
        2. Parallel simulations within each tree (virtual loss)
        """
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
                        node.state = self._compute_state(path)
                    leaves_to_eval.append((tree_idx, node, path))
            
            # Evaluate non-terminal leaves in one batch
            if leaves_to_eval:
                leaf_boards = torch.stack([node.state.flatten() 
                                          for _, node, _ in leaves_to_eval])
                
                with torch.no_grad():
                    policies, values = self.model.forward(leaf_boards)
                
                # Expand and backup
                for i, (tree_idx, node, path) in enumerate(leaves_to_eval):
                    # Check if terminal
                    self.env.boards[0] = node.state
                    self.env._check_winners_batch()
                    
                    if self.env.winners[0] != 0 or not (node.state[0, :] == 0).any():
                        # Terminal state
                        node.is_terminal = True
                        if self.env.winners[0] == 0:
                            value = 0.0  # Draw
                        else:
                            # Determine value from perspective
                            current_player = self._get_current_player(node.state)
                            winner = self.env.winners[0].item()
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
            
            # Handle terminal leaves
            for tree_idx, node, path in terminal_leaves:
                value = node.terminal_value
                self._backup_with_virtual_loss(path, value)
        
        # Extract final visit distributions
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
        
        return visit_distributions, action_probs
    
    def _select_leaf_with_virtual_loss(self, root: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        """Select leaf and apply virtual loss along the path"""
        node = root
        path = [node]
        
        while node.is_expanded() and not node.is_terminal:
            # Apply virtual loss to current node
            node.add_virtual_loss(self.virtual_loss_value)
            
            # Select child with best PUCT (including virtual loss)
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
        
        # Apply virtual loss to leaf
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
    
    def _compute_state(self, path: List[MCTSNode]) -> torch.Tensor:
        """Compute state by replaying actions"""
        state = path[0].state.clone()
        current_player = self._get_current_player(state)
        
        for i in range(1, len(path)):
            if path[i].action is not None:
                state = self._make_move(state, path[i].action, current_player)
                current_player = 3 - current_player
        
        return state
    
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


def benchmark_virtual_loss():
    """Compare performance with and without virtual loss"""
    import time
    
    # Dummy model for testing
    class RealisticModel:
        """Simulates a real neural network with inference time"""
        def __init__(self, device, inference_ms=15):
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