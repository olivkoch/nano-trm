"""
Batch MCTS for Connect Four - AlphaZero style
"""

import torch
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

from src.nn.environments.vectorized_c4_env import VectorizedC4State, VectorizedConnectFour
from src.nn.utils.constants import C4_EMPTY_CELL, C4_PLAYER1_CELL, C4_PLAYER2_CELL


@dataclass
class MCTSNode:
    """MCTS node for AlphaZero-style search"""
    state: torch.Tensor  # Board state
    parent: Optional['MCTSNode'] = None
    action: Optional[int] = None
    
    # Statistics
    visits: int = 0
    total_value: float = 0.0  # Sum of values
    prior: float = 0.0  # Prior probability from network
    
    # Children
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)
    
    # Terminal state
    is_terminal: bool = False
    terminal_value: Optional[float] = None
    
    def Q(self) -> float:
        """Average value (Q = W/N)"""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits
    
    def U(self, c_puct: float) -> float:
        """Upper confidence bound"""
        if self.parent is None:
            return 0.0
        return c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
    
    def puct_score(self, c_puct: float) -> float:
        """PUCT = Q + U"""
        return self.Q() + self.U(c_puct)
    
    def is_expanded(self) -> bool:
        """Check if node has been expanded"""
        return len(self.children) > 0

class BatchMCTS:
    """MCTS following AlphaZero paper"""
    
    def __init__(
        self,
        model,  # TRMConnectFourModule
        c_puct: float = 1.0,
        num_simulations: int = 800,
        dirichlet_alpha: float = 0.3,
        exploration_fraction: float = 0.25,
        device: str = "cpu"
    ):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.device = device
        
        # Environment for simulation
        self.env = VectorizedConnectFour(n_envs=1, device=device)

    def get_action_probs_batch_parallel(
        self,
        boards: List[torch.Tensor],
        legal_moves_list: List[torch.Tensor],
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Run MCTS on multiple positions truly in parallel
        """
        n_positions = len(boards)
        
        # Create root nodes for all positions
        roots = []
        for i, (board, legal_moves) in enumerate(zip(boards, legal_moves_list)):
            root = MCTSNode(state=board.clone())
            if self.exploration_fraction > 0:
                self._add_dirichlet_noise(root, legal_moves)
            roots.append(root)
        
        # Run simulations for all trees in parallel
        for sim in range(self.num_simulations):
            # Collect all leaf nodes to evaluate
            leaves_to_eval = []
            paths_to_backup = []
            
            for root_idx, root in enumerate(roots):
                # Select path to leaf for this tree
                node = root
                path = [node]
                
                while node.is_expanded() and not node.is_terminal:
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
                
                if not node.is_terminal:
                    leaves_to_eval.append((root_idx, node, path))
            
            if not leaves_to_eval:
                continue
            
            # Batch evaluate all leaves
            leaf_boards = []
            for _, node, _ in leaves_to_eval:
                if node.state is None:
                    node.state = self._compute_state(path)
                leaf_boards.append(node.state.flatten())
            
            if leaf_boards:
                # Single batched forward pass for all leaves
                leaf_boards_tensor = torch.stack(leaf_boards)
                with torch.no_grad():
                    policies, values = self.model.forward(leaf_boards_tensor)
                
                # Process results and backup
                for i, (root_idx, node, path) in enumerate(leaves_to_eval):
                    policy = policies[i]
                    value = values[i].item()
                    
                    # Get legal moves for this node
                    node_legal_moves = node.state[0, :] == C4_EMPTY_CELL
                    
                    # Apply legal move mask
                    policy = policy * node_legal_moves.float()
                    if policy.sum() > 0:
                        policy = policy / policy.sum()
                    
                    # Expand node
                    if not node.is_expanded():
                        for action in range(7):
                            if node_legal_moves[action]:
                                child = MCTSNode(
                                    state=None,
                                    parent=node,
                                    action=action,
                                    prior=policy[action].item()
                                )
                                node.children[action] = child
                    
                    # Backup
                    self._backup(path, value)
        
        # Extract probabilities for all roots
        all_probs = torch.zeros(n_positions, 7, device=self.device)
        for i, root in enumerate(roots):
            visits = torch.zeros(7, device=self.device)
            for action, child in root.children.items():
                visits[action] = child.visits
            
            if temperature == 0:
                probs = torch.zeros(7, device=self.device)
                if visits.sum() > 0:
                    probs[visits.argmax()] = 1.0
            else:
                if visits.sum() > 0:
                    visits_temp = visits ** (1.0 / temperature)
                    probs = visits_temp / visits_temp.sum()
                else:
                    probs = legal_moves_list[i].float() / legal_moves_list[i].sum()
            
            all_probs[i] = probs
        
        return all_probs    
    
    def get_action_probs_batch(
        self,
        boards: List[torch.Tensor],
        legal_moves_list: List[torch.Tensor],
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Run MCTS on multiple positions in parallel
        
        Returns:
            probs: (n_positions, 7) action probabilities
        """
        n_positions = len(boards)
        all_probs = torch.zeros(n_positions, 7, device=self.device)
        
        # Run MCTS for each position
        # TODO: Could further optimize by batching neural network calls
        # across different MCTS trees
        for i, (board, legal_moves) in enumerate(zip(boards, legal_moves_list)):
            all_probs[i] = self.get_action_probs(board, legal_moves, temperature)
        
        return all_probs
    
    def get_action_probs(
        self, 
        board: torch.Tensor, 
        legal_moves: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Run MCTS simulations and return action probabilities
        
        Args:
            board: (6, 7) current board state
            legal_moves: (7,) boolean mask of legal moves
            temperature: Temperature for action selection
            
        Returns:
            probs: (7,) action probabilities
        """
        # Create root node
        root = MCTSNode(state=board.clone())
        
        # Add Dirichlet noise to root for exploration
        if self.exploration_fraction > 0:
            self._add_dirichlet_noise(root, legal_moves)
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Start from root
            node = root
            path = [node]
            
            # Select down to a leaf
            while node.is_expanded() and not node.is_terminal:
                # Select child with highest PUCT score
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
            
            # Get legal moves for current node
            if node is root:
                node_legal_moves = legal_moves
            else:
                # Compute legal moves for this node's state
                if node.state is not None:
                    node_legal_moves = node.state[0, :] == C4_EMPTY_CELL
                else:
                    # Need to compute the state first
                    node.state = self._compute_state(path)
                    node_legal_moves = node.state[0, :] == C4_EMPTY_CELL
            
            # Expand and evaluate
            value = self._expand_and_evaluate(node, node_legal_moves)
            
            # Backup
            self._backup(path, value)
        
        # Get visit counts for actions
        visits = torch.zeros(7, device=self.device)
        for action, child in root.children.items():
            visits[action] = child.visits
        
        # Apply temperature
        if temperature == 0:
            # Deterministic: choose most visited
            probs = torch.zeros(7, device=self.device)
            if visits.sum() > 0:
                probs[visits.argmax()] = 1.0
        else:
            # Stochastic with temperature
            if visits.sum() > 0:
                visits_temp = visits ** (1.0 / temperature)
                probs = visits_temp / visits_temp.sum()
            else:
                # Fallback to uniform
                probs = legal_moves.float() / legal_moves.sum()
        
        return probs
    
    def _compute_state(self, path: List[MCTSNode]) -> torch.Tensor:
        """Compute board state by replaying actions from root"""
        # Start from root state
        state = path[0].state.clone()
        current_player = 1 if (state == C4_PLAYER1_CELL).sum() == (state == C4_PLAYER2_CELL).sum() else 2
        
        # Apply each action in the path
        for i in range(1, len(path)):
            if path[i].action is not None:
                state = self._make_move(state, path[i].action, current_player)
                current_player = 3 - current_player
        
        return state
    
    def _add_dirichlet_noise(self, node: MCTSNode, legal_moves: torch.Tensor):
        """Add Dirichlet noise to priors for exploration"""
        # Get network policy
        with torch.no_grad():
            policy, _ = self.model.forward(node.state.flatten().unsqueeze(0))
            policy = policy.squeeze(0)
        
        # Apply legal move mask
        policy = policy * legal_moves.float()
        if policy.sum() > 0:
            policy = policy / policy.sum()
        else:
            policy = legal_moves.float() / legal_moves.sum()
        
        # Add Dirichlet noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * 7)
        noise = torch.tensor(noise, dtype=torch.float32, device=self.device) * legal_moves.float()
        noise = noise / (noise.sum() + 1e-8)
        
        # Mix policy with noise
        policy = (1 - self.exploration_fraction) * policy + self.exploration_fraction * noise
        
        # Create children with noisy priors
        for action in range(7):
            if legal_moves[action]:
                child = MCTSNode(
                    state=None,  # Will be set when visited
                    parent=node,
                    action=action,
                    prior=policy[action].item()
                )
                node.children[action] = child
    
    def _expand_and_evaluate(self, node: MCTSNode, legal_moves: torch.Tensor) -> float:
        """Expand node and return value estimate"""
        # Check if terminal
        self.env.boards[0] = node.state
        self.env._check_winners_batch()
        
        if self.env.winners[0] != 0:
            # Terminal - return actual value
            node.is_terminal = True
            winner = self.env.winners[0].item()
            # Value from current player's perspective
            current_player = 1 if (node.state == C4_PLAYER1_CELL).sum() == (node.state == C4_PLAYER2_CELL).sum() else 2
            if winner == current_player:
                node.terminal_value = 1.0
                return 1.0
            else:
                node.terminal_value = -1.0
                return -1.0
        
        if not legal_moves.any():
            # Draw
            node.is_terminal = True
            node.terminal_value = 0.0
            return 0.0
        
        # Get network evaluation
        with torch.no_grad():
            policy, value = self.model.forward(node.state.flatten().unsqueeze(0))
            policy = policy.squeeze(0)
            value = value.item()
        
        # Apply legal move mask
        policy = policy * legal_moves.float()
        if policy.sum() > 0:
            policy = policy / policy.sum()
        else:
            policy = legal_moves.float() / legal_moves.sum()
        
        # Create children if not already expanded
        if not node.is_expanded():
            for action in range(7):
                if legal_moves[action]:
                    child = MCTSNode(
                        state=None,  # Will be set when visited
                        parent=node,
                        action=action,
                        prior=policy[action].item()
                    )
                    node.children[action] = child
        
        return value
    
    def _backup(self, path: List[MCTSNode], value: float):
        """Backup value through path"""
        # Backup alternates sign (opponent's perspective)
        for node in reversed(path):
            node.visits += 1
            node.total_value += value
            value = -value  # Flip for opponent
    
    def _make_move(self, board: torch.Tensor, action: int, player: int) -> torch.Tensor:
        """Make a move on the board"""
        new_board = board.clone()
        # Find lowest empty row in column
        col_values = new_board[:, action]
        empty_rows = (col_values == C4_EMPTY_CELL).nonzero(as_tuple=True)[0]
        
        if len(empty_rows) > 0:
            row = empty_rows[-1].item()
            player_token = C4_PLAYER1_CELL if player == 1 else C4_PLAYER2_CELL
            new_board[row, action] = player_token
        
        return new_board
