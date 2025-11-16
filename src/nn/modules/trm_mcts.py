"""
Monte Carlo Tree Search for Connect Four with TRM
Leverages TRM's recursive reasoning for position evaluation
"""

import torch
import math
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np

from connectfour_env import ConnectFourEnv, ConnectFourState


@dataclass
class MCTSNode:
    """MCTS node with TRM evaluation"""
    state: ConnectFourState
    parent: Optional['MCTSNode'] = None
    action: Optional[int] = None
    
    # MCTS statistics
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    
    # Children
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)
    
    # TRM-specific: store hidden state for reuse
    trm_hidden_state: Optional[torch.Tensor] = None
    
    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0
    
    def ucb_score(self, c_puct: float = 1.0) -> float:
        """UCB score with exploration bonus"""
        if self.visits == 0:
            return float('inf')
        
        exploration = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.value + exploration


class TRM_MCTS:
    """MCTS that uses TRM's recursive reasoning"""
    
    def __init__(
        self,
        trm_wrapper,  # TRMConnectFourWrapper
        c_puct: float = 1.5,
        num_simulations: int = 100,
        num_trm_iterations: int = 3,  # How many TRM iterations per evaluation
        device: str = "cpu",
        temperature: float = 1.0
    ):
        self.trm_wrapper = trm_wrapper
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.num_trm_iterations = num_trm_iterations
        self.device = device
        self.temperature = temperature
    
    def search(self, root_state: ConnectFourState) -> Dict[int, float]:
        """Run MCTS simulations"""
        root = MCTSNode(state=root_state)
        
        # Get initial evaluation
        state_tensor = root_state.to_trm_input()
        policy, value = self.trm_wrapper.get_policy_value(
            state_tensor,
            root_state.legal_moves,
            num_iterations=self.num_trm_iterations
        )
        
        # Initialize root children with priors
        for action in range(7):
            if root_state.legal_moves[action]:
                root.children[action] = MCTSNode(
                    state=None,
                    parent=root,
                    action=action,
                    prior=policy[action].item()
                )
        
        # Run simulations
        for sim in range(self.num_simulations):
            self._simulate(root)
        
        # Return visit counts
        visits = {
            action: child.visits
            for action, child in root.children.items()
        }
        
        return visits
    
    def _simulate(self, root: MCTSNode):
        """Run one simulation"""
        node = root
        path = [node]
        
        # Selection
        while node.is_expanded and not node.state.is_terminal:
            # Select best child by UCB
            action, child = max(
                node.children.items(),
                key=lambda x: x[1].ucb_score(self.c_puct)
            )
            
            # Create state if needed
            if child.state is None:
                env = ConnectFourEnv(device=self.device)
                env.board = node.state.board.clone()
                env.current_player = node.state.current_player
                env.move_count = node.state.move_count
                child.state = env.make_move(action)
            
            node = child
            path.append(node)
        
        # Evaluation
        if not node.state.is_terminal:
            # Use TRM for evaluation
            state_tensor = node.state.to_trm_input()
            
            # More iterations for critical positions
            num_iters = self.num_trm_iterations
            if node.state.move_count > 20:  # Late game
                num_iters = self.num_trm_iterations * 2
            
            policy, value = self.trm_wrapper.get_policy_value(
                state_tensor,
                node.state.legal_moves,
                num_iterations=num_iters
            )
            
            # Expand node
            for action in range(7):
                if node.state.legal_moves[action]:
                    node.children[action] = MCTSNode(
                        state=None,
                        parent=node,
                        action=action,
                        prior=policy[action].item()
                    )
        else:
            # Terminal node
            if node.state.winner == node.state.current_player:
                value = 1.0
            elif node.state.winner is not None:
                value = -1.0
            else:
                value = 0.0  # Draw
        
        # Backup
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value  # Flip for opponent
    
    def get_action_probabilities(self, state: ConnectFourState) -> torch.Tensor:
        """Get action probabilities from MCTS"""
        visits = self.search(state)
        
        # Convert to probabilities
        probs = torch.zeros(7, dtype=torch.float32)
        total_visits = sum(visits.values())
        
        if total_visits > 0:
            for action, visit_count in visits.items():
                probs[action] = visit_count / total_visits
            
            # Apply temperature
            if self.temperature > 0:
                probs = probs ** (1 / self.temperature)
                probs = probs / probs.sum()
        
        return probs
    
    def select_action(self, state: ConnectFourState, deterministic: bool = False) -> int:
        """Select action using MCTS"""
        visits = self.search(state)
        
        if deterministic or self.temperature == 0:
            # Select most visited
            return max(visits.items(), key=lambda x: x[1])[0]
        else:
            # Sample from distribution
            actions = list(visits.keys())
            counts = [visits[a] for a in actions]
            
            # Apply temperature
            counts = np.array(counts) ** (1 / self.temperature)
            probs = counts / counts.sum()
            
            return np.random.choice(actions, p=probs)