"""
Batch MCTS for Connect Four - evaluates multiple positions in parallel
"""

import torch
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

from src.nn.environments.vectorized_c4_env import VectorizedC4State, VectorizedConnectFour


@dataclass
class BatchMCTSNode:
    """MCTS node that supports batch operations"""
    # For vectorized states, we store the index into the batch
    env_id: Optional[int] = None  # Index in vectorized environment
    parent: Optional['BatchMCTSNode'] = None
    action: Optional[int] = None
    
    # MCTS statistics
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    
    # Children
    children: Dict[int, 'BatchMCTSNode'] = field(default_factory=dict)
    
    # For virtual loss during parallel search
    virtual_loss: int = 0
    
    # Terminal state flag
    is_terminal: bool = False
    terminal_value: Optional[float] = None
    
    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0
    
    def ucb_score(self, c_puct: float = 1.0) -> float:
        """UCB score with exploration bonus and virtual loss"""
        if self.visits == 0:
            return float('inf')
        
        # Include virtual loss to encourage exploration of different paths
        total_visits = self.visits + self.virtual_loss
        exploration = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + total_visits)
        
        # Adjust value by virtual loss
        adjusted_value = (self.value_sum - self.virtual_loss) / total_visits
        return adjusted_value + exploration


class BatchMCTS:
    """Batch MCTS that evaluates multiple leaf nodes in parallel using vectorized environments"""
    
    def __init__(
        self,
        trm_model,  # TRMConnectFourModule
        c_puct: float = 1.5,
        num_simulations: int = 50,
        batch_size: int = 8,  # Number of positions to evaluate together
        device: str = "cpu",
        temperature: float = 1.0,
        use_compile: bool = True
    ):
        self.trm_model = trm_model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.batch_size = min(batch_size, num_simulations)
        self.device = device
        self.temperature = temperature
        
        # Create vectorized environment for parallel simulation
        self.vec_env = VectorizedConnectFour(n_envs=batch_size, device=device)
        
        # Compile the model for faster inference if requested
        if use_compile and hasattr(torch, 'compile'):
            try:
                self.get_policy_value_batch = torch.compile(
                    self._get_policy_value_batch_impl,
                    mode="reduce-overhead" if device == "cuda" else "default"
                )
            except:
                # Fallback if compilation fails (e.g., on some MPS setups)
                self.get_policy_value_batch = self._get_policy_value_batch_impl
        else:
            self.get_policy_value_batch = self._get_policy_value_batch_impl
    
    def _get_policy_value_batch_impl(
        self, 
        states: torch.Tensor, 
        legal_moves: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implementation of batch policy/value evaluation"""
        with torch.no_grad():
            batch_size = states.shape[0]
            
            # Create batch for TRM
            batch = {
                'input': states.to(self.device),
                'output': torch.zeros_like(states).to(self.device),
                'puzzle_identifiers': torch.zeros(batch_size, dtype=torch.long, device=self.device)
            }
            
            # Initialize carry and run TRM
            carry = self.trm_model.base_trm.initial_carry(batch)
            carry, _ = self.trm_model.base_trm(carry, batch)
            
            # Extract hidden states
            if hasattr(self.trm_model.base_trm, 'puzzle_emb_len') and self.trm_model.base_trm.puzzle_emb_len > 0:
                hidden_states = carry.inner_carry.z_H[:, self.trm_model.base_trm.puzzle_emb_len]
            else:
                hidden_states = carry.inner_carry.z_H[:, 0]
            
            # Get policies and values
            policy_logits = self.trm_model.policy_head(hidden_states)  # (batch_size, 7)
            values = self.trm_model.value_head(hidden_states).squeeze(-1)  # (batch_size,)
            
            # Apply legal move masks
            illegal_mask = ~legal_moves.to(self.device)
            policy_logits = policy_logits.masked_fill(illegal_mask, -float('inf'))
            
            # Convert to probabilities
            policies = torch.softmax(policy_logits, dim=-1)
            
            return policies, values
    
    def search_vectorized(self, root_states: VectorizedC4State) -> List[Dict[int, float]]:
        """Run MCTS on multiple root states in parallel"""
        n_roots = root_states.n_envs
        
        # Create root nodes for each environment
        roots = []
        for env_id in range(n_roots):
            root = BatchMCTSNode(env_id=env_id)
            roots.append(root)
        
        # Initial batch evaluation of all roots
        state_tensors = root_states.to_trm_input()
        policies, values = self.get_policy_value_batch(state_tensors, root_states.legal_moves)
        
        # Initialize root children with priors
        for env_id, root in enumerate(roots):
            for action in range(7):
                if root_states.legal_moves[env_id, action]:
                    root.children[action] = BatchMCTSNode(
                        parent=root,
                        action=action,
                        prior=policies[env_id, action].item()
                    )
        
        # Run simulations
        for sim in range(self.num_simulations):
            # Select paths for all roots
            paths = []
            leaf_env_ids = []
            
            for env_id, root in enumerate(roots):
                path = self._select_path(root)
                if path:
                    paths.append(path)
                    leaf_env_ids.append(env_id)
            
            if not paths:
                break
            
            # Batch evaluate and backup
            self._batch_evaluate_and_backup_vectorized(paths, leaf_env_ids, root_states)
        
        # Return visit counts for each root
        all_visits = []
        for root in roots:
            visits = {}
            for action, child in root.children.items():
                visits[action] = child.visits
            all_visits.append(visits)
        
        return all_visits
    
    def search(self, root_state_single) -> Dict[int, float]:
        """Run MCTS on a single state (compatibility with non-vectorized code)"""
        # Convert single state to vectorized format
        self.vec_env.reset()
        self.vec_env.boards[0] = root_state_single.board
        self.vec_env.current_players[0] = root_state_single.current_player
        self.vec_env.move_counts[0] = root_state_single.move_count
        self.vec_env.is_terminal[0] = root_state_single.is_terminal
        
        if root_state_single.is_terminal:
            if root_state_single.winner is not None:
                self.vec_env.winners[0] = root_state_single.winner
        
        vec_state = self.vec_env.get_state()
        
        # Use only the first environment
        vec_state.boards = vec_state.boards[:1]
        vec_state.current_players = vec_state.current_players[:1]
        vec_state.move_counts = vec_state.move_counts[:1]
        vec_state.winners = vec_state.winners[:1]
        vec_state.is_terminal = vec_state.is_terminal[:1]
        vec_state.legal_moves = vec_state.legal_moves[:1]
        
        # Run vectorized search
        visits_list = self.search_vectorized(vec_state)
        return visits_list[0]
    
    def _select_path(self, root: BatchMCTSNode) -> List[BatchMCTSNode]:
        """Select a path from root to leaf"""
        path = []
        node = root
        
        while True:
            path.append(node)
            
            if node.is_terminal:
                return path
            
            if not node.is_expanded:
                return path
            
            # Select best child
            best_score = -float('inf')
            best_child = None
            
            for child in node.children.values():
                score = child.ucb_score(self.c_puct)
                if score > best_score:
                    best_score = score
                    best_child = child
            
            if best_child is None:
                return path
            
            # Apply virtual loss
            best_child.virtual_loss += 1
            
            node = best_child
    
    def _batch_evaluate_and_backup_vectorized(
        self,
        paths: List[List[BatchMCTSNode]],
        leaf_env_ids: List[int],
        root_states: VectorizedC4State
    ):
        """Evaluate multiple leaf nodes using vectorized environment"""
        if not paths:
            return
        
        # Reset vectorized environment with root states
        self.vec_env.boards[:len(leaf_env_ids)] = root_states.boards[leaf_env_ids]
        self.vec_env.current_players[:len(leaf_env_ids)] = root_states.current_players[leaf_env_ids]
        self.vec_env.move_counts[:len(leaf_env_ids)] = root_states.move_counts[leaf_env_ids]
        self.vec_env.winners[:len(leaf_env_ids)] = root_states.winners[leaf_env_ids]
        self.vec_env.is_terminal[:len(leaf_env_ids)] = root_states.is_terminal[leaf_env_ids]
        
        # Apply actions along paths to reach leaf states
        for i, path in enumerate(paths):
            self.vec_env.boards[i] = root_states.boards[leaf_env_ids[i]]
            self.vec_env.current_players[i] = root_states.current_players[leaf_env_ids[i]]
            
            # Apply all actions in the path (except root)
            for node in path[1:]:
                if node.action is not None and not self.vec_env.is_terminal[i]:
                    # Make the move
                    col = node.action
                    col_values = self.vec_env.boards[i, :, col]
                    empty_rows = (col_values == 0).nonzero(as_tuple=True)[0]
                    
                    if len(empty_rows) > 0:
                        row = empty_rows[-1].item()
                        self.vec_env.boards[i, row, col] = self.vec_env.current_players[i]
                        self.vec_env.current_players[i] = 3 - self.vec_env.current_players[i]
                        self.vec_env.move_counts[i] += 1
        
        # Check for terminal states
        self.vec_env._check_winners_batch()
        board_full = (self.vec_env.boards[:len(paths)] != 0).all(dim=(1, 2))
        self.vec_env.is_terminal[:len(paths)] = (self.vec_env.winners[:len(paths)] != 0) | board_full
        
        # Get current states
        current_states = self.vec_env.get_state()
        
        # Separate terminal and non-terminal
        terminal_mask = current_states.is_terminal[:len(paths)]
        non_terminal_indices = (~terminal_mask).nonzero(as_tuple=True)[0]
        
        # Initialize values
        values = torch.zeros(len(paths), device=self.device)
        
        # Handle terminal states
        for i in range(len(paths)):
            if terminal_mask[i]:
                leaf = paths[i][-1]
                leaf.is_terminal = True
                
                # Calculate value from the perspective of the player who just moved
                winner = current_states.winners[i].item()
                if winner == 0:
                    values[i] = 0.0  # Draw
                else:
                    # The value is from the perspective of the player to move
                    # If the current player is the winner, value is -1 (opponent won)
                    # If the current player is not the winner, value is 1 (we won)
                    if current_states.current_players[i].item() == winner:
                        values[i] = -1.0
                    else:
                        values[i] = 1.0
                
                leaf.terminal_value = values[i]
        
        # Evaluate non-terminal states
        if len(non_terminal_indices) > 0:
            non_terminal_states = current_states.boards[non_terminal_indices]
            state_tensors = (non_terminal_states.flatten(1) + 1).long()
            legal_moves = current_states.legal_moves[non_terminal_indices]
            
            policies, pred_values = self.get_policy_value_batch(state_tensors, legal_moves)
            
            # Update values and expand nodes
            for idx, i in enumerate(non_terminal_indices):
                values[i] = pred_values[idx].item()
                leaf = paths[i][-1]
                
                # Expand the leaf node
                if not leaf.is_expanded:
                    for action in range(7):
                        if legal_moves[idx, action]:
                            leaf.children[action] = BatchMCTSNode(
                                parent=leaf,
                                action=action,
                                prior=policies[idx, action].item()
                            )
        
        # Backup values through paths
        for i, path in enumerate(paths):
            current_value = values[i]
            
            for node in reversed(path):
                # Remove virtual loss
                if node != path[0]:  # Don't remove virtual loss from root
                    node.virtual_loss -= 1
                
                # Update statistics
                node.visits += 1
                node.value_sum += current_value
                current_value = -current_value  # Flip for opponent
    
    def get_action_probabilities(
        self,
        state,  # Can be single ConnectFourState or VectorizedC4State
        temperature: float = None
    ):
        """Get action probabilities for a state"""
        if temperature is None:
            temperature = self.temperature
        
        # Check if it's a vectorized state
        if hasattr(state, 'n_envs'):
            # Vectorized
            visits_list = self.search_vectorized(state)
            
            # Convert visits to probabilities for each environment
            all_probs = []
            for visits in visits_list:
                visits_tensor = torch.zeros(7)
                for action, count in visits.items():
                    visits_tensor[action] = count
                
                if temperature == 0:
                    probs = torch.zeros(7)
                    probs[torch.argmax(visits_tensor)] = 1.0
                else:
                    visits_tensor = visits_tensor ** (1.0 / temperature)
                    probs = visits_tensor / visits_tensor.sum()
                
                all_probs.append(probs)
            
            return torch.stack(all_probs) if len(all_probs) > 1 else all_probs[0]
        else:
            # Single state
            visits = self.search(state)
            
            visits_tensor = torch.zeros(7)
            for action, count in visits.items():
                visits_tensor[action] = count
            
            if temperature == 0:
                probs = torch.zeros(7)
                probs[torch.argmax(visits_tensor)] = 1.0
            else:
                visits_tensor = visits_tensor ** (1.0 / temperature)
                probs = visits_tensor / visits_tensor.sum()
            
            return probs
    
    def select_action(
        self,
        state,
        deterministic: bool = False
    ) -> int:
        """Select an action for a state"""
        temp = 0 if deterministic else self.temperature
        probs = self.get_action_probabilities(state, temperature=temp)
        
        if deterministic:
            return torch.argmax(probs).item()
        else:
            return torch.multinomial(probs, 1).item()