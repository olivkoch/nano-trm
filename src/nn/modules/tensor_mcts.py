"""
Fully Optimized Tensor-Based MCTS for Connect Four

All operations are vectorized for maximum GPU utilization.
No Python loops over individual games/simulations.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class TensorMCTSConfig:
    """Configuration for tensor MCTS"""
    n_actions: int = 7
    max_nodes_per_tree: int = 2000
    max_depth: int = 50
    c_puct: float = 1.0
    virtual_loss_weight: float = 3.0
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25


class TensorMCTS:
    """
    Fully vectorized MCTS with zero Python loops over games/simulations.
    """
    
    def __init__(
        self,
        model,
        n_trees: int,
        config: Optional[TensorMCTSConfig] = None,
        device: str = "cpu"
    ):
        self.model = model
        self.n_trees = n_trees
        self.config = config or TensorMCTSConfig()
        self.device = device
        
        self._allocate_tensors()
    
    def _allocate_tensors(self):
        """Pre-allocate all tensor storage"""
        n = self.n_trees
        max_nodes = self.config.max_nodes_per_tree
        n_actions = self.config.n_actions
        
        # Node statistics
        self.visits = torch.zeros(n, max_nodes, device=self.device)
        self.total_value = torch.zeros(n, max_nodes, device=self.device)
        self.virtual_loss = torch.zeros(n, max_nodes, device=self.device)
        
        # Prior probabilities
        self.priors = torch.zeros(n, max_nodes, n_actions, device=self.device)
        
        # Tree structure
        self.children = torch.full((n, max_nodes, n_actions), -1, 
                                   dtype=torch.long, device=self.device)
        self.parent = torch.full((n, max_nodes), -1, dtype=torch.long, device=self.device)
        self.parent_action = torch.full((n, max_nodes), -1, dtype=torch.long, device=self.device)
        
        # Board states (use long for discrete game states)
        self.states = torch.zeros(n, max_nodes, 6, 7, dtype=torch.long, device=self.device)
        self.has_state = torch.zeros(n, max_nodes, dtype=torch.bool, device=self.device)
        
        # Current player at each node
        self.current_player = torch.zeros(n, max_nodes, dtype=torch.long, device=self.device)
        
        # Terminal info
        self.is_terminal = torch.zeros(n, max_nodes, dtype=torch.bool, device=self.device)
        self.terminal_value = torch.zeros(n, max_nodes, device=self.device)
        
        # Node allocation tracking
        self.next_node_idx = torch.ones(n, dtype=torch.long, device=self.device)
        
        # Batch indices
        self.batch_idx = torch.arange(n, device=self.device)
    
    def reset(self, root_states: torch.Tensor, root_policies: torch.Tensor, 
              legal_masks: torch.Tensor, root_players: torch.Tensor):
        """Initialize trees with root states."""
        # Zero out everything
        self.visits.zero_()
        self.total_value.zero_()
        self.virtual_loss.zero_()
        self.priors.zero_()
        self.children.fill_(-1)
        self.parent.fill_(-1)
        self.parent_action.fill_(-1)
        self.states.zero_()
        self.has_state.zero_()
        self.current_player.zero_()
        self.is_terminal.zero_()
        self.terminal_value.zero_()
        self.next_node_idx.fill_(1)
        
        # Set root node (ensure long dtype for discrete game state)
        self.states[:, 0] = root_states.long()
        self.has_state[:, 0] = True
        self.current_player[:, 0] = root_players
        
        # Apply legal mask and normalize
        masked_policy = root_policies * legal_masks.float()
        masked_policy = masked_policy / (masked_policy.sum(dim=1, keepdim=True) + 1e-8)
        
        # Add Dirichlet noise
        if self.config.exploration_fraction > 0:
            if "mps" in str(self.device):
                noise_np = np.random.dirichlet(
                    [self.config.dirichlet_alpha] * self.config.n_actions, 
                    size=self.n_trees
                )
                noise = torch.from_numpy(noise_np).float().to(self.device)
            else:
                noise = torch.distributions.Dirichlet(
                    torch.full((self.config.n_actions,), self.config.dirichlet_alpha, 
                              device=self.device)
                ).sample((self.n_trees,))
            noise = noise * legal_masks.float()
            noise = noise / (noise.sum(dim=1, keepdim=True) + 1e-8)
            
            masked_policy = ((1 - self.config.exploration_fraction) * masked_policy + 
                            self.config.exploration_fraction * noise)
        
        self.priors[:, 0] = masked_policy
        
        # Expand root
        self._expand_roots(legal_masks, root_players)
    
    def _expand_roots(self, legal_masks: torch.Tensor, root_players: torch.Tensor):
        """Expand root nodes - vectorized"""
        child_player = 3 - root_players
        
        for action in range(self.config.n_actions):
            is_legal = legal_masks.bool()[:, action]
            child_idx = self.next_node_idx.clone()
            
            self.children[:, 0, action] = torch.where(
                is_legal, child_idx, torch.full_like(child_idx, -1)
            )
            
            self.parent[self.batch_idx, child_idx] = torch.where(
                is_legal, 
                torch.zeros_like(child_idx),
                self.parent[self.batch_idx, child_idx]
            )
            self.parent_action[self.batch_idx, child_idx] = torch.where(
                is_legal,
                torch.full_like(child_idx, action),
                self.parent_action[self.batch_idx, child_idx]
            )
            
            self.current_player[self.batch_idx, child_idx] = torch.where(
                is_legal,
                child_player,
                self.current_player[self.batch_idx, child_idx]
            )
            
            self.next_node_idx += is_legal.long()
    
    def run_simulations(self, num_simulations: int, parallel_sims: int = 16,
                       verbose: bool = False) -> torch.Tensor:
        """
        Run MCTS simulations with batched processing.
        
        Key insight from AlphaZero: Virtual losses prevent parallel simulations
        from exploring the same path. They must be:
        1. Added during selection (to discourage re-selection)
        2. Completely reverted after backup (not just decremented)
        """
        num_batches = (num_simulations + parallel_sims - 1) // parallel_sims
        
        for batch_idx in range(num_batches):
            sims_this_batch = min(parallel_sims, num_simulations - batch_idx * parallel_sims)
            
            # Select all leaves for this batch
            all_leaves = []
            all_paths = []
            all_path_lengths = []
            all_vl_applied = []  # Track where virtual loss was applied
            
            for sim in range(sims_this_batch):
                leaves, paths, path_lengths = self._batch_select_leaves()
                all_leaves.append(leaves)
                all_paths.append(paths)
                all_path_lengths.append(path_lengths)
                # Track which nodes had VL applied (all nodes in the path)
                all_vl_applied.append((paths.clone(), path_lengths.clone()))
            
            # Stack and process together
            leaves_tensor = torch.stack(all_leaves, dim=0)
            paths_tensor = torch.stack(all_paths, dim=0)
            lengths_tensor = torch.stack(all_path_lengths, dim=0)
            
            # Process batch of simulations
            self._materialize_states_vectorized(leaves_tensor, paths_tensor, lengths_tensor)
            self._batch_evaluate_and_expand(leaves_tensor)
            self._batch_backup_vectorized(leaves_tensor, paths_tensor, lengths_tensor)
            
            # CRITICAL: Revert virtual losses after backup (not just decrement!)
            self._revert_virtual_losses(all_vl_applied)
        
        return self._get_visit_distributions()
    
    def _batch_select_leaves(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select one leaf from each tree using PUCT with virtual loss.
        
        Note: Virtual losses are added to discourage parallel simulations from
        selecting the same path. In AlphaZero, child Q values are evaluated from
        the opponent's perspective, so we use -Q when computing PUCT scores.
        """
        max_depth = self.config.max_depth
        vl_weight = self.config.virtual_loss_weight
        c_puct = self.config.c_puct
        
        current = torch.zeros(self.n_trees, dtype=torch.long, device=self.device)
        paths = torch.zeros(self.n_trees, max_depth, dtype=torch.long, device=self.device)
        path_lengths = torch.zeros(self.n_trees, dtype=torch.long, device=self.device)
        
        for depth in range(max_depth):
            paths[:, depth] = current
            path_lengths += 1
            
            current_children = self.children[self.batch_idx, current]
            has_children = (current_children >= 0).any(dim=1)
            is_term = self.is_terminal[self.batch_idx, current]
            
            should_continue = has_children & ~is_term
            if not should_continue.any():
                break
            
            self.virtual_loss[self.batch_idx, current] += vl_weight * should_continue.float()
            
            valid_children = current_children >= 0
            safe_indices = current_children.clamp(min=0)
            
            tree_expand = self.batch_idx.unsqueeze(1).expand(-1, self.config.n_actions)
            
            child_visits = self.visits[tree_expand, safe_indices]
            child_values = self.total_value[tree_expand, safe_indices]
            child_vl = self.virtual_loss[tree_expand, safe_indices]
            child_priors = self.priors[self.batch_idx, current]
            
            effective_visits = child_visits + child_vl
            effective_value = child_values - child_vl
            
            # Child Q values are from opponent's perspective, so flip sign
            # This ensures we select moves that are good for the current player
            Q = torch.where(
                effective_visits > 0,
                -effective_value / effective_visits,  # Note the negative sign!
                torch.zeros_like(effective_value)
            )
            
            parent_visits = self.visits[self.batch_idx, current].unsqueeze(1)
            U = c_puct * child_priors * torch.sqrt(parent_visits + 1) / (1 + effective_visits)
            
            puct = Q + U
            puct = torch.where(valid_children, puct, torch.full_like(puct, -1e9))
            
            best_actions = puct.argmax(dim=1)
            next_nodes = self.children[self.batch_idx, current, best_actions]
            
            current = torch.where(should_continue, next_nodes, current)
        
        self.virtual_loss[self.batch_idx, current] += vl_weight
        
        return current, paths, path_lengths
    
    def _revert_virtual_losses(self, vl_applied_list):
        """
        Completely revert virtual losses after backup.
        
        This is critical for correctness: virtual loss is a temporary discourage
        mechanism, not a permanent penalty. After backup completes, we must
        fully restore the original statistics.
        """
        vl_weight = self.config.virtual_loss_weight
        
        for paths, path_lengths in vl_applied_list:
            max_len = path_lengths.max().item()
            
            for depth in range(max_len):
                active = path_lengths > depth
                if not active.any():
                    break
                
                node_idx = paths[:, depth]
                self.virtual_loss[self.batch_idx[active], node_idx[active]] -= vl_weight
        
        # Ensure no negative virtual losses (shouldn't happen, but safety check)
        self.virtual_loss.clamp_(min=0)
    
    def _make_moves_vectorized(self, boards: torch.Tensor, actions: torch.Tensor,
                               players: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorized move making - NO PYTHON LOOPS.
        
        Args:
            boards: (batch, 6, 7) board states
            actions: (batch,) column indices
            players: (batch,) player numbers (1 or 2)
            active: (batch,) boolean mask of which boards to update
        """
        new_boards = boards.clone()
        
        if not active.any():
            return new_boards
        
        # Work only with active boards
        active_indices = active.nonzero(as_tuple=True)[0]
        active_boards = new_boards[active_indices]
        active_actions = actions[active_indices]
        active_players = players[active_indices]
        n_active = active_indices.shape[0]
        
        # For each board, extract the selected column
        # Shape: (n_active, 6)
        batch_range = torch.arange(n_active, device=self.device)
        columns = active_boards[batch_range, :, active_actions]
        
        # Find the lowest empty row in each column (highest index where value is 0)
        # empty_mask: (n_active, 6) - True where cell is empty
        empty_mask = (columns == 0)
        
        # Create row indices (0 to 5) for each position
        row_indices = torch.arange(6, device=self.device).unsqueeze(0).expand(n_active, -1)
        
        # Set non-empty positions to -1, then find max (gives us lowest empty row)
        valid_rows = torch.where(empty_mask, row_indices, torch.tensor(-1, device=self.device))
        target_rows = valid_rows.max(dim=1)[0]  # (n_active,)
        
        # Only update where we found a valid row (target_row >= 0)
        valid_moves = target_rows >= 0
        
        if valid_moves.any():
            valid_batch = batch_range[valid_moves]
            valid_rows_idx = target_rows[valid_moves]
            valid_cols = active_actions[valid_moves]
            valid_players_val = active_players[valid_moves]
            
            # Place pieces using advanced indexing
            active_boards[valid_batch, valid_rows_idx, valid_cols] = valid_players_val
        
        # Write back to original tensor
        new_boards[active_indices] = active_boards
        return new_boards
    
    def _materialize_states_vectorized(self, leaves: torch.Tensor, paths: torch.Tensor,
                                      path_lengths: torch.Tensor):
        """
        Fully vectorized state materialization.
        
        Args:
            leaves: (n_sims, n_trees) leaf node indices
            paths: (n_sims, n_trees, max_depth) paths taken
            path_lengths: (n_sims, n_trees) length of each path
        """
        n_sims, n_trees = leaves.shape
        max_depth = paths.shape[2]
        
        # Find which leaves need states
        flat_tree_idx = self.batch_idx.unsqueeze(0).expand(n_sims, -1)
        flat_leaves = leaves
        needs_state = ~self.has_state[flat_tree_idx, flat_leaves]  # (n_sims, n_trees)
        
        if not needs_state.any():
            return
        
        # Start from root states - broadcast to all simulations
        current_states = self.states[:, 0].unsqueeze(0).expand(n_sims, -1, -1, -1).clone()
        current_players = self.current_player[:, 0].unsqueeze(0).expand(n_sims, -1).clone()
        
        # Process each depth level
        for depth in range(1, max_depth):
            # Active mask: which (sim, tree) pairs are at this depth and need states
            active = (path_lengths > depth) & needs_state
            
            if not active.any():
                break
            
            # Get node indices for this depth
            node_indices = paths[:, :, depth]  # (n_sims, n_trees)
            
            # Get actions for these nodes
            actions = self.parent_action[flat_tree_idx, node_indices]  # (n_sims, n_trees)
            
            # Flatten everything to apply moves in one batch
            flat_states = current_states.reshape(n_sims * n_trees, 6, 7)
            flat_actions = actions.reshape(-1)
            flat_players = current_players.reshape(-1)
            flat_active = active.reshape(-1)
            
            # Apply all moves at once
            flat_states = self._make_moves_vectorized(
                flat_states, flat_actions, flat_players, flat_active
            )
            
            # Reshape back
            current_states = flat_states.reshape(n_sims, n_trees, 6, 7)
            
            # Cache states where needed - use scatter for efficiency
            if active.any():
                active_sims, active_trees = active.nonzero(as_tuple=True)
                active_nodes = node_indices[active_sims, active_trees]
                self.states[active_trees, active_nodes] = current_states[active_sims, active_trees]
                self.has_state[active_trees, active_nodes] = True
            
            # Switch players
            current_players = torch.where(active, 3 - current_players, current_players)
    
    def _batch_evaluate_and_expand(self, leaves: torch.Tensor):
        """
        Evaluate leaf positions and expand non-terminal nodes.
        
        Important: If a node was selected multiple times despite virtual losses
        (can happen in parallel search), we should only expand it once.
        """
        n_sims, n_trees = leaves.shape
        
        flat_leaves = leaves.flatten()
        flat_tree_idx = self.batch_idx.unsqueeze(0).expand(n_sims, -1).flatten()
        
        leaf_states = self.states[flat_tree_idx, flat_leaves]
        leaf_players = self.current_player[flat_tree_idx, flat_leaves]
        
        already_terminal = self.is_terminal[flat_tree_idx, flat_leaves]
        already_expanded = (self.children[flat_tree_idx, flat_leaves] >= 0).any(dim=1)
        
        full_values = torch.zeros(flat_leaves.shape[0], device=self.device)
        if already_terminal.any():
            full_values[already_terminal] = self.terminal_value[
                flat_tree_idx[already_terminal], 
                flat_leaves[already_terminal]
            ]
        
        full_is_term = already_terminal.clone()
        full_policies = torch.zeros(flat_leaves.shape[0], self.config.n_actions, 
                                device=self.device)
        
        needs_eval = ~already_terminal & ~already_expanded  # Skip if already expanded!
        
        if needs_eval.any():
            eval_states = leaf_states[needs_eval]
            eval_players = leaf_players[needs_eval]
            
            is_term, winners = self._check_terminal_batch(eval_states)
            
            # Terminal values from perspective of current player
            term_values = torch.where(
                winners == 0,
                torch.zeros(eval_states.shape[0], device=self.device),
                torch.where(
                    winners == eval_players,
                    torch.ones(eval_states.shape[0], device=self.device),
                    -torch.ones(eval_states.shape[0], device=self.device)
                )
            )
            
            # Neural network evaluation for non-terminal
            non_term = ~is_term
            nn_values = torch.zeros(eval_states.shape[0], device=self.device)
            nn_policies = torch.zeros(eval_states.shape[0], self.config.n_actions, 
                                    device=self.device)
            
            if non_term.any():
                with torch.no_grad():
                    policies, values = self.model.forward(
                        eval_states[non_term].flatten(start_dim=1).float(),  # Convert to float for NN
                        eval_players[non_term]
                    )
                    policies = policies.float()
                    values = values.float()
                nn_values[non_term] = values.squeeze(-1) if values.dim() > 1 else values
                nn_policies[non_term] = policies
            
            final_values = torch.where(is_term, term_values, nn_values)
            
            full_is_term[needs_eval] = is_term
            full_values[needs_eval] = final_values
            full_policies[needs_eval] = nn_policies
            
            self.is_terminal[flat_tree_idx[needs_eval], flat_leaves[needs_eval]] = is_term
            
            newly_terminal = needs_eval.clone()
            newly_terminal[needs_eval] = is_term
            if newly_terminal.any():
                self.terminal_value[
                    flat_tree_idx[newly_terminal], 
                    flat_leaves[newly_terminal]
                ] = term_values[is_term]
            
            # Expand non-terminal leaves that aren't already expanded
            expand_mask = needs_eval & ~full_is_term
            if expand_mask.any():
                self._expand_nodes(
                    flat_tree_idx[expand_mask],
                    flat_leaves[expand_mask],
                    full_policies[expand_mask],
                    leaf_states[expand_mask],
                    leaf_players[expand_mask]
                )
        
        # For already-expanded nodes, use their stored values
        if already_expanded.any() and not already_terminal[already_expanded].all():
            # These nodes have been evaluated before, use mean Q value
            expanded_nodes = already_expanded & ~already_terminal
            if expanded_nodes.any():
                node_visits = self.visits[flat_tree_idx[expanded_nodes], flat_leaves[expanded_nodes]]
                node_values = self.total_value[flat_tree_idx[expanded_nodes], flat_leaves[expanded_nodes]]
                full_values[expanded_nodes] = torch.where(
                    node_visits > 0,
                    node_values / node_visits,
                    torch.zeros_like(node_values)
                )
        
        self._last_leaf_values = full_values.view(n_sims, n_trees)
    
    def _expand_nodes(self, tree_indices: torch.Tensor, node_indices: torch.Tensor,
                     policies: torch.Tensor, states: torch.Tensor,
                     node_players: torch.Tensor):
        """Expand nodes with children - vectorized"""
        n_nodes = tree_indices.shape[0]
        
        legal_masks = states[:, 0, :] == 0
        
        masked_policies = policies * legal_masks.float()
        masked_policies = masked_policies / (masked_policies.sum(dim=1, keepdim=True) + 1e-8)
        
        self.priors[tree_indices, node_indices] = masked_policies
        
        child_players = 3 - node_players
        
        for action in range(self.config.n_actions):
            is_legal = legal_masks[:, action]
            if not is_legal.any():
                continue
            
            legal_tree_idx = tree_indices[is_legal]
            legal_node_idx = node_indices[is_legal]
            legal_child_players = child_players[is_legal]
            child_idx = self.next_node_idx[legal_tree_idx]
            
            self.children[legal_tree_idx, legal_node_idx, action] = child_idx
            self.parent[legal_tree_idx, child_idx] = legal_node_idx
            self.parent_action[legal_tree_idx, child_idx] = action
            self.current_player[legal_tree_idx, child_idx] = legal_child_players
            
            self.next_node_idx[legal_tree_idx] += 1
    
    def _batch_backup_vectorized(self, leaves: torch.Tensor, paths: torch.Tensor,
                                 path_lengths: torch.Tensor):
        """
        Fully vectorized backup using scatter operations.
        
        Propagates values up the tree, flipping signs at each level because
        in two-player zero-sum games, a good position for one player is bad
        for the opponent.
        
        Note: Virtual loss is NOT decremented here - it's reverted separately
        in _revert_virtual_losses() after all backups complete.
        
        Args:
            leaves: (n_sims, n_trees) leaf nodes
            paths: (n_sims, n_trees, max_depth) paths
            path_lengths: (n_sims, n_trees) path lengths
        """
        n_sims, n_trees = leaves.shape
        max_depth = paths.shape[2]
        
        values = self._last_leaf_values.clone()  # (n_sims, n_trees)
        
        # Process each depth level
        for depth in range(max_depth - 1, -1, -1):
            # Active mask
            active = path_lengths > depth  # (n_sims, n_trees)
            
            if not active.any():
                continue
            
            # Get node indices at this depth
            node_idx = paths[:, :, depth]  # (n_sims, n_trees)
            
            # Flatten active updates
            active_sims, active_trees = active.nonzero(as_tuple=True)
            active_nodes = node_idx[active_sims, active_trees]
            active_values = values[active_sims, active_trees]
            
            # Use index_add_ for efficient scatter-add
            # Create linear indices: tree * max_nodes + node
            linear_indices = active_trees * self.config.max_nodes_per_tree + active_nodes
            
            # Update visits
            ones = torch.ones_like(active_values)
            self.visits.view(-1).index_add_(0, linear_indices, ones)
            
            # Update total values
            self.total_value.view(-1).index_add_(0, linear_indices, active_values)
            
            # Flip values for opponent perspective
            values = -values
    
    def _check_terminal_batch(self, boards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check for terminal states in batch - fully vectorized."""
        batch_size = boards.shape[0]
        device = boards.device
        
        board_full = ~(boards[:, 0, :] == 0).any(dim=1)
        winners = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        
        # Horizontal
        for row in range(6):
            for col in range(4):
                window = boards[:, row, col:col+4]
                player = window[:, 0]
                all_same = (window == player.unsqueeze(1)).all(dim=1)
                valid = player != 0
                has_win = all_same & valid
                winners = torch.where(has_win, player, winners)
        
        # Vertical
        for row in range(3):
            for col in range(7):
                window = boards[:, row:row+4, col]
                player = window[:, 0]
                all_same = (window == player.unsqueeze(1)).all(dim=1)
                valid = player != 0
                has_win = all_same & valid
                winners = torch.where(has_win, player, winners)
        
        # Diagonal down-right
        for row in range(3):
            for col in range(4):
                cells = torch.stack([
                    boards[:, row, col],
                    boards[:, row+1, col+1],
                    boards[:, row+2, col+2],
                    boards[:, row+3, col+3]
                ], dim=1)
                player = cells[:, 0]
                all_same = (cells == player.unsqueeze(1)).all(dim=1)
                valid = player != 0
                has_win = all_same & valid
                winners = torch.where(has_win, player, winners)
        
        # Diagonal down-left
        for row in range(3):
            for col in range(4):
                cells = torch.stack([
                    boards[:, row, col+3],
                    boards[:, row+1, col+2],
                    boards[:, row+2, col+1],
                    boards[:, row+3, col]
                ], dim=1)
                player = cells[:, 0]
                all_same = (cells == player.unsqueeze(1)).all(dim=1)
                valid = player != 0
                has_win = all_same & valid
                winners = torch.where(has_win, player, winners)
        
        winners = torch.where(board_full & (winners == -1), 
                             torch.zeros_like(winners), winners)
        
        is_terminal = winners >= 0
        
        return is_terminal, winners
    
    def _get_visit_distributions(self) -> torch.Tensor:
        """Get normalized visit distributions from root"""
        root_children = self.children[:, 0]
        valid = root_children >= 0
        safe_idx = root_children.clamp(min=0)
        
        tree_expand = self.batch_idx.unsqueeze(1).expand(-1, self.config.n_actions)
        child_visits = self.visits[tree_expand, safe_idx]
        child_visits = torch.where(valid, child_visits, torch.zeros_like(child_visits))
        
        total = child_visits.sum(dim=1, keepdim=True)
        distributions = child_visits / (total + 1e-8)
        
        return distributions
    
    def get_action_probs(self, temperature: float = 1.0) -> torch.Tensor:
        """Get action probabilities with temperature"""
        distributions = self._get_visit_distributions()
        
        if temperature == 0:
            probs = torch.zeros_like(distributions)
            probs.scatter_(1, distributions.argmax(dim=1, keepdim=True), 1.0)
            return probs
        else:
            visits_temp = distributions ** (1.0 / temperature)
            return visits_temp / (visits_temp.sum(dim=1, keepdim=True) + 1e-8)


class TensorMCTSWrapper:
    """
    Wrapper providing interface compatible with existing code.
    """
    
    def __init__(
        self,
        model,
        c_puct: float = 1.0,
        num_simulations: int = 800,
        parallel_simulations: int = 16,
        dirichlet_alpha: float = 0.3,
        exploration_fraction: float = 0.25,
        virtual_loss_value: float = 3.0,
        device: str = "cpu"
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.parallel_simulations = parallel_simulations
        self.device = device
        
        self.config = TensorMCTSConfig(
            c_puct=c_puct,
            virtual_loss_weight=virtual_loss_value,
            dirichlet_alpha=dirichlet_alpha,
            exploration_fraction=exploration_fraction
        )
        
        self._cached_mcts = {}
    
    def get_action_probs_batch_parallel(
        self,
        boards: List[torch.Tensor],
        legal_moves_list: List[torch.Tensor],
        temperature: float = 1.0,
        current_players: torch.Tensor = None,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run MCTS on batch of positions.
        
        Args:
            boards: List of (6, 7) board tensors
            legal_moves_list: List of (7,) legal move masks
            temperature: Temperature for action selection
            current_players: (n_positions,) tensor of current players (1 or 2)
        """
        n_positions = len(boards)
        
        if n_positions not in self._cached_mcts:
            self._cached_mcts[n_positions] = TensorMCTS(
                model=self.model,
                n_trees=n_positions,
                config=self.config,
                device=self.device
            )
        
        mcts = self._cached_mcts[n_positions]
        mcts.model = self.model
        
        # Stack inputs and ensure correct dtypes
        boards_tensor = torch.stack(boards).long()  # Boards should be long (discrete values)
        if boards_tensor.dim() == 2:
            boards_tensor = boards_tensor.view(n_positions, 6, 7)
        legal_masks = torch.stack(legal_moves_list)
        
        # Handle current_players
        if current_players is None:
            p1_count = (boards_tensor == 1).sum(dim=(1, 2))
            p2_count = (boards_tensor == 2).sum(dim=(1, 2))
            current_players = torch.where(
                p1_count == p2_count,
                torch.ones(n_positions, dtype=torch.long, device=self.device),
                torch.full((n_positions,), 2, dtype=torch.long, device=self.device)
            )
        else:
            current_players = current_players.to(self.device)
        
        # Get initial policies from model
        with torch.no_grad():
            policies, _ = self.model.forward(
                boards_tensor.flatten(start_dim=1).float(),  # Convert to float for NN
                current_players
            )
        
        # Initialize trees
        mcts.reset(boards_tensor, policies, legal_masks, current_players)
        
        # Run simulations
        start_time = time.time()
        visit_distributions = mcts.run_simulations(
            self.num_simulations, 
            self.parallel_simulations,
            verbose=verbose
        )
        elapsed = time.time() - start_time
        
        if verbose:
            sims_per_sec = (self.num_simulations * n_positions) / elapsed
            print(f"MCTS: {self.num_simulations} sims x {n_positions} positions "
                  f"in {elapsed:.3f}s ({sims_per_sec:.0f} sims/sec)")
        
        action_probs = mcts.get_action_probs(temperature)
        
        return visit_distributions, action_probs


def benchmark_comparison():
    """Compare optimized vs original implementation"""
    print("Benchmarking Optimized TensorMCTS")
    print("="*60)
    
    class DummyModel:
        def __init__(self, device):
            self.device = device
            
        def forward(self, boards, current_players):
            batch_size = boards.shape[0]
            policies = torch.ones(batch_size, 7, device=self.device) / 7
            values = torch.zeros(batch_size, device=self.device)
            return policies, values
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    model = DummyModel(device)
    
    # Test different batch sizes
    for n_positions in [1, 4, 16, 64]:
        print(f"\nBatch size: {n_positions}")
        print("-" * 40)
        
        boards = [torch.zeros(6, 7, device=device) for _ in range(n_positions)]
        legal = [torch.ones(7, dtype=torch.bool, device=device) for _ in range(n_positions)]
        players = torch.ones(n_positions, dtype=torch.long, device=device)
        
        wrapper = TensorMCTSWrapper(
            model=model,
            num_simulations=100,
            parallel_simulations=16,
            device=device
        )
        
        # Warmup
        _, _ = wrapper.get_action_probs_batch_parallel(boards, legal, current_players=players)
        
        # Timed runs
        times = []
        for _ in range(5):
            start = time.time()
            _, _ = wrapper.get_action_probs_batch_parallel(boards, legal, current_players=players)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        total_sims = 100 * n_positions
        sims_per_sec = total_sims / avg_time
        
        print(f"  Time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"  Throughput: {sims_per_sec:.0f} simulations/sec")
        print(f"  Per position: {avg_time/n_positions*1000:.1f}ms")


if __name__ == "__main__":
    benchmark_comparison()