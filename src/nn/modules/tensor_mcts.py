"""
Tensor-Based MCTS for Connect Four

Instead of Python objects (MCTSNode), entire trees are stored as contiguous tensors.
This enables fully vectorized selection/expansion/backup across all trees simultaneously.

Key insight: MCTS tree operations are essentially sparse matrix operations.
By pre-allocating fixed-size tensors, we eliminate:
- Python object overhead
- Dictionary lookups
- Memory allocation during search
- GC pressure

Expected speedup: 10-100x for tree operations (NN forward pass becomes the bottleneck)
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
    max_nodes_per_tree: int = 2000  # ~50 moves × ~20 sims × safety margin
    max_depth: int = 50
    c_puct: float = 1.0
    virtual_loss_weight: float = 3.0
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25


class TensorMCTS:
    """
    Fully tensorized MCTS for parallel game tree search.
    
    All trees are stored in contiguous tensors of shape (n_trees, max_nodes, ...).
    This enables vectorized operations across all trees simultaneously.
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
        
        # Prior probabilities at each node (for selecting among children)
        self.priors = torch.zeros(n, max_nodes, n_actions, device=self.device)
        
        # Tree structure: children[tree, node, action] = child_node_index (-1 if none)
        self.children = torch.full((n, max_nodes, n_actions), -1, 
                                   dtype=torch.long, device=self.device)
        
        # Parent tracking for backup
        self.parent = torch.full((n, max_nodes), -1, dtype=torch.long, device=self.device)
        self.parent_action = torch.full((n, max_nodes), -1, dtype=torch.long, device=self.device)
        
        # Board states (6x7 for Connect Four)
        self.states = torch.zeros(n, max_nodes, 6, 7, device=self.device)
        self.has_state = torch.zeros(n, max_nodes, dtype=torch.bool, device=self.device)
        
        # Terminal info
        self.is_terminal = torch.zeros(n, max_nodes, dtype=torch.bool, device=self.device)
        self.terminal_value = torch.zeros(n, max_nodes, device=self.device)
        
        # Node allocation tracking
        self.next_node_idx = torch.ones(n, dtype=torch.long, device=self.device)
        
        # Batch indices for efficient indexing
        self.batch_idx = torch.arange(n, device=self.device)
    
    def reset(self, root_states: torch.Tensor, root_policies: torch.Tensor, 
              legal_masks: torch.Tensor):
        """
        Initialize trees with root states.
        
        Args:
            root_states: (n_trees, 6, 7) initial board states
            root_policies: (n_trees, 7) policy priors from neural net
            legal_masks: (n_trees, 7) legal move masks
        """
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
        self.is_terminal.zero_()
        self.terminal_value.zero_()
        self.next_node_idx.fill_(1)
        
        # Set root node (index 0) for each tree
        self.states[:, 0] = root_states
        self.has_state[:, 0] = True
        
        # Apply legal mask and normalize
        masked_policy = root_policies * legal_masks.float()
        masked_policy = masked_policy / (masked_policy.sum(dim=1, keepdim=True) + 1e-8)
        
        # Add Dirichlet noise for exploration
        if self.config.exploration_fraction > 0:

            if "mps" in str(self.device): # could be "mps:0"
                noise_np = np.random.dirichlet([self.config.dirichlet_alpha] * self.config.n_actions, size=self.n_trees)
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
        
        # Expand root immediately
        self._expand_roots(legal_masks)
    
    def _expand_roots(self, legal_masks: torch.Tensor):
        """Expand root nodes with children for legal actions"""
        for action in range(self.config.n_actions):
            is_legal = legal_masks[:, action]
            
            # Allocate child nodes where legal
            child_idx = self.next_node_idx.clone()
            
            # Set children pointers
            self.children[:, 0, action] = torch.where(
                is_legal, child_idx, torch.full_like(child_idx, -1)
            )
            
            # Set parent info for children
            self.parent[self.batch_idx, child_idx] = torch.where(
                is_legal, 
                torch.zeros_like(child_idx),  # parent is root (0)
                self.parent[self.batch_idx, child_idx]
            )
            self.parent_action[self.batch_idx, child_idx] = torch.where(
                is_legal,
                torch.full_like(child_idx, action),
                self.parent_action[self.batch_idx, child_idx]
            )
            
            # Increment allocation counter
            self.next_node_idx += is_legal.long()
    
    def run_simulations(self, num_simulations: int, parallel_sims: int = 8,
                       verbose: bool = False) -> torch.Tensor:
        """
        Run MCTS simulations on all trees.
        
        Args:
            num_simulations: Total simulations per tree
            parallel_sims: Simulations to run in parallel (with virtual loss)
            verbose: Print timing info
            
        Returns:
            visit_distributions: (n_trees, n_actions) normalized visit counts
        """
        num_batches = (num_simulations + parallel_sims - 1) // parallel_sims
        
        for batch_idx in range(num_batches):
            sims_this_batch = min(parallel_sims, num_simulations - batch_idx * parallel_sims)
            
            t0 = time.time()
            
            # Collect leaves from all trees (parallel within each tree too)
            all_leaves = []
            all_paths = []
            all_path_lengths = []
            
            for sim in range(sims_this_batch):
                leaves, paths, path_lengths = self._batch_select_leaves()
                all_leaves.append(leaves)
                all_paths.append(paths)
                all_path_lengths.append(path_lengths)
            
            # Stack: (sims_this_batch, n_trees)
            leaves_tensor = torch.stack(all_leaves, dim=0)
            paths_tensor = torch.stack(all_paths, dim=0)
            lengths_tensor = torch.stack(all_path_lengths, dim=0)
            
            if verbose:
                t1 = time.time()
                print(f"  Selection: {t1-t0:.4f}s")
            
            # Compute states for leaves that need them
            t0 = time.time()
            self._materialize_states(leaves_tensor, paths_tensor, lengths_tensor)
            if verbose:
                t1 = time.time()
                print(f"  State materialization: {t1-t0:.4f}s")
            
            # Batch evaluate all leaves
            t0 = time.time()
            self._batch_evaluate_and_expand(leaves_tensor)
            if verbose:
                t1 = time.time()
                print(f"  NN eval + expand: {t1-t0:.4f}s")
            
            # Backup all paths
            t0 = time.time()
            self._batch_backup(leaves_tensor, paths_tensor, lengths_tensor)
            if verbose:
                t1 = time.time()
                print(f"  Backup: {t1-t0:.4f}s")
        
        return self._get_visit_distributions()
    
    def _batch_select_leaves(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select one leaf from each tree using PUCT with virtual loss.
        
        Returns:
            leaves: (n_trees,) selected leaf node indices
            paths: (n_trees, max_depth) node indices along path
            path_lengths: (n_trees,) actual path lengths
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
            
            # Check if expanded (has any valid children)
            current_children = self.children[self.batch_idx, current]  # (n_trees, n_actions)
            has_children = (current_children >= 0).any(dim=1)  # (n_trees,)
            
            # Check terminal
            is_term = self.is_terminal[self.batch_idx, current]
            
            # Continue if expanded and not terminal
            should_continue = has_children & ~is_term
            if not should_continue.any():
                break
            
            # Apply virtual loss to current node
            self.virtual_loss[self.batch_idx, current] += vl_weight * should_continue.float()
            
            # Compute PUCT scores for children
            valid_children = current_children >= 0
            safe_indices = current_children.clamp(min=0)
            
            # Gather child stats using advanced indexing
            tree_expand = self.batch_idx.unsqueeze(1).expand(-1, self.config.n_actions)
            
            child_visits = self.visits[tree_expand, safe_indices]
            child_values = self.total_value[tree_expand, safe_indices]
            child_vl = self.virtual_loss[tree_expand, safe_indices]
            child_priors = self.priors[self.batch_idx, current]
            
            # Q with virtual loss (VL assumes loss = -1)
            effective_visits = child_visits + child_vl
            effective_value = child_values - child_vl
            Q = torch.where(
                effective_visits > 0,
                effective_value / effective_visits,
                torch.zeros_like(effective_value)
            )
            
            # U (exploration bonus)
            parent_visits = self.visits[self.batch_idx, current].unsqueeze(1)
            U = c_puct * child_priors * torch.sqrt(parent_visits + 1) / (1 + effective_visits)
            
            # PUCT score with mask for invalid children
            puct = Q + U
            puct = torch.where(valid_children, puct, torch.full_like(puct, -1e9))
            
            # Select best action
            best_actions = puct.argmax(dim=1)
            next_nodes = self.children[self.batch_idx, current, best_actions]
            
            # Only advance where should_continue
            current = torch.where(should_continue, next_nodes, current)
        
        # Apply virtual loss to leaves
        self.virtual_loss[self.batch_idx, current] += vl_weight
        
        return current, paths, path_lengths
    
    def _materialize_states(self, leaves: torch.Tensor, paths: torch.Tensor,
                           path_lengths: torch.Tensor):
        """
        Compute board states for leaf nodes by replaying moves from root.
        
        Args:
            leaves: (n_sims, n_trees) leaf node indices
            paths: (n_sims, n_trees, max_depth) paths
            path_lengths: (n_sims, n_trees) path lengths
        
        This is vectorized where possible but has some sequential dependency
        due to move application.
        """
        n_sims, n_trees = leaves.shape
        
        # For each tree, find leaves that don't have states materialized
        # Need to properly index: for each (sim, tree), check has_state[tree, leaf]
        needs_state = torch.zeros(n_sims, n_trees, dtype=torch.bool, device=self.device)
        for sim_idx in range(n_sims):
            needs_state[sim_idx] = ~self.has_state[self.batch_idx, leaves[sim_idx]]
        
        if not needs_state.any():
            return
        
        # For leaves needing states, replay from root
        # This is per-simulation, so we iterate over sims
        n_sims = leaves.shape[0]
        
        for sim_idx in range(n_sims):
            sim_leaves = leaves[sim_idx]  # (n_trees,)
            sim_paths = paths[sim_idx]    # (n_trees, max_depth)
            sim_lengths = path_lengths[sim_idx]  # (n_trees,)
            sim_needs = needs_state[sim_idx]  # (n_trees,)
            
            if not sim_needs.any():
                continue
            
            # Start from root state
            current_states = self.states[:, 0].clone()  # (n_trees, 6, 7)
            
            # Determine current player from root
            p1_count = (current_states == 1).sum(dim=(1, 2))
            p2_count = (current_states == 2).sum(dim=(1, 2))
            current_player = torch.where(p1_count == p2_count, 
                                        torch.ones(self.n_trees, device=self.device),
                                        torch.full((self.n_trees,), 2, device=self.device))
            
            max_len = sim_lengths.max().item()
            
            for depth in range(1, max_len):
                active = (sim_lengths > depth) & sim_needs
                if not active.any():
                    break
                
                # Get node at this depth
                node_idx = sim_paths[:, depth]
                
                # Get action taken to reach this node
                action = self.parent_action[self.batch_idx, node_idx]
                
                # Make move (vectorized)
                current_states = self._make_moves_batch(
                    current_states, action, current_player.long(), active
                )
                
                # Switch player
                current_player = torch.where(active, 3 - current_player, current_player)
                
                # Cache state at this node
                self.states[self.batch_idx[active], node_idx[active]] = current_states[active]
                self.has_state[self.batch_idx[active], node_idx[active]] = True
    
    def _make_moves_batch(self, boards: torch.Tensor, actions: torch.Tensor,
                         players: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """
        Make moves on multiple boards simultaneously.
        
        Args:
            boards: (n_trees, 6, 7) current board states
            actions: (n_trees,) column indices
            players: (n_trees,) player making move (1 or 2)
            active: (n_trees,) which boards to update
            
        Returns:
            new_boards: (n_trees, 6, 7) updated boards
        """
        new_boards = boards.clone()
        
        # For each active board, find lowest empty row in the action column
        for i in range(self.n_trees):
            if not active[i]:
                continue
            
            col = actions[i].item()
            if col < 0:  # Invalid action
                continue
                
            player = players[i].item()
            
            # Find lowest empty row
            col_vals = new_boards[i, :, col]
            empty_rows = (col_vals == 0).nonzero(as_tuple=True)[0]
            
            if len(empty_rows) > 0:
                row = empty_rows[-1].item()
                new_boards[i, row, col] = player
        
        return new_boards
    
    def _batch_evaluate_and_expand(self, leaves: torch.Tensor):
        """
        Evaluate leaf positions with neural network and expand.
        
        Args:
            leaves: (n_sims, n_trees) leaf node indices
        """
        n_sims, n_trees = leaves.shape
        
        # Flatten leaves for batch evaluation
        flat_leaves = leaves.flatten()  # (n_sims * n_trees,)
        flat_tree_idx = self.batch_idx.unsqueeze(0).expand(n_sims, -1).flatten()
        
        # Get states for all leaves
        leaf_states = self.states[flat_tree_idx, flat_leaves]  # (n_sims * n_trees, 6, 7)
        
        # Check which are already terminal
        already_terminal = self.is_terminal[flat_tree_idx, flat_leaves]
        
        # Only evaluate non-terminal leaves
        needs_eval = ~already_terminal
        
        if needs_eval.any():
            eval_states = leaf_states[needs_eval]  # (?, 6, 7)
            
            # Check for terminal states
            is_term, winners = self._check_terminal_batch(eval_states)
            
            # Get current player for value perspective
            p1_count = (eval_states == 1).sum(dim=(1, 2))
            p2_count = (eval_states == 2).sum(dim=(1, 2))
            current_player = torch.where(p1_count == p2_count,
                                        torch.ones(eval_states.shape[0], device=self.device),
                                        torch.full((eval_states.shape[0],), 2, device=self.device))
            
            # Compute terminal values
            term_values = torch.zeros(eval_states.shape[0], device=self.device)
            term_values = torch.where(
                winners == 0,  # Draw
                torch.zeros_like(term_values),
                torch.where(
                    winners == current_player,  # Win
                    torch.ones_like(term_values),
                    -torch.ones_like(term_values)  # Loss
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
                        eval_states[non_term].flatten(start_dim=1)
                    )
                nn_values[non_term] = values.squeeze(-1) if values.dim() > 1 else values
                nn_policies[non_term] = policies
            
            # Combine terminal and NN values
            final_values = torch.where(is_term, term_values, nn_values)
            
            # Map back to full leaf tensor
            full_is_term = torch.zeros(flat_leaves.shape[0], dtype=torch.bool, device=self.device)
            full_values = torch.zeros(flat_leaves.shape[0], device=self.device)
            full_policies = torch.zeros(flat_leaves.shape[0], self.config.n_actions, 
                                       device=self.device)
            
            full_is_term[needs_eval] = is_term
            full_values[needs_eval] = final_values
            full_policies[needs_eval] = nn_policies
            
            # Update terminal status
            self.is_terminal[flat_tree_idx[needs_eval], flat_leaves[needs_eval]] = is_term
            self.terminal_value[flat_tree_idx[needs_eval & full_is_term], 
                               flat_leaves[needs_eval & full_is_term]] = term_values[is_term]
            
            # Expand non-terminal leaves
            expand_mask = needs_eval & ~full_is_term
            if expand_mask.any():
                self._expand_nodes(
                    flat_tree_idx[expand_mask],
                    flat_leaves[expand_mask],
                    full_policies[expand_mask],
                    leaf_states[expand_mask]
                )
            
            # Store values for backup (we need to return these)
            # For now, store in a temporary attribute
            self._last_leaf_values = full_values.view(n_sims, n_trees)
        else:
            # All terminal - use stored terminal values
            self._last_leaf_values = self.terminal_value[flat_tree_idx, flat_leaves].view(n_sims, n_trees)
    
    def _expand_nodes(self, tree_indices: torch.Tensor, node_indices: torch.Tensor,
                     policies: torch.Tensor, states: torch.Tensor):
        """Expand nodes with children"""
        n_nodes = tree_indices.shape[0]
        
        # Get legal moves from states
        legal_masks = states[:, 0, :] == 0  # Top row empty = legal
        
        # Mask and normalize policies
        masked_policies = policies * legal_masks.float()
        masked_policies = masked_policies / (masked_policies.sum(dim=1, keepdim=True) + 1e-8)
        
        # Store priors
        self.priors[tree_indices, node_indices] = masked_policies
        
        # Create children for each legal action
        for action in range(self.config.n_actions):
            is_legal = legal_masks[:, action]
            if not is_legal.any():
                continue
            
            # Get next available node indices for each tree
            legal_tree_idx = tree_indices[is_legal]
            legal_node_idx = node_indices[is_legal]
            child_idx = self.next_node_idx[legal_tree_idx]
            
            # Set child pointers
            self.children[legal_tree_idx, legal_node_idx, action] = child_idx
            
            # Set parent info
            self.parent[legal_tree_idx, child_idx] = legal_node_idx
            self.parent_action[legal_tree_idx, child_idx] = action
            
            # Increment allocation counter
            self.next_node_idx[legal_tree_idx] += 1
    
    def _batch_backup(self, leaves: torch.Tensor, paths: torch.Tensor,
                     path_lengths: torch.Tensor):
        """
        Backup values along all paths.
        
        Args:
            leaves: (n_sims, n_trees) leaf indices
            paths: (n_sims, n_trees, max_depth) paths
            path_lengths: (n_sims, n_trees) path lengths
        """
        n_sims = leaves.shape[0]
        vl_weight = self.config.virtual_loss_weight
        
        # Get leaf values from last evaluation
        values = self._last_leaf_values  # (n_sims, n_trees)
        
        for sim_idx in range(n_sims):
            sim_paths = paths[sim_idx]  # (n_trees, max_depth)
            sim_lengths = path_lengths[sim_idx]  # (n_trees,)
            sim_values = values[sim_idx].clone()  # (n_trees,)
            
            max_len = sim_lengths.max().item()
            
            # Backup from leaf to root
            for depth in range(max_len - 1, -1, -1):
                active = sim_lengths > depth
                node_idx = sim_paths[:, depth]
                
                # Update statistics
                self.visits[self.batch_idx, node_idx] += active.float()
                self.total_value[self.batch_idx, node_idx] += sim_values * active.float()
                
                # Remove virtual loss
                self.virtual_loss[self.batch_idx, node_idx] -= vl_weight * active.float()
                
                # Flip value for opponent
                sim_values = -sim_values
        
        # Clamp virtual loss to non-negative
        self.virtual_loss.clamp_(min=0)
    
    def _check_terminal_batch(self, boards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check for terminal states in batch.
        
        Returns:
            is_terminal: (batch,) bool
            winners: (batch,) 0=draw, 1=p1, 2=p2, -1=ongoing
        """
        batch_size = boards.shape[0]
        device = boards.device
        
        # Check for full board
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
        
        # Set draws
        winners = torch.where(board_full & (winners == -1), 
                             torch.zeros_like(winners), winners)
        
        is_terminal = winners >= 0
        
        return is_terminal, winners
    
    def _get_visit_distributions(self) -> torch.Tensor:
        """Get normalized visit distributions from root"""
        root_children = self.children[:, 0]  # (n_trees, n_actions)
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
            # Greedy
            probs = torch.zeros_like(distributions)
            probs.scatter_(1, distributions.argmax(dim=1, keepdim=True), 1.0)
            return probs
        else:
            # Temperature scaling
            visits_temp = distributions ** (1.0 / temperature)
            return visits_temp / (visits_temp.sum(dim=1, keepdim=True) + 1e-8)


class TensorMCTSWrapper:
    """
    Wrapper to provide similar interface to BatchMCTSWithVirtualLoss.
    
    This handles dynamic batch sizes by creating new TensorMCTS instances
    as needed (or reusing if same size).
    """
    
    def __init__(
        self,
        model,
        c_puct: float = 1.0,
        num_simulations: int = 800,
        parallel_simulations: int = 8,
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
        
        self._cached_mcts = {}  # Cache by n_trees
    
    def get_action_probs_batch_parallel(
        self,
        boards: List[torch.Tensor],
        legal_moves_list: List[torch.Tensor],
        temperature: float = 1.0,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run MCTS on batch of positions.
        
        Compatible with BatchMCTSWithVirtualLoss interface.
        """
        n_positions = len(boards)
        
        # Get or create TensorMCTS for this batch size
        if n_positions not in self._cached_mcts:
            self._cached_mcts[n_positions] = TensorMCTS(
                model=self.model,
                n_trees=n_positions,
                config=self.config,
                device=self.device
            )
        
        mcts = self._cached_mcts[n_positions]
        mcts.model = self.model  # Update model reference
        
        # Stack inputs
        boards_tensor = torch.stack(boards)  # (n, 6, 7)
        if boards_tensor.dim() == 2:
            boards_tensor = boards_tensor.view(n_positions, 6, 7)
        legal_masks = torch.stack(legal_moves_list)  # (n, 7)
        
        # Get initial policies from model
        with torch.no_grad():
            policies, _ = self.model.forward(boards_tensor.flatten(start_dim=1))
        
        # Initialize trees
        mcts.reset(boards_tensor, policies, legal_masks)
        
        # Run simulations
        visit_distributions = mcts.run_simulations(
            self.num_simulations, 
            self.parallel_simulations,
            verbose=verbose
        )
        
        # Get action probs with temperature
        action_probs = mcts.get_action_probs(temperature)
        
        return visit_distributions, action_probs


def benchmark_comparison():
    """Compare tensor MCTS vs original implementation"""
    
    print("=" * 60)
    print("MCTS Benchmark: Tensor-Based vs Object-Based")
    print("=" * 60)
    
    # Dummy model
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
    
    # Test configs
    configs = [
        (32, 20, 4),   # Small batch, few sims
        (128, 20, 8),  # Medium batch
        (512, 20, 8),  # Large batch (your use case)
        (512, 50, 8),  # More simulations
    ]
    
    for n_games, n_sims, parallel in configs:
        print(f"\n--- {n_games} games, {n_sims} sims, {parallel} parallel ---")
        
        # Create test boards
        boards = [torch.zeros(6, 7, device=device) for _ in range(n_games)]
        legal = [torch.ones(7, dtype=torch.bool, device=device) for _ in range(n_games)]
        
        # Tensor MCTS
        tensor_mcts = TensorMCTSWrapper(
            model=model,
            num_simulations=n_sims,
            parallel_simulations=parallel,
            device=device
        )
        
        # Warmup
        _ = tensor_mcts.get_action_probs_batch_parallel(boards[:8], legal[:8])
        
        # Benchmark
        if device == "cuda":
            torch.cuda.synchronize()
        
        t0 = time.time()
        visit_dist, action_probs = tensor_mcts.get_action_probs_batch_parallel(
            boards, legal, temperature=1.0
        )
        
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        
        print(f"Tensor MCTS: {t1-t0:.3f}s ({n_games/(t1-t0):.1f} games/sec)")
        print(f"  Visit dist shape: {visit_dist.shape}")
        print(f"  Sample distribution: {visit_dist[0].cpu().numpy().round(3)}")


if __name__ == "__main__":
    benchmark_comparison()