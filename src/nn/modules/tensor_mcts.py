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
import math


@dataclass
class TensorMCTSConfig:
    """Configuration for tensor MCTS"""
    n_actions: int = 7
    max_nodes_per_tree: int = 5000
    max_depth: int = 50
    c_puct_base: float = 19652.0
    c_puct_init: float = 1.25
    virtual_loss_weight: float = 1.0
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
        
        with torch.no_grad():
            self.visits = torch.zeros(n, max_nodes, device=self.device)
            self.total_value = torch.zeros(n, max_nodes, device=self.device)
            self.virtual_loss = torch.zeros(n, max_nodes, device=self.device)
            self.priors = torch.zeros(n, max_nodes, n_actions, device=self.device)
            self.children = torch.full((n, max_nodes, n_actions), -1, dtype=torch.long, device=self.device)
            self.parent = torch.full((n, max_nodes), -1, dtype=torch.long, device=self.device)
            self.parent_action = torch.full((n, max_nodes), -1, dtype=torch.long, device=self.device)
            self.states = torch.zeros(n, max_nodes, 6, 7, dtype=torch.long, device=self.device)
            self.has_state = torch.zeros(n, max_nodes, dtype=torch.bool, device=self.device)
            self.current_player = torch.zeros(n, max_nodes, dtype=torch.long, device=self.device)
            self.is_terminal = torch.zeros(n, max_nodes, dtype=torch.bool, device=self.device)
            self.terminal_value = torch.zeros(n, max_nodes, device=self.device)
            self.next_node_idx = torch.ones(n, dtype=torch.long, device=self.device)
            self.batch_idx = torch.arange(n, device=self.device)
    
    def reset(self, root_states: torch.Tensor, root_policies: torch.Tensor, 
              legal_masks: torch.Tensor, root_players: torch.Tensor,
              root_values: Optional[torch.Tensor] = None):
        """Initialize trees with root states."""
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
        
        self.states[:, 0] = root_states.long()
        self.has_state[:, 0] = True
        self.current_player[:, 0] = root_players
        
        if root_values is not None:
            self.visits[:, 0] = 1
            self.total_value[:, 0] = root_values.detach().squeeze() # avoid mem leak
        
        masked_policy = root_policies.detach() * legal_masks.float() # avoid mem leak
        masked_policy = masked_policy / (masked_policy.sum(dim=1, keepdim=True) + 1e-8)
        
        if self.config.exploration_fraction > 0:
            if "mps" in str(self.device):
                noise_np = np.random.dirichlet([self.config.dirichlet_alpha] * self.config.n_actions, size=self.n_trees)
                noise = torch.from_numpy(noise_np).float().to(self.device)
            else:
                noise = torch.distributions.Dirichlet(torch.full((self.config.n_actions,), self.config.dirichlet_alpha, device=self.device)).sample((self.n_trees,))
            noise = noise * legal_masks.float()
            noise = noise / (noise.sum(dim=1, keepdim=True) + 1e-8)
            masked_policy = ((1 - self.config.exploration_fraction) * masked_policy + self.config.exploration_fraction * noise)
        
        self.priors[:, 0] = masked_policy
        self._expand_roots(legal_masks, root_players)
    
    def _expand_roots(self, legal_masks: torch.Tensor, root_players: torch.Tensor):
        child_player = 3 - root_players
        for action in range(self.config.n_actions):
            is_legal = legal_masks.bool()[:, action]
            child_idx = self.next_node_idx.clone()
            
            # Check capacity before expanding
            has_space = child_idx < self.config.max_nodes_per_tree
            can_expand = is_legal & has_space

            self.children[:, 0, action] = torch.where(can_expand, child_idx, torch.full_like(child_idx, -1))
        
            # Only update nodes that have space
            if can_expand.any():
                expand_idx = self.batch_idx[can_expand]
                expand_child = child_idx[can_expand]
                self.parent[expand_idx, expand_child] = 0
                self.parent_action[expand_idx, expand_child] = action
                self.current_player[expand_idx, expand_child] = child_player[can_expand]
                    
            self.next_node_idx += can_expand.long()
    
    def _compute_pb_c(self, parent_visits: torch.Tensor) -> torch.Tensor:
        base = self.config.c_puct_base
        init = self.config.c_puct_init
        N = parent_visits.squeeze(-1) if parent_visits.dim() > 1 else parent_visits
        return torch.log((1 + N + base) / base) + init
    
    def run_simulations(self, num_simulations: int, parallel_sims: int = 16, verbose: bool = False) -> torch.Tensor:
        num_batches = (num_simulations + parallel_sims - 1) // parallel_sims
        for _ in range(num_batches):
            sims_this_batch = min(parallel_sims, num_simulations)
            num_simulations -= sims_this_batch
            
            all_leaves = []
            all_paths = []
            all_path_lengths = []
            all_vl_applied = []
            
            for _ in range(sims_this_batch):
                leaves, paths, path_lengths = self._batch_select_leaves()
                all_leaves.append(leaves)
                all_paths.append(paths)
                all_path_lengths.append(path_lengths)
                all_vl_applied.append((paths.clone(), path_lengths.clone()))
            
            leaves_tensor = torch.stack(all_leaves, dim=0)
            paths_tensor = torch.stack(all_paths, dim=0)
            lengths_tensor = torch.stack(all_path_lengths, dim=0)
            
            self._materialize_states_vectorized(leaves_tensor, paths_tensor, lengths_tensor)
            self._batch_evaluate_and_expand(leaves_tensor)
            self._batch_backup_vectorized(leaves_tensor, paths_tensor, lengths_tensor)
            self._revert_virtual_losses(all_vl_applied)
            
        return self._get_visit_distributions()
    
    def _batch_select_leaves(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_depth = self.config.max_depth
        vl_weight = self.config.virtual_loss_weight
        
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
            
            if depth > 0: 
                self.virtual_loss[self.batch_idx, current] += vl_weight * should_continue.float()
            
            valid_children = current_children >= 0
            safe_indices = current_children.clamp(min=0)
            tree_expand = self.batch_idx.unsqueeze(1).expand(-1, self.config.n_actions)
            
            # --- Ghost Stats Fix ---
            child_visits = self.visits[tree_expand, safe_indices]
            child_visits = torch.where(valid_children, child_visits, torch.zeros_like(child_visits))
            
            child_values = self.total_value[tree_expand, safe_indices]
            child_values = torch.where(valid_children, child_values, torch.zeros_like(child_values))
            
            child_vl = self.virtual_loss[tree_expand, safe_indices]
            child_vl = torch.where(valid_children, child_vl, torch.zeros_like(child_vl))
            # -----------------------
            
            child_priors = self.priors[self.batch_idx, current]
            
            effective_W = child_values + child_vl
            Q = -(effective_W / torch.clamp(child_visits, min=1))
            
            parent_visits = self.visits[self.batch_idx, current]
            pb_c = self._compute_pb_c(parent_visits)
            U = pb_c.unsqueeze(1) * child_priors * (torch.sqrt(parent_visits.unsqueeze(1)) / (1 + child_visits))
            
            puct = Q + U
            puct = torch.where(valid_children, puct, torch.full_like(puct, -1e9))
            
            best_actions = puct.argmax(dim=1)
            next_nodes = self.children[self.batch_idx, current, best_actions]
            
            current = torch.where(should_continue, next_nodes, current)
        
        leaf_is_term = self.is_terminal[self.batch_idx, current]
        vl_update = torch.where(
            leaf_is_term,
            torch.zeros(self.n_trees, dtype=torch.float, device=self.device),
            torch.tensor(vl_weight, dtype=torch.float, device=self.device)
        )
        self.virtual_loss[self.batch_idx, current] += vl_update
        
        return current, paths, path_lengths
    
    def _revert_virtual_losses(self, vl_applied_list):
        vl_weight = self.config.virtual_loss_weight
        for paths, path_lengths in vl_applied_list:
            max_len = path_lengths.max().item()
            for depth in range(1, max_len):
                active = path_lengths > depth
                if not active.any(): break
                node_idx = paths[:, depth]
                self.virtual_loss[self.batch_idx[active], node_idx[active]] -= vl_weight
            
            sim_indices = torch.arange(paths.shape[0], device=self.device)
            leaf_depths = path_lengths - 1
            leaf_nodes = paths[sim_indices, leaf_depths]
            self.virtual_loss[self.batch_idx, leaf_nodes] -= vl_weight
            
        self.virtual_loss.clamp_(min=0)
    
    def _make_moves_vectorized(self, boards, actions, players, active):
        new_boards = boards.clone()
        if not active.any(): return new_boards
        
        active_indices = active.nonzero(as_tuple=True)[0]
        active_boards = new_boards[active_indices]
        active_actions = actions[active_indices]
        active_players = players[active_indices]
        n_active = active_indices.shape[0]
        
        batch_range = torch.arange(n_active, device=self.device)
        columns = active_boards[batch_range, :, active_actions]
        
        empty_mask = (columns == 0)
        row_indices = torch.arange(6, device=self.device).unsqueeze(0).expand(n_active, -1)
        valid_rows = torch.where(empty_mask, row_indices, torch.tensor(-1, device=self.device))
        target_rows = valid_rows.max(dim=1)[0]
        
        valid_moves = target_rows >= 0
        if valid_moves.any():
            valid_batch = batch_range[valid_moves]
            valid_rows_idx = target_rows[valid_moves]
            valid_cols = active_actions[valid_moves]
            valid_players_val = active_players[valid_moves]
            active_boards[valid_batch, valid_rows_idx, valid_cols] = valid_players_val
        
        new_boards[active_indices] = active_boards
        return new_boards
    
    def _materialize_states_vectorized(self, leaves, paths, path_lengths):
        n_sims, n_trees = leaves.shape
        max_depth = paths.shape[2]
        flat_tree_idx = self.batch_idx.unsqueeze(0).expand(n_sims, -1)
        flat_leaves = leaves
        needs_state = ~self.has_state[flat_tree_idx, flat_leaves]
        
        if not needs_state.any(): return
        
        current_states = self.states[:, 0].unsqueeze(0).expand(n_sims, -1, -1, -1).clone()
        current_players = self.current_player[:, 0].unsqueeze(0).expand(n_sims, -1).clone()
        
        for depth in range(1, max_depth):
            active = (path_lengths > depth) & needs_state
            if not active.any(): break
            
            node_indices = paths[:, :, depth]
            actions = self.parent_action[flat_tree_idx, node_indices]
            
            flat_states = current_states.reshape(n_sims * n_trees, 6, 7)
            flat_actions = actions.reshape(-1)
            flat_players = current_players.reshape(-1)
            flat_active = active.reshape(-1)
            
            flat_states = self._make_moves_vectorized(flat_states, flat_actions, flat_players, flat_active)
            current_states = flat_states.reshape(n_sims, n_trees, 6, 7)
            
            if active.any():
                active_sims, active_trees = active.nonzero(as_tuple=True)
                active_nodes = node_indices[active_sims, active_trees]
                self.states[active_trees, active_nodes] = current_states[active_sims, active_trees]
                self.has_state[active_trees, active_nodes] = True
            
            current_players = torch.where(active, 3 - current_players, current_players)
    
    def _batch_evaluate_and_expand(self, leaves):
        n_sims, n_trees = leaves.shape
        flat_leaves = leaves.flatten()
        flat_tree_idx = self.batch_idx.unsqueeze(0).expand(n_sims, -1).flatten()
        
        leaf_states = self.states[flat_tree_idx, flat_leaves]
        leaf_players = self.current_player[flat_tree_idx, flat_leaves]
        already_terminal = self.is_terminal[flat_tree_idx, flat_leaves]
        already_expanded = (self.children[flat_tree_idx, flat_leaves] >= 0).any(dim=1)
        
        full_values = torch.zeros(flat_leaves.shape[0], device=self.device)
        if already_terminal.any():
            full_values[already_terminal] = self.terminal_value[flat_tree_idx[already_terminal], flat_leaves[already_terminal]]
        
        full_is_term = already_terminal.clone()
        full_policies = torch.zeros(flat_leaves.shape[0], self.config.n_actions, device=self.device)
        needs_eval = ~already_terminal & ~already_expanded
        
        if needs_eval.any():
            eval_states = leaf_states[needs_eval]
            eval_players = leaf_players[needs_eval]
            
            is_term, winners = self._check_terminal_batch(eval_states)
            
            term_values = torch.where(winners == 0, 0.0, torch.where(winners == eval_players, 1.0, -1.0))
            non_term = ~is_term
            nn_values = torch.zeros(eval_states.shape[0], device=self.device)
            nn_policies = torch.zeros(eval_states.shape[0], self.config.n_actions, device=self.device)
            
            if non_term.any():
                with torch.no_grad():
                    policies, values = self.model.forward(eval_states[non_term].flatten(start_dim=1).float(), eval_players[non_term])
                    policies = policies.detach().float() # avoid mem leak
                    values = values.detach().float() # avoid mem leak
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
                self.terminal_value[flat_tree_idx[newly_terminal], flat_leaves[newly_terminal]] = term_values[is_term]
            
            expand_mask = needs_eval & ~full_is_term
            if expand_mask.any():
                self._expand_nodes(flat_tree_idx[expand_mask], flat_leaves[expand_mask], full_policies[expand_mask], leaf_states[expand_mask], leaf_players[expand_mask])
        
        # CRITICAL FIX: Match mcts_v2 behavior - skip backup for already-expanded nodes
        # In mcts_v2's parallel_uct_search:
        #   if leaf.is_expanded:
        #       continue  # Skip backup entirely
        # We do NOT compute or backup values for already_expanded nodes.
        self._should_backup = (~already_expanded).view(n_sims, n_trees)
        self._last_leaf_values = full_values.view(n_sims, n_trees)
        
    def _expand_nodes(self, tree_indices, node_indices, policies, states, node_players):
        legal_masks = states[:, 0, :] == 0
        masked_policies = policies * legal_masks.float()
        masked_policies = masked_policies / (masked_policies.sum(dim=1, keepdim=True) + 1e-8)
        self.priors[tree_indices, node_indices] = masked_policies
        child_players = 3 - node_players
        
        for action in range(self.config.n_actions):
            is_legal = legal_masks[:, action]
            if not is_legal.any(): continue
            legal_tree_idx = tree_indices[is_legal]
            legal_node_idx = node_indices[is_legal]
            legal_child_players = child_players[is_legal]
            child_idx = self.next_node_idx[legal_tree_idx]
            has_space = child_idx < self.config.max_nodes_per_tree
            if not has_space.any(): continue
            legal_tree_idx = legal_tree_idx[has_space]
            legal_node_idx = legal_node_idx[has_space]
            legal_child_players = legal_child_players[has_space]
            child_idx = child_idx[has_space]
            self.children[legal_tree_idx, legal_node_idx, action] = child_idx
            self.parent[legal_tree_idx, child_idx] = legal_node_idx
            self.parent_action[legal_tree_idx, child_idx] = action
            self.current_player[legal_tree_idx, child_idx] = legal_child_players
            self.next_node_idx[legal_tree_idx] += 1
    
    def _batch_backup_vectorized(self, leaves: torch.Tensor, paths: torch.Tensor, path_lengths: torch.Tensor):
        n_sims, n_trees = leaves.shape
        max_depth = paths.shape[2]
        values = self._last_leaf_values.clone()
        
        for depth in range(max_depth - 1, -1, -1):
            # Only backup nodes where should_backup is True (not already_expanded)
            active = (path_lengths > depth) & self._should_backup
            
            if not active.any():
                values = -values  # Still flip sign
                continue
            
            node_idx = paths[:, :, depth]
            active_sims, active_trees = active.nonzero(as_tuple=True)
            active_nodes = node_idx[active_sims, active_trees]
            active_values = values[active_sims, active_trees]
            
            linear_indices = active_trees * self.config.max_nodes_per_tree + active_nodes
            ones = torch.ones_like(active_values)
            self.visits.view(-1).index_add_(0, linear_indices, ones)
            self.total_value.view(-1).index_add_(0, linear_indices, active_values)
            
            values = -values
    
    def _check_terminal_batch(self, boards):
        batch_size = boards.shape[0]
        device = boards.device
        board_full = ~(boards[:, 0, :] == 0).any(dim=1)
        winners = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        
        for row in range(6):
            for col in range(4):
                window = boards[:, row, col:col+4]
                player = window[:, 0]
                all_same = (window == player.unsqueeze(1)).all(dim=1)
                valid = player != 0
                has_win = all_same & valid
                winners = torch.where(has_win, player, winners)
        for row in range(3):
            for col in range(7):
                window = boards[:, row:row+4, col]
                player = window[:, 0]
                all_same = (window == player.unsqueeze(1)).all(dim=1)
                valid = player != 0
                has_win = all_same & valid
                winners = torch.where(has_win, player, winners)
        for row in range(3):
            for col in range(4):
                cells = torch.stack([boards[:, row+i, col+i] for i in range(4)], dim=1)
                player = cells[:, 0]
                all_same = (cells == player.unsqueeze(1)).all(dim=1)
                valid = player != 0
                has_win = all_same & valid
                winners = torch.where(has_win, player, winners)
                cells = torch.stack([boards[:, row+i, col+3-i] for i in range(4)], dim=1)
                player = cells[:, 0]
                all_same = (cells == player.unsqueeze(1)).all(dim=1)
                valid = player != 0
                has_win = all_same & valid
                winners = torch.where(has_win, player, winners)
        
        winners = torch.where(board_full & (winners == -1), torch.zeros_like(winners), winners)
        is_terminal = winners >= 0
        return is_terminal, winners
    
    def _get_visit_distributions(self):
        root_children = self.children[:, 0]
        valid = root_children >= 0
        safe_idx = root_children.clamp(min=0)
        tree_expand = self.batch_idx.unsqueeze(1).expand(-1, self.config.n_actions)
        child_visits = self.visits[tree_expand, safe_idx]
        child_visits = torch.where(valid, child_visits, torch.zeros_like(child_visits))
        total = child_visits.sum(dim=1, keepdim=True)
        return child_visits / (total + 1e-8)
    
    def get_action_probs(self, temperature=1.0):
        distributions = self._get_visit_distributions()
        if temperature == 0:
            probs = torch.zeros_like(distributions)
            probs.scatter_(1, distributions.argmax(dim=1, keepdim=True), 1.0)
            return probs
        exp = max(1.0, min(5.0, 1.0 / temperature))
        visits_temp = distributions ** exp
        return visits_temp / (visits_temp.sum(dim=1, keepdim=True) + 1e-8)


class TensorMCTSWrapper:
    def __init__(self, model, c_puct=1.25, c_puct_base=19652.0, num_simulations=800, parallel_simulations=16, dirichlet_alpha=0.3, exploration_fraction=0.25, virtual_loss_value=1.0, device="cpu"):
        self.model = model
        self.num_simulations = num_simulations
        self.parallel_simulations = parallel_simulations
        self.device = device
        self.config = TensorMCTSConfig(c_puct_base=c_puct_base, c_puct_init=c_puct, virtual_loss_weight=virtual_loss_value, dirichlet_alpha=dirichlet_alpha, exploration_fraction=exploration_fraction)
        self._cached_mcts = {}
    
    def get_action_probs_batch_parallel(self, boards, legal_moves_list, temperature=1.0, current_players=None, verbose=False):
        n_positions = len(boards)
        if n_positions not in self._cached_mcts:
            self._cached_mcts[n_positions] = TensorMCTS(model=self.model, n_trees=n_positions, config=self.config, device=self.device)
        mcts = self._cached_mcts[n_positions]
        mcts.model = self.model
        boards_tensor = torch.stack(boards).long()
        if boards_tensor.dim() == 2: boards_tensor = boards_tensor.view(n_positions, 6, 7)
        legal_masks = torch.stack(legal_moves_list)
        if current_players is None:
            p1_count = (boards_tensor == 1).sum(dim=(1, 2))
            p2_count = (boards_tensor == 2).sum(dim=(1, 2))
            current_players = torch.where(p1_count == p2_count, torch.ones(n_positions, dtype=torch.long, device=self.device), torch.full((n_positions,), 2, dtype=torch.long, device=self.device))
        else:
            current_players = current_players.to(self.device)
        
        with torch.no_grad():
            policies, values = self.model.forward(boards_tensor.flatten(start_dim=1).float(), current_players)
        
        mcts.reset(boards_tensor, policies, legal_masks, current_players, root_values=values)
        mcts.run_simulations(self.num_simulations, self.parallel_simulations, verbose=verbose)
        return mcts._get_visit_distributions(), mcts.get_action_probs(temperature)

def benchmark_comparison():
    print("Benchmarking Optimized TensorMCTS")

if __name__ == "__main__":
    benchmark_comparison()