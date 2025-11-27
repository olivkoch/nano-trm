"""
Tensor-Based MCTS for Connect Four - With Current Player Tracking

Key addition: current_player is tracked at every node so the model
always knows whose perspective to evaluate from.
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
    Fully tensorized MCTS with current player tracking.
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
        
        # Board states
        self.states = torch.zeros(n, max_nodes, 6, 7, device=self.device)
        self.has_state = torch.zeros(n, max_nodes, dtype=torch.bool, device=self.device)
        
        # NEW: Current player at each node (1 or 2) - the player TO MOVE
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
        """
        Initialize trees with root states.
        
        Args:
            root_states: (n_trees, 6, 7) initial board states
            root_policies: (n_trees, 7) policy priors from neural net
            legal_masks: (n_trees, 7) legal move masks
            root_players: (n_trees,) current player at root (1 or 2)
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
        self.current_player.zero_()
        self.is_terminal.zero_()
        self.terminal_value.zero_()
        self.next_node_idx.fill_(1)
        
        # Set root node
        self.states[:, 0] = root_states
        self.has_state[:, 0] = True
        self.current_player[:, 0] = root_players  # NEW: store who moves at root
        
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
        """Expand root nodes - children have opposite player"""
        child_player = 3 - root_players  # Opponent moves at children
        
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
            
            # NEW: Set current_player for child nodes
            self.current_player[self.batch_idx, child_idx] = torch.where(
                is_legal,
                child_player,
                self.current_player[self.batch_idx, child_idx]
            )
            
            self.next_node_idx += is_legal.long()
    
    def run_simulations(self, num_simulations: int, parallel_sims: int = 8,
                       verbose: bool = False) -> torch.Tensor:
        """Run MCTS simulations on all trees."""
        num_batches = (num_simulations + parallel_sims - 1) // parallel_sims
        
        for batch_idx in range(num_batches):
            sims_this_batch = min(parallel_sims, num_simulations - batch_idx * parallel_sims)
            
            all_leaves = []
            all_paths = []
            all_path_lengths = []
            
            for sim in range(sims_this_batch):
                leaves, paths, path_lengths = self._batch_select_leaves()
                all_leaves.append(leaves)
                all_paths.append(paths)
                all_path_lengths.append(path_lengths)
            
            leaves_tensor = torch.stack(all_leaves, dim=0)
            paths_tensor = torch.stack(all_paths, dim=0)
            lengths_tensor = torch.stack(all_path_lengths, dim=0)
            
            self._materialize_states(leaves_tensor, paths_tensor, lengths_tensor)
            self._batch_evaluate_and_expand(leaves_tensor)
            self._batch_backup(leaves_tensor, paths_tensor, lengths_tensor)
        
        return self._get_visit_distributions()
    
    def _batch_select_leaves(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select one leaf from each tree using PUCT with virtual loss."""
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
            Q = torch.where(
                effective_visits > 0,
                -effective_value / effective_visits,
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
    
    def _materialize_states(self, leaves: torch.Tensor, paths: torch.Tensor,
                           path_lengths: torch.Tensor):
        """Compute board states for leaf nodes by replaying moves from root."""
        n_sims, n_trees = leaves.shape
        
        needs_state = torch.zeros(n_sims, n_trees, dtype=torch.bool, device=self.device)
        for sim_idx in range(n_sims):
            needs_state[sim_idx] = ~self.has_state[self.batch_idx, leaves[sim_idx]]
        
        if not needs_state.any():
            return
        
        for sim_idx in range(n_sims):
            sim_leaves = leaves[sim_idx]
            sim_paths = paths[sim_idx]
            sim_lengths = path_lengths[sim_idx]
            sim_needs = needs_state[sim_idx]
            
            if not sim_needs.any():
                continue
            
            current_states = self.states[:, 0].clone()
            # NEW: Use stored current_player from root instead of inferring
            current_player = self.current_player[:, 0].clone()
                        
            max_len = sim_lengths.max().item()
            
            for depth in range(1, max_len):
                active = (sim_lengths > depth) & sim_needs
                if not active.any():
                    break
                
                node_idx = sim_paths[:, depth]
                action = self.parent_action[self.batch_idx, node_idx]
                
                # Make move with current player
                current_states = self._make_moves_batch(
                    current_states, action, current_player, active
                )
                
                # Cache state
                self.states[self.batch_idx[active], node_idx[active]] = current_states[active]
                self.has_state[self.batch_idx[active], node_idx[active]] = True
                
                # Switch player
                current_player = torch.where(active, 3 - current_player, current_player)

    def _make_moves_batch(self, boards: torch.Tensor, actions: torch.Tensor,
                         players: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Make moves on multiple boards simultaneously."""
        new_boards = boards.clone()
        
        for i in range(self.n_trees):
            if not active[i]:
                continue
            
            col = actions[i].item()
            if col < 0:
                continue
                
            player = players[i].item()
            col_vals = new_boards[i, :, col]
            empty_rows = (col_vals == 0).nonzero(as_tuple=True)[0]
            
            if len(empty_rows) > 0:
                row = empty_rows[-1].item()
                new_boards[i, row, col] = player
        
        return new_boards
    
    def _batch_evaluate_and_expand(self, leaves: torch.Tensor):
        """Evaluate leaf positions with neural network and expand."""
        n_sims, n_trees = leaves.shape
        
        flat_leaves = leaves.flatten()
        flat_tree_idx = self.batch_idx.unsqueeze(0).expand(n_sims, -1).flatten()
        
        leaf_states = self.states[flat_tree_idx, flat_leaves]
        # NEW: Get current_player for each leaf
        leaf_players = self.current_player[flat_tree_idx, flat_leaves]
        
        already_terminal = self.is_terminal[flat_tree_idx, flat_leaves]
        
        full_values = torch.zeros(flat_leaves.shape[0], device=self.device)
        if already_terminal.any():
            full_values[already_terminal] = self.terminal_value[
                flat_tree_idx[already_terminal], 
                flat_leaves[already_terminal]
            ]
        
        full_is_term = already_terminal.clone()
        full_policies = torch.zeros(flat_leaves.shape[0], self.config.n_actions, 
                                device=self.device)
        
        needs_eval = ~already_terminal
        
        if needs_eval.any():
            eval_states = leaf_states[needs_eval]
            eval_players = leaf_players[needs_eval]  # NEW
            
            is_term, winners = self._check_terminal_batch(eval_states)
            
            # Terminal values from perspective of current player (player to move)
            # If current player won, value = 1; if opponent won, value = -1
            term_values = torch.where(
                winners == 0,  # Draw
                torch.zeros(eval_states.shape[0], device=self.device),
                torch.where(
                    winners == eval_players,  # Current player won
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
                    # NEW: Pass current_player to model
                    policies, values = self.model.forward(
                        eval_states[non_term].flatten(start_dim=1),
                        eval_players[non_term]
                    )
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
            
            # Expand non-terminal leaves
            expand_mask = needs_eval & ~full_is_term
            if expand_mask.any():
                self._expand_nodes(
                    flat_tree_idx[expand_mask],
                    flat_leaves[expand_mask],
                    full_policies[expand_mask],
                    leaf_states[expand_mask],
                    leaf_players[expand_mask]  # NEW: pass current players
                )
        
        self._last_leaf_values = full_values.view(n_sims, n_trees)
    
    def _expand_nodes(self, tree_indices: torch.Tensor, node_indices: torch.Tensor,
                     policies: torch.Tensor, states: torch.Tensor,
                     node_players: torch.Tensor):  # NEW parameter
        """Expand nodes with children"""
        n_nodes = tree_indices.shape[0]
        
        legal_masks = states[:, 0, :] == 0
        
        masked_policies = policies * legal_masks.float()
        masked_policies = masked_policies / (masked_policies.sum(dim=1, keepdim=True) + 1e-8)
        
        self.priors[tree_indices, node_indices] = masked_policies
        
        # Children have opposite player
        child_players = 3 - node_players
        
        for action in range(self.config.n_actions):
            is_legal = legal_masks[:, action]
            if not is_legal.any():
                continue
            
            legal_tree_idx = tree_indices[is_legal]
            legal_node_idx = node_indices[is_legal]
            legal_child_players = child_players[is_legal]  # NEW
            child_idx = self.next_node_idx[legal_tree_idx]
            
            self.children[legal_tree_idx, legal_node_idx, action] = child_idx
            self.parent[legal_tree_idx, child_idx] = legal_node_idx
            self.parent_action[legal_tree_idx, child_idx] = action
            
            # NEW: Set current_player for children
            self.current_player[legal_tree_idx, child_idx] = legal_child_players
            
            self.next_node_idx[legal_tree_idx] += 1
    
    def _batch_backup(self, leaves: torch.Tensor, paths: torch.Tensor,
                     path_lengths: torch.Tensor):
        """Backup values along all paths."""
        n_sims = leaves.shape[0]
        vl_weight = self.config.virtual_loss_weight
        
        values = self._last_leaf_values
        
        for sim_idx in range(n_sims):
            sim_paths = paths[sim_idx]
            sim_lengths = path_lengths[sim_idx]
            sim_values = values[sim_idx].clone()
            
            max_len = sim_lengths.max().item()
            
            for depth in range(max_len - 1, -1, -1):
                active = sim_lengths > depth
                node_idx = sim_paths[:, depth]
                
                self.visits[self.batch_idx, node_idx] += active.float()
                self.total_value[self.batch_idx, node_idx] += sim_values * active.float()
                self.virtual_loss[self.batch_idx, node_idx] -= vl_weight * active.float()
                
                # Flip value for opponent
                sim_values = -sim_values
        
        self.virtual_loss.clamp_(min=0)
    
    def _check_terminal_batch(self, boards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check for terminal states in batch."""
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
    Now accepts current_players parameter.
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
        
        self._cached_mcts = {}
    
    def get_action_probs_batch_parallel(
        self,
        boards: List[torch.Tensor],
        legal_moves_list: List[torch.Tensor],
        temperature: float = 1.0,
        current_players: torch.Tensor = None,  # NEW parameter
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run MCTS on batch of positions.
        
        Args:
            boards: List of (6, 7) board tensors
            legal_moves_list: List of (7,) legal move masks
            temperature: Temperature for action selection
            current_players: (n_positions,) tensor of current players (1 or 2)
                           If None, inferred from piece counts (legacy behavior)
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
        
        # Stack inputs
        boards_tensor = torch.stack(boards)
        if boards_tensor.dim() == 2:
            boards_tensor = boards_tensor.view(n_positions, 6, 7)
        legal_masks = torch.stack(legal_moves_list)
        
        # Handle current_players
        if current_players is None:
            # Legacy: infer from piece counts
            p1_count = (boards_tensor == 1).sum(dim=(1, 2))
            p2_count = (boards_tensor == 2).sum(dim=(1, 2))
            current_players = torch.where(
                p1_count == p2_count,
                torch.ones(n_positions, dtype=torch.long, device=self.device),
                torch.full((n_positions,), 2, dtype=torch.long, device=self.device)
            )
        else:
            current_players = current_players.to(self.device)
        
        # Get initial policies from model - NOW WITH current_players
        with torch.no_grad():
            policies, _ = self.model.forward(
                boards_tensor.flatten(start_dim=1),
                current_players
            )
        
        # Initialize trees with current_players
        mcts.reset(boards_tensor, policies, legal_masks, current_players)
        
        # Run simulations
        visit_distributions = mcts.run_simulations(
            self.num_simulations, 
            self.parallel_simulations,
            verbose=verbose
        )
        
        action_probs = mcts.get_action_probs(temperature)
        
        return visit_distributions, action_probs


def test_current_player_tracking():
    """Test that current_player is correctly tracked through the tree"""
    
    class DebugModel:
        """Model that prints what it receives"""
        def __init__(self, device):
            self.device = device
            self.call_count = 0
            
        def forward(self, boards, current_players):
            self.call_count += 1
            batch_size = boards.shape[0]
            print(f"  Model call {self.call_count}: batch_size={batch_size}, "
                  f"players={current_players.tolist()}")
            
            policies = torch.ones(batch_size, 7, device=self.device) / 7
            values = torch.zeros(batch_size, device=self.device)
            return policies, values
    
    device = "cpu"
    model = DebugModel(device)
    
    # Create a simple position where it's player 1's turn
    board = torch.zeros(6, 7, device=device)
    legal = torch.ones(7, dtype=torch.bool, device=device)
    current_player = torch.tensor([1], dtype=torch.long, device=device)
    
    print("Testing with player 1 to move...")
    wrapper = TensorMCTSWrapper(
        model=model,
        num_simulations=5,
        parallel_simulations=1,
        device=device
    )
    
    visit_dist, action_probs = wrapper.get_action_probs_batch_parallel(
        [board], [legal], temperature=1.0, current_players=current_player
    )
    
    print(f"\nVisit distribution: {visit_dist[0].numpy().round(3)}")
    print(f"Action probs: {action_probs[0].numpy().round(3)}")
    
    # Now test with player 2 to move
    print("\n" + "="*50)
    print("Testing with player 2 to move...")
    model.call_count = 0
    
    # Add one piece so it's player 2's turn
    board2 = board.clone()
    board2[5, 0] = 1  # Player 1 played column 0
    current_player2 = torch.tensor([2], dtype=torch.long, device=device)
    
    visit_dist2, action_probs2 = wrapper.get_action_probs_batch_parallel(
        [board2], [legal], temperature=1.0, current_players=current_player2
    )
    
    print(f"\nVisit distribution: {visit_dist2[0].numpy().round(3)}")
    

if __name__ == "__main__":
    test_current_player_tracking()