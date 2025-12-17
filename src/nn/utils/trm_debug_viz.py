"""
TRM Training Debug Visualization

Generates diagnostic PNGs showing:
- Input board state
- Predicted policy logits (column probabilities)
- Predicted board reconstruction
- Ground truth board

Usage:
    In your training step, call:
        debug_viz_training_step(batch, outputs, labels, step, save_dir="debug_viz", every_n_steps=500)
    
    To disable, set DEBUG_VIZ_ENABLED = False or simply don't call the function.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from typing import Dict, Optional

# Global toggle - set to False to disable all visualization
DEBUG_VIZ_ENABLED = True

# Board dimensions for Connect 4
BOARD_ROWS = 6
BOARD_COLS = 7

# Color scheme
COLORS = {
    'empty': '#1a1a2e',      # Dark blue-gray
    'player1': '#e63946',     # Red
    'player2': '#f4d35e',     # Yellow
    'background': '#0f0f1a',  # Very dark
    'grid': '#3d3d5c',        # Grid lines
    'text': '#ffffff',        # White text
}

# Cell value mapping (adjust based on your encoding)
# Typically: 0 = empty, 1 = player 1, 2 = player 2
CELL_COLORS = ListedColormap([COLORS['empty'], COLORS['player1'], COLORS['player2']])


def board_to_grid(board_tensor: torch.Tensor) -> np.ndarray:
    """Convert flat board tensor to 2D grid."""
    board = board_tensor.detach().float().cpu().numpy()
    if board.ndim == 1:
        board = board.reshape(BOARD_ROWS, BOARD_COLS)
    return board.astype(int)


def draw_board(ax, board: np.ndarray, title: str, show_values: bool = False):
    """Draw a Connect 4 board on the given axes."""
    ax.set_facecolor(COLORS['background'])
    
    # Draw grid background
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            cell_val = int(board[row, col]) if board[row, col] >= 0 else 0
            
            # Clamp to valid range
            cell_val = max(0, min(2, cell_val))
            
            # Choose color
            if cell_val == 0:
                color = COLORS['empty']
            elif cell_val == 1:
                color = COLORS['player1']
            else:
                color = COLORS['player2']
            
            # Draw circle for piece
            circle = plt.Circle((col + 0.5, BOARD_ROWS - row - 0.5), 0.4,
                               color=color, ec=COLORS['grid'], linewidth=1.5)
            ax.add_patch(circle)
            
            # Optionally show numeric values
            if show_values and board[row, col] >= 0:
                ax.text(col + 0.5, BOARD_ROWS - row - 0.5, str(int(board[row, col])),
                       ha='center', va='center', fontsize=8, color='white', alpha=0.7)
    
    # Set limits and appearance
    ax.set_xlim(0, BOARD_COLS)
    ax.set_ylim(0, BOARD_ROWS)
    ax.set_aspect('equal')
    ax.set_title(title, color=COLORS['text'], fontsize=11, fontweight='bold', pad=10)
    
    # Column labels
    for col in range(BOARD_COLS):
        ax.text(col + 0.5, -0.3, str(col), ha='center', va='center',
               color=COLORS['text'], fontsize=9, alpha=0.7)
    
    ax.axis('off')


def draw_policy(ax, policy_logits: torch.Tensor, title: str):
    """Draw policy logits as a bar chart."""
    ax.set_facecolor(COLORS['background'])
    
    logits = policy_logits.detach().float().cpu().numpy()
    probs = torch.softmax(policy_logits.float(), dim=-1).detach().cpu().numpy()
    
    cols = np.arange(BOARD_COLS)
    
    # Create gradient colors based on probability
    colors = plt.cm.YlOrRd(probs / (probs.max() + 1e-8))
    
    bars = ax.bar(cols, probs, color=colors, edgecolor=COLORS['grid'], linewidth=1)
    
    # Add probability labels on bars
    for i, (bar, prob, logit) in enumerate(zip(bars, probs, logits)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{prob:.2f}', ha='center', va='bottom',
               color=COLORS['text'], fontsize=8)
        # Show logit value below bar
        ax.text(bar.get_x() + bar.get_width()/2., -0.08,
               f'{logit:.1f}', ha='center', va='top',
               color=COLORS['text'], fontsize=7, alpha=0.6)
    
    ax.set_xlim(-0.5, BOARD_COLS - 0.5)
    ax.set_ylim(-0.15, 1.1)
    ax.set_xticks(cols)
    ax.set_xticklabels([str(c) for c in cols], color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'])
    ax.set_title(title, color=COLORS['text'], fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Probability', color=COLORS['text'], fontsize=9)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])


def draw_reconstruction(ax, logits: torch.Tensor, labels: torch.Tensor, title: str):
    """Draw predicted vs actual board reconstruction."""
    ax.set_facecolor(COLORS['background'])
    
    # Get predictions (cast to float32 for numpy compatibility)
    preds = logits.argmax(dim=-1).detach().float().cpu().numpy()
    targets = labels.detach().float().cpu().numpy()
    
    preds_grid = preds.reshape(BOARD_ROWS, BOARD_COLS)
    targets_grid = targets.reshape(BOARD_ROWS, BOARD_COLS)
    
    # Draw grid with prediction/correctness indicators
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            pred_val = int(preds_grid[row, col])
            target_val = int(targets_grid[row, col])
            
            # Clamp values
            pred_val = max(0, min(2, pred_val))
            
            # Choose color based on prediction
            if pred_val == 0:
                color = COLORS['empty']
            elif pred_val == 1:
                color = COLORS['player1']
            else:
                color = COLORS['player2']
            
            # Draw main circle
            circle = plt.Circle((col + 0.5, BOARD_ROWS - row - 0.5), 0.4,
                               color=color, ec=COLORS['grid'], linewidth=1.5)
            ax.add_patch(circle)
            
            # Add correctness indicator
            is_correct = (pred_val == target_val) or (target_val == -100)  # ignore padding
            if not is_correct:
                # Draw X for incorrect
                ax.plot([col + 0.25, col + 0.75], [BOARD_ROWS - row - 0.75, BOARD_ROWS - row - 0.25],
                       color='white', linewidth=2, alpha=0.8)
                ax.plot([col + 0.25, col + 0.75], [BOARD_ROWS - row - 0.25, BOARD_ROWS - row - 0.75],
                       color='white', linewidth=2, alpha=0.8)
            
            # Show predicted value
            ax.text(col + 0.5, BOARD_ROWS - row - 0.5, str(pred_val),
                   ha='center', va='center', fontsize=8, color='white', alpha=0.8)
    
    # Compute accuracy
    valid_mask = targets >= 0
    if valid_mask.sum() > 0:
        accuracy = ((preds == targets) & valid_mask).sum() / valid_mask.sum()
    else:
        accuracy = 0.0
    
    ax.set_xlim(0, BOARD_COLS)
    ax.set_ylim(0, BOARD_ROWS)
    ax.set_aspect('equal')
    ax.set_title(f'{title}\nAccuracy: {accuracy:.1%}', color=COLORS['text'], 
                fontsize=11, fontweight='bold', pad=10)
    ax.axis('off')


def debug_viz_training_step(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    step: int,
    save_dir: str = "debug_viz",
    every_n_steps: int = 500,
    sample_idx: int = 0,
    current_player: Optional[torch.Tensor] = None,
) -> Optional[str]:
    """
    Generate debug visualization during training.
    
    Args:
        batch: Training batch containing 'boards', 'current_player', etc.
        outputs: Model outputs containing 'logits', 'policy_logits', 'value'
        labels: Ground truth labels for board reconstruction
        step: Current training step
        save_dir: Directory to save visualizations
        every_n_steps: Save every N steps (0 to save every step)
        sample_idx: Which sample in the batch to visualize
        current_player: Current player tensor (optional, extracted from batch if not provided)
    
    Returns:
        Path to saved image, or None if skipped
    """
    if not DEBUG_VIZ_ENABLED:
        return None
    
    # Always save first few steps, then every N steps
    should_save = (step <= 1) or (every_n_steps <= 0) or (step % every_n_steps == 0)
    if not should_save:
        return None
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data for single sample
    board = batch["boards"][sample_idx]
    policy_logits = outputs["policy_logits"][sample_idx]
    recon_logits = outputs["logits"][sample_idx]
    sample_labels = labels[sample_idx]
    value = outputs["value"][sample_idx].float().item()
    
    # print("Inside debug_viz_training_step:")
    # print("Input:\n", board.reshape(6,7))
    # print("Label:\n", sample_labels.reshape(6,7))
    # print("Prediction:\n", torch.argmax(recon_logits, dim=-1).reshape(6,7))
    
    if current_player is None:
        current_player = batch.get("current_player", torch.tensor([1]))
    
    # Handle both full batch tensor and single value
    if torch.is_tensor(current_player):
        if current_player.numel() > 1:
            current_player = current_player[sample_idx]
        player = current_player.item()
    else:
        player = current_player
    
    # Create figure with dark theme
    fig = plt.figure(figsize=(16, 5), facecolor=COLORS['background'])
    
    # Create subplots
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.3)
    
    # 1. Input Board
    ax1 = fig.add_subplot(gs[0])
    input_grid = board_to_grid(board)
    draw_board(ax1, input_grid, f'Input Board\nPlayer {player} to move')
    
    # 2. Policy Logits
    ax2 = fig.add_subplot(gs[1])
    draw_policy(ax2, policy_logits, 'Policy Prediction')
    
    # 3. Predicted Board Reconstruction  
    ax3 = fig.add_subplot(gs[2])
    draw_reconstruction(ax3, recon_logits, sample_labels, 'Predicted Board')
    
    # 4. Ground Truth Board
    ax4 = fig.add_subplot(gs[3])
    gt_grid = board_to_grid(sample_labels)
    draw_board(ax4, gt_grid, 'Ground Truth', show_values=True)
    
    # Add step and value info
    fig.suptitle(f'Step {step} | Predicted Value: {value:.3f}', 
                color=COLORS['text'], fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    save_path = os.path.join(save_dir, f'debug_step_{step:06d}.png')
    print(f"[DEBUG_VIZ] Generating visualization to {save_path}...")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                facecolor=COLORS['background'], edgecolor='none')
    plt.close(fig)
    
    return save_path


def debug_viz_single(
    board: torch.Tensor,
    policy_logits: torch.Tensor,
    recon_logits: torch.Tensor,
    labels: torch.Tensor,
    value: float,
    player: int = 1,
    save_path: str = "debug_single.png",
) -> str:
    """
    Generate visualization for a single sample (for testing/inference).
    """
    fig = plt.figure(figsize=(16, 5), facecolor=COLORS['background'])
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    draw_board(ax1, board_to_grid(board), f'Input Board\nPlayer {player}')
    
    ax2 = fig.add_subplot(gs[1])
    draw_policy(ax2, policy_logits, 'Policy Prediction')
    
    ax3 = fig.add_subplot(gs[2])
    draw_reconstruction(ax3, recon_logits, labels, 'Predicted Board')
    
    ax4 = fig.add_subplot(gs[3])
    draw_board(ax4, board_to_grid(labels), 'Ground Truth', show_values=True)
    
    fig.suptitle(f'Value: {value:.3f}', color=COLORS['text'], fontsize=14, fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
               facecolor=COLORS['background'], edgecolor='none')
    plt.close(fig)
    
    return save_path


# ============================================================================
# Integration snippet for trm_c4.py
# ============================================================================
"""
Add to your imports:
    from trm_debug_viz import debug_viz_training_step, DEBUG_VIZ_ENABLED

In compute_loss_and_metrics, after getting outputs, add:

    # Debug visualization (enable/disable via DEBUG_VIZ_ENABLED)
    if DEBUG_VIZ_ENABLED and hasattr(self, 'manual_step'):
        debug_viz_training_step(
            batch=batch,
            outputs=outputs,
            labels=labels,
            step=self.manual_step,
            save_dir="debug_viz",
            every_n_steps=500,  # Adjust frequency
            current_player=new_carry.current_data.get("current_player"),
        )

Or to quickly toggle off:
    from trm_debug_viz import DEBUG_VIZ_ENABLED
    # At module level or in __init__:
    import trm_debug_viz
    trm_debug_viz.DEBUG_VIZ_ENABLED = False
"""


if __name__ == "__main__":
    # Test visualization with dummy data
    print("Testing debug visualization...")
    
    # Create dummy data
    board = torch.zeros(42, dtype=torch.long)
    board[35:42] = torch.tensor([1, 2, 1, 0, 0, 2, 1])  # Bottom row
    board[28:35] = torch.tensor([0, 1, 2, 0, 0, 0, 0])  # Second row
    
    policy_logits = torch.randn(7)
    policy_logits[3] = 3.0  # Make column 3 preferred
    
    recon_logits = torch.randn(42, 3)
    # Make predictions mostly correct
    for i in range(42):
        recon_logits[i, board[i]] += 5.0
    
    labels = board.clone()
    
    path = debug_viz_single(
        board=board,
        policy_logits=policy_logits,
        recon_logits=recon_logits,
        labels=labels,
        value=0.35,
        player=1,
        save_path="test_debug_viz.png"
    )
    print(f"Saved test visualization to: {path}")