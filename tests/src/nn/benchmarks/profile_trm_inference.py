"""
TRM Inference Profiler
Profiles TRM inference time against key hyperparameters:
- H_cycles, L_cycles, num_layers, N_supervision_val, hidden_size

Run from project root: python trm_profiler.py
"""

import torch
import torch.nn as nn
import torch._dynamo
import time
import sys
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

torch.set_float32_matmul_precision('high')

# Import TRM
try:
    from src.nn.models.trm import TRMModule
except ImportError:
    print("Error: Could not import TRMModule. Run this script from the project root.")
    sys.exit(1)

# ==========================================
# Configuration
# ==========================================
INPUT_SIZE = 32
BATCH_SIZE = 512
SEQ_LEN = INPUT_SIZE * INPUT_SIZE  # 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_BF16 = True
TARGET_DTYPE = torch.bfloat16

# Warmup and timing iterations
WARMUP_ITERS = 10
TIMING_ITERS = 20

# Base TRM config (will be modified for each sweep)
BASE_CONFIG = {
    "vocab_size": 256,
    "hidden_size": 256,
    "num_layers": 2,
    "num_heads": 8,
    "H_cycles": 2,
    "L_cycles": 2,
    "N_supervision_val": 4,
    "puzzle_emb_dim": 0,
    "puzzle_emb_len": 0,
    "pos_emb_type": "1d",
    "use_mlp_t": False,
    "forward_dtype": TARGET_DTYPE,
    "seq_len": SEQ_LEN,
    "max_grid_size": INPUT_SIZE,
}

# Hyperparameter sweep ranges
SWEEPS = {
    "H_cycles": [1, 2, 3, 4, 6, 8],
    "L_cycles": [1, 2, 3, 4, 6, 8],
    "num_layers": [1, 2, 3, 4, 6],
    "N_supervision_val": [1, 2, 4, 8, 12, 16],
    "hidden_size": [128, 192, 256, 384, 512],
}

print(f"{'='*60}")
print(f"TRM Inference Profiler")
print(f"{'='*60}")
print(f"Device: {DEVICE.upper()}")
print(f"Precision: {'BF16' if USE_BF16 else 'FP32'}")
print(f"Input: {INPUT_SIZE}x{INPUT_SIZE} (seq_len={SEQ_LEN})")
print(f"Batch size: {BATCH_SIZE}")
print(f"{'='*60}\n")


# ==========================================
# TRM Adapter for Inference
# ==========================================
class TRMInferenceAdapter(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.model = TRMModule(**config)
        self.n_supervision = config.get("N_supervision_val", 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run full inference with N_supervision iterations."""
        # x is (B, seq_len) token indices
        x_tokens = x.to(torch.int32)
        dummy_ids = torch.zeros((x.size(0),), dtype=torch.long, device=x.device)
        
        batch_data = {"input": x_tokens, "puzzle_identifiers": dummy_ids}
        carry = self.model.initial_carry(batch_data)
        
        # Run N_supervision iterations (simulating full inference)
        for _ in range(self.n_supervision):
            carry, outputs = self.model.forward(carry, batch_data)
        
        return outputs["logits"]


# ==========================================
# Profiling Functions
# ==========================================
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def profile_model(model: nn.Module, x: torch.Tensor, 
                  warmup: int = WARMUP_ITERS, 
                  iters: int = TIMING_ITERS) -> Tuple[float, float]:
    """
    Profile model inference time.
    Returns: (mean_time_ms, std_time_ms)
    """
    model.eval()
    times = []
    
    try:
        with torch.no_grad():
            # Warmup
            for _ in range(warmup):
                _ = model(x)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            
            # Timing
            for _ in range(iters):
                if DEVICE == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(x)
                if DEVICE == 'cuda':
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)  # ms
        
        return np.mean(times), np.std(times)
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return float('inf'), 0.0
        raise


def build_and_profile(config: dict, x: torch.Tensor) -> Dict:
    """Build model with config and profile it."""
    # Reset dynamo cache to avoid recompilation issues across configs
    torch._dynamo.reset()
    
    # Adjust num_heads based on hidden_size
    hidden = config.get("hidden_size", 256)
    config["num_heads"] = min(8, hidden // 32)  # At least 32 dims per head
    
    try:
        model = TRMInferenceAdapter(config).to(DEVICE)
        
        # Compile for faster inference
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="default")
            except:
                pass  # Skip compilation if it fails
        
        params = count_params(model)
        mean_ms, std_ms = profile_model(model, x)
        
        # Cleanup
        del model
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        
        return {
            "params": params,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "throughput": BATCH_SIZE / (mean_ms / 1000) if mean_ms < float('inf') else 0,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return {"params": 0, "mean_ms": float('inf'), "std_ms": 0, "throughput": 0}


# ==========================================
# Main Profiling Loop
# ==========================================
def run_sweeps() -> Dict[str, List[Dict]]:
    """Run all hyperparameter sweeps."""
    results = {}
    
    # Create dummy input (B, seq_len) - flattened tokens
    x = torch.randint(0, 256, (BATCH_SIZE, SEQ_LEN)).to(DEVICE)
    
    for param_name, values in SWEEPS.items():
        print(f"\n{'='*60}")
        print(f"Sweeping: {param_name}")
        print(f"{'='*60}")
        
        sweep_results = []
        
        for val in values:
            # Create config with this parameter varied
            config = BASE_CONFIG.copy()
            config[param_name] = val
            
            print(f"  {param_name}={val}...", end=" ", flush=True)
            result = build_and_profile(config, x)
            result["value"] = val
            sweep_results.append(result)
            
            if result["mean_ms"] < float('inf'):
                print(f"mean={result['mean_ms']:.2f}ms, std={result['std_ms']:.2f}ms, "
                      f"throughput={result['throughput']:.1f} samples/s, "
                      f"params={result['params']/1e6:.2f}M")
            else:
                print("OOM")
        
        results[param_name] = sweep_results
    
    return results


# ==========================================
# Visualization
# ==========================================
def plot_results(results: Dict[str, List[Dict]], save_path: str = "trm_profile.png"):
    """Create visualization of profiling results with plots and tables."""
    
    # Set up the figure with a dark theme
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#1a1a2e')
    
    # Create grid: 2 rows x 5 columns (plots on top, tables below)
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 0.6], hspace=0.4, wspace=0.3)
    
    # Color palette - vibrant on dark
    colors = {
        "H_cycles": "#00d4ff",
        "L_cycles": "#ff6b6b", 
        "num_layers": "#4ecdc4",
        "N_supervision_val": "#ffe66d",
        "hidden_size": "#c56cf0",
    }
    
    param_labels = {
        "H_cycles": "H-Cycles",
        "L_cycles": "L-Cycles",
        "num_layers": "Layers",
        "N_supervision_val": "N_supervision",
        "hidden_size": "Hidden Size",
    }
    
    for idx, (param_name, sweep_results) in enumerate(results.items()):
        # Plot on top row
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor('#16213e')
        
        values = [r["value"] for r in sweep_results]
        throughputs = [r["throughput"] for r in sweep_results]
        times = [r["mean_ms"] for r in sweep_results]
        
        # Filter out OOM results
        valid = [(v, tp, t) for v, tp, t in zip(values, throughputs, times) if t < float('inf')]
        if not valid:
            ax.text(0.5, 0.5, "All OOM", ha='center', va='center', fontsize=14, color='red')
            continue
        
        values, throughputs, times = zip(*valid)
        
        color = colors.get(param_name, "#ffffff")
        
        # Plot throughput
        ax.plot(values, throughputs, 
               marker='o', markersize=8, 
               linewidth=2.5,
               color=color, markerfacecolor=color,
               markeredgecolor='white', markeredgewidth=1.5,
               alpha=0.9)
        
        ax.set_xlabel(param_labels.get(param_name, param_name), fontsize=10, fontweight='bold')
        ax.set_ylabel("Throughput (samples/s)", fontsize=9, fontweight='bold')
        ax.set_title(f"{param_name}", fontsize=11, fontweight='bold', color=color, pad=8)
        
        # Grid styling
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#444')
        ax.spines['bottom'].set_color('#444')
        
        # Set y-axis to log scale
        ax.set_yscale('log')
        
        # Table on bottom row
        ax_table = fig.add_subplot(gs[1, idx])
        ax_table.set_facecolor('#16213e')
        ax_table.axis('off')
        
        # Prepare table data
        table_data = []
        for r in sweep_results:
            if r["mean_ms"] < float('inf'):
                table_data.append([
                    f"{r['value']}",
                    f"{r['throughput']:.0f}",
                    f"{r['mean_ms']:.1f}",
                    f"{r['params']/1e6:.2f}M"
                ])
            else:
                table_data.append([f"{r['value']}", "OOM", "-", "-"])
        
        table = ax_table.table(
            cellText=table_data,
            colLabels=['Value', 'Tput', 'ms', 'Params'],
            loc='center',
            cellLoc='center',
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)
        
        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor('#0f3460' if row == 0 else '#16213e')
            cell.set_edgecolor('#444')
            cell.set_text_props(color='white' if row == 0 else '#e0e0e0')
            if row == 0:
                cell.set_text_props(fontweight='bold', color=color)
    
    plt.suptitle("TRM Hyperparameter Profiling", 
                fontsize=16, fontweight='bold', 
                color='white', y=0.98)
    
    plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', edgecolor='none', 
                bbox_inches='tight', pad_inches=0.3)
    print(f"\nPlot saved to: {save_path}")
    
    return fig


def plot_scaling_analysis(results: Dict[str, List[Dict]], save_path: str = "trm_scaling.png"):
    """Create a scaling analysis plot showing throughput vs parameters/iterations."""
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')
    
    colors = {
        "H_cycles": "#00d4ff",
        "L_cycles": "#ff6b6b", 
        "num_layers": "#4ecdc4",
        "N_supervision_val": "#ffe66d",
        "hidden_size": "#c56cf0",
    }
    
    # Left plot: Throughput vs Parameter count (for hidden_size and num_layers)
    ax1 = axes[0]
    ax1.set_facecolor('#16213e')
    
    for param_name in ["hidden_size", "num_layers"]:
        if param_name not in results:
            continue
        sweep = results[param_name]
        params = [r["params"]/1e6 for r in sweep if r["mean_ms"] < float('inf')]
        throughputs = [r["throughput"] for r in sweep if r["mean_ms"] < float('inf')]
        
        if params:
            ax1.scatter(params, throughputs, s=100, label=param_name, 
                       color=colors[param_name], alpha=0.8, edgecolors='white', linewidth=1)
            ax1.plot(params, throughputs, color=colors[param_name], alpha=0.5, linewidth=2)
    
    ax1.set_xlabel("Parameters (M)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Throughput (samples/s)", fontsize=12, fontweight='bold')
    ax1.set_title("Throughput vs Model Size", fontsize=13, fontweight='bold', color='white')
    ax1.legend(framealpha=0.3)
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_yscale('log')
    
    # Right plot: Throughput vs Compute multiplier (for cycles)
    ax2 = axes[1]
    ax2.set_facecolor('#16213e')
    
    for param_name in ["H_cycles", "L_cycles", "N_supervision_val"]:
        if param_name not in results:
            continue
        sweep = results[param_name]
        values = [r["value"] for r in sweep if r["mean_ms"] < float('inf')]
        throughputs = [r["throughput"] for r in sweep if r["mean_ms"] < float('inf')]
        
        if values:
            ax2.scatter(values, throughputs, s=100, label=param_name,
                       color=colors[param_name], alpha=0.8, edgecolors='white', linewidth=1)
            ax2.plot(values, throughputs, color=colors[param_name], alpha=0.5, linewidth=2)
    
    ax2.set_xlabel("Iteration Count", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Throughput (samples/s)", fontsize=12, fontweight='bold')
    ax2.set_title("Throughput vs Iteration Depth", fontsize=13, fontweight='bold', color='white')
    ax2.legend(framealpha=0.3)
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.set_yscale('log')
    
    plt.suptitle("TRM Scaling Analysis", fontsize=16, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', edgecolor='none',
                bbox_inches='tight', pad_inches=0.3)
    print(f"Scaling plot saved to: {save_path}")
    
    return fig


def plot_compute_breakdown(results: Dict[str, List[Dict]], save_path: str = "trm_compute_breakdown.png"):
    """
    Create a compute breakdown analysis showing how different iteration types
    contribute to total compute (normalized throughput curves).
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    
    colors = {"H_cycles": "#00d4ff", "L_cycles": "#ff6b6b", "N_supervision_val": "#ffe66d"}
    
    h_data = results.get("H_cycles", [])
    l_data = results.get("L_cycles", [])
    n_data = results.get("N_supervision_val", [])
    
    if h_data and l_data and n_data:
        # Get baseline throughput (value=1)
        h_base = next((r["throughput"] for r in h_data if r["value"] == 1), None)
        l_base = next((r["throughput"] for r in l_data if r["value"] == 1), None)
        n_base = next((r["throughput"] for r in n_data if r["value"] == 1), None)
        
        if all([h_base, l_base, n_base]):
            # Plot normalized throughput curves (relative to value=1)
            for name, data, base in [("H_cycles", h_data, h_base), 
                                      ("L_cycles", l_data, l_base),
                                      ("N_supervision_val", n_data, n_base)]:
                values = [r["value"] for r in data if r["mean_ms"] < float('inf')]
                # Normalized: base/current (so higher is better, 1.0 at value=1)
                normalized = [base/r["throughput"] for r in data if r["mean_ms"] < float('inf')]
                
                ax.plot(values, normalized, marker='o', markersize=8, 
                       linewidth=2.5, label=name, color=colors[name])
                
                # Plot linear scaling reference
                ax.plot(values, values, '--', alpha=0.3, color=colors[name])
            
            ax.set_xlabel("Parameter Value", fontsize=12, fontweight='bold')
            ax.set_ylabel("Slowdown Factor (relative to value=1)", fontsize=12, fontweight='bold')
            ax.set_title("Iteration Scaling Efficiency", fontsize=14, fontweight='bold', color='white')
            ax.legend(framealpha=0.3, loc='upper left')
            ax.grid(True, alpha=0.2, linestyle='--')
            
            # Add annotation
            ax.text(0.95, 0.05, "Dashed lines = linear scaling\nBelow line = sublinear (good)\nAbove line = superlinear (bad)",
                   transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
                   color='#888', bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', edgecolor='none',
                bbox_inches='tight', pad_inches=0.3)
    print(f"Compute breakdown saved to: {save_path}")
    
    return fig


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    # Run profiling
    results = run_sweeps()
    
    # Save raw results
    results_serializable = {
        k: [{kk: (vv if not isinstance(vv, float) or vv != float('inf') else "inf") 
             for kk, vv in r.items()} for r in v]
        for k, v in results.items()
    }
    
    with open("trm_profile_results.json", "w") as f:
        json.dump(results_serializable, f, indent=2)
    print("\nResults saved to: trm_profile_results.json")
    
    # Generate plots
    plot_results(results, "trm_profile.png")
    plot_scaling_analysis(results, "trm_scaling.png")
    plot_compute_breakdown(results, "trm_compute_breakdown.png")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Parameter':<20} {'Value':<10} {'Time (ms)':<15} {'Throughput':<15} {'Params':<12}")
    print("-"*80)
    
    for param_name, sweep_results in results.items():
        for r in sweep_results:
            if r["mean_ms"] < float('inf'):
                print(f"{param_name:<20} {r['value']:<10} {r['mean_ms']:<15.2f} "
                      f"{r['throughput']:<15.1f} {r['params']/1e6:<12.2f}M")
            else:
                print(f"{param_name:<20} {r['value']:<10} {'OOM':<15} {'-':<15} {'-':<12}")
    
    print("="*80)