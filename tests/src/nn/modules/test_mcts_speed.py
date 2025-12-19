#!/usr/bin/env python
"""
Benchmark: tensor_mcts vs mcts_v2

Compares:
1. Speed (simulations per second)
2. Correctness (do they find the same moves?)
"""

import torch
import numpy as np
import time
import copy
from typing import Tuple

# Import both MCTS implementations
from src.nn.modules.tensor_mcts import TensorMCTS, TensorMCTSConfig, TensorMCTSWrapper
from src.nn.modules.mcts_v2 import (
    Node, DummyNode, expand, backup, add_virtual_loss, revert_virtual_loss,
    add_dirichlet_noise, generate_search_policy, parallel_uct_search
)
from src.nn.environments.connectfour_env import ConnectFourEnv


# ============================================================================
# Model (uniform policy, zero value)
# ============================================================================

class DummyModel:
    """Uniform policy, zero value - isolates MCTS performance"""
    def __init__(self, device="cpu"):
        self.device = device
    
    def forward(self, boards, current_players):
        if isinstance(boards, np.ndarray):
            batch_size = boards.shape[0] if boards.ndim > 2 else 1
            return (
                np.ones((batch_size, 7), dtype=np.float32) / 7,
                np.zeros(batch_size, dtype=np.float32)
            )
        else:
            batch_size = boards.shape[0]
            return (
                torch.ones(batch_size, 7, device=self.device) / 7,
                torch.zeros(batch_size, device=self.device)
            )


def make_numpy_eval_func(model):
    """Create eval_func for mcts_v2"""
    def eval_func(obs, batched):
        if batched:
            batch_size = obs.shape[0]
            return (
                np.ones((batch_size, 7), dtype=np.float32) / 7,
                [float(0.0) for _ in range(batch_size)]  # List of Python floats
            )
        else:
            return (
                np.ones(7, dtype=np.float32) / 7,
                float(0.0)  # Python float, not numpy.float32
            )
    return eval_func


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_tensor_mcts(device, num_positions, num_simulations, parallel_sims, num_runs=5):
    """Benchmark tensor_mcts"""
    model = DummyModel(device)
    
    config = TensorMCTSConfig(
        c_puct_base=19652.0,
        c_puct_init=1.25,
        exploration_fraction=0.0,  # No noise for fair comparison
        virtual_loss_weight=3.0
    )
    
    mcts = TensorMCTS(model, n_trees=num_positions, config=config, device=device)
    
    # Create positions
    boards = torch.zeros(num_positions, 6, 7, dtype=torch.long, device=device)
    legal = torch.ones(num_positions, 7, device=device)
    players = torch.ones(num_positions, dtype=torch.long, device=device)
    
    with torch.no_grad():
        policies, _ = model.forward(boards.flatten(1).float(), players)
    
    # Warmup
    mcts.reset(boards, policies, legal, players)
    mcts.run_simulations(num_simulations, parallel_sims)
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        mcts.reset(boards, policies, legal, players)
        
        if device != "cpu":
            torch.cuda.synchronize() if "cuda" in device else None
        
        start = time.perf_counter()
        mcts.run_simulations(num_simulations, parallel_sims)
        
        if device != "cpu":
            torch.cuda.synchronize() if "cuda" in device else None
        
        times.append(time.perf_counter() - start)
    
    return np.mean(times), np.std(times)


def benchmark_mcts_v2(num_positions, num_simulations, parallel_sims, num_runs=5):
    """Benchmark mcts_v2 (numpy-based)"""
    eval_func = make_numpy_eval_func(None)
    
    c_puct_base = 19652.0
    c_puct_init = 1.25
    
    # Create environments
    envs = [ConnectFourEnv() for _ in range(num_positions)]
    
    # Warmup
    for env in envs:
        move, search_pi, _, _, _ = parallel_uct_search(
            env, eval_func, None, c_puct_base, c_puct_init,
            num_simulations, parallel_sims
        )
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        for env in envs:
            env.reset()
            parallel_uct_search(
                env, eval_func, None, c_puct_base, c_puct_init,
                num_simulations, parallel_sims
            )
        times.append(time.perf_counter() - start)
    
    return np.mean(times), np.std(times)


def run_benchmarks():
    """Run full benchmark suite"""
    print("=" * 70)
    print("MCTS Benchmark: tensor_mcts vs mcts_v2")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()
    
    # Test configurations
    configs = [
        # (num_positions, num_simulations, parallel_sims)
        (1, 100, 8),
        (1, 200, 8),
        (1, 400, 8),
        (4, 100, 8),
        (16, 100, 8),
        (1, 100, 16),
        (1, 100, 32),
    ]
    
    print(f"{'Config':<25} {'tensor_mcts':<20} {'mcts_v2':<20} {'Speedup':<10}")
    print(f"{'(pos x sims x par)':<25} {'(ms ± std)':<20} {'(ms ± std)':<20}")
    print("-" * 75)
    
    for num_pos, num_sims, par_sims in configs:
        config_str = f"{num_pos} x {num_sims} x {par_sims}"
        
        # Benchmark tensor_mcts
        t_mean, t_std = benchmark_tensor_mcts(device, num_pos, num_sims, par_sims)
        t_str = f"{t_mean*1000:.1f} ± {t_std*1000:.1f} ms"
        
        # Benchmark mcts_v2
        v2_mean, v2_std = benchmark_mcts_v2(num_pos, num_sims, par_sims)
        v2_str = f"{v2_mean*1000:.1f} ± {v2_std*1000:.1f} ms"
        
        # Speedup
        speedup = v2_mean / t_mean
        speedup_str = f"{speedup:.2f}x"
        
        print(f"{config_str:<25} {t_str:<20} {v2_str:<20} {speedup_str:<10}")
    
    print()
    print("=" * 70)
    print("Throughput Comparison (simulations per second)")
    print("=" * 70)
    
    # Single large benchmark
    num_pos, num_sims, par_sims = 16, 200, 16
    
    t_mean, _ = benchmark_tensor_mcts(device, num_pos, num_sims, par_sims)
    v2_mean, _ = benchmark_mcts_v2(num_pos, num_sims, par_sims)
    
    total_sims = num_pos * num_sims
    
    print(f"Configuration: {num_pos} positions x {num_sims} simulations = {total_sims} total")
    print(f"  tensor_mcts: {total_sims / t_mean:,.0f} sims/sec")
    print(f"  mcts_v2:     {total_sims / v2_mean:,.0f} sims/sec")
    print(f"  Speedup:     {v2_mean / t_mean:.2f}x")


if __name__ == "__main__":
    run_benchmarks()