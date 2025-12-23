#!/usr/bin/env python3
"""
Summarize GPU benchmark results from multiple JSON files.

Reads benchmark_*.json files and produces summary tables by architecture and GPU.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import click


def load_benchmark_files(directory: Path) -> list[dict]:
    """Load all benchmark JSON files from directory."""
    results = []
    files = sorted(directory.glob("benchmark_*.json"))
    
    if not files:
        click.echo(f"No benchmark_*.json files found in {directory}", err=True)
        return results
    
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
                results.extend(data)
                click.echo(f"Loaded {len(data)} results from {f.name}")
        except Exception as e:
            click.echo(f"Error loading {f}: {e}", err=True)
    
    return results


def summarize_results(results: list[dict]) -> dict:
    """Organize results by architecture, variant, GPU, and compile mode."""
    # Structure: arch -> variant -> gpu -> compile -> result
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for r in results:
        arch = r["architecture"]
        variant = r["model_variant"]
        gpu = r["gpu_name"].replace("NVIDIA ", "")
        compiled = "compiled" if r["use_compile"] else "eager"
        
        summary[arch][variant][gpu][compiled] = r
    
    return summary


def print_summary_by_architecture(summary: dict):
    """Print summary tables organized by architecture."""
    
    for arch in sorted(summary.keys()):
        variants = summary[arch]
        
        # Collect all GPUs across variants
        all_gpus = set()
        for variant_data in variants.values():
            all_gpus.update(variant_data.keys())
        all_gpus = sorted(all_gpus)
        
        click.echo(f"\n{'='*100}")
        click.echo(f"ARCHITECTURE: {arch.upper()}")
        click.echo(f"{'='*100}")
        
        for variant in sorted(variants.keys()):
            gpu_data = variants[variant]
            
            click.echo(f"\n  Variant: {variant}")
            click.echo(f"  {'-'*96}")
            click.echo(f"  {'GPU':<25} {'Batch':<7} {'Eager (img/s)':<15} {'Compiled (img/s)':<17} {'Speedup':<10} {'Mem (GB)':<10}")
            click.echo(f"  {'-'*96}")
            
            for gpu in all_gpus:
                if gpu not in gpu_data:
                    continue
                
                compiles = gpu_data[gpu]
                eager = compiles.get("eager", {})
                compiled = compiles.get("compiled", {})
                
                eager_throughput = eager.get("samples_per_sec", 0)
                compiled_throughput = compiled.get("samples_per_sec", 0)
                
                batch = eager.get("batch_size") or compiled.get("batch_size", "-")
                mem = eager.get("gpu_memory_peak_gb") or compiled.get("gpu_memory_peak_gb", 0)
                
                if eager_throughput and compiled_throughput:
                    speedup = f"{compiled_throughput / eager_throughput:.2f}x"
                else:
                    speedup = "-"
                
                eager_str = f"{eager_throughput:.1f}" if eager_throughput else "-"
                compiled_str = f"{compiled_throughput:.1f}" if compiled_throughput else "-"
                mem_str = f"{mem:.1f}" if mem else "-"
                
                click.echo(f"  {gpu:<25} {batch:<7} {eager_str:<15} {compiled_str:<17} {speedup:<10} {mem_str:<10}")


def print_summary_by_gpu(summary: dict):
    """Print summary tables organized by GPU."""
    
    # Reorganize: gpu -> arch -> variant -> compile -> result
    by_gpu = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for arch, variants in summary.items():
        for variant, gpus in variants.items():
            for gpu, compiles in gpus.items():
                for compile_mode, result in compiles.items():
                    by_gpu[gpu][arch][variant][compile_mode] = result
    
    for gpu in sorted(by_gpu.keys()):
        arch_data = by_gpu[gpu]
        
        click.echo(f"\n{'='*100}")
        click.echo(f"GPU: {gpu}")
        click.echo(f"{'='*100}")
        click.echo(f"{'Arch':<12} {'Variant':<10} {'Batch':<7} {'Eager (img/s)':<15} {'Compiled (img/s)':<17} {'Speedup':<10} {'Mem (GB)':<10}")
        click.echo(f"{'-'*100}")
        
        for arch in sorted(arch_data.keys()):
            variants = arch_data[arch]
            
            for variant in sorted(variants.keys()):
                compiles = variants[variant]
                
                eager = compiles.get("eager", {})
                compiled = compiles.get("compiled", {})
                
                eager_throughput = eager.get("samples_per_sec", 0)
                compiled_throughput = compiled.get("samples_per_sec", 0)
                
                batch = eager.get("batch_size") or compiled.get("batch_size", "-")
                mem = eager.get("gpu_memory_peak_gb") or compiled.get("gpu_memory_peak_gb", 0)
                
                if eager_throughput and compiled_throughput:
                    speedup = f"{compiled_throughput / eager_throughput:.2f}x"
                else:
                    speedup = "-"
                
                eager_str = f"{eager_throughput:.1f}" if eager_throughput else "-"
                compiled_str = f"{compiled_throughput:.1f}" if compiled_throughput else "-"
                mem_str = f"{mem:.1f}" if mem else "-"
                
                click.echo(f"{arch:<12} {variant:<10} {batch:<7} {eager_str:<15} {compiled_str:<17} {speedup:<10} {mem_str:<10}")


def print_comparison_matrix(summary: dict):
    """Print GPU comparison matrix for each architecture/variant."""
    
    click.echo(f"\n{'='*100}")
    click.echo("GPU COMPARISON MATRIX (Compiled throughput, img/s)")
    click.echo(f"{'='*100}")
    
    # Collect all GPUs
    all_gpus = set()
    for variants in summary.values():
        for gpu_data in variants.values():
            all_gpus.update(gpu_data.keys())
    all_gpus = sorted(all_gpus)
    
    # Dynamic column width based on GPU names
    gpu_width = max(10, max(len(g) for g in all_gpus) + 1)
    
    # Header
    header = f"{'Arch/Variant':<20}"
    for gpu in all_gpus:
        header += f" {gpu:>{gpu_width}}"
    click.echo(header)
    click.echo("-" * len(header))
    
    for arch in sorted(summary.keys()):
        for variant in sorted(summary[arch].keys()):
            row = f"{arch}/{variant:<13}"
            
            for gpu in all_gpus:
                gpu_data = summary[arch][variant].get(gpu, {})
                compiled = gpu_data.get("compiled", {})
                eager = gpu_data.get("eager", {})
                
                # Prefer compiled, fall back to eager
                throughput = compiled.get("samples_per_sec") or eager.get("samples_per_sec")
                
                if throughput:
                    row += f" {throughput:>{gpu_width}.1f}"
                else:
                    row += f" {'-':>{gpu_width}}"
            
            click.echo(row)


def print_gpu_speedup_summary(summary: dict):
    """Print average speedup for each GPU relative to the slowest GPU."""
    
    click.echo(f"\n{'='*100}")
    click.echo("GPU SPEEDUP SUMMARY (relative to slowest GPU, averaged across all models)")
    click.echo(f"{'='*100}")
    
    # Collect all GPUs
    all_gpus = set()
    for variants in summary.values():
        for gpu_data in variants.values():
            all_gpus.update(gpu_data.keys())
    all_gpus = sorted(all_gpus)
    
    if len(all_gpus) < 2:
        click.echo("Need at least 2 GPUs to compare.")
        return
    
    # Collect throughput for each (arch, variant, gpu) combination
    # Structure: (arch, variant) -> {gpu: throughput}
    model_throughputs = defaultdict(dict)
    
    for arch, variants in summary.items():
        for variant, gpus in variants.items():
            key = (arch, variant)
            for gpu, compiles in gpus.items():
                # Prefer compiled, fall back to eager
                compiled = compiles.get("compiled", {})
                eager = compiles.get("eager", {})
                throughput = compiled.get("samples_per_sec") or eager.get("samples_per_sec")
                if throughput:
                    model_throughputs[key][gpu] = throughput
    
    # For each model, find the slowest GPU and calculate speedups
    # gpu -> list of speedups across models
    gpu_speedups = defaultdict(list)
    
    for (arch, variant), gpu_data in model_throughputs.items():
        if len(gpu_data) < 2:
            continue  # Need at least 2 GPUs for comparison
        
        # Find slowest throughput for this model
        min_throughput = min(gpu_data.values())
        
        for gpu, throughput in gpu_data.items():
            speedup = throughput / min_throughput
            gpu_speedups[gpu].append({
                "model": f"{arch}/{variant}",
                "speedup": speedup,
                "throughput": throughput,
            })
    
    if not gpu_speedups:
        click.echo("No comparable results found.")
        return
    
    # Calculate average speedup per GPU
    gpu_avg_speedups = {}
    for gpu, speedups in gpu_speedups.items():
        avg = sum(s["speedup"] for s in speedups) / len(speedups)
        gpu_avg_speedups[gpu] = {
            "avg_speedup": avg,
            "num_models": len(speedups),
            "details": speedups,
        }
        
    # Sort GPUs by average speedup (descending)
    sorted_gpus = sorted(gpu_avg_speedups.keys(), key=lambda g: gpu_avg_speedups[g]["avg_speedup"], reverse=True)
    
    # Find the baseline (slowest) GPU
    baseline_gpu = sorted_gpus[-1]
    
    # Print summary table
    click.echo(f"\n{'GPU':<25} {'Avg Speedup':>12} {'Models':>8}")
    click.echo("-" * 50)
    
    for gpu in sorted_gpus:
        data = gpu_avg_speedups[gpu]
        speedup_str = f"{data['avg_speedup']:.2f}x"
        click.echo(f"{gpu:<25} {speedup_str:>12} {data['num_models']:>8}")
    
    click.echo("-" * 50)
    click.echo(f"Baseline (1.00x): {baseline_gpu}")
    
    # Print detailed breakdown
    click.echo(f"\n{'Detailed speedups by model:'}")
    click.echo("-" * 80)
    
    # Get all models
    all_models = sorted(set(s["model"] for speedups in gpu_speedups.values() for s in speedups))
    
    # Dynamic column width
    gpu_width = max(12, max(len(g) for g in sorted_gpus) + 1)
    
    header = f"{'Model':<20}"
    for gpu in sorted_gpus:
        header += f" {gpu:>{gpu_width}}"
    click.echo(header)
    click.echo("-" * len(header))
    
    for model in all_models:
        row = f"{model:<20}"
        for gpu in sorted_gpus:
            # Find speedup for this gpu/model
            speedup_data = next((s for s in gpu_speedups[gpu] if s["model"] == model), None)
            if speedup_data:
                row += f" {speedup_data['speedup']:>{gpu_width}.2f}x"
            else:
                row += f" {'-':>{gpu_width}}"
        click.echo(row)


def export_csv(summary: dict, output_path: Path):
    """Export results to CSV."""
    rows = []
    
    for arch, variants in summary.items():
        for variant, gpus in variants.items():
            for gpu, compiles in gpus.items():
                for compile_mode, r in compiles.items():
                    rows.append({
                        "architecture": arch,
                        "variant": variant,
                        "gpu": gpu,
                        "mode": compile_mode,
                        "batch_size": r.get("batch_size"),
                        "samples_per_sec": r.get("samples_per_sec"),
                        "gpu_memory_peak_gb": r.get("gpu_memory_peak_gb"),
                        "time_per_epoch_sec": r.get("time_per_epoch_sec"),
                    })
    
    if not rows:
        return
    
    import csv
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    click.echo(f"\nCSV exported to: {output_path}")


@click.command()
@click.argument("directory", type=click.Path(exists=True), default=".")
@click.option("--by-gpu", is_flag=True, help="Organize output by GPU instead of architecture")
@click.option("--matrix", is_flag=True, help="Show GPU comparison matrix")
@click.option("--speedup", is_flag=True, help="Show average GPU speedup relative to slowest GPU")
@click.option("--csv", "csv_path", type=click.Path(), help="Export to CSV file")
@click.option("--all", "show_all", is_flag=True, help="Show all views")
def main(directory: str, by_gpu: bool, matrix: bool, speedup: bool, csv_path: str, show_all: bool):
    """Summarize GPU benchmark results.
    
    Reads all benchmark_*.json files from DIRECTORY (default: current directory)
    and prints summary tables.
    
    Examples:
    
    \b
    # Default: summarize by architecture
    python summarize_benchmarks.py ./results/
    
    \b
    # Summarize by GPU
    python summarize_benchmarks.py ./results/ --by-gpu
    
    \b
    # Show comparison matrix
    python summarize_benchmarks.py ./results/ --matrix
    
    \b
    # Show GPU speedup comparison
    python summarize_benchmarks.py ./results/ --speedup
    
    \b
    # Show everything and export CSV
    python summarize_benchmarks.py ./results/ --all --csv results.csv
    """
    dir_path = Path(directory)
    results = load_benchmark_files(dir_path)
    
    if not results:
        sys.exit(1)
    
    click.echo(f"\nTotal results loaded: {len(results)}")
    
    summary = summarize_results(results)
    
    if show_all:
        print_summary_by_architecture(summary)
        print_summary_by_gpu(summary)
        print_comparison_matrix(summary)
        print_gpu_speedup_summary(summary)
    elif by_gpu:
        print_summary_by_gpu(summary)
    elif matrix:
        print_comparison_matrix(summary)
    elif speedup:
        print_gpu_speedup_summary(summary)
    else:
        print_summary_by_architecture(summary)
    
    if csv_path:
        export_csv(summary, Path(csv_path))


if __name__ == "__main__":
    main()