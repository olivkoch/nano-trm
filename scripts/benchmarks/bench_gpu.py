#!/usr/bin/env python3
"""
GPU Benchmarking Script for Deep Learning Training

Supports multiple architectures:
- ConvNeXt (classification)
- ResNet (classification)  
- Diffusion UNet (denoising)

Automatically finds and uses the maximum batch size for each configuration.
Uses ImageNette (10-class ImageNet subset) by default for classification benchmarks.
"""

import gc
import json
import shutil
import tarfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import click

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("WARNING: timm not installed. Install with: pip install timm")

try:
    from diffusers import UNet2DModel
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False


IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGENETTE_DIR = Path.home() / ".cache" / "imagenette2-320"

# Model architecture definitions
ARCHITECTURES = {
    "convnext": {
        "variants": ["base", "large", "xlarge", "xxlarge"],
        "default_variants": ["base", "large", "xlarge", "xxlarge"],
        "task": "classification",
    },
    "resnet": {
        "variants": ["50", "101", "152"],
        "default_variants": ["50", "101", "152"],
        "task": "classification",
    },
    "diffusion": {
        "variants": ["small", "medium", "large"],
        "default_variants": ["small", "medium", "large"],
        "task": "diffusion",
    },
}

# Diffusion UNet configurations (channels, layers, attention resolutions)
DIFFUSION_CONFIGS = {
    "small": {
        "block_out_channels": (128, 256, 256, 256),
        "layers_per_block": 2,
        "down_block_types": ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        "up_block_types": ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    },
    "medium": {
        "block_out_channels": (128, 256, 512, 512),
        "layers_per_block": 2,
        "down_block_types": ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        "up_block_types": ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    },
    "large": {
        "block_out_channels": (256, 512, 768, 768),
        "layers_per_block": 3,
        "down_block_types": ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        "up_block_types": ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    },
}


def create_model(
    arch: str,
    variant: str,
    num_classes: int = 10,
) -> nn.Module:
    """Create a model for the given architecture and variant."""
    if arch == "convnext":
        if not HAS_TIMM:
            raise ImportError("timm required for ConvNeXt. Install with: pip install timm")
        model_name = f"convnext_{variant}"
        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    
    elif arch == "resnet":
        if not HAS_TIMM:
            raise ImportError("timm required for ResNet. Install with: pip install timm")
        model_name = f"resnet{variant}"
        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    
    elif arch == "diffusion":
        if not HAS_DIFFUSERS:
            raise ImportError("diffusers required for Diffusion. Install with: pip install diffusers")
        config = DIFFUSION_CONFIGS[variant]
        return UNet2DModel(
            sample_size=64,  # 64x64 images for diffusion
            in_channels=3,
            out_channels=3,
            **config,
        )
    
    else:
        raise ValueError(f"Unknown architecture: {arch}")

import os

DEFAULT_ARCH = "convnext"
DEFAULT_EPOCHS = 10  # ~30 min total for 4 models × 2 compile options on H100
DEFAULT_NUM_WORKERS = min(4, os.cpu_count() or 4)
DEFAULT_PREFETCH_FACTOR = 2
IMAGENET_SIZE = (224, 224)
DIFFUSION_SIZE = (64, 64)
IMAGENETTE_NUM_CLASSES = 10
IMAGENET_NUM_CLASSES = 1000

# Batch sizes to try (descending order - will use first that works)
BATCH_SIZE_CANDIDATES = [512, 384, 256, 192, 128, 96, 64, 48, 32, 16, 8]


def download_imagenette(target_dir: Path = IMAGENETTE_DIR) -> Path:
    """Download and extract ImageNette dataset."""
    if target_dir.exists() and (target_dir / "train").exists():
        print(f"ImageNette already downloaded at {target_dir}")
        return target_dir
    
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    tgz_path = target_dir.parent / "imagenette2-320.tgz"
    
    if not tgz_path.exists():
        print(f"Downloading ImageNette (~160MB)...")
        
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            pct = min(100, downloaded * 100 // total_size)
            print(f"\r  Progress: {pct}% ({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)", end="", flush=True)
        
        urlretrieve(IMAGENETTE_URL, tgz_path, reporthook=progress_hook)
        print()
    
    print(f"Extracting to {target_dir}...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(target_dir.parent)
    
    print(f"ImageNette ready at {target_dir}")
    return target_dir


@dataclass
class BenchmarkResult:
    architecture: str
    model_variant: str
    batch_size: int
    use_compile: bool
    compile_mode: Optional[str]
    use_amp: bool
    num_epochs: int
    total_time_sec: float
    time_per_epoch_sec: float
    samples_per_sec: float
    gpu_name: str
    gpu_memory_peak_gb: float
    num_workers: int
    prefetch_factor: int
    dataset_type: str
    num_samples: int


class SyntheticDataset(Dataset):
    """Synthetic dataset for benchmarking classification or diffusion."""
    
    def __init__(
        self,
        num_samples: int = 100000,
        num_classes: int = IMAGENETTE_NUM_CLASSES,
        task: str = "classification",
        image_size: tuple[int, int] = IMAGENET_SIZE,
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.task = task
        self.image_size = image_size
        self.image_shape = (3, *image_size)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.task == "diffusion":
            # For diffusion: return image and noise target
            image = torch.randn(self.image_shape, dtype=torch.float32)
            noise = torch.randn(self.image_shape, dtype=torch.float32)
            timestep = torch.randint(0, 1000, (1,)).item()
            return image, noise, timestep
        else:
            # For classification: return image and label
            image = torch.randn(self.image_shape, dtype=torch.float32)
            label = idx % self.num_classes
            return image, label


def get_imagenet_transforms():
    """Standard ImageNet training transforms."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = DEFAULT_NUM_WORKERS,
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR,
) -> DataLoader:
    """Create an optimized DataLoader for GPU training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def get_gpu_info() -> tuple[str, float]:
    """Get GPU name and peak memory usage."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.max_memory_allocated(0) / (1024**3)
        return gpu_name, memory_gb
    return "CPU", 0.0


def clear_gpu_memory():
    """Clear GPU memory and caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def test_batch_size(
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    task: str = "classification",
    num_classes: int = 10,
    image_size: tuple[int, int] = IMAGENET_SIZE,
) -> bool:
    """Test if a batch size fits in memory by running one forward+backward pass."""
    clear_gpu_memory()
    
    try:
        if task == "diffusion":
            dummy_input = torch.randn(batch_size, 3, *image_size, device=device)
            dummy_timesteps = torch.randint(0, 1000, (batch_size,), device=device)
            dummy_target = torch.randn(batch_size, 3, *image_size, device=device)
            
            with autocast('cuda', enabled=use_amp):
                output = model(dummy_input, dummy_timesteps).sample
                loss = nn.functional.mse_loss(output, dummy_target)
        else:
            dummy_input = torch.randn(batch_size, 3, *image_size, device=device)
            dummy_target = torch.randint(0, num_classes, (batch_size,), device=device)
            criterion = nn.CrossEntropyLoss()
            
            with autocast('cuda', enabled=use_amp):
                output = model(dummy_input)
                loss = criterion(output, dummy_target)
        
        loss.backward()
        torch.cuda.synchronize()
        
        del dummy_input, dummy_target, output, loss
        clear_gpu_memory()
        return True
        
    except RuntimeError as e:
        error_str = str(e).lower()
        if "out of memory" in error_str:
            clear_gpu_memory()
            return False
        # Handle CUDA assertion errors (can happen with torch.compile on some GPUs)
        if "cuda error" in error_str or "device-side assert" in error_str:
            print(f"\n    CUDA error (trying smaller batch)...", end=" ")
            # Reset CUDA context
            torch.cuda.synchronize()
            clear_gpu_memory()
            return False
        raise
    except Exception as e:
        # Catch torch.AcceleratorError and other CUDA-related errors
        error_str = str(e).lower()
        if "cuda" in error_str or "accelerator" in error_str:
            print(f"\n    CUDA error (trying smaller batch)...", end=" ")
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            clear_gpu_memory()
            return False
        raise


def find_max_batch_size(
    arch: str,
    variant: str,
    device: torch.device,
    use_amp: bool,
    use_compile: bool,
    compile_mode: str,
    num_classes: int = IMAGENETTE_NUM_CLASSES,
) -> Optional[int]:
    """Find the maximum batch size that fits in GPU memory.
    
    Tests with uncompiled model to avoid torch.compile inductor bugs.
    If compiled training needs more memory, OOM retry logic handles it.
    """
    
    task = ARCHITECTURES[arch]["task"]
    image_size = DIFFUSION_SIZE if task == "diffusion" else IMAGENET_SIZE
    
    print(f"  Finding max batch size for {arch}/{variant} (compile={use_compile})...")
    
    # Always test with uncompiled model to avoid torch.compile inductor bugs
    model = create_model(arch, variant, num_classes=num_classes).to(device)
    model.train()
    
    max_batch = None
    for bs in BATCH_SIZE_CANDIDATES:
        print(f"    Trying batch_size={bs}...", end=" ", flush=True)
        if test_batch_size(model, bs, device, use_amp, task, num_classes, image_size):
            print("OK")
            max_batch = bs
            break
        else:
            print("OOM")
    
    del model
    clear_gpu_memory()
    
    if max_batch:
        print(f"  -> Max batch size: {max_batch}")
    else:
        print(f"  -> No valid batch size found!")
    
    return max_batch


def run_training_benchmark(
    arch: str,
    variant: str,
    batch_size: int,
    num_epochs: int,
    dataset: Dataset,
    num_classes: int,
    use_compile: bool,
    compile_mode: str,
    use_amp: bool,
    num_workers: int,
    prefetch_factor: int,
) -> Optional[BenchmarkResult]:
    """Run the actual training benchmark with a known-good batch size.
    
    Returns None if OOM occurs during training.
    """
    
    clear_gpu_memory()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = ARCHITECTURES[arch]["task"]
    
    try:
        # Create model
        model = create_model(arch, variant, num_classes=num_classes).to(device)
        
        if use_compile:
            model = torch.compile(model, mode=compile_mode)
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
    
        # Setup training
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        scaler = GradScaler('cuda') if use_amp else None
        
        if task == "classification":
            criterion = nn.CrossEntropyLoss()
    
        # Warmup (triggers compilation)
        print("  Warming up...", flush=True)
        model.train()
        
        for batch in dataloader:
            if task == "diffusion":
                images, noise, timesteps = batch
                images = images.to(device, non_blocking=True)
                noise = noise.to(device, non_blocking=True)
                timesteps = torch.tensor([timesteps] * images.size(0), device=device) if isinstance(timesteps, int) else timesteps.to(device, non_blocking=True)
                
                with autocast('cuda', enabled=use_amp):
                    outputs = model(images, timesteps).sample
                    loss = nn.functional.mse_loss(outputs, noise)
            else:
                images, labels = batch
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with autocast('cuda', enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            break
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        torch.cuda.reset_peak_memory_stats()
        
        # Run benchmark
        print(f"  Training {num_epochs} epochs...", flush=True)
        total_samples = 0
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()
            epoch_samples = 0
            
            for batch in dataloader:
                if task == "diffusion":
                    images, noise, timesteps = batch
                    images = images.to(device, non_blocking=True)
                    noise = noise.to(device, non_blocking=True)
                    timesteps = torch.tensor([timesteps] * images.size(0), device=device) if isinstance(timesteps, int) else timesteps.to(device, non_blocking=True)
                    
                    with autocast('cuda', enabled=use_amp):
                        outputs = model(images, timesteps).sample
                        loss = nn.functional.mse_loss(outputs, noise)
                    
                    batch_samples = images.size(0)
                else:
                    images, labels = batch
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    with autocast('cuda', enabled=use_amp):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    batch_samples = images.size(0)
                
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                epoch_samples += batch_samples
            
            total_samples += epoch_samples
            epoch_time = time.perf_counter() - epoch_start
            print(f"    Epoch {epoch+1}/{num_epochs}: {epoch_time:.2f}s ({epoch_samples/epoch_time:.1f} img/s)")
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.perf_counter() - start_time
        
        gpu_name, gpu_memory_peak = get_gpu_info()
        
        result = BenchmarkResult(
            architecture=arch,
            model_variant=variant,
            batch_size=batch_size,
            use_compile=use_compile,
            compile_mode=compile_mode if use_compile else None,
            use_amp=use_amp,
            num_epochs=num_epochs,
            total_time_sec=round(total_time, 2),
            time_per_epoch_sec=round(total_time / num_epochs, 2),
            samples_per_sec=round(total_samples / total_time, 1),
            gpu_name=gpu_name,
            gpu_memory_peak_gb=round(gpu_memory_peak, 2),
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            dataset_type="synthetic" if isinstance(dataset, SyntheticDataset) else "real",
            num_samples=len(dataset),
        )
        
        del model
        clear_gpu_memory()
        
        return result
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  OOM during training with batch_size={batch_size}")
            clear_gpu_memory()
            return None
        raise


def run_full_benchmark(
    arch: str,
    variants: list[str],
    num_epochs: int,
    imagenet_path: Optional[str],
    synthetic_samples: int,
    use_synthetic: bool,
    use_compile_options: list[bool],
    compile_mode: str,
    use_amp: bool,
    num_workers: int,
    prefetch_factor: int,
) -> list[BenchmarkResult]:
    """Run full benchmark suite."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = ARCHITECTURES[arch]["task"]
    
    # Create dataset based on task
    if task == "diffusion":
        # Diffusion always uses synthetic data (noise prediction)
        print(f"Using synthetic diffusion dataset ({synthetic_samples} samples, 64x64)")
        dataset = SyntheticDataset(
            num_samples=synthetic_samples,
            task="diffusion",
            image_size=DIFFUSION_SIZE,
        )
        num_classes = 0  # Not used for diffusion
    elif use_synthetic:
        print(f"Using synthetic classification dataset ({synthetic_samples} samples)")
        dataset = SyntheticDataset(num_samples=synthetic_samples, task="classification")
        num_classes = IMAGENETTE_NUM_CLASSES
    elif imagenet_path and Path(imagenet_path).exists():
        print(f"Using ImageNet from: {imagenet_path}")
        dataset = ImageFolder(
            root=Path(imagenet_path) / "train",
            transform=get_imagenet_transforms()
        )
        num_classes = len(dataset.classes)
    else:
        # Download and use ImageNette by default
        imagenette_path = download_imagenette()
        print(f"Using ImageNette ({imagenette_path})")
        dataset = ImageFolder(
            root=imagenette_path / "train",
            transform=get_imagenet_transforms()
        )
        num_classes = IMAGENETTE_NUM_CLASSES
    
    if task == "diffusion":
        print(f"Dataset: {len(dataset)} samples (diffusion)")
    else:
        print(f"Dataset: {len(dataset)} images, {num_classes} classes")
    
    results = []
    configs = [(v, c) for v in variants for c in use_compile_options]
    
    for i, (variant, use_compile) in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(configs)}] {arch.upper()}/{variant} | compile={use_compile}")
        print(f"{'='*60}")
        
        # Find max batch size
        max_bs = find_max_batch_size(
            arch=arch,
            variant=variant,
            device=device,
            use_amp=use_amp,
            use_compile=use_compile,
            compile_mode=compile_mode,
            num_classes=num_classes,
        )
        
        if max_bs is None:
            print(f"  Skipping - no valid batch size found")
            continue
        
        # Try running benchmark, reduce batch size on OOM
        # torch.compile can use more memory during actual training
        result = None
        batch_size = max_bs
        
        while result is None and batch_size >= 8:
            result = run_training_benchmark(
                arch=arch,
                variant=variant,
                batch_size=batch_size,
                num_epochs=num_epochs,
                dataset=dataset,
                num_classes=num_classes,
                use_compile=use_compile,
                compile_mode=compile_mode,
                use_amp=use_amp,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )
            
            if result is None:
                # Reduce batch size and retry
                batch_size = int(batch_size * 0.75)
                batch_size = max(8, batch_size - (batch_size % 8))  # Round down to multiple of 8
                print(f"  Retrying with batch_size={batch_size}...")
        
        if result:
            results.append(result)
            print(f"  -> {result.samples_per_sec:.1f} img/s | {result.gpu_memory_peak_gb:.1f} GB")
        else:
            print(f"  Skipping - could not find working batch size")
    
    return results


def print_results_table(results: list[BenchmarkResult]):
    """Print results summary table."""
    if not results:
        print("No results.")
        return
    
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    print(f"{'Arch':<10} {'Variant':<8} {'Batch':<7} {'Compile':<12} {'Total(s)':<10} {'Per Epoch':<10} {'Img/s':<10} {'Mem(GB)':<8}")
    print("-"*100)
    
    for r in results:
        compile_str = r.compile_mode if r.use_compile else "-"
        print(f"{r.architecture:<10} {r.model_variant:<8} {r.batch_size:<7} {compile_str:<12} {r.total_time_sec:<10.1f} {r.time_per_epoch_sec:<10.1f} {r.samples_per_sec:<10.1f} {r.gpu_memory_peak_gb:<8.1f}")
    
    print("="*100)
    print(f"GPU: {results[0].gpu_name} | AMP: {results[0].use_amp}")
    print(f"Dataset: {results[0].dataset_type} ({results[0].num_samples} samples) | {results[0].num_epochs} epochs")


def save_results(results: list[BenchmarkResult], output_path: str):
    """Save results to JSON."""
    if not results:
        print("No results to save.")
        return
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to: {output_path}")


def get_output_filename(arch: str) -> str:
    """Generate output filename based on architecture and GPU."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        # Clean GPU name for filename: "NVIDIA H100 80GB HBM3" -> "H100_80GB"
        gpu_clean = gpu_name.replace("NVIDIA ", "").replace(" ", "_")
        # Simplify common patterns
        gpu_clean = gpu_clean.replace("GeForce_", "").replace("Tesla_", "").replace("Quadro_", "")
    else:
        gpu_clean = "CPU"
    
    return f"benchmark_{arch}_{gpu_clean}.json"


class VariantType(click.ParamType):
    """Custom click type that validates variants against selected architecture."""
    name = "variant"
    
    def convert(self, value, param, ctx):
        return value  # Validation happens in main()


@click.command()
@click.option(
    "--arch", "-a",
    type=click.Choice(list(ARCHITECTURES.keys())),
    default=DEFAULT_ARCH,
    help="Model architecture to benchmark",
)
@click.option(
    "--variants", "-v",
    multiple=True,
    help="Model variants to benchmark (architecture-specific, e.g., 'base,large' for convnext)",
)
@click.option("--epochs", "-e", default=DEFAULT_EPOCHS, help="Training epochs per benchmark")
@click.option("--imagenet-path", type=click.Path(exists=True), help="Path to full ImageNet (uses ImageNette by default)")
@click.option("--synthetic", is_flag=True, help="Use synthetic data instead of ImageNette")
@click.option("--synthetic-samples", default=50000, help="Synthetic dataset size (if --synthetic)")
@click.option("--no-compile", is_flag=True, help="Skip torch.compile benchmarks")
@click.option("--compile-only", is_flag=True, help="Only run torch.compile benchmarks")
@click.option(
    "--compile-mode",
    type=click.Choice(["default", "reduce-overhead", "max-autotune"]),
    default="default",
    help="torch.compile mode",
)
@click.option("--no-amp", is_flag=True, help="Disable mixed precision")
@click.option("--num-workers", default=DEFAULT_NUM_WORKERS, help="DataLoader workers")
@click.option("--prefetch-factor", default=DEFAULT_PREFETCH_FACTOR, help="DataLoader prefetch factor")
@click.option("--output", "-o", default=None, help="Output JSON file (default: benchmark_<arch>_<gpu>.json)")
def main(
    arch: str,
    variants: tuple[str, ...],
    epochs: int,
    imagenet_path: str | None,
    synthetic: bool,
    synthetic_samples: int,
    no_compile: bool,
    compile_only: bool,
    compile_mode: str,
    no_amp: bool,
    num_workers: int,
    prefetch_factor: int,
    output: str | None,
):
    """Benchmark GPU training performance across architectures.
    
    Supported architectures:
    
    \b
    - convnext: ConvNeXt models (variants: base, large, xlarge, xxlarge)
    - resnet: ResNet models (variants: 50, 101, 152)
    - diffusion: Diffusion UNet (variants: small, medium, large)
    
    Examples:
    
    \b
    # ConvNeXt (default)
    python bench_gpu.py
    
    \b
    # ResNet variants
    python bench_gpu.py --arch resnet --variants 50,101,152
    
    \b
    # Diffusion UNet
    python bench_gpu.py --arch diffusion --variants small,large
    """
    
    # Generate output filename if not provided
    if output is None:
        output = get_output_filename(arch)
    
    arch_info = ARCHITECTURES[arch]
    valid_variants = arch_info["variants"]
    default_variants = arch_info["default_variants"]
    
    # Parse and validate variants
    if variants:
        # Handle comma-separated variants
        parsed_variants = []
        for v in variants:
            parsed_variants.extend(v.split(","))
        variants = parsed_variants
        
        # Validate
        invalid = [v for v in variants if v not in valid_variants]
        if invalid:
            raise click.BadParameter(
                f"Invalid variants for {arch}: {invalid}. Valid: {valid_variants}"
            )
    else:
        variants = default_variants
    
    if no_compile:
        compile_options = [False]
    elif compile_only:
        compile_options = [True]
    else:
        compile_options = [False, True]
    
    click.echo("=" * 60)
    click.echo(f"GPU Benchmark: {arch.upper()}")
    click.echo("=" * 60)
    click.echo(f"Variants: {variants}")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"Compile: {compile_options} (mode={compile_mode})")
    click.echo(f"AMP: {not no_amp}")
    click.echo(f"Workers: {num_workers}, Prefetch: {prefetch_factor}")
    click.echo(f"Output: {output}")
    
    if torch.cuda.is_available():
        click.echo(f"GPU: {torch.cuda.get_device_name(0)}")
        click.echo(f"CUDA: {torch.version.cuda}")
    else:
        click.secho("WARNING: No GPU - running on CPU", fg="yellow")
    click.echo(f"PyTorch: {torch.__version__}")
    
    # Check dependencies
    if arch in ("convnext", "resnet") and not HAS_TIMM:
        raise click.ClickException("timm required for ConvNeXt/ResNet. Install with: pip install timm")
    if arch == "diffusion" and not HAS_DIFFUSERS:
        raise click.ClickException("diffusers required for Diffusion. Install with: pip install diffusers")
    
    results = run_full_benchmark(
        arch=arch,
        variants=list(variants),
        num_epochs=epochs,
        imagenet_path=imagenet_path,
        synthetic_samples=synthetic_samples,
        use_synthetic=synthetic,
        use_compile_options=compile_options,
        compile_mode=compile_mode,
        use_amp=not no_amp,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    
    # Save results first, then print table
    save_results(results, output)
    print_results_table(results)


if __name__ == "__main__":
    main()