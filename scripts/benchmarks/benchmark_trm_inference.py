import torch
import torch.nn as nn
import time
import sys
import os
from torchvision.models import resnet18, resnet50, efficientnet_b2
import torch.nn.functional as F
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64  # Increase from default 8

torch.set_float32_matmul_precision('high')

# Import your local TRM
try:
    from src.nn.models.trm import TRMModule
except ImportError:
    print("Error: Could not import TRMModule. Run this script from the project root.")
    sys.exit()

# ==========================================
# Configuration
# ==========================================
INPUT_SIZES = [32, 46, 64, 90, 128]
BATCH_SIZES = [32]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 256      
DIFFUSION_STEPS = 8

# Precision Flags
USE_BF16 = os.environ.get('USE_BF16', '0') == '1'
TARGET_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

print(f"--- Benchmarking on {DEVICE.upper()} ---")
print(f"--- Precision: {'BFLOAT16' if USE_BF16 else 'FLOAT32'} ---")

# TRM CONFIGURATIONS
TRM_CONFIGS = {
    "TRM-Light": {
        "hidden_size": 128, "num_layers": 1, "num_heads": 4, 
        "H_cycles": 1, "L_cycles": 1, "N_supervision_val": 2
    },
    "TRM-Medium": {
        "hidden_size": 256, "num_layers": 2, "num_heads": 8,
        "H_cycles": 2, "L_cycles": 2, "N_supervision_val": 4
    },
    "TRM-Heavy": {
        "hidden_size": 512, "num_layers": 4, "num_heads": 8,
        "H_cycles": 3, "L_cycles": 6, "N_supervision_val": 16
    }
}

# Shared TRM defaults
TRM_DEFAULTS = {
    "vocab_size": 256,
    "puzzle_emb_dim": 0,    
    "puzzle_emb_len": 0,    
    "pos_emb_type": "1d",
    "use_mlp_t": False,
    "forward_dtype": TARGET_DTYPE, 
}

# ==========================================
# 1. Models
# ==========================================

class BasicWideCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, 1, 1), nn.BatchNorm2d(1024), nn.ReLU(), 
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class TRM_Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = TRMModule(**config)

    def forward(self, x):
        # Flatten (B, 1, H, W) -> (B, H*W)
        x_tokens = x[:, 0, :, :].flatten(1).to(torch.int32)
        dummy_ids = torch.zeros((x.size(0),), dtype=torch.long, device=x.device)
        
        batch_data = {"input": x_tokens, "puzzle_identifiers": dummy_ids}
        carry = self.model.initial_carry(batch_data)
        
        # TRM runs internally in mixed precision due to forward_dtype
        _, outputs = self.model.forward(carry, batch_data)
        
        logits = outputs["logits"].mean(dim=1) 
        return logits

# ==========================================
# 1. Standard DiT Implementation, adapted for SRM
# ==========================================
class DiTBlock(nn.Module):
    """
    Standard DiT Block using Scaled Dot Product Attention (Flash Attention).
    Ref: Peebles & Xie, 2022
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 1. Input Norms
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # 2. Attention Projections (Merged QKV for speed)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        
        # 3. MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )
        
        # 4. Adaptive Norm (adaLN)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, t):
        B, N, C = x.shape
        
        # Get adaptive parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        
        # --- ATTENTION BLOCK ---
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        # QKV Projection
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim)
        # Permute to (3, B, Heads, N, Dim) for easy unpacking
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Flash Attention (Memory Efficient)
        # This replaces the O(N^2) matrix with O(N) memory kernel
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        
        # Reshape back and project
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.proj(attn_out)
        
        # Residual Connection
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # --- MLP BLOCK ---
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x

class DiT_Generator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x_embed = nn.Linear(1, hidden_size) 
        self.pos_embed = nn.Parameter(torch.zeros(1, input_size*input_size, hidden_size))
        self.t_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(num_layers)])
        self.final_layer = nn.Linear(hidden_size, 1)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t):
        x = self.x_embed(x) + self.pos_embed
        t = self.t_embed(t)
        for block in self.blocks:
            x = block(x, t)
        return self.final_layer(x)

    @torch.no_grad()
    def generate(self, batch_size, steps, device='cuda'):
        """
        mode='standard': Standard DiT (Parallel). Denoise whole image 'steps' times.
        mode='srm': SRM-style (Sequential). Divide image into parts and generate sequentially.
                    This mimics "Reasoning" by fixing one part before generating the next.
        """
        N = self.input_size * self.input_size
        x = torch.randn(batch_size, N, 1, device=device)
        
        for i in range(steps):
            t_input = torch.full((batch_size, self.hidden_size), i/steps, device=device)
            _ = self(x, t_input)
    
# ==========================================
# 2. Setup & Compilation
# ==========================================

def build_models(input_size: int):
    models = {}

    # --- A. CNN Baselines (Keep in Float32, we use Autocast for them) ---
    m_basic = BasicWideCNN(num_classes=NUM_CLASSES).to(DEVICE)
    models['Basic CNN'] = m_basic

    m_resnet18 = resnet18(num_classes=NUM_CLASSES)
    m_resnet18.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False) 
    models['ResNet18'] = m_resnet18.to(DEVICE)
    
    m_resnet50 = resnet50(num_classes=NUM_CLASSES)
    m_resnet50.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False) 
    models['ResNet50'] = m_resnet50.to(DEVICE)

    m_eff = efficientnet_b2(num_classes=NUM_CLASSES)
    m_eff.features[0][0] = nn.Conv2d(1, 32, 3, 2, 1, bias=False) 
    models['EfficientNet-B2'] = m_eff.to(DEVICE)

    dit_model = DiT_Generator(input_size=input_size, hidden_size=256, num_layers=2, num_heads=8).to(DEVICE).eval()
    models['DiT-Medium'] = dit_model

    # --- B. TRM Variants (Internal Casting) ---
    for name, specific_config in TRM_CONFIGS.items():
        full_config = {**TRM_DEFAULTS, **specific_config}
        full_config['seq_len'] = input_size * input_size
        full_config['max_grid_size'] = input_size
        
        trm_adapter = TRM_Adapter(full_config).to(DEVICE)
        models[name] = trm_adapter

    # --- C. Compilation ---
    for name, model in models.items():
        model.eval()
        if DEVICE == 'cuda':
            if "TRM" in name:
                trm_mod = model.model
                # Only compile inner loop to allow dynamic casting logic to persist
                trm_mod.inner_forward = torch.compile(trm_mod.inner_forward, mode="default")
            else:
                models[name] = torch.compile(model, mode="reduce-overhead")
    return models

# ==========================================
# 3. Execution Loop
# ==========================================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark(model, x, use_autocast, iters=5, warmup=10): 
    try:
        # Helper context manager: applies autocast OR does nothing (nullcontext)
        context = torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16) if use_autocast else torch.no_grad()
        # Note: torch.no_grad() is just a dummy context here since we wrap everything in no_grad anyway
        
        # Warmup
        with torch.no_grad():
            with context:
                for _ in range(warmup): _ = model(x)
        if DEVICE == 'cuda': torch.cuda.synchronize()
        
        # Timing
        start = time.time()
        with torch.no_grad():
            with context:
                for _ in range(iters): _ = model(x)
        if DEVICE == 'cuda': torch.cuda.synchronize()
        
        avg_time = ((time.time() - start) / iters)
        throughput = x.size(0) / avg_time
        return throughput
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower(): return "OOM"
        return f"Error: {str(e)[:50]}"
    except Exception as e:
        return f"Error: {str(e)[:50]}"

def benchmark_generative(model, x, steps, use_autocast, iters=5, warmup=5):
    b_size = x.size(0)
    context = torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16) if use_autocast else torch.no_grad()
    
    with torch.no_grad(), context:
        for _ in range(warmup): model.generate(b_size, steps, device=DEVICE)
        if DEVICE == 'cuda': torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters): model.generate(b_size, steps, device=DEVICE)
        if DEVICE == 'cuda': torch.cuda.synchronize()
        
    return b_size / ((time.time() - start) / iters)

# --- Main Loop ---
for input_size in INPUT_SIZES:
    models = build_models(input_size)
    for b_size in BATCH_SIZES:
        try:
            # Inputs are Float32.
            # CNNs: Autocast will handle downcasting.
            # TRM: Adapter casts to Int32; Embeddings output BF16 internally.
            dummy_input = torch.randint(0, 256, (b_size, 1, input_size, input_size)).float().to(DEVICE)
        except RuntimeError:
            print(f"  CRITICAL: Cannot allocate input tensor for batch {b_size}. Skipping.")
            continue

        print("-" * 80)
        print(f"Size: {input_size}x{input_size}, Batch: {b_size}, Precision: {'BF16' if USE_BF16 else 'FP32'}")
        print(f"{'Model':<20} | {'Params':<12} | {'Throughput (samples/s)':<25} | {'Mode':<10}")
        print("-" * 80)

        for name, model in models.items():
            try:
                if hasattr(model, '_orig_mod'): p_count = count_params(model._orig_mod)
                else: p_count = count_params(model)
            except: p_count = 0
            
            # DECISION LOGIC:
            # If CNN: Use Autocast (Standard PyTorch way)
            # If TRM: Use Native (Internal casting via forward_dtype)
            use_autocast = (USE_BF16 and "TRM" not in name)
            mode_str = "Native" if "TRM" in name else ("Autocast" if use_autocast else "FP32")

            if "DiT" in name:
                 # Pass dummy_input just for batch size reference, DiT generates its own noise internally
                throughput = benchmark_generative(model, dummy_input, steps=DIFFUSION_STEPS, use_autocast=use_autocast, iters=5, warmup=10)
            else:
                throughput = benchmark(model, dummy_input, use_autocast=use_autocast, iters=5, warmup=10)
            
            throughput_str = f"{throughput:.2f}" if isinstance(throughput, float) else throughput
            
            if p_count > 1e6: p_str = f"{p_count/1e6:.2f} M"
            else: p_str = f"{p_count/1e3:.0f} K"
                
            print(f"{name:<20} | {p_str:<12} | {throughput_str:<25} | {mode_str:<10}")

        print("-" * 80)
        del dummy_input
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()