"""
TRM Compilation Strategy Comparison (Final Fix)
Includes:
1. Baseline
2. Radical Fusion
3. TorchAO Int8 (Dynamic)
4. Manual Int8 (Pure PyTorch)
"""

import torch
import time
import copy
import numpy as np
import torch._dynamo
import torch._inductor.config
import torchao
from torchao.quantization import quant_api

torch.set_float32_matmul_precision('high')

from src.nn.models.trm import TRMModule, TRMInnerCarry

# ==========================================
# Configuration
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32 
SEQ_LEN = 1024
CONFIG = {
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
    "forward_dtype": torch.bfloat16,
    "seq_len": SEQ_LEN,
    "max_grid_size": 32,
}

WARMUP_ITERS = 10
TIMING_ITERS = 50

# ==========================================
# Test Harness
# ==========================================
class TRMRunner:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, x):
        x_tokens = x.to(torch.int32)
        dummy_ids = torch.zeros((x.size(0),), dtype=torch.long, device=x.device)
        batch_data = {"input": x_tokens, "puzzle_identifiers": dummy_ids}
        
        # Reset carry structure
        carry = self.model.initial_carry(batch_data)
        
        # Run supervision steps
        for _ in range(self.model.hparams.N_supervision_val):
            # Safe to call even if no graph is active
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            
            carry, outputs = self.model.forward(carry, batch_data)
            
        return outputs["logits"]

def measure_throughput(runner, x, name, warmup=WARMUP_ITERS, iters=TIMING_ITERS):
    print(f"[{name}] Warming up...")
    try:
        with torch.no_grad():
            for _ in range(warmup):
                _ = runner(x)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            
            print(f"[{name}] Timing {iters} iterations...")
            times = []
            for _ in range(iters):
                if DEVICE == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = runner(x)
                if DEVICE == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
        
        mean_time = np.mean(times)
        throughput = BATCH_SIZE / mean_time
        # Throughput, Latency, StdDev
        return throughput, mean_time * 1000, np.std(times) * 1000
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[{name}] CRASHED: {e}")
        return 0.0, 0.0, 0.0

# ==========================================
# 2. Radical Optimization (Full Fusion)
# ==========================================
def apply_radical_optimizations(base_model):
    model = copy.deepcopy(base_model)
    torch._dynamo.config.cache_size_limit = 128
    torch._inductor.config.triton.cudagraphs = True
    torch._inductor.config.epilogue_fusion = True

    def full_recurrent_op(z_H, z_L, input_embeddings, cos, sin):
        seq_info = {'cos_sin': (cos, sin)}
        for _ in range(model.hparams.H_cycles):
            injection = z_H + input_embeddings
            for _ in range(model.hparams.L_cycles):
                z_L = model.lenet(z_L, injection, **seq_info)
            z_H = model.lenet(z_H, z_L, **seq_info)
        return z_H, z_L

    print("  > Compiling Full Recurrent Op (max-autotune + fullgraph)...")
    compiled_full_op = torch.compile(full_recurrent_op, mode="max-autotune", fullgraph=True)

    def radical_inner_forward(carry, batch):
        seq_info = dict(cos_sin=model.pos_embedding() if hasattr(model, "pos_embedding") else None)
        cos, sin = seq_info['cos_sin']
        input_embeddings = model._input_embeddings(batch["input"], batch["puzzle_identifiers"])
        z_H, z_L = compiled_full_op(carry.z_H, carry.z_L, input_embeddings, cos, sin)
        new_carry = TRMInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = model.lm_head(z_H)[:, model.puzzle_emb_len :]
        q_logits = model.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, q_logits[..., 0]

    model.inner_forward = radical_inner_forward
    return model

# ==========================================
# 3. TorchAO Int8
# ==========================================
def apply_torchao_int8(base_model):
    print("  > Applying Int8DynamicActivationInt8WeightConfig (TorchAO)...")
    model = copy.deepcopy(base_model)
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig
    # Quantize in-place
    torchao.quantization.quantize_(model, Int8DynamicActivationInt8WeightConfig())
    return model

# ==========================================
# 4. Manual Int8 Implementation
# ==========================================
class Int8LinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=torch.int8), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(torch.ones(out_features), requires_grad=False)
        if bias: self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else: self.bias = None
    
    @classmethod
    def from_float(cls, linear_layer):
        int8_layer = cls(linear_layer.in_features, linear_layer.out_features, bias=linear_layer.bias is not None)
        with torch.no_grad():
            weight = linear_layer.weight.data.float()
            scale = weight.abs().max(dim=1).values / 127.0
            scale = scale.clamp(min=1e-5)
            weight_int8 = (weight / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
            int8_layer.weight.data = weight_int8
            int8_layer.weight_scale.data = scale
            if linear_layer.bias is not None:
                int8_layer.bias.data = linear_layer.bias.data.float()
        return int8_layer
    
    def forward(self, x):
        # Flatten for matrix mult
        x_flat = x.reshape(-1, self.in_features)
        x_scale = x_flat.abs().max() / 127.0
        x_scale = x_scale.clamp(min=1e-5)
        x_int8 = (x_flat / x_scale).round().clamp(-128, 127).to(torch.int8)
        
        out_int32 = torch._int_mm(x_int8, self.weight.t())
        out = out_int32.float() * (x_scale * self.weight_scale)
        
        if self.bias is not None: 
            out = out + self.bias
        return out.view(*x.shape[:-1], self.out_features).to(x.dtype)

def apply_manual_int8(base_model):
    print("  > Converting Linear layers to Manual Int8...")
    model = copy.deepcopy(base_model)
    layers_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "lm_head" in name or "q_head" in name: continue
            layers_to_replace.append(name)
            
    for name in layers_to_replace:
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        child_name = name.rsplit('.', 1)[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        original_layer = getattr(parent, child_name)
        int8_layer = Int8LinearLayer.from_float(original_layer)
        setattr(parent, child_name, int8_layer)
    return model

import torch
import torch.nn as nn
from src.nn.modules.trm_block import ReasoningBlock

class FusedReasoningBlock(nn.Module):
    """
    A block that aggressively fuses the MLP and Attention sub-graphs.
    """
    def __init__(self, original_block: ReasoningBlock):
        super().__init__()
        self.config = original_block.config
        self.self_attn = original_block.self_attn
        self.mlp = original_block.mlp
        self.norm_eps = original_block.norm_eps
        
        # JIT Compile the sub-components independently
        # This encourages "Vertical Fusion" within the block
        # rather than trying to compile the whole recurrent loop at once.
        self.fused_mlp_step = torch.compile(self._mlp_forward_logic, mode="max-autotune")

    def _mlp_forward_logic(self, x):
        """
        The goal: Fuse RMSNorm -> Up -> Act -> Down -> Residual
        Input: x (before norm)
        Output: x + MLP(Norm(x))
        """
        # 1. RMSNorm logic (inline for fusion)
        input_dtype = x.dtype
        x_f32 = x.float()
        var = x_f32.pow(2).mean(-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(var + self.norm_eps)
        x_norm = x_norm.to(input_dtype)
        
        # 2. MLP logic
        # SwiGLU: (Silu(xW_g) * xW_u) W_d
        # Inductor is VERY good at fusing element-wise ops into the matmul epilogue
        return x + self.mlp(x_norm)

    def forward(self, cos_sin, hidden_states):
        # 1. Attention Step
        # We don't fuse Attn with MLP because of the global sync barrier
        attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
        
        # 2. Residual + Norm (Manual RMSNorm inline helps fusion)
        # Re-implementing RMSNorm inline here allows the compiler to
        # fuse the addition from Attn output directly into the Norm normalization
        input_dtype = hidden_states.dtype
        resid = hidden_states + attn_out
        
        # 3. Fused MLP Step
        # We pass the residual state. The function handles Norm + MLP + Residual
        hidden_states = self.fused_mlp_step(resid)
            
        return hidden_states

def apply_vertical_fusion(model):
    """Replaces ReasoningBlocks with FusedReasoningBlocks"""
    for i, layer in enumerate(model.lenet.layers):
        model.lenet.layers[i] = FusedReasoningBlock(layer)
    return model

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    print(f"{'='*60}")
    print("TRM Compilation Strategy Comparison (Final Fix)")
    print(f"{'='*60}")
    
    x = torch.randint(0, 256, (BATCH_SIZE, SEQ_LEN)).to(DEVICE)
    results = {}
    
    # 1. Baseline
    print("1. Testing: Whole model compile...")
    m1 = TRMModule(**CONFIG).to(DEVICE).eval()
    m1 = torch.compile(m1, mode="default")
    r1 = TRMRunner(m1)
    results["Whole Model"] = measure_throughput(r1, x, "WholeModel")
    
    # 2. Radical Kernel
    print("\n2. Testing: Radical Kernel...")
    m2 = TRMModule(**CONFIG).to(DEVICE).eval()
    m2 = apply_radical_optimizations(m2)
    r2 = TRMRunner(m2)
    results["Radical Kernel"] = measure_throughput(r2, x, "Radical")
    
    # 3. TorchAO Int8
    print("\n3. Testing: TorchAO Int8...")
    try:
        m3 = TRMModule(**CONFIG).to(DEVICE).eval()
        m3 = apply_torchao_int8(m3)
        # FIX: Compile ONLY inner_forward to avoid graph safety crashes in the loop
        m3.inner_forward = torch.compile(m3.inner_forward, mode="max-autotune")
        r3 = TRMRunner(m3)
        results["TorchAO Int8"] = measure_throughput(r3, x, "TorchAO")
    except Exception as e:
        print(f"TorchAO Failed: {e}")

    # 4. Manual Int8
    print("\n4. Testing: Manual Int8...")
    try:
        m4 = TRMModule(**CONFIG).to(DEVICE).eval()
        m4 = apply_manual_int8(m4)
        # FIX: Compile ONLY inner_forward to avoid graph safety crashes
        m4.inner_forward = torch.compile(m4.inner_forward, mode="max-autotune")
        r4 = TRMRunner(m4)
        results["Manual Int8"] = measure_throughput(r4, x, "ManualInt8")
    except Exception as e:
         print(f"Manual Int8 Failed: {e}")

    # ---------------------------------------------------------
    # Test 5: Vertical Fusion (End-to-End Block Optimization)
    # ---------------------------------------------------------
    print("Testing: Vertical Fusion (Block Rewriting)...")
    try:
        model5 = TRMModule(**CONFIG).to(DEVICE).eval()
        
        # Apply the architectural change
        model5 = apply_vertical_fusion(model5)
        
        # We still need to compile the outer loop for the recurrence
        model5.inner_forward = torch.compile(model5.inner_forward, mode="max-autotune")
        
        runner5 = TRMRunner(model5)
        
        tp5, ms5, std5 = measure_throughput(runner5, x, "Vertical Fusion", warmup=20)
        results["Vertical Fusion"] = (tp5, ms5, std5)
        print(f"  Result: {tp5:.1f} samp/s | {ms5:.2f} ms\n")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Failed: {e}\n")
        
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    baseline = results.get("Whole Model", (0,))[0]
    for name, (tp, ms, std) in results.items():
        rel = f"({tp/baseline:.2f}x)" if baseline > 0 and tp > 0 else ""
        print(f"{name:<20} {tp:>10.1f} s/s {ms:>10.2f} ms {rel}")
    print(f"{'='*60}")