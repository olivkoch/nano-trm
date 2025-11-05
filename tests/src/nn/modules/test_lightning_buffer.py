# Test without Lightning
import torch

from src.nn.modules.sparse_embeddings import (
    CastedSparseEmbedding,
    CastedSparseEmbeddingSignSGD_Distributed,
)

# Create embedding module directly
emb = CastedSparseEmbedding(
    num_embeddings=100, embedding_dim=64, batch_size=32, init_std=0.0, cast_to=torch.float32
)

# Check buffer status
for name, buf in emb.named_buffers():
    print(f"{name}: requires_grad={buf.requires_grad}, is_leaf={buf.is_leaf}")

# Try creating optimizer directly (no Lightning)
opt = CastedSparseEmbeddingSignSGD_Distributed(
    list(emb.buffers()), lr=0.01, weight_decay=0.01, world_size=1
)
print("Optimizer created successfully without Lightning!")
