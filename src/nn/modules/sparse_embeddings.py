import torch
import torch.nn as nn

class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 init_std: float = 0.0, cast_to=torch.float32, batch_size=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.cast_to = cast_to
        
        # Note: batch_size parameter exists in reference but seems unused
        
        # Regular embedding with sparse gradients
        self.embedding = nn.Embedding(
            num_embeddings, 
            embedding_dim, 
            sparse=True
        )
        
        # Initialize
        if init_std == 0:
            nn.init.zeros_(self.embedding.weight)
        else:
            nn.init.normal_(self.embedding.weight, std=init_std)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        return embeddings.to(self.cast_to)