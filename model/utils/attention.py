import torch
import torch.nn as nn
import math
from typing import Optional

def get_eps(
    x: torch.Tensor,
    eps16: float = torch.finfo(torch.float16).min,
    eps32: float = torch.finfo(torch.float32).min,
    eps64: float = torch.finfo(torch.float64).min
) -> float:
    if x.dtype == torch.float16:
        return eps16
    elif x.dtype == torch.float32:
        return eps32
    elif x.dtype == torch.float64:
        return eps64
    else:
        return -torch.inf    

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.scale_factor = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout_p)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def scaled_dot_production_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scale_factor

        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask, get_eps(attn_scores))

        attn_weights = torch.softmax(attn_weights, dim=3)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, v)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, query_length, _ = q.size()
        key_length = k.size(1)

        # Projection
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Split Heads
        q = q.view([batch_size, query_length, self.n_heads, self.head_dim]).transpose(1, 2).contiguous()
        k = k.view([batch_size, key_length, self.n_heads, self.head_dim]).transpose(1, 2).contiguous()
        v = v.view([batch_size, key_length, self.n_heads, self.head_dim]).transpose(1, 2).contiguous()

        # Attention
        attn_context = self.scaled_dot_production_attention(q, k, v, attn_mask)
        attn_context = attn_context.transpose(1, 2).contiguous().view([batch_size, query_length, self.embedding_dim])
        attn_context = self.out_proj(attn_context)
        return attn_context