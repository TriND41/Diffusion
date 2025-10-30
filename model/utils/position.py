import torch
import torch.nn as nn
from typing import Optional, Union

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_positions: Optional[int] = None) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        if max_positions is None:
            pe = torch.empty(0)
        else:
            pe = self.__encode(max_positions)

        self.register_buffer('pe', pe)

    def __encode(self, length: int, device: Union[str, int] = 'cpu') -> torch.Tensor:
        positions = torch.arange(length, dtype=torch.float, device=device).unsqueeze(1)
        freqs = 1.0 / (10000 ** torch.arange(0, self.embedding_dim//2, dtype=torch.float, device=device)).unsqueeze(0)
        angles = torch.matmul(positions, freqs)

        pe = torch.zeros([length, self.embedding_dim], dtype=torch.float, device=device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        if self.pe.numel() == 0 or length > self.pe.size(0):
            self.pe = self.__encode(length, device=x.device)
        return self.pe[:length].unsqueeze(0)