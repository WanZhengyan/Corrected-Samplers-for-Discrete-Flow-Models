import torch
from torch import nn, Tensor

# Pretrained model for toy datasets
class ToyMLP(nn.Module):
    def __init__(
        self, vocab_size: int = 16, hidden_dim=256, length=2, time_dim=1):
        super().__init__()
        self.length = length
        self.time_dim = time_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.time_embedding = nn.Linear(1, 1)
        self.token_embedding = torch.nn.Embedding(self.vocab_size, hidden_dim)

        self.block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim * length + self.time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.vocab_size * length)
        )
    def forward(self, x, t):
        x = self.token_embedding(x)
        B, N, d = x.shape
        x = x.reshape(B, N * d)
        if self.time_dim == 1: # if time_dim == 0 then no time embedding
            t = self.time_embedding(t.unsqueeze(-1)) # shape: (B, time_dim)
            h = torch.cat([x, t], dim=1)
        else:
            h = x
        h = self.block(h)
        h = h.reshape(B, N, self.vocab_size)
        return h
