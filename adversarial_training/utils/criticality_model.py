"""Task-agnostic criticality classifier for LIBERO init states.

Architecture
------------

Single shared input projection (init states from any task are right-padded
to the global max dim) -> deep residual pre-LN MLP trunk -> binary head.

Pipeline::

       init_state (B, max_D)      <- caller right-pads with zeros
              |
              v  Linear(max_D, H)
              |
              v  h0 in R^H
              |
              v  ResMLPBlock x depth
              |
              v  LayerNorm + Linear(H, 2)
              v
            logits (B, 2)

With hidden=1024, expansion=4, depth=12 the trunk lands around ~100M
parameters. The task identifier is intentionally not an input — the model
infers task structure from the padded state vector itself.

Output: ``logits[:, 1]`` is the (un-normalized) failure score.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CriticalityModelConfig:
    input_dim: int                            # padded init-state dim (max across all tasks)
    hidden_dim: int = 1024
    expansion: int = 4                        # feed-forward expansion factor
    depth: int = 12                           # number of residual blocks
    dropout: float = 0.1
    num_classes: int = 2


class ResMLPBlock(nn.Module):
    """Pre-LN residual MLP block:  h <- h + W2(GeLU(W1(LN(h))))."""

    def __init__(self, hidden: int, expansion: int, dropout: float):
        super().__init__()
        mid = hidden * expansion
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, mid)
        self.fc2 = nn.Linear(mid, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.drop(self.fc2(F.gelu(self.fc1(self.norm(h)))))


class CriticalityModel(nn.Module):
    """Task-agnostic deep residual MLP binary classifier."""

    def __init__(self, config: CriticalityModelConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.blocks = nn.ModuleList([
            ResMLPBlock(config.hidden_dim, config.expansion, config.dropout)
            for _ in range(config.depth)
        ])
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.num_classes)

    @property
    def input_dim(self) -> int:
        return self.config.input_dim

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, init_state: torch.Tensor) -> torch.Tensor:
        """init_state: (B, input_dim). Returns logits (B, num_classes)."""
        h = self.input_proj(init_state)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.final_norm(h))

    @torch.no_grad()
    def criticality_score(self, init_state: torch.Tensor) -> torch.Tensor:
        """Return P(failure) in [0, 1], shape (B,)."""
        return F.softmax(self.forward(init_state), dim=-1)[:, 1]


# ---------- self-check ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = CriticalityModelConfig(input_dim=92)
    model = CriticalityModel(cfg)
    x = torch.randn(8, 92)
    logits = model(x)
    assert logits.shape == (8, 2)
    n_params = sum(p.numel() for p in model.parameters())
    proj_params = sum(p.numel() for p in model.input_proj.parameters())
    trunk_params = sum(p.numel() for p in model.blocks.parameters())
    head_params = sum(p.numel() for p in model.head.parameters()) \
        + sum(p.numel() for p in model.final_norm.parameters())
    print(f"#params total : {n_params/1e6:.2f}M")
    print(f"  input_proj  : {proj_params/1e6:.4f}M")
    print(f"  trunk ({cfg.depth} blocks): {trunk_params/1e6:.2f}M")
    print(f"  head        : {head_params/1e6:.4f}M")
