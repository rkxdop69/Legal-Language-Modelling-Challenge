"""
Phase 6: Abstractive Legal Summarization
Architecture: Section-Aware LLaMA Summarizer adapted from Phase 1 Pretrained Foundation.
"""
import torch
import torch.nn as nn
from typing import Optional

class LoRMALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.scaling = alpha / r
        in_dim = base_layer.in_features
        out_dim = base_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_dim))
        self.lora_B = nn.Parameter(torch.randn(out_dim, r) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        mult_update = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        return base_out * (1.0 + torch.matmul(x, mult_update.t()) / (base_out + 1e-6))
