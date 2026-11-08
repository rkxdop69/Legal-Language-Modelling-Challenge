"""
Phase 3: Rhetorical Role Prediction (RR)
Model Architecture: Pretrained LLaMA with Discourse Relative Position Projection
and Parameter-Efficient Adapters (LoRMA & rsLoRA).
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

class LLaMAForRhetoricalRole(nn.Module):
    def __init__(self, llama_model: nn.Module, hidden_size: int = 768, num_classes: int = 13, dropout: float = 0.15):
        super().__init__()
        self.llama = llama_model
        self.pos_proj = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 64))
        self.classifier = nn.Sequential(nn.Linear(hidden_size + 64, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, rel_pos: torch.Tensor, labels: Optional[torch.Tensor] = None):
        outputs = self.llama(input_ids, attention_mask=attention_mask)
        hidden = outputs["hidden_states"] if isinstance(outputs, dict) else outputs[0]
        pooled = (hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True).clamp(min=1e-6)
        pos_feat = self.pos_proj(rel_pos)
        combined = torch.cat([pooled, pos_feat], dim=-1)
        logits = self.classifier(combined)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}
