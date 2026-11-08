"""
Phase 5: Legal Statute Identification (LSI)
Architecture: Multi-Label LLaMA Classifier with Independent Sigmoid Activations
and Parameter-Efficient Adaptation (LoRMA & rsLoRA).
"""
import torch
import torch.nn as nn
from typing import Optional

class LLaMAForLSI(nn.Module):
    def __init__(self, llama_model: nn.Module, hidden_size: int = 768, num_classes: int = 100, dropout: float = 0.2):
        super().__init__()
        self.llama = llama_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
        outputs = self.llama(input_ids, attention_mask=attention_mask)
        hidden = outputs["hidden_states"] if isinstance(outputs, dict) else outputs[0]
        pooled = (hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True).clamp(min=1e-6)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        return {"loss": loss, "logits": logits}
