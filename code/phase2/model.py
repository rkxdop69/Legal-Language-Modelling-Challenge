"""
Phase 2: Legal Named Entity Recognition (L-NER)
Model Architecture: Pretrained LLaMA Backbone with Bidirectional Attention Adapter
and Token Classification Head for 12 Legal Entity Classes.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

NUM_TAGS = 25  # 12 entity types * 2 (B/I) + 1 (O)

def create_bidirectional_mask(attention_mask_2d: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (1.0 - attention_mask_2d[:, None, None, :].to(dtype=dtype)) * float("-inf")

class LLaMAForLNER(nn.Module):
    def __init__(self, llama_model: nn.Module, hidden_size: int = 768, num_classes: int = NUM_TAGS, dropout: float = 0.1):
        super().__init__()
        self.llama = llama_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
        mask_4d = create_bidirectional_mask(attention_mask, input_ids.device)
        outputs = self.llama(input_ids, attention_mask=mask_4d)
        hidden_states = outputs["hidden_states"] if isinstance(outputs, dict) else outputs[0]
        sequence_output = self.dropout(hidden_states)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits}
