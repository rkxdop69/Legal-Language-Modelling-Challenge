"""
Phase 4: Court Judgment Prediction (CJPE)
Architecture: Hierarchical Chunk Attention Pooling LLaMA
Processes 4 non-overlapping structural chunks (2048 tokens context)
with cross-chunk multi-head attention aggregation.
"""
import torch
import torch.nn as nn
from typing import Optional

class HierarchicalChunkAttention(nn.Module):
    def __init__(self, hidden_size: int = 768, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

    def forward(self, chunk_embeddings: torch.Tensor) -> torch.Tensor:
        bsz = chunk_embeddings.size(0)
        query = self.query_token.expand(bsz, -1, -1)
        attn_out, _ = self.mha(query, chunk_embeddings, chunk_embeddings)
        return attn_out.squeeze(1)

class LLaMAForCJPE(nn.Module):
    def __init__(self, llama_model: nn.Module, hidden_size: int = 768, dropout: float = 0.2):
        super().__init__()
        self.llama = llama_model
        self.hierarchical_pool = HierarchicalChunkAttention(hidden_size=hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, chunk_input_ids: torch.Tensor, chunk_masks: torch.Tensor, labels: Optional[torch.Tensor] = None):
        bsz, num_chunks, seq_len = chunk_input_ids.shape
        flat_ids = chunk_input_ids.view(bsz * num_chunks, seq_len)
        flat_mask = chunk_masks.view(bsz * num_chunks, seq_len)

        outputs = self.llama(flat_ids, attention_mask=flat_mask)
        hidden = outputs["hidden_states"] if isinstance(outputs, dict) else outputs[0]
        chunk_pooled = (hidden * flat_mask.unsqueeze(-1)).sum(1) / flat_mask.sum(1, keepdim=True).clamp(min=1e-6)
        chunk_embeddings = chunk_pooled.view(bsz, num_chunks, -1)

        doc_embedding = self.hierarchical_pool(chunk_embeddings)
        logits = self.classifier(doc_embedding).squeeze(-1)

        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        return {"loss": loss, "logits": logits}
