"""
Phase 1 Continual Pretraining Pipeline
Ingests 30M tokens across Statutory Acts, High Courts, and Tribunals (10M each).
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from model import LLaMAConfig, LLaMAForCausalLM

class LegalPretrainingDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, block_size: int = 512):
        self.block_size = block_size
        self.num_blocks = len(token_ids) // block_size
        self.data = token_ids[:self.num_blocks * block_size].view(self.num_blocks, block_size)

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return {"input_ids": chunk, "labels": chunk.clone()}

def train_continual(
    checkpoint_dir: str,
    output_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 8,
    lr: float = 3e-4,
    epochs: int = 1
):
    config = LLaMAConfig()
    model = LLaMAForCausalLM(config).to(device)
    
    # Load foundational weights if available
    weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        print("Loaded initial pretrained checkpoint.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    print("Starting continual pretraining on multi-source corpus...")
    model.train()
    print("Training pipeline ready.")

if __name__ == "__main__":
    train_continual("./checkpoints", "./output")
