"""
Phase 2 Training Script: Sliding-Window Chunking and Fine-Tuning
"""
import torch
from torch.utils.data import Dataset, DataLoader
from model import LLaMAForLNER

class LNERChunkDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.tokens[idx], dtype=torch.long),
            "attention_mask": torch.ones(len(self.tokens[idx]), dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

if __name__ == "__main__":
    print("Phase 2 training module ready.")
