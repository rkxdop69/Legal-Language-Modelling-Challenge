"""
Phase 3 Training Script with Smoothed Inverse-Frequency Class Weighting
"""
import torch

def compute_class_weights(class_counts, gamma=0.35):
    total = sum(class_counts.values())
    weights = [(total / class_counts.get(c, 1)) ** gamma for c in range(13)]
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights / weights.mean()

if __name__ == "__main__":
    print("Phase 3 training module ready.")
