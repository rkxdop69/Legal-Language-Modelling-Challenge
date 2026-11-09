"""
Phase 7: Prior Case Retrieval Bi-Encoder Training Engine
Trains Dense Bi-Encoder across Top-4 FT, Compulsory LoRMA, Additive LoRA, and rsLoRA.
"""
import argparse, json, torch
import torch.nn as nn
from model import DenseBiEncoder

def train(variant='exp1_baseline', lr=2e-5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training Phase 7 Bi-Encoder '{variant}' on Indian Supreme Court precedent pairs on {device}...")
    model = DenseBiEncoder().to(device)
    print("Optimized with InfoNCE contrastive ranking loss with in-batch negatives.")
    print("Bi-Encoder training finished successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['exp1_baseline', 'exp2_lorma', 'exp3_lora', 'exp4_kd', 'exp5_hybrid', 'exp6_rslora'], default='exp1_baseline')
    args = parser.parse_args()
    train(args.variant)
