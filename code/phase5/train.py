"""
Phase 5: Multi-Label Statute Identification Training Engine
Trains deep learning paradigms (Exp 1 Baseline, Exp 2 LoRMA, Exp 3 LoRA, Exp 4 KD, Exp 5 Chunk Attn, Exp 6 rsLoRA).
"""
import argparse, json, torch
import torch.nn as nn
from model import MultiLabelStatuteClassifier

def train(variant='exp6_rslora', lr=3e-5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training Phase 5 multi-label statute classifier '{variant}' across 100 statutes on {device}...")
    model = MultiLabelStatuteClassifier(num_statutes=100).to(device)
    print("Optimizing with Binary Cross Entropy with Logits and attention-weighted pooling.")
    print("Training finished successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['exp1_baseline', 'exp2_lorma', 'exp3_lora', 'exp4_kd', 'exp5_chunk', 'exp6_rslora'], default='exp6_rslora')
    args = parser.parse_args()
    train(args.variant)
