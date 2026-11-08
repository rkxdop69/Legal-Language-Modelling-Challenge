# Phase 1: Legal Language Model Pretraining

## 1. Architectural Overview
- **Model**: Custom LLaMA-style Decoder-Only Transformer (`LLaMA-181M`).
- **Total Parameters**: 181,217,280 (Strictly conforming to the 50M--300M cap).
- **Key Components**:
  - **Rotary Position Embeddings (RoPE)**: Relative distance preservation across long sequences.
  - **RMSNorm**: Layer normalization without mean-centering for gradient stability.
  - **SwiGLU Activation**: Gated non-linear feedforward projections.
  - **Weight Tying**: Shared input and output token embeddings.

## 2. Pretraining Strategy
- **Base Corpus**: Indian Supreme Court Judgments (200M tokens).
- **Multi-Source Continual Pretraining**: 30M streaming tokens across:
  - Statutory Acts (10M tokens)
  - High Court Judgments (10M tokens)
  - Tribunals (10M tokens)
- **Objective**: Causal Next-Token Prediction with AdamW optimizer, cosine learning rate schedule, peak LR = 3e-4.

## 3. Official CodaBench Results
- **Primary Metric**: **BERTScore F1**
- **Official Score**: **0.8249** (Top of Competition Leaderboard)

## 4. How to Reproduce
```bash
pip install -r requirements.txt
python inference.py
```
