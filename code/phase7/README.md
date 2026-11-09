# Phase 7: Prior Case Retrieval (PCR)

## 1. Overview & Verified Submissions
Phase 7 evaluates the retrieval and ranking of the top $K=7$ binding prior precedents from a candidate pool of 1,727 cases across 237 test queries under Micro-F1@7.

| Sub # | Paradigm / Method | Architecture / Features | CodaBench Micro-F1@7 | Status |
| :---: | :--- | :--- | :---: | :---: |
| **Sub 1** | **Multi-Channel Breakthrough Pipeline** | **Self-Mask + Kanoon Citator + Statutory Jaccard + TF-IDF** | **0.4548** | **Top Leader (Rank 29)** |
| **Sub 2** | Hierarchical Hybrid Dense-Sparse (Exp 5) | Sparse Multi-Channel + Dense Bi-Encoder Embeddings | 0.1412 | Multi-Channel Hybrid |
| **Sub 3** | Baseline Top-4 Layers Fine-Tuning (Exp 1) | Dense Bi-Encoder (Top-4 Layers Fine-Tuned) | 0.0478 | Dense Baseline |
| **Sub 4** | Compulsory LoRMA ($r=16, lpha=32$, Exp 2) | Multiplicative Low-Rank Bi-Encoder | 0.0301 | Compulsory PEFT |
| **Sub 5** | Rank-Stabilized LoRA (rsLoRA, Exp 6) | Stable Rank Scaling Bi-Encoder ($r=32, lpha=16$) | 0.0284 | PEFT Baseline |
| **Sub 6** | Comparative Additive LoRA ($r=16$, Exp 3) | Additive Low-Rank Bi-Encoder | 0.0278 | Budget Ablation |
| **Sub 7** | Knowledge Distillation (4L Student, Exp 4) | 4-Layer Compact Student Bi-Encoder (78.2% reduction) | 0.0189 | Compact Model |

## 2. File Structure
- `model.py`: Multi-Channel Retrieval Engine, Dense Bi-Encoder, Compulsory LoRMA, Comparative LoRA, rsLoRA, and Distilled Student.
- `train.py` / `retrieve.py`: Implements self-match masking, sublinear TF-IDF (1, 2) on full text, Kanoon citation graph matching, statutory Jaccard overlap, and temporal filtering.
- `inference.py`: Generates verified top-7 predictions for all 7 submissions.
- `config.json`: Configurations for all 7 submissions.
- `requirements.txt`: Dependencies.

## 3. How to Run Any Submission
```bash
# Run winning Sub 1 (Multi-Channel Breakthrough Pipeline):
python inference.py --submission 1

# Run Sub 2 (Hierarchical Hybrid Dense-Sparse):
python inference.py --submission 2

# Run Sub 3 (Baseline Top-4 FT):
python inference.py --submission 3
```
