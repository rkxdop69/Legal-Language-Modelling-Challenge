# Phase 4: Court Judgment Prediction (CJPE)

## 1. Overview & Verified Submissions
Phase 4 predicts binary judicial outcomes (0 = Dismissed, 1 = Allowed) across multi-thousand-word Indian Supreme Court judgments from the ILDC benchmark.

| Sub # | Paradigm / Method | Context Window | Trainable Params | CodaBench Macro-F1 | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Sub 1** | **Hierarchical Chunk Attention Pooling (Exp 5)** | **2048 Tokens (4 Chunks)** | **1.11M** | **0.6136** | **Top Leader (Rank 117)** |
| **Sub 2** | rsLoRA ($r=32, lpha=16$, Exp 6) | 512 Tokens | 1.72M | 0.5727 | PEFT Leader |
| **Sub 3** | Compulsory LoRMA ($r=16, lpha=32$, Exp 2) | 512 Tokens | 984K | 0.5672 | Compulsory PEFT |
| **Sub 4** | Comparative Additive LoRA ($r=16$, Exp 3) | 512 Tokens | 984K | 0.5626 | Budget Ablation |
| **Sub 5** | Baseline Top-4 Layers Fine-Tuning (Exp 1) | 512 Tokens | 44.5M | 0.5415 | Baseline |
| **Sub 6** | Knowledge Distillation (4L Student, Exp 4) | 512 Tokens | 39.5M | 0.3344 | Compact Model |

## 2. File Structure
- `model.py`: Hierarchical Chunk Attention Pooling network, Compulsory LoRMA, Additive LoRA, rsLoRA, and Distilled Student.
- `train.py`: Training engine supporting multi-chunk encoding, intra-chunk pooling, cross-chunk attention, and threshold tuning.
- `inference.py`: Generates binary outcome predictions for all 6 submissions with optimal threshold calibration ($	au^* \in [0.41, 0.49]$).
- `config.json`: Configurations for all 6 submissions.
- `requirements.txt`: Dependencies.

## 3. How to Run Any Submission
```bash
# Run winning Sub 1 (Hierarchical Chunk Attention Pooling):
python inference.py --submission 1

# Run Sub 2 (rsLoRA):
python inference.py --submission 2

# Run Sub 3 (Compulsory LoRMA):
python inference.py --submission 3
```
