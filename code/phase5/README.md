# Phase 5: Legal Statute Identification (LSI)

## 1. Overview & Verified Submissions
Phase 5 identifies applicable statutory enactments across 100 candidate Indian statutes as a multi-label classification problem under severe class imbalance.

| Sub # | Paradigm / Method | Corpus Scale | Key Innovation | CodaBench Macro-F1 | Status |
| :---: | :--- | :---: | :--- | :---: | :---: |
| **Sub 1** | **Master Calibrated Full-Corpus Pipeline** | **42,750 cases** | 60,000 n-grams + adaptive per-statute thresholds $	au_c$ | **0.4014** | **Top Leader (Rank 33)** |
| **Sub 2** | rsLoRA ($r=32, lpha=16$, Exp 6) | 3,500 cases | Stable rank scaling $\gamma = lpha/\sqrt{r}$ | 0.2685 | PEFT Leader |
| **Sub 3** | Hierarchical Chunk Attention (Exp 5) | 3,500 cases | 4-Chunk 2048-token cross-chunk attention pooling | 0.2653 | Long Context |
| **Sub 4** | Compulsory LoRMA ($r=16, lpha=32$, Exp 2) | 3,500 cases | Multiplicative low-rank adapter gating | 0.2625 | Compulsory PEFT |
| **Sub 5** | Comparative Additive LoRA ($r=16$, Exp 3) | 3,500 cases | Additive low-rank update | 0.2614 | Budget Ablation |
| **Sub 6** | Knowledge Distillation (4L Student, Exp 4) | 3,500 cases | 4-layer compact student model (78.2% compression) | 0.2317 | Compact Model |
| **Sub 7** | Baseline Top-4 Layers Fine-Tuning (Exp 1) | 3,500 cases | Attention-weighted pooling multi-label head | 0.2118 | Baseline |

## 2. File Structure
- `model.py`: Multi-label classification models, Compulsory LoRMA, Additive LoRA, rsLoRA, KD Student, Chunk Attention, and Master Calibrated Feature Engine.
- `train.py`: Trains all deep learning adaptation paradigms (Exp 1..6).
- `train_improved.py`: Implements full 42,750-case corpus scaling, sublinear TF-IDF, and grid-search per-statute threshold calibration ($	au_c \in [0.15, 0.88]$).
- `inference.py`: Generates multi-label predictions for all 7 submissions with non-empty argmax guarantee.
- `config.json`: Configurations for all 7 submissions.
- `requirements.txt`: Dependencies.

## 3. How to Run Any Submission
```bash
# Run winning Sub 1 (Master Calibrated Full-Corpus Pipeline):
python inference.py --submission 1

# Run Sub 2 (rsLoRA):
python inference.py --submission 2

# Run Sub 4 (Compulsory LoRMA):
python inference.py --submission 4
```
