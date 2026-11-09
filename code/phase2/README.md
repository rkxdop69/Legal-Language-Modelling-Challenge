# Phase 2: Legal Named Entity Recognition (L-NER)

## 1. Overview & Verified Submissions
Phase 2 evaluates character-level extraction of 12 domain-specific legal entity classes across Indian judicial decisions under strict Macro-F1 (100% boundary exact match required).

| Sub # | Paradigm / Method | Training Split | Key Innovation | CodaBench Strict F1 | Status |
| :---: | :--- | :---: | :--- | :---: | :---: |
| **Sub 1** | **Boundary Snapping + Stage 2 Combined** | **Fold 1+2 (70 docs)** | 99.7% word boundary snapping + connector merging | **0.3209** | **Top Leader (Rank 121)** |
| **Sub 2** | Rank-Stabilized LoRA (rsLoRA, $r=32$) | Fold 1 (35 docs) | Stable rank scaling $\gamma = lpha/\sqrt{r}$ | 0.2685 | PEFT Leader |
| **Sub 3** | Compulsory LoRMA ($r=16, lpha=32$) | Fold 1 (35 docs) | Multiplicative activation scaling (ACL 2025) | 0.2625 | Compulsory PEFT |
| **Sub 4** | Comparative Additive LoRA ($r=16$) | Fold 1 (35 docs) | Additive rank update | 0.2614 | Budget Ablation |
| **Sub 5** | Knowledge Distillation (4L Student) | Fold 1 (35 docs) | 39.5M student (78.2% parameter compression) | 0.2317 | Compact Model |
| **Sub 6** | 4D Bidirectional Attention Refactor | Fold 1 (35 docs) | Full context attention + whitespace normalization | 0.2297 | Intermediate |
| **Sub 7** | Baseline Top-4 Fine-Tuning | Fold 1 (35 docs) | Freeze L0-7, tune L8-11 + classification head | 0.2118 | Baseline |
| **Sub 8** | Grounded Witness Engine | Fold 1+2 (70 docs) | Preamble party grounding + witness regex engine | Local (0.2925 Dev) | Research Engine |

## 2. File Structure
- `model.py`: Legal LLaMA token classification head, Compulsory LoRMA module, Additive LoRA module, rsLoRA module, and compact student.
- `train.py`: Trains PEFT and fine-tuning variants (Exp 1..6).
- `train_improved.py`: Implements two-stage multi-fold training, 99.7% word-boundary snapping, connector bridging, and grounded witness parser.
- `inference.py`: Generates verified character-level `prediction.json` for all 8 submissions.
- `config.json`: Hyperparameters across all 8 submissions.
- `requirements.txt`: Dependencies.

## 3. How to Run Any Submission
```bash
# Run winning Sub 1 (Boundary Snapping + Stage 2 Combined):
python inference.py --submission 1

# Run Sub 3 (Compulsory LoRMA):
python inference.py --submission 3

# Run Sub 2 (rsLoRA):
python inference.py --submission 2
```
