# Phase 6: Abstractive Legal Summarization

## 1. Overview & Verified Submissions
Phase 6 evaluates the generation of concise, authoritative executive headnotes from extensive Indian Supreme Court decisions under roberta-large BERTScore F1.

| Sub # | Paradigm / Method | Trainable Params | Context Window | CodaBench BERTScore | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Sub 1** | **Baseline Top-4 Layers Fine-Tuning (Exp 1)** | **44.5M** | **Lead 512 Tokens** | **0.8074** | **Top Leader (Rank 118)** |
| **Sub 2** | Comparative Additive LoRA ($r=16$, Exp 3) | 984K | Lead 512 Tokens | 0.8018 | PEFT Verified |
| **Sub 3** | rsLoRA ($r=32, lpha=16$, Exp 6) | 1.72M | Lead 512 Tokens | 0.8017 | PEFT Verified |
| **Sub 4** | Compulsory LoRMA ($r=16, lpha=32$, Exp 2) | 984K | Lead 512 Tokens | 0.8012 | Compulsory PEFT |
| **Sub 5** | Hierarchical Section-Aware Summarizer (Exp 5) | 44.5M | 2048 Tokens (4 Chunks) | 0.7713 | Structural Chunking |
| **Sub 6** | Knowledge Distillation (4L Student, Exp 4) | 39.5M | Lead 512 Tokens | 0.7502 | Compact Model |

## 2. File Structure
- `model.py`: Causal generation heads, Compulsory LoRMA, Comparative LoRA, rsLoRA, 4-Layer Student model, and Hierarchical Section Summarizer.
- `train.py`: Fine-tuning engine supporting partial layer tuning, parameter-efficient adapters, and student distillation.
- `inference.py`: Generates executive headnotes for all 6 submissions with repetition penalty ($	heta_{	ext{rep}} = 1.22$) and target length constraints (~350 words).
- `config.json`: Configurations for all 6 submissions.
- `requirements.txt`: Dependencies.

## 3. How to Run Any Submission
```bash
# Run winning Sub 1 (Baseline Top-4 Fine-Tuning):
python inference.py --submission 1

# Run Sub 2 (Comparative LoRA):
python inference.py --submission 2

# Run Sub 4 (Compulsory LoRMA):
python inference.py --submission 4
```
