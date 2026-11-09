# Phase 1: Legal Language Model Pretraining

## 1. Overview & Verified Submissions
Phase 1 benchmarks legal foundation language models trained strictly from scratch on the Indian legal corpus under a 200M token budget, generating an exact 128-word continuation for each 512-word prompt in `test.json` (1,000 test cases).

| Sub # | Architecture / Variant | Pretrain Budget | Decoding Setup | CodaBench BERTScore | Leaderboard Rank |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Sub 1** | **LLaMA-181M (RoPE, RMSNorm, SwiGLU)** | **200M SC Tokens** | $T=0.7, p=0.90, 	ext{rep}=1.15$ | **0.8249** | **Top Leader (Rank 118)** |
| **Sub 2** | GPT-2 Baseline (182M, Post-LN) | 200M SC Tokens | $T=0.7, p=0.90, 	ext{rep}=1.15$ | 0.8242 | Verified |
| **Sub 3** | Mistral SWA (181M, Window=256) | 200M SC Tokens | $T=0.7, p=0.90, 	ext{rep}=1.15$ | 0.8234 | Verified |
| **Sub 4** | LLaMA Multi-Source Continual | 230M Tokens (+30M) | $T=0.7, p=0.90, 	ext{rep}=1.15$ | 0.8232 | Continual Adaptation |
| **Sub 5** | LLaMA Multi-Source Continual | 230M Tokens (+30M) | $T=0.5, p=0.90, 	ext{rep}=1.15$ | 0.8218 | Low Temperature |
| **Sub 6** | LLaMA Multi-Source Continual | 230M Tokens (+30M) | $T=0.9, p=0.90, 	ext{rep}=1.15$ | 0.8208 | High Temperature |
| **Sub 7** | BART Seq2Seq (158M, 6-Enc/8-Dec) | 200M SC Tokens | $T=0.7, p=0.90, 	ext{rep}=1.15$ | 0.8092 | Seq2Seq Baseline |
| **Sub 8** | Scratch LLaMA Foundation Baseline | 200M SC Tokens | $T=0.7, p=0.90, 	ext{rep}=1.00$ | 0.8208 | Unpenalized Greedy |

## 2. File Structure
- `model.py`: Full PyTorch implementations of all four benchmarked architectures (LLaMA-181M, GPT-2-182M, Mistral SWA-181M, and BART-158M).
- `train.py`: Pretraining engine supporting zero-copy binary token streaming and FP16/BF16 training loops.
- `continual_pretrain.py`: Streaming multi-source continual pretraining across Statutory Acts, High Courts, and Tribunals (30M tokens).
- `inference.py`: Inference engine supporting all 8 submission configurations, temperature controls ($T \in \{0.5, 0.7, 0.9\}$), repetition penalties, and exact 128-word post-processing.
- `config.json`: Unified hyperparameters for all 8 submissions.
- `requirements.txt`: Dependencies.

## 3. How to Run Any Submission
```bash
# Run winning Sub 1 (LLaMA-181M, T=0.7):
python inference.py --submission 1

# Run Sub 2 (GPT-2 Baseline):
python inference.py --submission 2

# Run Sub 4, 5, or 6 (Continual Pretrained LLaMA with T=0.7, 0.5, or 0.9):
python inference.py --submission 4  # T=0.7
python inference.py --submission 5  # T=0.5
python inference.py --submission 6  # T=0.9
```
