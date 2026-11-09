# Phase 3: Rhetorical Role Prediction (RR)

## 1. Overview & Verified Submissions
Phase 3 addresses 13-class sentence-level rhetorical role prediction across Corporate Law (CL) and Information Technology (IT) disputes.

| Sub # | Paradigm / Method | Trainable Params | Key Architectural Innovation | CodaBench Macro-F1 | Status |
| :---: | :--- | :---: | :--- | :---: | :---: |
| **Sub 1** | **Structural Discourse Calibration Engine** | **1.97M** | Auxiliary boundary shift loss + coram dissent parsing | **0.5735** | **Top Leader (Rank 17)** |
| **Sub 2** | Master Calibrated Discourse Ensemble | 1.97M | Multi-model sequence voting + preamble smoothing | 0.5735 | Final Master |
| **Sub 3** | Advanced Hierarchical Ensemble | 2.15M | Multi-model probability fusion + BiLSTM | 0.5030 | Breakthrough |
| **Sub 4** | Rank-Stabilized LoRA (rsLoRA, $r=32$) | 1.97M | Stable rank scaling $\gamma = lpha/\sqrt{r}$ | 0.4161 | PEFT Leader |
| **Sub 5** | Compulsory LoRMA ($r=16, lpha=32$) | 984K | Multiplicative adapter gating (ACL 2025) | 0.4089 | Compulsory PEFT |
| **Sub 6** | Comparative Additive LoRA ($r=16$) | 984K | Additive rank update | 0.4036 | Budget Ablation |
| **Sub 7** | $IA^3$ Vector-Valued Scaling | 185K | Key/value activation multipliers (NeurIPS 2022) | 0.3991 | Parameter Minimal |
| **Sub 8** | Standalone Hierarchical BiLSTM | 1.15M | 2-layer BiLSTM over sentence embeddings | 0.3628 (Dev) | Sequence Baseline |
| **Sub 9** | Knowledge Distillation (4L Student) | 39.5M | 4-layer compact student model | 0.3522 | Compact Model |
| **Sub 10** | Context-Aware Baseline Fine-Tuning | 93.0M | Top-4 layers + discourse position projection | 0.2943 | Baseline |

## 2. File Structure
- `model.py`: Discourse sequence model, Compulsory LoRMA, Additive LoRA, rsLoRA, IA3, Distilled Student, and Hierarchical BiLSTM.
- `train.py`: Trains all pointwise adaptation variants (Exp 1..7).
- `train_improved.py`: Trains hierarchical sequence model, auxiliary label shift predictor, coram dissent filter, and ensemble blender.
- `inference.py`: Generates verified sentence-level `prediction.json` for all 10 submissions.
- `config.json`: Configurations for all 10 submissions.
- `requirements.txt`: Dependencies.

## 3. How to Run Any Submission
```bash
# Run winning Sub 1 (Structural Discourse Calibration Engine):
python inference.py --submission 1

# Run Sub 3 (Advanced Hierarchical Ensemble):
python inference.py --submission 3

# Run Sub 5 (Compulsory LoRMA):
python inference.py --submission 5
```
