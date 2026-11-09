# CS781 Capstone Project: In-House Legal Intelligence System
### An End-to-End, Confidential, Self-Hosted Legal NLP Stack for Indian Jurisprudence

[![Course](https://img.shields.io/badge/Course-CS781%20LLM%20Capstone-blue.svg)](https://www.cse.iitk.ac.in)
[![Institution](https://img.shields.io/badge/IIT%20Kanpur-CSE%20Department-red.svg)](https://www.iitk.ac.in)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

---

## 1. Executive Summary

This repository presents the end-to-end design, implementation, and empirical evaluation of an **In-House Legal Intelligence System** developed for the **CS781 Capstone Project** at the **Indian Institute of Technology Kanpur (IIT Kanpur)**. 

In institutional law firms, corporate legal departments, and judicial registries, legal data is governed by strict confidentiality, attorney-client privilege, and statutory data-localization mandates. Deploying third-party commercial APIs (such as OpenAI or Anthropic) introduces unacceptable data leakage and compliance risks. Consequently, this system is engineered strictly as a **compact, self-hosted, confidential neural legal stack** that executes on modest on-premises compute (single consumer or workstation GPU).

The system addresses **seven core computational legal tasks** spanning the entire judicial lifecycle:
1. **Phase 1: Legal Language Model Pretraining** — Custom 181.2M LLaMA decoder trained strictly from scratch and continually pretrained on 30M multi-source legal tokens.
2. **Phase 2: Legal Named Entity Recognition (L-NER)** — Extraction of 12 legal entity classes under strict character offset boundaries.
3. **Phase 3: Rhetorical Role Prediction (RR)** — Sentence-level judicial discourse segmentation across 13 rhetorical functional roles.
4. **Phase 4: Court Judgment Prediction (CJPE)** — Binary judicial disposition prediction (*Dismissed* vs. *Allowed*) over long-document apex court opinions.
5. **Phase 5: Legal Statute Identification (LSI)** — Multi-label identification of applicable legislative acts and statutory provisions.
6. **Phase 6: Abstractive Legal Summarization** — Generation of concise, legally faithful executive headnotes from multi-page judicial judgments.
7. **Phase 7: Prior Case Retrieval (PCR)** — Precedent retrieval ranking the top $K=7$ relevant prior cases from a 1,727-case candidate pool for 237 test queries.

---

## 2. Official Competition Leaderboard Results

All downstream phases (Phases 2–7) adapt strictly from our **Phase 1 Pretrained Foundation Model**, satisfying all competition constraints (50M–300M parameters, no external pretrained weights, no external legal data, minimum 6 distinct submissions per phase).

| Phase | Task | Primary Metric | Official CodaBench Score | Leaderboard Rank | Key Architecture / Winning Method | Valid Submissions |
|:---:|:---|:---:|:---:|:---:|:---|:---:|
| **1** | Legal LM Pretraining | BERTScore F1 | **0.8249** | 118 | Custom LLaMA-181M (RoPE, RMSNorm, SwiGLU, 200M Scratch + 30M Multi-Source Continual) | **8** |
| **2** | Legal Named Entity Recognition (L-NER) | Strict Macro-F1 | **0.3209** | 121 | Word-Boundary Snapping + Legal Connector Bridging (Fold 1+2 Stage 2 Fine-Tuning) | **8** |
| **3** | Rhetorical Role Prediction (RR) | Macro-F1 | **0.5735** | 17 | Structural Discourse Calibration Engine (Hierarchical BiLSTM + Boundary Shift Loss) | **10** |
| **4** | Court Judgment Prediction (CJPE) | Macro-F1 | **0.6136** | 117 | Hierarchical Chunk Attention Pooling (4 chunks $\times$ 512 tokens = 2048 tokens) | **6** |
| **5** | Legal Statute Identification (LSI) | Macro-F1 | **0.4014** | 33 | Master Calibrated Full-Corpus Pipeline (Adaptive Per-Statute Thresholding $\tau_c$) | **7** |
| **6** | Abstractive Legal Summarization | BERTScore F1 | **0.8074** | 118 | Baseline Top-4 Layers Fine-Tuning (Lead 512 Windowing, Repetition Penalty 1.22) | **6** |
| **7** | Prior Case Retrieval (PCR) | Micro-F1@7 | **0.4548** | 29 | Multi-Channel Breakthrough Pipeline (Self-Mask + Kanoon Citator + Statutory Jaccard) | **7** |
| **Total** | **All Seven Phases** | — | — | — | **52 Valid Submissions Evaluated on Official CodaBench Platform** | **52** |

---

## 3. System Architecture & Foundation Transfer

The entire pipeline is unified by a single foundational backbone:

```
               [Phase 1 Pretrained Legal LLaMA-181M Foundation Backbone]
                     (12 Layers, 12 Heads, d=768, RoPE, RMSNorm, SwiGLU)
                                        │
     ┌──────────────────┬───────────────┼───────────────┬──────────────────┐
     ▼                  ▼               ▼               ▼                  ▼
[Phase 2 L-NER]  [Phase 3 RR]    [Phase 4 CJPE]   [Phase 5 LSI]      [Phase 6 Summarizer]
Token Clf Head   Discourse Proj  Hierarchical     Multi-Label Head   Causal Continuation
(Strict Offsets) (13 Roles)      Chunk Attention  (Sigmoids + LoRMA) (Lead-Tail Window)
     │
     └──────────────────────────────────┬──────────────────────────────────┘
                                        ▼
                         [Phase 7 Prior Case Retrieval]
                          Multi-Channel Hybrid Engine:
                  Self-Match Mask + Kanoon Citator + Statutory Jaccard
```

---

## 4. Phase-by-Phase Methodologies & Implementation Details

### Phase 1: Legal Language Model Pretraining
- **Architecture**: Custom decoder-only Transformer with 12 layers, 12 attention heads, hidden dimension $d_{\text{model}}=768$, intermediate dimension $d_{\text{ff}}=2048$, and Rotary Position Embeddings (RoPE).
- **Parameter Verification**: Total parameters = 181,217,280, strictly within the mandatory 50M--300M cap.
- **Corpus & Training**: Pretrained strictly from scratch on Indian Supreme Court judgments, followed by streaming continual pretraining on 30M multi-source tokens (10M Statutory Acts, 10M High Court judgments, 10M Tribunals) using AdamW, cosine annealing with warmup, peak learning rate $3 \times 10^{-4}$.
- **Inference**: Temperature $T=0.7$, top-$p=0.90$, repetition penalty 1.15, generating exact 128-word continuations for 512-word prompts.
- **Official Score**: **0.8249 BERTScore F1**.

### Phase 2: Legal Named Entity Recognition (L-NER)
- **Task**: Extract 12 legal entity classes (`COURT`, `PETITIONER`, `RESPONDENT`, `JUDGE`, `LAWYER`, `DATE`, `ORG`, `GPE`, `STATUTE`, `PROVISION`, `PRECEDENT`, `CASE_NUMBER`) with 100% exact character offset match.
- **Methodology**:
  - Converted causal attention to full bidirectional context using custom 4D attention masks.
  - 512-token sliding window with 128-token stride.
  - Word-boundary snapping: Expands/contracts token offsets to strict whitespace/punctuation boundaries.
  - Connector bridging: Merges fragmented spans separated by legal connectors (`"and"`, `","`, `"v."`).
  - Confidence-based Non-Maximum Suppression (NMS) resolves multi-window overlaps.
- **Official Score**: **0.3209 Strict Macro-F1**.

### Phase 3: Rhetorical Role Prediction (RR)
- **Task**: Classify every sentence in Indian judicial opinions into one of 13 official rhetorical roles.
- **Methodology**:
  - Encodes sentences with 2-layer MLP discourse position projection ($idx / N$).
  - Smoothed inverse-frequency class-weighted cross-entropy ($w_c = (N / N_c)^{0.35}$) prevents minority role starvation (`Dissent`, `PrecedentOverruled`).
  - Structural discourse calibration engine enforces legal transition dynamics.
  - Systematic benchmarking of 6 adaptation paradigms: LoRMA, additive LoRA, rsLoRA, Knowledge Distillation, $IA^3$, and baseline fine-tuning.
- **Official Score**: **0.5735 Macro-F1**.

### Phase 4: Court Judgment Prediction (CJPE)
- **Task**: Binary outcome prediction (*Dismissed 0* vs. *Allowed 1*) from extensive Indian judicial judgments (ILDC corpus).
- **Methodology**:
  - Overcame sequence truncation via **Hierarchical Chunk Attention Pooling**: Splits each case into 4 non-overlapping structural chunks spanning 2048 tokens.
  - Multi-head cross-chunk attention pooling dynamically upweights decisive sections while dampening procedural boilerplate.
  - Calibrated decision thresholding ($\tau^* = 0.50$) on validation set.
- **Official Score**: **0.6136 Macro-F1** (0.7279 precision on positive outcomes).

### Phase 5: Legal Statute Identification (LSI)
- **Task**: Multi-label prediction of applicable statutory enactments and sections from judicial dispute facts.
- **Methodology**:
  - Multi-label binary cross-entropy head with independent sigmoid activations.
  - Head-tail segment windowing captures statutory citations situated in both preambles and operative orders.
  - Frequency-calibrated adaptive per-statute thresholding resolves the long-tail distribution.
  - Compulsory LoRMA adaptation achieves competitive accuracy with 99.5% parameter reduction.
- **Official Score**: **0.4014 Macro-F1**.

### Phase 6: Abstractive Legal Summarization
- **Task**: Compress lengthy Supreme Court judgments (~4,368 words) into concise executive headnotes (~350 words).
- **Methodology**:
  - Section-aware lead-tail windowing extracts core procedural context and final operative disposition.
  - Repetition penalty 1.22 suppresses cyclical repetition of judicial boilerplate.
  - Evaluated under BERTScore F1 with `roberta-large` contextual embeddings to measure semantic fidelity.
- **Official Score**: **0.8074 BERTScore F1**.

### Phase 7: Prior Case Retrieval (PCR)
- **Task**: Retrieve and rank the top $K=7$ most relevant prior case precedents from a 1,727-case candidate pool for 237 test queries.
- **Root Cause Analysis & Breakthrough Architecture**:
  1. **Enforced Self-Match Masking**: In the raw corpus, each query appears in the candidate pool. Setting $S(q, c) = -\infty$ whenever $cid == qid$ rescued Rank 1 from being wasted as an automatic false positive.
  2. **Full-Text Representation**: Sublinear TF-IDF (1, 2) on full 8,500+ word judgments preserved critical statutory sections and doctrine discarded by 160-word truncation.
  3. **Indian Kanoon Citator Graph**: Exploited `CITATOR INFO:` headers and formal law report citations (`AIR`, `SCC`, `SCR`) to form explicit bipartite precedent links (1,558 links in test set).
  4. **Statutory Provision Jaccard**: Quantified statutory overlap across shared sections and enactments.
  5. **Temporal Admissibility Filter**: Penalized candidate cases decided after the query case.
- **Official Score**: **0.4548 Micro-F1@7** (jumping from initial 0.02–0.14 baselines).

---

## 5. Comparative Analysis of Adaptation Paradigms

Across all downstream phases, we conducted controlled, head-to-head benchmarking of diverse parameter-efficient fine-tuning (PEFT) and compression paradigms:

| Paradigm | Formulation | Trainable Params | Compression | Empirical Behavior & Findings |
|:---|:---:|:---:|:---:|:---|
| **Baseline FT** | Top 4 Transformer Layers | 44.5M | 24.5% | Strong baseline, but prone to overfitting on small legal datasets. |
| **LoRMA** (Compulsory) | $W = W_0 \odot (I + B A)$ | **984K** | **99.5%** | Consistently matched or outperformed additive LoRA (+25% gain in NER), demonstrating superior gating of legal vocabulary. |
| **Additive LoRA** | $W = W_0 + \frac{\alpha}{r} B A$ | 984K | 99.5% | Stable adaptation, but slightly less expressive than multiplicative gating on long legal sequences. |
| **rsLoRA** | $W = W_0 + \frac{\alpha}{\sqrt{r}} B A$ | 1.7M | 99.1% | Rank stabilization enabled scaling to $r=32$ without gradient vanishing, winning Phase 3 (0.4161 initial score). |
| **Knowledge Distillation** | 12L Teacher $\to$ 4L Student | **39.5M** | **78.2%** | Achieved 3x faster GPU inference with 75% capacity retention; ideal for resource-constrained edge deployments. |
| **Hierarchical Pooling** | Multi-Chunk Cross-Attention | 1.1M | 99.4% | Broke the sequence truncation ceiling in Phase 4 (0.6136 Macro-F1), proving vital for apex court judgments. |

---

## 6. Directory Structure

```
.
├── code/
│   ├── phase1/                 # Pretrained Legal LLaMA-181M & Continual Pretraining
│   │   ├── model.py            # Custom RoPE, RMSNorm, SwiGLU LLaMA implementation
│   │   ├── continual_pretrain.py # 30M token multi-source pretraining script
│   │   ├── inference.py        # 128-word continuation generator
│   │   ├── config.json         # Architecture hyperparameters
│   │   ├── requirements.txt    # Python dependencies
│   │   └── README.md           # Phase 1 documentation
│   ├── phase2/                 # Legal Named Entity Recognition (L-NER)
│   ├── phase3/                 # Rhetorical Role Prediction (RR)
│   ├── phase4/                 # Court Judgment Prediction (CJPE)
│   ├── phase5/                 # Legal Statute Identification (LSI)
│   ├── phase6/                 # Abstractive Legal Summarization
│   └── phase7/                 # Prior Case Retrieval (PCR Breakthrough Pipeline)
├── report/
│   └── CS781-CP-Rajat-Kumar-230839-report.pdf
├── eval_scripts/               # Official competition evaluation scripts
└── README.md                   # Master repository documentation
```

---

## 7. Installation & Reproduction

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ with CUDA support
- 16 GB+ System RAM (GPU with 8 GB+ VRAM recommended)

### Quick Start
```bash
# Clone repository
git clone https://github.com/rkxdop69/Legal-Language-Modelling-Challenge.git
cd CS781-Legal-Intelligence-System

# Create virtual environment
conda create -n legal_ai python=3.10 -y
conda activate legal_ai

# Install dependencies for any phase (e.g., Phase 7)
cd code/phase7
pip install -r requirements.txt

# Run inference
python inference.py
```

---

## 8. Acknowledgments & Citations

This work was conducted as part of the CS781 Capstone Project at IIT Kanpur. We acknowledge the open-source legal NLP community and the authors of the following landmark datasets and algorithms:
- **ILDC**: Malik et al., *ILDC for CJPE: Indian Legal Documents Corpus*, ACL-IJCNLP 2021.
- **IN-Abs**: Bhattacharya et al., *Comparative Analysis of Legal Summarization*, 2019.
- **LoRMA**: Bihany, Patel, & Modi, *Low-Rank Multiplicative Adaptation of Large Language Models*, ACL Findings 2025.
- **rsLoRA**: Dam et al., *Rank-Stabilized LoRA*, 2023.
- **BERTScore**: Zhang et al., *BERTScore: Evaluating Text Generation with BERT*, ICLR 2020.
