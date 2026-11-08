# CS-781 Capstone Project: Phase 2 — Legal Named Entity Recognition (L-NER)
## Comprehensive Architectural Design Choices, Adaptation Methodologies & Empirical Analysis

---

### 1. Executive Summary

Phase 2 of the **CS-781 Capstone Project** focuses on **Legal Named Entity Recognition (L-NER)** in Indian judicial decisions. Unlike generic named entity recognition (which targets coarse categories like Person, Location, and Organization), Indian legal adjudication demands precise boundary extraction across **12 domain-specific judicial, statutory, and procedural categories**:
1. `APP` (Appellant / Petitioner)
2. `RESP` (Respondent / Defendant / State)
3. `A.COUNSEL` (Appellant's Legal Counsel)
4. `R.COUNSEL` (Respondent's Legal Counsel)
5. `JUDGE` (Presiding Judge / Bench Members)
6. `WIT` (Witness / Deponent)
7. `AUTH` (Public Authority / Statutory Body)
8. `COURT` (Court of Law / Judicial Forum)
9. `STAT` (Statute / Act / Constitutional Provision)
10. `PREC` (Precedent / Judicial Case Citation)
11. `DATE` (Date of Judgment / Occurrence / Hearing)
12. `CASENO` (Case Number / Appeal Identification)

#### Strict Evaluation Protocol:
Predictions must be submitted as exact **character-level spans**:
$$\text{Entity Span} = [\text{start\_char}, \text{end\_char}, \text{label}]$$
Under the official competition metric (**Strict Macro-F1**, evaluated by `eval_scripts/evaluate_phase2.py`), an extracted entity is considered correct **only if its start character, end character, and entity label match the gold annotation with 100% precision**. Any boundary shift—even by a single space or punctuation mark—results in an F1 score of **zero** for that prediction.

#### Mandatory Project Requirements Satisfied (Capstone Document Section 8.2):
1. **Foundation Model Transfer**: All experiments start strictly from our **LLaMA-Style model continually pretrained on 230 Million tokens** (Phase 1 CodaBench Leader: 0.8249 BERTScore).
2. **Four Methodologically Distinct Adaptation Experiments**:
   - **Experiment 1**: Baseline Fine-Tuning (Top 4 Transformer Layers + Linear Head)
   - **Experiment 2**: **Compulsory LoRMA** (Low-Rank Multiplicative Adaptation, Bihany, Patel, & Modi, ACL Findings 2025)
   - **Experiment 3**: Comparative Additive LoRA
   - **Experiment 4**: **Mandatory Knowledge Distillation** (Teacher $\to$ Compact 4-Layer Student)
3. **Reproducibility & Verification**: Clean sliding-window chunking, bidirectional attention conversion, boundary whitespace normalization, connector-bridged span merging, and automated CodaBench ZIP archive generation.

---

### 2. Legal Entity Categories & Domain Taxonomy

Indian judicial decisions possess unique syntactic structures characterized by procedural preambles, statutory citations, and multi-party adversarial representations:

```mermaid
graph TD
    JudicialDoc["Indian Legal Judgment"]
    
    subgraph Parties ["Litigant Parties & Advocates"]
        APP["APP: Appellant / Petitioner"]
        RESP["RESP: Respondent / State"]
        A_COUNSEL["A.COUNSEL: Appellant's Counsel"]
        R_COUNSEL["R.COUNSEL: Respondent's Counsel"]
    end
    
    subgraph Adjudicators ["Bench & Forum"]
        JUDGE["JUDGE: Justice / Magistrate"]
        COURT["COURT: Apex / High Court / Tribunal"]
        AUTH["AUTH: Statutory Authority / Commission"]
        WIT["WIT: Prosecution / Defense Witness"]
    end
    
    subgraph LawAndCase ["Legal Authorities & Identifiers"]
        STAT["STAT: Acts, Sections, Codes"]
        PREC["PREC: Prior Cited Judgments"]
        CASENO["CASENO: Writ / SLP / Criminal Appeal No."]
        DATE["DATE: Occurrence / Pronouncement"]
    end

    JudicialDoc --> Parties
    JudicialDoc --> Adjudicators
    JudicialDoc --> LawAndCase
```

#### Category Definitions & Linguistic Challenges:
- **`APP` vs `RESP`**: Distinguishing between appellant and respondent requires contextual sensitivity to document preambles (e.g. *"Between: X, Appellant and Y, Respondent"*) versus body paragraphs where either party may be referred to passively.
- **`A.COUNSEL` vs `R.COUNSEL`**: Both are lawyers, often listed together in appearance memos (e.g. *"Mr. P.P. Rao, Senior Advocate for the Appellant; Mr. Harish Salve for the Respondent"*). The model must utilize syntactic proximity to party labels.
- **`STAT` vs `PREC`**: Statutes cite codified legislation (e.g. *"Section 302 of the Indian Penal Code, 1860"*), while precedents cite prior rulings (e.g. *"Maneka Gandhi v. Union of India, AIR 1978 SC 597"*).
- **`CASENO`**: Irregular alphanumeric formatting across various registries (e.g. *"Criminal Appeal No. 92/2015"*, *"W.P.(C) 4412/2021"*).

---

### 3. Root Cause Analysis: From 0.0035 to High-Performance Macro-F1

Our quantitative and qualitative audit uncovered **4 core failure modes** that initially caused submissions to score ~0.0035, and established the solutions implemented in our refactored pipeline:

#### 3.1 Critical BPE Token Offset Whitespace Inclusion (5.2% vs 98.7% Exact Match)
* **The Failure**: In Byte-Pair Encoding (BPE) tokenizers used by LLaMA and GPT-2, tokens for words preceded by a space include the leading space as part of the token (e.g., token `" High"` has offset `(15, 20)`, spanning the space character at index 15 and `"High"` at 16..19). However, human gold annotations in legal corpora never annotate leading spaces (`(16, 20)`).
* **The Impact**: Under strict character matching, raw token offsets matched gold boundaries on only **5.2% of entities**, automatically disqualifying **94.8% of correct predictions**.
* **The Solution**: We implemented dynamic whitespace stripping against the original document text:
  ```python
  s_clean, e_clean = cur_start, cur_end
  while s_clean < e_clean and doc_text[s_clean].isspace():
      s_clean += 1
  while e_clean > s_clean and doc_text[e_clean - 1].isspace():
      e_clean -= 1
  ```
  This single normalization increased exact boundary reconstruction from **5.2% to 98.69% across all 4,120 gold entities**.

#### 3.2 "One-Way Blindness" (Causal Masking in a Bidirectional Task)
* **The Failure**: LLaMA's decoder architecture enforces lower-triangular causal attention by default ($j \le i$), preventing tokens from looking ahead. In NER, future context is indispensable (e.g., in *"Dr. B. R. Sharma, learned counsel appearing for the appellant"*, the title *"learned counsel"* occurs *after* the name).
* **The Solution**: We converted LLaMA's attention into a full 4D bidirectional attention mask:
  $$\text{mask}_{4D} = (1.0 - \text{mask}_{2D}) \times \text{min\_dtype}$$
  This allowed every token to attend bidirectionally across past and future non-padding tokens.

#### 3.3 The False-Alarm Flooding ($w_O = 0.2$ Artificial Downweighting)
* **The Failure**: Because 90%+ of tokens are non-entity tokens (`O`), setting $w_O = 0.2$ in cross-entropy penalized false negatives 5x more than false alarms. The model responded by guessing entity tags on almost every punctuation mark and number, creating over **8,600 false positives** against only 24 true positives!
* **The Solution**: We restored balanced cross-entropy with `ignore_index=-100` and introduced confidence thresholding, slashing false positives by over 93%.

#### 3.4 Subword Fragmentation & Word-Boundary Snapping (99.7% Rule)
* **The Empirical Discovery**: An audit of all 4,120 gold entities revealed that **99.68% of entities start at word boundaries** and **98.23% end at word boundaries**.
* **The Problem**: Raw BPE token classification frequently predicted entities on subword fragments (e.g., `'ANG'` inside `'BANGALORE'`, `'erjee'` inside `'Banerjee'`).
* **The Solution**: We introduced **word-boundary snapping**: any span starting or ending inside a word is expanded to the full enclosing alphanumeric boundary, while trimming trailing punctuation (`,`, `;`, `:`).
* **Connector-Bridged Span Merging**: Adjacent predicted spans of the same label separated by small gaps (`-`, `/`, `&`, `of`, `and`, `v.`) are unified into continuous entity phrases (e.g., `'APPEAL NOS. 3964'` + `'71 OF 2007'` $\to$ `'APPEAL NOS. 3964-71 OF 2007'`).
* **Document Co-Reference Entity Propagation**: Verified party names established in the preamble are propagated across the judgment, with strict legal stopword filtering (`'appellant'`, `'respondent'`, `'court'`, `'advocate'`).

---

### 4. Foundation Representation Transfer from Phase 1

Rather than relying on generic out-of-domain BERT or RoBERTa weights, our L-NER pipeline transfers representation directly from our **continually pretrained LLaMA-230M Foundation Model**:
1. **Domain-Specific Legal Semantic Space**: The 12-layer, 960-dimensional transformer was pretrained on Apex Court judgments, High Court rulings, Statutory Acts, and Tribunal decisions. The attention heads have already learned Indian legal collocations and judicial discourse structures.
2. **RoPE Rotational Invariance**: Rotary Position Embeddings (RoPE) provide relative positional representations, crucial for recognizing entities whose legal roles depend on relative distance to trigger phrases (e.g. *"Advocate appearing on behalf of..."*).
3. **SwiGLU Gating & RMSNorm**: Provide stable gradient propagation through the classification head without catastrophic disruption to underlying legal token representations.

---

### 5. Architectural Exploration: The Four Adaptation Paradigms

```mermaid
graph TD
    Input["Input Token Chunks (512 tokens)"] --> Backbone["LLaMA-230M Transformer Backbone (12 Layers, d=960)"]
    
    subgraph AdaptationParadigms ["4 Mandatory Adaptation Experiments"]
        Exp1["Exp 1: Baseline Fine-Tuning<br/>• Freeze Bottom 8 Layers<br/>• Train Top 4 Layers + Linear Head"]
        Exp2["Exp 2: Compulsory LoRMA (ACL 2025)<br/>• Freeze All Backbone Weights<br/>• Multiplicative Adapter: y = W_0 x, y_adapt = y + (alpha/r)*B*(A*y)<br/>• Operating on d_out Activation Space"]
        Exp3["Exp 3: Comparative Additive LoRA<br/>• Freeze All Backbone Weights<br/>• Additive Adapter: h = W_0 x + (alpha/r)*B*(A*x)<br/>• Operating on d_in Input Space"]
        Exp4["Exp 4: Knowledge Distillation<br/>• Teacher: Adapted LLaMA-181M<br/>• Student: Compact 4-Layer LLaMA (d=512)<br/>• Loss: (1-alpha)*L_CE + alpha*T^2*L_KL"]
    end
    
    Backbone --> Exp1
    Backbone --> Exp2
    Backbone --> Exp3
    Backbone --> Exp4

    Exp1 --> Head1["Linear Classifier (25 BIO Tags)"]
    Exp2 --> Head2["Linear Classifier (25 BIO Tags)"]
    Exp3 --> Head3["Linear Classifier (25 BIO Tags)"]
    Exp4 --> Head4["Linear Classifier (25 BIO Tags)"]
```

#### 5.1 Experiment 1: Baseline Fine-Tuning
- **Mechanism**: Freezes the bottom 8 transformer blocks ($L_1 \dots L_8$) and fine-tunes the top 4 blocks ($L_9 \dots L_{12}$) along with the classification head ($d=960 \to 25$).
- **Optimizer**: AdamW, $\text{lr}=5 \times 10^{-5}$, cosine decay, weight decay 0.01.

#### 5.2 Experiment 2: Compulsory LoRMA (Low-Rank Multiplicative Adaptation)
- **Paper Reference**: *LoRMA: Low-Rank Multiplicative Adaptation for LLMs* (Harsh Bihany, Shubham Patel, and Ashutosh Modi, ACL Findings 2025).
- **Core Formulation**: Standard LoRA uses additive updates ($W = W_0 + \Delta W$). In contrast, LoRMA reformulates adaptation as a **multiplicative transformation on the output activation space**:
  $$y = W_0 x$$
  $$\Delta y = \frac{\alpha}{r} B (A y)$$
  $$y_{\text{adapted}} = y + \Delta y = \left(I + \frac{\alpha}{r} B A\right) W_0 x$$
  where $A \in \mathbb{R}^{r \times d_{\text{out}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$.
- **Initialization**: $A \sim \text{Kaiming Uniform}$, $B = 0$. Thus $\Delta y = 0$ at step 0, preserving exact pretrained representations.
- **Placement**: Query (`q_proj`) and Value (`v_proj`) projections across all 12 transformer layers ($r=16, \alpha=32$).

#### 5.3 Experiment 3: Comparative Additive LoRA
- **Formulation**: Standard low-rank additive update:
  $$h = W_0 x + \frac{\alpha}{r} B (A x)$$
  where $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$ ($r=16, \alpha=32$).
- **Benchmarking Purpose**: Rigorously compares multiplicative vs additive rank updates under identical parameter budgets (~1.8M params, 1.0% footprint).

#### 5.4 Experiment 4: Mandatory Knowledge Distillation
- **Motivation**: Section 8.2 mandates testing knowledge transfer from a large legal model to a compact student to optimize memory and latency.
- **Student Architecture**: 4-Layer LLaMA model ($d=512$, 8 attention heads, intermediate FFN 1,536, ~39.38M parameters — a **78.2% parameter reduction**).
- **Objective Function**:
  $$\mathcal{L} = (1 - \alpha_{\text{kd}}) \mathcal{L}_{\text{CE}}(y_{\text{student}}, y_{\text{gold}}) + \alpha_{\text{kd}} T^2 \mathcal{L}_{\text{KL}}\left(\sigma\left(\frac{z_{\text{student}}}{T}\right), \sigma\left(\frac{z_{\text{teacher}}}{T}\right)\right)$$
  configured with temperature $T=2.0$ and $\alpha_{\text{kd}} = 0.50$.

---

### 6. Empirical Validation & Comparative Analysis

#### 6.1 Benchmark Results on Held-Out Fold 2 (`eval_scripts/evaluate_phase2.py`)
All models were trained on Fold 1 (35 legal cases, 4,120 reference entities) and evaluated against the private ground truth of Fold 2 (35 legal cases, 4,273 reference entities) under Strict Macro-F1:

| Experiment | Adaptation Method | Trainable Params | Strict Macro-F1 | Strict Precision | Strict Recall | Submission Package | Spans in Fold 3 |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Exp 1 (Enhanced)** | **Baseline Fine-Tuning** (Top 4 Layers + Head + Refinements) | ~48.2M (26.6%) | **0.2678** | **0.2497** | **0.3069** | `submission_phase2_baseline_ft.zip` | 5,767 |
| **Exp 1 (Raw)** | Baseline Fine-Tuning (Without Refinements) | ~48.2M (26.6%) | **0.1926** | 0.1746 | 0.2208 | — | 5,844 |
| **Exp 2** | **Compulsory LoRMA** ($r=16, \alpha=32$) | **~1.8M (1.0%)** | **0.1279** | 0.1224 | 0.1393 | `submission_phase2_compulsory_lorma.zip` | 4,494 |
| **Exp 3** | **Comparative Additive LoRA** ($r=16, \alpha=32$) | ~1.8M (1.0%) | **0.1025** | 0.1064 | 0.1096 | `submission_phase2_comparative_lora.zip` | 4,403 |
| **Exp 4** | **Knowledge Distillation** (Student 4L, $d=512$) | ~39.4M (Student) | **0.1452** | 0.1516 | 0.1471 | `submission_phase2_knowledge_distillation.zip` | 3,637 |

#### 6.2 Detailed Class-Level Performance Breakdown (Enhanced Baseline Fine-Tuning):
| Entity Label | Strict Precision | Strict Recall | Strict F1-Score | True Positives (TP) | False Positives (FP) | False Negatives (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`COURT`** | **0.7029** | **0.7760** | **0.7376** | 433 | 183 | 125 |
| **`DATE`** | **0.7041** | **0.7743** | **0.7375** | 590 | 248 | 172 |
| **`STAT`** | **0.4689** | **0.5020** | **0.4849** | 369 | 418 | 366 |
| **`AUTH`** | **0.2651** | **0.2158** | **0.2379** | 101 | 280 | 367 |
| **`JUDGE`** | **0.1741** | **0.3525** | **0.2331** | 43 | 204 | 79 |
| **`A.COUNSEL`** | **0.1812** | **0.2320** | **0.2035** | 29 | 131 | 96 |
| **`CASENO`** | **0.1262** | **0.2018** | **0.1553** | 68 | 471 | 269 |
| **`PREC`** | **0.1061** | **0.2129** | **0.1416** | 119 | 1,003 | 440 |
| **`R.COUNSEL`** | **0.1098** | **0.1856** | **0.1379** | 18 | 146 | 79 |
| **`APP`** | **0.0696** | **0.1800** | **0.1004** | 36 | 481 | 164 |
| **`RESP`** | **0.0216** | **0.0421** | **0.0286** | 8 | 362 | 182 |
| **`WIT`** | **0.0667** | **0.0083** | **0.0148** | 1 | 14 | 119 |

#### 6.3 Key Analytical Findings:
1. **Zero-Class Resolution Across the Board**:
   * Prior to word-boundary snapping and entity propagation, 4 classes (`APP`, `RESP`, `A.COUNSEL`, `WIT`) suffered from 0 True Positives due to subword splitting.
   * With word-boundary repair and connector bridging, **all 12 classes achieved non-zero True Positives**, pushing Strict Macro-F1 on Fold 2 up to **0.2678**.
   * Top institutional categories reached exceptional exact-match scores: `COURT` reached **0.7376 F1** and `DATE` reached **0.7375 F1**.
2. **LoRMA Outperforms Additive LoRA (+25% Relative Gain)**:
   * Compulsory LoRMA achieved **0.1279 Macro-F1** versus Additive LoRA's **0.1025 Macro-F1** with identical parameter counts (~1.8M params).
   * LoRMA demonstrated superior capacity on statutory provisions (`STAT`: 0.1690 vs 0.1029) and judicial citations (`PREC`: 0.0524 vs 0.0389), validating the ACL 2025 paper's thesis that multiplicative rank scaling expands feature capacity on specialized token vocabularies.
3. **Knowledge Distillation Achieves Exceptional Entity Retention**:
   * The compact 4-layer student model (39.4M parameters, 78% compression) scored **0.1452 Macro-F1**, retaining **75.4% of the teacher's capacity** and outperforming both parameter-efficient adaptation methods (LoRMA and LoRA).
   * Notably, the student model achieved a remarkable **0.6982 F1 on `COURT`** (406 TPs), demonstrating that distillation effectively transfers institutional legal vocabulary into compact architectures.

---

#### 6.4 Official CodaBench Leaderboard Progression & Empirical Results

On the official **CodaBench CS-781 Capstone Evaluation Platform** (held-out private Fold 3 test set evaluated under Strict Macro-F1), our submissions have demonstrated massive, iterative performance leaps:

| Iteration / Submission | Pipeline Innovations | Training Data | Strict Macro-F1 (CodaBench Official) | Status |
| :---: | :--- | :---: | :---: | :---: |
| **Run 1 (Initial Starter)** | Causal Attention Masking, raw token offsets, $w_O = 0.2$ loss | Fold 1 (35 docs) | **0.0035** | Baseline failure due to BPE whitespace & causal masking |
| **Run 2 (Bidirectional + Normalize)** | 4D Bidirectional Attention, raw whitespace trimming, balanced CE loss | Fold 1 (35 docs) | **0.2297** | +65.6x gain from bidirectional attention & whitespace fix |
| **Run 3 (Boundary Snapping & 2-Stage)** | 99.7% Word-boundary snapping, connector bridging, combined 70-doc fine-tuning | Fold 1 + Fold 2 (70 docs) | **0.3209 (~0.32)** | **Final Official Submission & Competition Score** |
| **Run 4 (Next-Gen Grounded Engine)** | Confidence NMS, Preamble Party Grounding, High-Precision Witness (`WIT`) Parser | Fold 1 + Fold 2 (70 docs) | *Local Benchmark* | **Fold 2 Val Strict Macro-F1: 0.2925** (`WIT` jumped from 0.0148 to **0.3665**) |

```mermaid
graph LR
    R1["Run 1: Initial Starter<br/><b>0.0035</b> Strict Macro-F1"] -->|Bidirectional Attention + Whitespace Normalization| R2["Run 2: Attention Refactor<br/><b>0.2297</b> Strict Macro-F1"]
    R2 -->|Word-Boundary Snapping + Connector Merging + 70-Doc Stage 2| R3["Run 3: Final Official Submission<br/><b>0.3209 (~0.32)</b> Strict Macro-F1"]
    R3 -.->|Confidence NMS + Preamble Party Grounding + Witness Engine| R4["Run 4 (Local Research): Grounded Engine<br/>Fold 2 Val: <b>0.2925</b> | Architecture Documented"]
```

---

#### 6.5 Next-Generation Grounded Engine (Run 4 Innovations & Diagnostics)

To surpass the **0.3209** milestone, Run 4 targeted the largest remaining bottlenecks discovered in our class-level audit:

1. **Confidence-Ranked Non-Maximum Suppression (NMS)**:
   - *Previous Defect*: Overlapping sliding-window spans were sorted by `start_char` ascending, allowing low-confidence boundary fragments near window edges to overwrite full, high-confidence entity spans.
   - *Solution*: Spans are sorted strictly by `confidence` descending. High-confidence model predictions claim character spans first, suppressing spurious overlapping fragments.

2. **The Witness (`WIT`) Breakthrough (+24.7x Relative F1 Increase)**:
   - *Previous Defect*: Fold 1 contained only 67 witness tokens, resulting in near-zero recall (F1 = **0.0148**, 1 TP out of 120 reference entities).
   - *Discovery*: Indian judicial decisions mark witnesses using institutional identifiers (`PW`, `LW`, `DW`, `CW`).
   - *Solution*: A high-precision parser extracts witness names adjacent to `PW/LW/DW/CW` markers with legal title/stopword filtering (`Dr.`, `Constable`, `Shri`, `When`), propagating confirmed witness names across the document.
   - *Result*: On held-out Fold 2, **`WIT` Precision jumped to 0.4930, Recall jumped to 0.2917, and F1 reached 0.3665** (35 TPs vs 1 previously).

3. **Preamble Party Grounding & Role Disambiguation (`APP` vs `RESP` vs `AUTH`)**:
   - *Discovery*: Over 54% of documents have explicit adversarial preambles where text before `"VERSUS"` is Appellant (`APP`) and text after `"VERSUS"` is Respondent (`RESP`).
   - *Impact*: Eliminates role confusion where the model misclassified appellants as respondents or tagged public statutory entities (like `State of Bihar` or `Indian Bank`) as generic `AUTH`. False positives on party labels decreased by over 50%.

4. **Updated Held-Out Validation Benchmark (Fold 2)**:
   - **Stage 1 Val Strict Macro-F1**: **`0.292530`** (up from `0.267800`, +9.2% relative gain)
   - **Stage 1 Val Strict Macro-Precision**: **`0.280781`**
   - **Stage 1 Val Strict Macro-Recall**: **`0.324755`**

---

### 7. Two-Stage Training & Submission Strategy

To ensure both rigorous local validation and maximum test set performance on CodaBench:
1. **Stage 1 (Local Validation & Evaluation)**:
   - Models are trained on Fold 1 (35 cases) and evaluated against held-out Fold 2 (35 cases).
   - This provides verified Strict Macro-F1 metrics under official evaluation rules without data leakage.
2. **Stage 2 (Final Test Set Submission)**:
   - Models continue fine-tuning on **combined Fold 1 + Fold 2 (all 70 cases)** for 3 additional epochs.
   - Doubling the training data exposes the models to twice as many judicial names, statute types, and court formats, maximizing test set generalization on Fold 3.

---

### 8. Official Submission Packaging & Verification Protocol

Every submission ZIP created by the pipeline underwent programmatic verification matching `eval_scripts/evaluate_phase2.py`:
1. **Archive Content**: Strictly contains only `prediction.json` at the archive root.
2. **Document ID Match**: Contains exact keys for all 35 test cases in `Phase_2_LNER/Test/fold3-text.json`.
3. **Span Validity**:
   - Every span is an integer tuple $[s, e, l]$ with $0 \le s < e$.
   - $l$ belongs strictly to the 12 official labels.
   - Zero duplicate or overlapping spans per document.

#### Generated Submission Packages:
| Submission Archive | Path | Size | Description |
| :--- | :--- | :---: | :--- |
| **`submission.zip`** *(Root)* | `./submission.zip` | 42.3 KB | **Next-Gen Grounded Engine** (Run 4, 70 docs, Confidence NMS + Witness & Preamble Grounding) ready for CodaBench |
| **`submission_phase2_baseline_ft.zip`** | `./cs781_phase2_output/submissions/` | 42.3 KB | Next-Gen Grounded Engine Archive with Boundary & Witness Grounding |
| **`submission_phase2_compulsory_lorma.zip`** | `./cs781_phase2_output/submissions/` | 32.2 KB | **Compulsory LoRMA** Multiplicative Adaptation (ACL 2025) |
| **`submission_phase2_comparative_lora.zip`** | `./cs781_phase2_output/submissions/` | 31.3 KB | Comparative Additive LoRA |
| **`submission_phase2_knowledge_distillation.zip`** | `./cs781_phase2_output/submissions/` | 26.1 KB | Mandatory Knowledge Distillation into 39.4M Student |
