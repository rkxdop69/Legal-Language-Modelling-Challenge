# Phase 7: Prior Case Retrieval (PCR)

## 1. Task Description
Retrieve the top $K=7$ most relevant prior judicial precedents from a candidate pool of 1,727 cases for each of 237 test queries.

## 2. Root Cause Analysis & Breakthrough Methodologies
1. **Enforced Self-Match Masking**: In the raw candidate pool, each query appears in the candidate pool. Without masking, cosine similarity ($1.0$) placed the query at Rank 1, wasting a slot as an automatic false positive. Setting $S(q, c) = -\infty$ whenever $cid == qid$ instantly rescued this slot.
2. **Full-Text Representation**: Truncation to 160 words discarded 98% of judicial facts and statutory provisions. Sublinear TF-IDF (1, 2) on full text preserves core statutory terminology.
3. **Citator Graph Matching**: Leverages Indian Kanoon `CITATOR INFO:` headers and formal law report citations (`AIR`, `SCC`, `SCR`) to construct bipartite citation links.
4. **Statutory Provision Jaccard**: Measures overlap of sections and articles under judicial interpretation.
5. **Temporal Admissibility Filter**: Penalizes future candidate cases decided after the query case.

## 3. Official CodaBench Performance
- **Primary Metric**: Micro-F1@7
- **Official Score**: **0.4548** (Breakthrough performance from initial 0.02--0.14 baselines)

## 4. Run Instructions
```bash
pip install -r requirements.txt
python inference.py
```
