# Phase 4: Court Judgment Prediction (CJPE)

## 1. Task Description
Predict the definitive binary judicial disposition of Indian Supreme Court cases (0: Dismissed vs 1: Allowed).

## 2. Key Methodological Innovations
- **Hierarchical Chunk Attention Pooling**: Segments extensive judgments into 4 structural chunks spanning 2048 tokens, aggregated via multi-head cross-chunk attention pooling.
- **Overcoming Truncation**: Unlike naive head-truncation that clips the operative disposition, hierarchical pooling preserves evidence across the entire judicial opinion.
- **Calibrated Decision Thresholding**: Optimizes Macro-F1 across balanced class distributions.

## 3. Official CodaBench Performance
- **Primary Metric**: Macro-F1
- **Official Score**: **0.6136** (Top Active Leaderboard Submission)

## 4. Run Instructions
```bash
pip install -r requirements.txt
python inference.py
```
