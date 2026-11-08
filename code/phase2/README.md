# Phase 2: Legal Named Entity Recognition (L-NER)

## 1. Task Description
Extract mentions of 12 legal entity types from Indian judicial text with exact character-level offsets (`[start, end, label]`).

## 2. Methodology & Key Innovations
- **Bidirectional Attention Adapter**: Converts causal attention in the Phase 1 LLaMA backbone to full bidirectional context.
- **Sliding-Window Chunking**: 512 token windows with 128 token stride.
- **Strict Word-Boundary Snapping**: Eliminates subword offset shift artifacts.
- **Confidence NMS**: Resolves multi-window span overlaps.

## 3. Official CodaBench Performance
- **Metric**: Strict Macro-F1
- **Official Score**: **0.3209**

## 4. Run Instructions
```bash
pip install -r requirements.txt
python inference.py
```
