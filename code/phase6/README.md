# Phase 6: Abstractive Legal Summarization

## 1. Task Description
Compress extensive Indian Supreme Court judgments (~4,368 words average) into concise executive legal headnote summaries.

## 2. Key Methodological Innovations
- **Discourse-Guided Lead-Tail Windowing**: Captures both jurisdictional parties and the operative ratio decidendi.
- **Repetition Penalty Tuning**: Repetition penalty 1.22 prevents circular loops of statutory boilerplate.
- **BERTScore Alignment**: Directly targets semantic and contextual alignment rather than brittle n-gram surface overlap.

## 3. Official CodaBench Performance
- **Primary Metric**: BERTScore F1
- **Official Score**: **0.8074** (Top of Competition Leaderboard)

## 4. Run Instructions
```bash
pip install -r requirements.txt
python inference.py
```
