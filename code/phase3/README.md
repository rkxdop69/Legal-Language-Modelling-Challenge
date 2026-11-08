# Phase 3: Rhetorical Role Prediction (RR)

## 1. Task Description
Classify each sentence in Indian judicial decisions into one of 13 official judicial rhetorical roles.

## 2. Key Methodological Innovations
- **Discourse Relative Position Projections**: 2-layer MLP encoding relative position ($idx / N$).
- **Smoothed Inverse-Frequency Loss**: $w_c = (N/N_c)^{0.35}$ prevents minority class starvation (`Dissent`, `PrecedentOverruled`).
- **Discourse Calibration Engine**: Enforces structural transitions (e.g. `Argument` leads to `Analysis` and `Ratio`).

## 3. Official CodaBench Performance
- **Primary Metric**: Macro-F1
- **Official Score**: **0.5735**

## 4. Run Instructions
```bash
pip install -r requirements.txt
python inference.py
```
