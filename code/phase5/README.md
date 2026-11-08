# Phase 5: Legal Statute Identification (LSI)

## 1. Task Description
Predict the set of applicable statutory provisions and legislative enactments for Indian judicial decisions.

## 2. Key Methodological Innovations
- **Independent Sigmoids with Adaptive Thresholding**: Replaces brittle global thresholding with frequency-calibrated per-statute decision boundaries.
- **Head-Tail Multi-Segment Windowing**: Preserves statutory invocations situated in both the opening jurisdictional preamble and the closing ratio decidendi.
- **LoRMA Parameter Efficiency**: Achieves 0.4014 Macro-F1 with 99.5% fewer trainable parameters.

## 3. Official CodaBench Performance
- **Primary Metric**: Macro-F1
- **Official Score**: **0.4014**

## 4. Run Instructions
```bash
pip install -r requirements.txt
python inference.py
```
