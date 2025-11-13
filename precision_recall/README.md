# Precision–Recall Curve Visualisations

This script generates synthetic binary-classification datasets to demonstrate how precision–recall (PR) curves behave under common industry scenarios—spam detection, fraud detection, search ranking, recommender systems, anomaly detection, and more. Each subplot offers an annotated lesson that helps you diagnose model trade-offs.

## Getting Started

- Install dependencies: `numpy`, `matplotlib`, and `scikit-learn`.
- Execute the module:
  ```bash
  python precision_recall/precision_recall_curve_visualizations.py
  ```
- A 3×2 grid window appears, each subplot covering a different PR-curve story.

## Metrics Refresher

- **Precision** – Share of predicted positives that are actually positive; guards against false alarms.
- **Recall** – Share of actual positives the model correctly retrieves; guards against misses.
- **Precision–Recall Curve** – Plots precision vs. recall as the classification threshold sweeps from 1→0.
- **Baseline (Positive Rate)** – Horizontal reference equal to the class prior; any curve above it adds value over random guessing.

## Scenario Cheat Sheet

| Scenario | What You See | How to Interpret |
| --- | --- | --- |
| Good Model on Imbalanced Data | Curve stays near the top-right. | High confidence predictions despite few positives. |
| Weak Model on Imbalanced Data | Precision falls quickly as recall increases. | Model struggles; revisit features or architecture. |
| High Precision, Low Recall | Curve hugs top-left then drops sharply. | Conservative threshold; many positives missed. |
| High Recall, Low Precision | Curve achieves far recall but low precision. | Aggressive strategy; expect many false positives. |
| Severe Class Imbalance (1% Positives) | Lower baseline with steep gains early. | Highlights why PR curves beat ROC under extreme imbalance. |
| Overfitting: Train vs Validation | Train curve dominates validation curve. | Gap indicates memorisation; regularise or collect more data. |

## Key Terms & Tips

- **Threshold Tuning** – Choose operating point that balances business cost of false positives vs false negatives.
- **Calibrated Scores** – Well-calibrated probabilities make PR curves smoother and easier to interpret.
- **Class Imbalance** – When positives are rare, PR curves provide more insight than ROC curves.
- **Overfitting** – Divergence between train and validation curves warns that generalisation is degrading.
- **Sampling Strategy** – Adjust positive/negative sampling to mimic deployment class priors; otherwise PR curves may mislead.

Use these plots as quick intuition pumps when explaining PR trade-offs to stakeholders or teammates.
