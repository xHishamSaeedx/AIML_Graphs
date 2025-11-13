# ROC Curve Visualisations

This module fabricates several binary-classification scenarios so you can see how receiver operating characteristic (ROC) curves shift with model quality, overfitting, thresholds, and class balance. Each subplot is annotated so the story is easy to relay to non-ML teammates.

## How to Run

- Dependencies: `numpy`, `matplotlib`, and `scikit-learn`.
- Execute:
  ```bash
  python roc_curve/roc_curve_visualizations.py
  ```
- A 3×2 window of ROC plots appears, each highlighting a different takeaway.

## ROC Refresher

- **True Positive Rate (TPR / Recall / Sensitivity)** – Fraction of positives you catch.
- **False Positive Rate (FPR / 1 − Specificity)** – Fraction of negatives you mistakenly flag.
- **ROC Curve** – Traces TPR vs FPR as you sweep the classification threshold.
- **Area Under the Curve (AUC)** – Single-number score summarising ROC performance; 1.0 is perfect, 0.5 is random.
- **Chance Line** – Diagonal baseline (`y = x`). Curves above this line indicate real skill.

## Scenario Guide (Plain Speak)

| Scenario | What You See | Translation |
| --- | --- | --- |
| Good Model (AUC ≈ 0.95) | Curve hugs the top-left corner. | Model separates classes cleanly; almost always better than the baseline. |
| Weak Model (AUC ≈ 0.6) | Gently above the diagonal. | Barely helpful; may need richer features or more data. |
| Random Model (AUC ≈ 0.5) | Sits on the diagonal. | Classifier is guessing; AUC equals coin flip. |
| Overfitting: Train vs Validation | Train curve is stellar, validation is mediocre. | Memorisation on training data; expect disappointment in production. |
| Threshold-Sensitive Model | Good AUC but highlighted point shows high TPR/high FPR. | You can catch more positives but pay with lots of false alarms; threshold choice matters. |
| Balanced vs Imbalanced Dataset | Balanced curve outperforms, imbalanced lags. | ROC can hide imbalance pain; consider PR curves or per-class metrics. |

## Tips for Practice

- Compare ROC with precision–recall when positives are rare; AUC can look fine while precision collapses.
- Use ROC to choose operating points that match business constraints (e.g., acceptable false-alarm rate).
- Inspect train vs validation ROC gaps to catch overfitting early.
- AUC is threshold-independent, but deployments always need a fixed threshold—use annotations like the threshold-sensitive example to guide that choice.

These visuals double as quick slides for model reviews or onboarding sessions where you need to explain ROC behavior without diving into raw math.
