# Metric Curve Visualisations

This module generates synthetic training-versus-validation curves that illustrate how different metrics behave across computer vision, NLP, and speech tasks. The plots are meant to act as teaching aids: each subplot shows a common learning pattern and highlights what to look for when diagnosing models.

## Getting Started

- Install dependencies: `matplotlib` and `numpy`.
- Run the script from the project root:
  ```bash
  python accuracy_curves/metric_curve_visualizations.py
  ```
- A 4×2 grid of subplots will open, each showing a unique metric scenario with annotations.

## Metrics Explained

- **Accuracy** – Fraction of correct predictions. Higher is better.
- **F1 Score** – Harmonic mean of precision and recall. Useful when classes are imbalanced.
- **BLEU Score** – Measures n-gram overlap between generated and reference text in machine translation; higher indicates closer match.
- **ROUGE-L** – Recall-oriented metric based on longest common subsequence; common in summarisation tasks.
- **Word Error Rate (WER)** – Normalised edit distance between predicted and reference transcripts; lower is better.
- **Generic Metric** – Placeholder for any bounded score (e.g., AUC) to illustrate plateauing behaviour.

## Scenario Interpretations

| Scenario                    | What You See                                                        | How to Interpret                                                                           |
| --------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Accuracy Improving Normally | Training and validation accuracy rise steadily.                     | Healthy learning; both curves converge without divergence.                                 |
| Accuracy Overfitting        | Training accuracy keeps climbing while validation stalls or drops.  | Model memorises training data; apply regularisation, data augmentation, or early stopping. |
| F1 Score Improving Smoothly | Both curves trend upward with mild noise.                           | Classifier balances precision and recall over time.                                        |
| BLEU Jump Then Stabilise    | Sharp early increase followed by a plateau.                         | Typical MT behaviour; further gains need targeted improvements.                            |
| ROUGE-L Slow Growth         | Gradual, logarithmic improvement.                                   | Summarisation model improves slowly; longer training or curriculum learning may help.      |
| WER Decreasing Normally     | Training and validation WER both fall.                              | Speech model makes fewer errors; convergence looks healthy.                                |
| WER Overfitting             | Training WER keeps dropping while validation levels off or worsens. | Overfitting; consider regularisation or more data.                                         |
| Metric Saturation / Plateau | Both curves flatten early.                                          | Optimiser stuck; adjust learning rate, architecture, or loss scheduling.                   |

## Key Terms

- **Overfitting** – Model performs well on training data but poorly on unseen data due to memorisation.
- **Plateauing** – Metric stops improving despite continued training, indicating optimisation challenges.
- **Regularisation** – Techniques (dropout, weight decay) that penalise complexity to improve generalisation.
- **Early Stopping** – Halt training when validation performance stops improving to avoid overfitting.
- **Learning Rate (LR)** – Step size for gradient-based optimisers; too high causes divergence, too low causes plateaus.

Use the annotations in each subplot as quick reminders of the behaviours to monitor during your own training runs.
