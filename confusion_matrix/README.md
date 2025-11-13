# Confusion Matrix Visualisations

This module builds synthetic confusion matrices to show how different classification issues look in practice. Each heatmap is labelled to help you explain the story to teammates or stakeholders without diving into raw numbers.

## How to Run

- Install dependencies: `numpy`, `matplotlib`, and `seaborn`.
- Launch the script:
  ```bash
  python confusion_matrix/confusion_matrix_visualizations.py
  ```
- A 3×2 grid of heatmaps appears; each one is a separate scenario with a tooltip-style note underneath.

## Reading a Confusion Matrix (Quick Refresher)

- Rows are the *actual* classes; columns are the *predicted* classes.
- Diagonal cells show correct predictions. Off-diagonal cells show mistakes.
- Darker colours mean more samples in that cell.

## Scenario Guide (Plain Language)

| Scenario | What You See | What It Means |
| --- | --- | --- |
| Balanced Performance | Strong diagonal, light elsewhere. | Model gets most examples right; errors are rare and random. |
| Class Imbalance | First row dominates; lower rows are faint. | Model sees class A most often and struggles with the minority classes. |
| Neighbor Confusion | Bright blocks around classes B and C. | These two classes look similar to the model; review features or labelling. |
| Overfitting (Train) | Almost perfect diagonal. | Training data memorised; beware of a large gap to validation performance. |
| Overfitting (Validation) | Many off-diagonal cells lit up. | Once tested on new data, accuracy collapses; confirms the overfitting suspicion. |
| Mislabeled Training Data | Column B attracts counts from class C. | Likely label noise sending class C examples into class B; audit your dataset. |
| Underfitting | Every cell has similar counts. | Model acts like a guesser; capacity or features are insufficient. |

## Key Takeaways

- Focus on the diagonal vs. off-diagonal contrast to judge overall accuracy.
- Compare train vs. validation matrices to spot overfitting early.
- Watch for consistent error funnels (e.g., C → B) that hint at mislabeled data or overlapping classes.
- In imbalanced problems, normalise the matrix or inspect per-class recall so minority classes aren’t ignored.

Use these plots as quick visual aids when presenting model diagnostics or planning remediation steps.
