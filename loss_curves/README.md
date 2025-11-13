# Loss Curve Visualisations

This folder contains synthetic training runs that illustrate how loss curves behave under common deep-learning scenarios. Each subplot provides a short note to help you recognise the pattern during your own experiments.

## Running the Demo

- Install dependencies: `numpy` and `matplotlib`.
- Execute the script:
  ```bash
  python loss_curves/loss_curve_visualizations.py
  ```
- A 4×2 grid of line charts will open, one scenario per subplot with annotations in the corner.

## Loss Curve Basics

- **Training Loss** – Error on the mini-batches the model trains on; should trend downward when learning.
- **Validation Loss** – Error on held-out data; tracks how well the model generalises.
- **Healthy Trend** – Training and validation losses drop together and then stabilise.
- **Warning Signs** – Diverging curves, noisy spikes, or flat lines can signal problems.

## Scenario Guide

| Scenario | What You See | Interpretation |
| --- | --- | --- |
| Ideal Training | Both curves fall smoothly and stay close. | Model fits well and generalises. |
| Overfitting | Training loss keeps falling, validation loss turns upward. | Stop early, add regularisation, or gather more data. |
| Underfitting | Both losses stay high with little change. | Model capacity too low or not trained long enough. |
| High Learning Rate / Noisy | Loss zig-zags wildly. | Lower the learning rate or add scheduling/gradient clipping. |
| Too Low Learning Rate | Slow, shallow decrease. | Increase learning rate or use adaptive optimisers. |
| Data Leakage | Validation loss far below training loss. | Suspect contamination between train/validation splits. |
| Optimisation Plateau | Loss drops then flattens early. | Adjust optimiser, learning rate, or model initialisation. |

## Practical Tips

- Plot both training and validation losses on the same axis whenever possible.
- Monitor the gap between curves; a widening gap often means overfitting.
- Combine loss curves with accuracy or other metrics for a fuller picture.
- Use annotations from these scenarios as a quick checklist when reviewing experiment logs.

Feel free to adapt the generators to mimic your own datasets or to demonstrate additional behaviours like curriculum learning or regularisation effects.
