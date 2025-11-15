# Hyperparameter Tuning Landscape Plots

This module fabricates smooth-but-noisy hyperparameter search spaces so you can explain how tuning dashboards (Weights & Biases sweeps, Optuna studies, TensorBoard HParams, etc.) communicate model behavior. Running the script renders five figures and stores them in the local `figures/` directory along with on-plot annotations that highlight the big takeaways.

## How the Data Is Synthesized

- Each hyperparameter dimension (learning rate, batch size, depth, width, weight decay) is sampled on a grid or via Monte Carlo trials.
- Performance scores combine bell-shaped optima, monotonic penalties, and light random noise to mimic smooth but imperfect validation signals.
- Surfaces are clipped into `[0, 1]` to resemble accuracy or inverse loss.
- A fixed RNG seed keeps the visuals deterministic across runs.

## Key Hyperparameter Terms

- **Learning Rate (LR)** – Step size used by gradient descent optimizers. Very low LR stalls learning; very high LR diverges.
- **Batch Size** – Number of examples per gradient update. Tiny batches are noisy/unstable; huge batches can generalize poorly unless LR is tuned.
- **Model Depth** – Number of layers (e.g., transformer blocks). Extremely deep networks can overfit or become unstable without regularization.
- **Model Width** – Hidden dimension or neurons per layer. Wider models capture more detail but overfit when data is limited.
- **Weight Decay** – L2 regularization coefficient. Too little decay overfits; too much shrinks weights and underfits.

## Generated Visualizations

1. **Learning Rate vs Batch Size Heatmap (`learning_rate_vs_batch_size_heatmap.png`)**
   - Shows a diagonal ridge of stable accuracy where LR and batch size are balanced.
   - High LR or very small batch sizes push the model into noisy/unstable regions.

2. **Learning Rate vs Weight Decay Heatmap (`learning_rate_vs_weight_decay_heatmap.png`)**
   - Classic U-shape for LR (too low or too high hurts performance) combined with monotonic regularization.
   - Highlights the sweet spot at moderate LR and moderate weight decay.

3. **Architecture Search Surface (`architecture_search_surface.png`)**
   - 3D surface of depth × width.
   - Valley indicates balanced architectures; the outer walls show overfitting/instability in over-parameterized regions.

4. **Parallel Coordinates Plot (`hyperparameter_parallel_coordinates.png`)**
   - Each polyline is a trial; colors denote performance buckets.
   - Where the lines converge you find consistent “good” hyperparameter ranges.

5. **Scatter of Hyperparameter Trials (`hyperparameter_trials_scatter.png`)**
   - Optuna/W&B style scatter with LR on the x-axis and batch size on the y-axis (both log scales).
   - Point color encodes accuracy; clustering shows promising regions while the bottom-right (high LR + tiny batch) illustrates divergence.

## How to Reproduce

```bash
cd hyperparameter_tuning_landscape_plots
python hyperparameter_landscape_visualizations.py
```

The script saves all figures to `hyperparameter_tuning_landscape_plots/figures/` and also opens interactive windows if you have a GUI backend.

