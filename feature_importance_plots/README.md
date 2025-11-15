# Feature Importance Visualizations

This module demonstrates how ML engineers interpret feature importance using synthetic data, a tree-based model, and three complementary explainability techniques.

## Key Terms

- **Feature importance**: A score indicating how much a model relies on a given feature when making predictions.
- **Impurity importance**: The built-in measure from decision trees/ensembles that sums how much each feature decreases node impurity (Gini/entropy) across the forest.
- **Permutation importance**: A model-agnostic technique that permutes one feature at a time and measures the drop in validation performance—larger drops mean higher true reliance.
- **SHAP (SHapley Additive exPlanations)**: A unified game-theoretic framework that attributes each prediction to individual feature contributions.
- **SHAP values**: Signed contributions showing how each feature pushes a prediction above or below the baseline expectation; absolute values reflect impact magnitude.

## Dataset & Model

- Synthetic binary classification dataset with 1,000 samples and 10 engineered features (mix of informative, redundant, and noisy signals).
- Train/test split with stratification; RandomForestClassifier (400 trees, depth-limited) for stability across plots.

## Visualizations

1. **Tree-Based Feature Importance (Impurity)**
   - Horizontal bar chart sorted by the model’s built-in feature_importances_ values.
   - Highlights which features the trees split on most often.
   - Fast to compute but biased toward high-cardinality or noisy features.
   - Annotation reminds viewers that higher bars mean stronger reliance.

2. **Permutation Importance (More Reliable)**
   - Horizontal bar chart showing the mean validation accuracy drop when each feature is permuted.
   - Provides a model-agnostic check: if shuffling a feature barely changes accuracy, that feature contributes little predictive power.
   - Requires a trained model plus validation data, but mitigates bias from tree internals.
   - Annotation emphasizes that this reflects true predictive contribution.

3. **SHAP Summary Plot (Global Interpretability)**
   - Beeswarm-style scatter where each point is a feature value for a specific sample.
     - X-axis: SHAP value (positive drives prediction toward class 1, negative toward class 0).
     - Color: feature value (blue low, red high) to reveal how magnitude/direction vary with feature level.
     - Vertical spread: strength of interactions and variability across samples.
   - Annotation explains the color, spread, and position cues so users know how to read the chart.

4. **SHAP Feature Importance (Mean |SHAP| Values)**
   - Bar chart of average absolute SHAP values per feature.
   - Ranks features by overall influence while remaining faithful to SHAP’s additive explanations.
   - Useful for regulator-facing dashboards or high-level documentation.

## Running the Script

```bash
python feature_importance_visualizations.py
```

Requirements: `scikit-learn`, `seaborn`, `matplotlib`, and `shap` (compatible with `numpy<2`). The script generates all plots in sequence, saves PNG files under `feature_importance_plots/figures/`, and pauses until the figures are closed.

