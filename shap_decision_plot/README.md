## Individual Prediction Explanations

This module demonstrates how to explain a **single loan-risk prediction** using two complementary local interpretability techniques:

- `SHAP` (`shap.force_plot` + `shap.decision_plot`)
- `LIME` (local surrogate explainer)

The script trains a Gradient Boosting classifier on a synthetic consumer-loan dataset (1,500 samples, 10 base features + 7 one-hot columns) and saves all artifacts under `shap_decision_plot/figures`.

### Key Terms
- **Default probability** – model-estimated likelihood (0–1) that a customer will default.
- **Base value** – average model output on the training distribution; explanations show how features push the prediction away from this reference.
- **SHAP value** – marginal contribution of a feature toward the model output for a specific instance.
- **Force plot** – horizontal SHAP visualization with red (risk-increasing) vs. blue (risk-reducing) forces that push from the base value to the final prediction.
- **Decision plot** – cumulative SHAP line plot that shows how each feature adjusts the logit score and in what order.
- **LIME** – Local Interpretable Model-agnostic Explanations; it fits a weighted linear surrogate around the selected point and reports feature weights.
- **Local surrogate** – simplified model that mimics the black-box behavior in the neighborhood of one input.
- **One-hot encoding** – converts categorical entries (`region`, `loan_purpose`) into binary indicator columns so tree ensembles receive numeric input.

### Generated Artifacts
Running `python individual_prediction_explanations.py` creates:

| File | Description |
| --- | --- |
| `figures/shap_force_plot.html` | Interactive SHAP force explanation for the chosen borrower. |
| `figures/shap_decision_plot.png` | Static SHAP decision plot (all feature labels visible thanks to custom sizing/margins). |
| `figures/lime_local_explanation.png` | LIME bar chart of local feature weights. |
| `figures/lime_local_explanation.json` | Raw LIME weights for downstream use. |
| `figures/individual_prediction_summary.json` | Metadata containing the selected index, prediction, and artifact paths. |

> The image file written to `figures/shap_decision_plot.png` is also stored in the repository so you can preview the explanation without re-running the script.

### Dataset + Model
- **Use case**: retail lending default risk.
- **Numeric features**: credit score, annual income, loan amount, debt-to-income ratio, loan term (months), employment length (years), age, historical late payments.
- **Categorical features**: region (`Northeast`, `Midwest`, `South`, `West`) and loan purpose (`debt_consolidation`, `auto`, `home_improvement`, `small_business`).
- **Model**: `GradientBoostingClassifier` (scikit-learn) trained with default hyperparameters.

### What the SHAP Decision Plot Shows
For the example borrower (index `3`):
1. The line starts at the base value (`~0.23` default probability).  
2. Features with positive SHAP values bend the line to the right (higher risk). In the stored image you can observe:
   - `loan_amount` ≈ \$22K and **`credit_score` 631** push the prediction strongly upward.
   - Short `employment_length` (`3` years) and **1 late payment** add smaller boosts.
3. Features with negative SHAP values pull left (lower risk). For this borrower:
   - Lower `debt_to_income` (0.264), high `annual_income` (~\$100K), and a short `loan_term_months=36` all offset some risk.
4. The final value (~0.80) is the model’s estimated default probability after all SHAP contributions are applied in order.

### Quickstart
```bash
cd shap_decision_plot
pip install -r ../requirements.txt  # or pip install numpy pandas scikit-learn shap lime matplotlib
python individual_prediction_explanations.py
```

Open `figures/shap_decision_plot.png` to inspect the saved explanation or view the interactive `shap_force_plot.html` in a browser. Use the `sample_idx` variable inside the script to explore other individuals. Once regenerated, the figures folder will always contain the latest image so you can share or embed it directly.

