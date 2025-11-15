"""Individual prediction interpretability with SHAP and LIME on synthetic data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def generate_synthetic_loan_data(
    n_samples: int = 1500, random_state: int | None = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Create a tabular dataset that mimics a consumer loan risk scenario."""
    rng = np.random.default_rng(random_state)

    credit_score = rng.normal(680, 60, n_samples).clip(300, 850)
    annual_income = rng.lognormal(mean=10.8, sigma=0.4, size=n_samples)  # dollars
    loan_amount = rng.normal(18000, 6000, n_samples).clip(3000, 50000)
    debt_to_income = rng.normal(0.32, 0.08, n_samples).clip(0.05, 0.75)
    loan_term_months = rng.choice([36, 48, 60], size=n_samples, p=[0.4, 0.3, 0.3])
    employment_length = rng.integers(0, 31, n_samples)
    age = rng.integers(21, 70, n_samples)
    late_payments = rng.poisson(0.8, n_samples).clip(0, 10)

    regions = ["Northeast", "Midwest", "South", "West"]
    purposes = ["debt_consolidation", "auto", "home_improvement", "small_business"]
    region = rng.choice(regions, size=n_samples, p=[0.25, 0.2, 0.35, 0.2])
    loan_purpose = rng.choice(purposes, size=n_samples, p=[0.45, 0.2, 0.2, 0.15])

    region_risk = {"Northeast": 0.05, "Midwest": -0.02, "South": 0.08, "West": -0.03}
    purpose_risk = {
        "debt_consolidation": 0.04,
        "auto": -0.05,
        "home_improvement": -0.02,
        "small_business": 0.12,
    }

    base_logit = (
        0.006 * (700 - credit_score)
        + 0.00003 * (loan_amount - 15000)
        + 2.5 * (debt_to_income - 0.3)
        + 0.45 * late_payments
        - 0.000002 * (annual_income - 60000)
        + 0.015 * (loan_term_months - 48)
        - 0.03 * employment_length
        - 0.01 * (age - 40)
    )

    for idx in range(n_samples):
        base_logit[idx] += region_risk[region[idx]] + purpose_risk[loan_purpose[idx]]

    default_probability = 1 / (1 + np.exp(-base_logit))
    default_flag = rng.binomial(1, default_probability)

    features = pd.DataFrame(
        {
            "credit_score": credit_score,
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "debt_to_income": debt_to_income,
            "loan_term_months": loan_term_months,
            "employment_length": employment_length,
            "age": age,
            "late_payments": late_payments,
            "region": region,
            "loan_purpose": loan_purpose,
        }
    )

    return features, default_flag


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """One-hot encode categorical columns."""
    categorical_cols = ["region", "loan_purpose"]
    encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    encoded = encoded.astype(float)
    feature_names = encoded.columns.tolist()
    return encoded, feature_names


def train_model(X_train: pd.DataFrame, y_train: np.ndarray) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(random_state=0)
    model.fit(X_train, y_train)
    return model


def explain_with_shap(
    model: GradientBoostingClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
    sample_idx: int,
    output_dir: Path,
):
    """Generate SHAP force and decision plots for an individual prediction."""
    X_train_array = X_train.to_numpy(dtype=float)
    X_test_array = X_test.to_numpy(dtype=float)

    explainer = shap.TreeExplainer(
        model, data=X_train_array, feature_names=feature_names
    )
    shap_values = explainer.shap_values(X_test_array)

    base_value = explainer.expected_value
    instance_features = X_test.iloc[sample_idx]
    instance_shap = shap_values[sample_idx]

    force_plot = shap.force_plot(
        base_value,
        instance_shap,
        instance_features,
        feature_names=feature_names,
        matplotlib=False,
    )

    force_path = output_dir / "shap_force_plot.html"
    shap.save_html(force_path.as_posix(), force_plot)

    plt.figure(figsize=(10, 12))
    shap.decision_plot(
        base_value,
        instance_shap,
        features=instance_features.values,
        feature_names=feature_names,
        link="logit",
        auto_size_plot=False,
        show=False,
    )
    ax = plt.gca()
    ax.tick_params(axis="y", labelsize=11)
    plt.subplots_adjust(left=0.35, right=0.98, top=0.97, bottom=0.05)
    decision_path = output_dir / "shap_decision_plot.png"
    plt.tight_layout()
    plt.savefig(decision_path, dpi=200)
    plt.close()

    return {
        "force_plot": force_path.as_posix(),
        "decision_plot": decision_path.as_posix(),
        "base_value": float(base_value),
    }


def explain_with_lime(
    model: GradientBoostingClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
    sample_idx: int,
    output_dir: Path,
) -> dict:
    """Create a LIME local surrogate explanation and persist artifacts."""
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=["No Default", "High Risk"],
        mode="classification",
        discretize_continuous=True,
        random_state=7,
    )

    def predict_fn(data: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(data, columns=feature_names)
        return model.predict_proba(frame)

    lime_exp = explainer.explain_instance(
        data_row=X_test.values[sample_idx],
        predict_fn=predict_fn,
        num_features=8,
    )

    lime_fig = lime_exp.as_pyplot_figure()
    lime_plot_path = output_dir / "lime_local_explanation.png"
    lime_fig.tight_layout()
    lime_fig.savefig(lime_plot_path, dpi=200)
    plt.close(lime_fig)

    lime_text_path = output_dir / "lime_local_explanation.json"
    with open(lime_text_path, "w", encoding="utf-8") as fp:
        json.dump(lime_exp.as_list(), fp, indent=2)

    return {
        "lime_plot": lime_plot_path.as_posix(),
        "lime_weights": lime_text_path.as_posix(),
    }


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic loan-risk dataset...")
    raw_features, target = generate_synthetic_loan_data()
    features, feature_names = prepare_features(raw_features)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, stratify=target, random_state=8
    )

    print("Training Gradient Boosting classifier...")
    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nClassification report on hold-out set:")
    print(classification_report(y_test, y_pred, digits=3))

    sample_idx = 3
    pred_prob = model.predict_proba(X_test.iloc[[sample_idx]])[0, 1]
    print(
        f"\nSample index {sample_idx} predicted default probability: {pred_prob:.3f}"
    )

    print("\nGenerating SHAP explanations...")
    shap_artifacts = explain_with_shap(
        model, X_train, X_test, feature_names, sample_idx, output_dir
    )

    print("Generating LIME explanations...")
    lime_artifacts = explain_with_lime(
        model, X_train, X_test, feature_names, sample_idx, output_dir
    )

    summary = {
        "sample_index": sample_idx,
        "predicted_default_probability": float(pred_prob),
        "shap_outputs": shap_artifacts,
        "lime_outputs": lime_artifacts,
    }

    summary_path = output_dir / "individual_prediction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("\nArtifacts saved to:", output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


