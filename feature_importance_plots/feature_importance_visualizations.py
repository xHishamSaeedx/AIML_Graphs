"""
Feature importance interpretability demo.

This script generates a synthetic classification dataset, trains a tree-based
model, and visualizes three popular feature-importance techniques:
    1. Impurity-based (Gini) importance from the trained estimator.
    2. Permutation importance (model-agnostic, more faithful).
    3. SHAP summary plots (beeswarm + bar) for modern global interpretability.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore", category=UserWarning)
sns.set(style="whitegrid", context="talk")
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_synthetic_data(random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Create a tabular dataset with informative, redundant, and noise features."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_clusters_per_class=2,
        flip_y=0.03,
        class_sep=1.2,
        weights=[0.55, 0.45],
        random_state=random_state,
    )

    feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="target")


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Persist figures to disk with consistent settings."""
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")


def plot_impurity_importance(importances: pd.Series) -> None:
    """Plot horizontal bar chart for impurity-based feature importances."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        x=importances.values,
        y=importances.index,
        ax=ax,
        palette="Blues_r",
    )
    ax.set_title("Tree-Based Feature Importance (Impurity)")
    ax.set_xlabel("Importance (Gini decrease)")
    ax.set_ylabel("Feature")
    ax.text(
        0.02,
        0.95,
        "High value = model relies heavily on this feature",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    fig.tight_layout()
    save_figure(fig, "impurity_feature_importance.png")


def plot_permutation_importance(importances: pd.Series) -> None:
    """Plot horizontal bar chart for permutation importances."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        x=importances.values,
        y=importances.index,
        ax=ax,
        palette="viridis",
    )
    ax.set_title("Permutation Importance (More Reliable)")
    ax.set_xlabel("Mean accuracy drop when permuted")
    ax.set_ylabel("Feature")
    ax.text(
        0.02,
        0.95,
        "Permutation importance reveals true predictive contribution",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    fig.tight_layout()
    save_figure(fig, "permutation_feature_importance.png")


def plot_shap_summary(model: RandomForestClassifier, X: pd.DataFrame) -> None:
    """
    Generate SHAP beeswarm and bar plots with helpful annotations.

    SHAP explains how each feature pushes predictions up or down in aggregate.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Binary classifiers return a list; take the positive class explanations.
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values_to_plot,
        X,
        feature_names=X.columns,
        show=False,
        color_bar_label="Feature value",
    )
    ax = plt.gca()
    ax.set_title("SHAP Summary Plot (Global Interpretability)", fontsize=16)
    ax.text(
        0.02,
        0.02,
        "Color shows feature value (blue=low, red=high)\n"
        "Spread indicates interaction strength\n"
        "Position shows feature influence magnitude",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )
    plt.tight_layout()
    save_figure(plt.gcf(), "shap_summary_beeswarm.png")

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values_to_plot,
        X,
        feature_names=X.columns,
        plot_type="bar",
        show=False,
    )
    ax = plt.gca()
    ax.set_title("SHAP Feature Importance (Mean |SHAP| Values)", fontsize=16)
    ax.text(
        0.02,
        0.95,
        "Bars show average absolute impact\n"
        "Great for regulator-facing dashboards",
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )
    plt.tight_layout()
    save_figure(plt.gcf(), "shap_summary_bar.png")


def main() -> None:
    """Run the end-to-end feature importance tutorial."""
    X, y = create_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    impurity_importance = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=True)
    plot_impurity_importance(impurity_importance)

    perm_result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )
    permutation_importance_values = pd.Series(
        perm_result.importances_mean, index=X.columns
    ).sort_values(ascending=True)
    plot_permutation_importance(permutation_importance_values)

    plot_shap_summary(model, X_test)

    plt.show()


if __name__ == "__main__":
    main()

