"""
ROC Curve Visualizations
========================

This script synthesizes several classification scenarios and visualizes their
receiver operating characteristic (ROC) curves. Each scenario showcases a
teaching point such as model quality, overfitting, threshold sensitivity, and
dataset class balance.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(ax, y_true, scores, title, label="Model", show_baseline=True):
    """
    Helper to compute and plot a ROC curve on a given axis.
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    if show_baseline:
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Chance")

    ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {roc_auc:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    return fpr, tpr, thresholds, roc_auc


def generate_good_model(n_samples=500):
    """
    Strong separation between classes; AUC ~ 0.95.
    """
    negatives = np.clip(np.random.normal(loc=0.15, scale=0.05, size=n_samples // 2), 0, 1)
    positives = np.clip(np.random.normal(loc=0.85, scale=0.07, size=n_samples // 2), 0, 1)
    scores = np.concatenate([negatives, positives])
    labels = np.concatenate([np.zeros_like(negatives), np.ones_like(positives)])
    return labels, scores


def generate_weak_model(n_samples=600):
    """
    Slightly better than random; AUC ~ 0.6.
    """
    negatives = np.clip(np.random.normal(loc=0.45, scale=0.12, size=n_samples // 2), 0, 1)
    positives = np.clip(np.random.normal(loc=0.55, scale=0.12, size=n_samples // 2), 0, 1)
    scores = np.concatenate([negatives, positives])
    labels = np.concatenate([np.zeros_like(negatives), np.ones_like(positives)])
    return labels, scores


def generate_random_model(n_samples=600):
    """
    Scores are independent from labels; AUC ~ 0.5.
    """
    labels = np.random.randint(0, 2, size=n_samples)
    scores = np.random.rand(n_samples)
    return labels, scores


def generate_overfitting_case(train_samples=500, val_samples=300):
    """
    Training data gets near-perfect separation while validation remains weak.
    """
    train_neg = np.clip(np.random.normal(loc=0.05, scale=0.03, size=train_samples // 2), 0, 1)
    train_pos = np.clip(np.random.normal(loc=0.95, scale=0.03, size=train_samples // 2), 0, 1)
    val_neg = np.clip(np.random.normal(loc=0.40, scale=0.12, size=int(val_samples * 0.6)), 0, 1)
    val_pos = np.clip(np.random.normal(loc=0.62, scale=0.12, size=int(val_samples * 0.4)), 0, 1)

    train_scores = np.concatenate([train_neg, train_pos])
    train_labels = np.concatenate([np.zeros_like(train_neg), np.ones_like(train_pos)])

    val_scores = np.concatenate([val_neg, val_pos])
    val_labels = np.concatenate([np.zeros_like(val_neg), np.ones_like(val_pos)])

    return (train_labels, train_scores), (val_labels, val_scores)


def generate_threshold_sensitive_behavior(n_negatives=300, n_positives=220):
    """
    High sensitivity but poor specificity due to many negatives with high scores.
    """
    neg_low = np.clip(np.random.normal(loc=0.25, scale=0.08, size=int(n_negatives * 0.4)), 0, 1)
    neg_high = np.clip(np.random.normal(loc=0.70, scale=0.05, size=int(n_negatives * 0.6)), 0, 1)
    positives = np.clip(np.random.normal(loc=0.80, scale=0.08, size=n_positives), 0, 1)

    negatives = np.concatenate([neg_low, neg_high])
    scores = np.concatenate([negatives, positives])
    labels = np.concatenate([np.zeros_like(negatives), np.ones_like(positives)])
    return labels, scores


def generate_balanced_dataset(n_per_class=250):
    """
    Balanced positives and negatives with reasonably good separation.
    """
    negatives = np.clip(np.random.beta(a=2.2, b=5, size=n_per_class), 0, 1)
    positives = np.clip(np.random.beta(a=5, b=2.2, size=n_per_class), 0, 1)
    scores = np.concatenate([negatives, positives])
    labels = np.concatenate([np.zeros_like(negatives), np.ones_like(positives)])
    return labels, scores


def generate_imbalanced_dataset(n_negatives=900, n_positives=90):
    """
    Heavily imbalanced (pos:neg = 1:10) with modest separation.
    """
    negatives = np.clip(np.random.beta(a=2.5, b=7, size=n_negatives), 0, 1)
    positives = np.clip(np.random.beta(a=4, b=3, size=n_positives), 0, 1)
    scores = np.concatenate([negatives, positives])
    labels = np.concatenate([np.zeros_like(negatives), np.ones_like(positives)])
    return labels, scores


def main():
    np.random.seed(42)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()

    # Scenario 1: Good model
    y_good, scores_good = generate_good_model()
    plot_roc_curve(axes[0], y_good, scores_good, title="Good Model (AUC ≈ 0.95)")

    # Scenario 2: Weak model
    y_weak, scores_weak = generate_weak_model()
    plot_roc_curve(axes[1], y_weak, scores_weak, title="Weak Model (AUC ≈ 0.6)")

    # Scenario 3: Random model
    y_random, scores_random = generate_random_model()
    plot_roc_curve(axes[2], y_random, scores_random, title="Random Model (AUC ≈ 0.5)")

    # Scenario 4: Overfitting case (train vs validation)
    (y_train, scores_train), (y_val, scores_val) = generate_overfitting_case()
    plot_roc_curve(
        axes[3],
        y_train,
        scores_train,
        title="Overfitting: Train vs Validation",
        label="Train",
    )
    plot_roc_curve(
        axes[3],
        y_val,
        scores_val,
        title="Overfitting: Train vs Validation",
        label="Validation",
        show_baseline=False,
    )
    axes[3].legend(loc="lower right")

    # Scenario 5: Threshold-sensitive behavior
    y_thresh, scores_thresh = generate_threshold_sensitive_behavior()
    fpr, tpr, thresholds, _ = plot_roc_curve(
        axes[4],
        y_thresh,
        scores_thresh,
        title="Threshold-Sensitive Model",
    )

    # Highlight an aggressive threshold (prioritizing sensitivity)
    target_threshold = 0.45
    idx = np.argmin(np.abs(thresholds - target_threshold))
    axes[4].scatter(
        fpr[idx],
        tpr[idx],
        color="crimson",
        s=80,
        zorder=5,
        label="Aggressive Threshold",
    )
    axes[4].annotate(
        "High sensitivity\nLow specificity",
        xy=(fpr[idx], tpr[idx]),
        xytext=(fpr[idx] + 0.25, min(tpr[idx] + 0.1, 0.95)),
        arrowprops=dict(arrowstyle="->", color="crimson"),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="crimson", alpha=0.8),
    )
    axes[4].legend(loc="lower right")

    # Scenario 6: Balanced vs imbalanced dataset
    y_bal, scores_bal = generate_balanced_dataset()
    plot_roc_curve(
        axes[5],
        y_bal,
        scores_bal,
        title="Balanced vs Imbalanced Dataset",
        label="Balanced (1:1)",
    )
    y_imb, scores_imb = generate_imbalanced_dataset()
    plot_roc_curve(
        axes[5],
        y_imb,
        scores_imb,
        title="Balanced vs Imbalanced Dataset",
        label="Imbalanced (1:10)",
        show_baseline=False,
    )
    axes[5].annotate(
        "Few positives\n(look beyond ROC!)",
        xy=(0.15, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff4ce", ec="#c47f0e", alpha=0.9),
    )
    axes[5].legend(loc="lower right")

    fig.suptitle("Synthetic ROC Curve Scenarios", fontsize=20, y=0.94)
    fig.subplots_adjust(top=0.90, hspace=0.35, wspace=0.25)
    plt.show()


if __name__ == "__main__":
    main()

