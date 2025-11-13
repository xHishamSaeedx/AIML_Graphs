"""
Precision-Recall curve visualizations for a variety of synthetic model scenarios.

The goal of this script is to illustrate how practitioners can interpret PR curves
for common industry use-cases such as spam detection, fraud detection, search ranking,
recommender systems, and anomaly detection. Each scenario uses synthetic data to
highlight a specific behaviour or pitfall.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def make_dataset(n_samples, positive_rate, pos_dist, neg_dist, pos_noise=0.02, neg_noise=0.02):
    """
    Generate synthetic probabilities and labels for a binary classification scenario.

    Parameters
    ----------
    n_samples : int
        Number of samples to simulate.
    positive_rate : float
        Proportion of the positive class within the dataset.
    pos_dist : callable
        Callable that accepts `size` and returns raw scores for positive samples.
    neg_dist : callable
        Callable that accepts `size` and returns raw scores for negative samples.
    pos_noise : float
        Standard deviation of Gaussian noise added to positive scores.
    neg_noise : float
        Standard deviation of Gaussian noise added to negative scores.
    """
    n_pos = max(1, int(n_samples * positive_rate))
    n_neg = max(1, n_samples - n_pos)

    pos_scores = np.clip(pos_dist(size=n_pos) + np.random.normal(0, pos_noise, size=n_pos), 0, 1)
    neg_scores = np.clip(neg_dist(size=n_neg) + np.random.normal(0, neg_noise, size=n_neg), 0, 1)

    labels = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    scores = np.concatenate([pos_scores, neg_scores])

    # Shuffle to break any ordering bias.
    shuffle_idx = np.random.permutation(n_samples)
    return labels[shuffle_idx], scores[shuffle_idx]


def plot_pr_curve(ax, labels, scores, title, annotation, annotation_xy=(0.45, 0.25), legend_loc="lower left"):
    """Plot a single PR curve with baseline and annotation."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    baseline = labels.mean()

    ax.plot(recall, precision, label="Model", color="#1f77b4")
    ax.axhline(
        baseline,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Baseline ({baseline:.2f})",
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(loc=legend_loc)
    ax.annotate(annotation, xy=annotation_xy, xycoords="axes fraction", fontsize=9, bbox=dict(
        boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8
    ))


def plot_overfitting(ax, datasets, title, annotation, annotation_xy=(0.45, 0.25), legend_loc="lower left"):
    """Plot train vs validation PR curves to highlight overfitting."""
    baselines = []
    colors = {"Train": "#2ca02c", "Validation": "#d62728"}

    for name, (labels, scores) in datasets.items():
        precision, recall, _ = precision_recall_curve(labels, scores)
        ax.plot(recall, precision, label=f"{name} model", color=colors.get(name, None))
        baselines.append(labels.mean())

    baseline = float(np.mean(baselines))
    ax.axhline(
        baseline,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Baseline ({baseline:.2f})",
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
    ax.legend(loc=legend_loc)
    ax.annotate(annotation, xy=annotation_xy, xycoords="axes fraction", fontsize=9, bbox=dict(
        boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8
    ))


def main():
    np.random.seed(42)

    scenarios = []

    # 1. Good model on imbalanced dataset.
    labels, scores = make_dataset(
        n_samples=1500,
        positive_rate=0.15,
        pos_dist=lambda size: np.random.beta(10, 2, size=size),
        neg_dist=lambda size: np.random.beta(2, 10, size=size),
    )
    scenarios.append(dict(
        type="single",
        title="Good Model on Imbalanced Data",
        labels=labels,
        scores=scores,
        annotation="PR stays high → confident positives",
        annotation_xy=(0.45, 0.8),
        legend_loc="lower left",
    ))

    # 2. Weak model on imbalanced dataset.
    labels, scores = make_dataset(
        n_samples=1500,
        positive_rate=0.15,
        pos_dist=lambda size: np.random.beta(4, 4, size=size),
        neg_dist=lambda size: np.random.beta(3, 5, size=size),
        pos_noise=0.03,
        neg_noise=0.03,
    )
    scenarios.append(dict(
        type="single",
        title="Weak Model on Imbalanced Data",
        labels=labels,
        scores=scores,
        annotation="Precision drops → reassess model fit",
        annotation_xy=(0.4, 0.35),
        legend_loc="upper right",
    ))

    # 3. High precision, low recall model (conservative classifier).
    def conservative_positive(size):
        high_conf = int(size * 0.3)
        low_conf = size - high_conf
        high_scores = np.random.beta(12, 1.5, size=high_conf)
        low_scores = np.random.beta(2, 8, size=low_conf)
        return np.concatenate([high_scores, low_scores])

    labels, scores = make_dataset(
        n_samples=1500,
        positive_rate=0.15,
        pos_dist=conservative_positive,
        neg_dist=lambda size: np.random.beta(1.5, 8, size=size),
        pos_noise=0.015,
        neg_noise=0.02,
    )
    scenarios.append(dict(
        type="single",
        title="High Precision, Low Recall",
        labels=labels,
        scores=scores,
        annotation="High P but low R → conservative classifier",
        annotation_xy=(0.35, 0.7),
        legend_loc="lower left",
    ))

    # 4. High recall, low precision model (aggressive classifier).
    labels, scores = make_dataset(
        n_samples=1500,
        positive_rate=0.15,
        pos_dist=lambda size: np.random.beta(4, 3, size=size),
        neg_dist=lambda size: np.random.beta(1.2, 2.2, size=size),
        pos_noise=0.025,
        neg_noise=0.035,
    )
    scenarios.append(dict(
        type="single",
        title="High Recall, Low Precision",
        labels=labels,
        scores=scores,
        annotation="Aggressive model → many false alarms",
        annotation_xy=(0.55, 0.3),
        legend_loc="upper right",
    ))

    # 5. Severe class imbalance (1% positives).
    labels, scores = make_dataset(
        n_samples=5000,
        positive_rate=0.01,
        pos_dist=lambda size: np.random.beta(8, 2, size=size),
        neg_dist=lambda size: np.random.beta(1.5, 25, size=size),
        pos_noise=0.01,
        neg_noise=0.01,
    )
    scenarios.append(dict(
        type="single",
        title="Severe Class Imbalance (1% Positives)",
        labels=labels,
        scores=scores,
        annotation="Low baseline → PR beats ROC here",
        annotation_xy=(0.35, 0.55),
        legend_loc="upper right",
    ))

    # 6. Overfitting case (train vs validation).
    train_labels, train_scores = make_dataset(
        n_samples=1500,
        positive_rate=0.10,
        pos_dist=lambda size: np.random.beta(9, 2, size=size),
        neg_dist=lambda size: np.random.beta(2, 9, size=size),
        pos_noise=0.015,
        neg_noise=0.02,
    )
    val_labels, val_scores = make_dataset(
        n_samples=1500,
        positive_rate=0.10,
        pos_dist=lambda size: np.random.beta(4, 3, size=size),
        neg_dist=lambda size: np.random.beta(3, 4, size=size),
        pos_noise=0.03,
        neg_noise=0.04,
    )
    scenarios.append(dict(
        type="overfit",
        title="Overfitting: Train vs Validation",
        datasets={
            "Train": (train_labels, train_scores),
            "Validation": (val_labels, val_scores),
        },
        annotation="Gap signals overfitting risk",
        annotation_xy=(0.45, 0.5),
        legend_loc="lower left",
    ))

    fig, axes = plt.subplots(3, 2, figsize=(14, 16), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        if scenario["type"] == "single":
            plot_pr_curve(
                ax=ax,
                labels=scenario["labels"],
                scores=scenario["scores"],
                title=scenario["title"],
                annotation=scenario["annotation"],
                annotation_xy=scenario["annotation_xy"],
                legend_loc=scenario["legend_loc"],
            )
        else:
            plot_overfitting(
                ax=ax,
                datasets=scenario["datasets"],
                title=scenario["title"],
                annotation=scenario["annotation"],
                annotation_xy=scenario["annotation_xy"],
                legend_loc=scenario["legend_loc"],
            )

    fig.tight_layout(pad=4.0, h_pad=3.0, w_pad=2.5)
    plt.show()


if __name__ == "__main__":
    main()

