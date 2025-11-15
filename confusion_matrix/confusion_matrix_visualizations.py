"""
Generate synthetic confusion matrices that illustrate common model behaviors.

The visualizations highlight how ML teams interpret confusion matrices in
domains such as vision classification, intent detection, fraud detection, and
medical imaging. Each scenario captures a distinct diagnostic pattern—from
healthy performance to severe pathology—and shows what to look for.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_confusion_matrices():
    """
    Build confusion matrices for a variety of storytelling scenarios.

    Returns
    -------
    classes : list[str]
        The class labels shared by every scenario.
    scenarios : list[dict]
        A list of dictionaries containing the matrix, title, annotation, and
        where to place that annotation for plotting.
    """
    classes = ["A", "B", "C", "D"]

    scenarios = []

    # === Scenario 1: Balanced classification with good performance ===
    balanced_performance = np.array(
        [
            [85, 5, 4, 6],
            [6, 88, 3, 3],
            [4, 2, 90, 4],
            [3, 4, 5, 88],
        ],
        dtype=int,
    )
    scenarios.append(
        {
            "title": "Balanced Performance",
            "matrix": balanced_performance,
            "annotation_text": "Strong diagonal → confident, clean separation",
        }
    )

    # === Scenario 2: Class imbalance ===
    class_imbalance = np.array(
        [
            [480, 15, 5, 0],
            [160, 25, 8, 7],
            [140, 18, 25, 12],
            [120, 12, 8, 10],
        ],
        dtype=int,
    )
    scenarios.append(
        {
            "title": "Class Imbalance",
            "matrix": class_imbalance,
            "annotation_text": "Class A dominates → minority recalls collapse",
        }
    )

    # === Scenario 3: Confused neighboring classes ===
    confused_neighbors = np.array(
        [
            [70, 8, 10, 12],
            [10, 40, 45, 5],
            [8, 42, 38, 12],
            [9, 6, 10, 75],
        ],
        dtype=int,
    )
    scenarios.append(
        {
            "title": "Neighbor Confusion",
            "matrix": confused_neighbors,
            "annotation_text": "High confusion: B ↔ C → investigate shared features",
        }
    )

    # === Scenario 4: Severe overfitting / memorization ===
    overfit_train = np.array(
        [
            [97, 1, 1, 1],
            [0, 98, 1, 1],
            [1, 0, 98, 1],
            [0, 1, 0, 99],
        ],
        dtype=int,
    )
    scenarios.append(
        {
            "title": "Overfitting (Train)",
            "matrix": overfit_train,
            "annotation_text": "Near-perfect diagonal → memorization warning",
        }
    )

    overfit_validation = np.array(
        [
            [40, 25, 20, 15],
            [22, 35, 28, 15],
            [25, 30, 32, 13],
            [21, 18, 26, 35],
        ],
        dtype=int,
    )
    scenarios.append(
        {
            "title": "Overfitting (Validation)",
            "matrix": overfit_validation,
            "annotation_text": "Validation scatter → model fails to generalize",
        }
    )

    # === Scenario 5: Mislabeled data in training ===
    mislabeled_training = np.array(
        [
            [82, 10, 4, 4],
            [12, 70, 10, 8],
            [5, 88, 2, 5],
            [6, 15, 10, 69],
        ],
        dtype=int,
    )
    scenarios.append(
        {
            "title": "Mislabeled Training Data",
            "matrix": mislabeled_training,
            "annotation_text": "Class C → B funnel → audit upstream labels",
        }
    )

    # === Scenario 6: Underfitting ===
    underfitting = np.array(
        [
            [30, 28, 20, 22],
            [26, 25, 24, 25],
            [23, 26, 27, 24],
            [24, 22, 28, 26],
        ],
        dtype=int,
    )
    scenarios.append(
        {
            "title": "Underfitting",
            "matrix": underfitting,
            "annotation_text": "Counts flat → model guessing randomly",
        }
    )

    return classes, scenarios


def plot_confusion_matrices():
    """Render all configured confusion matrices with seaborn heatmaps."""
    sns.set_theme(style="whitegrid", font_scale=1.0)

    classes, scenarios = create_confusion_matrices()

    fig, axes = plt.subplots(3, 2, figsize=(14, 20))
    axes = axes.flatten()

    for ax, scenario in zip(axes, scenarios):
        matrix = scenario["matrix"]
        heatmap = sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            square=True,
            xticklabels=classes,
            yticklabels=classes,
            linewidths=0.5,
            linecolor="gray",
            cbar_kws={"shrink": 0.75},
            ax=ax,
        )
        heatmap.set_xlabel("Predicted")
        heatmap.set_ylabel("Actual")
        heatmap.set_title(scenario["title"])

        ax.text(
            0.5,
            -0.38,
            scenario["annotation_text"],
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85),
        )

    fig.subplots_adjust(top=0.95, bottom=0.12, hspace=0.85, wspace=0.4, left=0.08, right=0.95)

    output_path = Path(__file__).with_suffix(".png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Confusion matrix figure saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    plot_confusion_matrices()

