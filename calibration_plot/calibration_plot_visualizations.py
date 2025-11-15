"""
Calibration curve visualizations for multiple synthetic scenarios.

The goal is to demonstrate how calibration diagnostics surface issues such as
overconfidence, underconfidence, data imbalance, and overfitting for safety-
critical ML systems (LLMs, AVs, finance, medical, etc.).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

sns.set_theme(style="whitegrid", context="talk")

RNG = np.random.default_rng(42)


def _generate_labels(probabilities: np.ndarray, bias: float = 0.0, noise: float = 0.02) -> np.ndarray:
    """Sample binary labels given predicted probabilities plus optional bias/noise."""
    actual = np.clip(probabilities + bias + RNG.normal(0, noise, size=probabilities.shape), 0, 1)
    return RNG.binomial(1, actual)


def plot_reliability_curve(
    ax: plt.Axes,
    probs: np.ndarray,
    labels: np.ndarray,
    title: str,
    annotation: str,
    annotation_pos: tuple[float, float],
    color: str = "C0",
    bins: int = 10,
    strategy: str = "uniform",
    extra_curves: list[dict[str, object]] | None = None,
) -> None:
    """Plot a calibration curve with diagonal reference and optional extra curves."""
    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=bins, strategy=strategy)
    ax.plot(mean_pred, frac_pos, marker="o", label="Model", color=color)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    if extra_curves:
        for curve in extra_curves:
            frac, mean = calibration_curve(
                curve["labels"], curve["probs"], n_bins=curve.get("bins", bins), strategy=curve.get("strategy", strategy)
            )
            ax.plot(mean, frac, marker="o", label=curve["label"], color=curve["color"])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Frequency")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.annotate(
        annotation,
        xy=annotation_pos,
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"),
        fontsize=11,
    )


def main() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    axes = axes.ravel()

    # 1. Perfectly calibrated model -------------------------------------------------
    perfect_probs = RNG.beta(2.5, 2.5, size=6000)
    perfect_labels = _generate_labels(perfect_probs, bias=0.0, noise=0.015)
    plot_reliability_curve(
        axes[0],
        perfect_probs,
        perfect_labels,
        title="Perfectly Calibrated Model",
        annotation="Close to diagonal → probabilities reflect reality",
        annotation_pos=(0.05, 0.8),
        color="C2",
    )

    # 2. Overconfident model --------------------------------------------------------
    overconfident_probs = np.clip(RNG.beta(6, 1.8, size=5000) * 0.9 + 0.05, 0, 1)
    overconfident_labels = _generate_labels(overconfident_probs, bias=-0.18, noise=0.02)
    plot_reliability_curve(
        axes[1],
        overconfident_probs,
        overconfident_labels,
        title="Overconfident Model",
        annotation="Curve below diagonal → predicted risk overstated",
        annotation_pos=(0.4, 0.2),
        color="C3",
    )

    # 3. Underconfident model -------------------------------------------------------
    underconfident_probs = np.clip(RNG.beta(1.4, 5, size=5000) * 0.8, 0, 1)
    underconfident_labels = _generate_labels(underconfident_probs, bias=0.22, noise=0.025)
    plot_reliability_curve(
        axes[2],
        underconfident_probs,
        underconfident_labels,
        title="Underconfident Model",
        annotation="Curve above diagonal → probabilities too conservative",
        annotation_pos=(0.35, 0.75),
        color="C0",
    )

    # 4. High-variance / poorly calibrated model ------------------------------------
    hv_probs_components = np.concatenate(
        [
            RNG.beta(0.8, 4, size=300),
            RNG.beta(4, 0.8, size=300),
            RNG.uniform(0.2, 0.8, size=200),
        ]
    )
    hv_probs = np.clip(hv_probs_components + RNG.normal(0, 0.05, size=hv_probs_components.shape), 0, 1)
    hv_labels = _generate_labels(hv_probs, bias=RNG.normal(0, 0.05, size=hv_probs.shape), noise=0.08)
    plot_reliability_curve(
        axes[3],
        hv_probs,
        hv_labels,
        title="High-Variance / Noisy Calibration",
        annotation="Small data + noise → wiggly reliability curve",
        annotation_pos=(0.05, 0.2),
        color="C4",
        bins=8,
    )

    # 5. Overfitting case (Train vs. Validation) ------------------------------------
    train_probs = np.clip(RNG.beta(3.5, 2, size=4000), 0, 1)
    train_labels = _generate_labels(train_probs, bias=0.02, noise=0.015)
    val_probs = np.clip(train_probs + RNG.normal(0, 0.15, size=train_probs.shape), 0, 1)
    val_labels = _generate_labels(val_probs, bias=-0.15, noise=0.04)
    plot_reliability_curve(
        axes[4],
        train_probs,
        train_labels,
        title="Overfitting: Train vs Validation",
        annotation="Validation drifts below diagonal → overfitting risk",
        annotation_pos=(0.35, 0.2),
        color="C2",
        extra_curves=[
            {"probs": val_probs, "labels": val_labels, "label": "Validation", "color": "C1"},
        ],
    )

    # 6. Severe class imbalance -----------------------------------------------------
    imbalance_probs = np.concatenate(
        [
            RNG.beta(0.4, 8, size=4500),
            RNG.beta(2, 6, size=400),
            RNG.beta(4, 1.2, size=100),
        ]
    )
    imbalance_probs = np.clip(imbalance_probs, 0, 1)
    base_rate = 0.05
    imbalance_actual = np.clip(
        base_rate + 0.9 * imbalance_probs + RNG.normal(0, 0.02, size=imbalance_probs.shape), 0, 1
    )
    imbalance_labels = RNG.binomial(1, imbalance_actual)
    plot_reliability_curve(
        axes[5],
        imbalance_probs,
        imbalance_labels,
        title="Severe Class Imbalance",
        annotation="Most bins low-probability → calibration exposes scarcity",
        annotation_pos=(0.02, 0.25),
        color="C5",
        bins=12,
        strategy="quantile",
    )

    fig.suptitle("Calibration Curve Scenarios for Trustworthy ML", fontsize=22, y=0.98)
    fig.tight_layout(pad=3.0, w_pad=2.5, h_pad=3.0)

    output_path = Path(__file__).with_suffix(".png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Calibration plot figure saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()

