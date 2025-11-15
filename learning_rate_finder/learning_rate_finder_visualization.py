"""
Learning Rate Finder visualization with synthetic data.

This script mimics the FastAI / PyTorch LR Finder workflow by sweeping
learning rates on a log scale, tracking synthetic loss values, and
annotating key regions practitioners inspect when selecting a learning rate.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RNG_SEED = 42
NUM_POINTS = 250
LR_MIN = 1e-6
LR_MAX = 1e0

np.random.seed(RNG_SEED)
LR_VALUES = np.logspace(np.log10(LR_MIN), np.log10(LR_MAX), NUM_POINTS)
OUTPUT_PATH = Path(__file__).with_name("learning_rate_finder_visualizations.png")


# -----------------------------------------------------------------------------
# Synthetic loss generation
# -----------------------------------------------------------------------------
def synthetic_loss_curve(
    lr_values: np.ndarray,
    optimal_lr: float,
    base_loss: float = 3.0,
    descent_depth: float = 1.6,
    curve_smoothness: float = 0.8,
    divergence_scale: float = 14.0,
    divergence_power: float = 2.3,
    noise_scale: float = 0.02,
) -> np.ndarray:
    """
    Create a synthetic loss curve that decreases before the optimal learning rate
    and diverges afterward, with optional noise for realism.
    """
    log_lr = np.log10(lr_values)
    log_opt = np.log10(optimal_lr)
    log_min = log_lr.min()
    log_max = log_lr.max()

    loss = np.empty_like(lr_values)
    before_mask = log_lr <= log_opt
    after_mask = ~before_mask

    # Smooth descent region (too-low to optimal LR)
    normalized_before = np.clip(
        (log_lr[before_mask] - log_min) / (log_opt - log_min + 1e-12), 0.0, 1.0
    )
    loss_before = base_loss - descent_depth * normalized_before**curve_smoothness
    loss[before_mask] = loss_before

    # Divergence beyond optimal LR
    normalized_after = np.clip(
        (log_lr[after_mask] - log_opt) / (log_max - log_opt + 1e-12), 0.0, 1.0
    )
    loss_after = (base_loss - descent_depth) + divergence_scale * (
        normalized_after**divergence_power
    )
    loss[after_mask] = loss_after

    noise = np.random.normal(0.0, noise_scale * base_loss, size=lr_values.size)
    return loss + noise


# -----------------------------------------------------------------------------
# Scenario definitions
# -----------------------------------------------------------------------------
SCENARIOS = {
    "Well-behaved model": {
        "optimal_lr": 2e-3,
        "descent_depth": 1.7,
        "noise_scale": 0.015,
        "curve_smoothness": 0.85,
    },
    "Noisy batches": {
        "optimal_lr": 1e-3,
        "descent_depth": 1.4,
        "noise_scale": 0.05,
        "divergence_scale": 11.0,
    },
    "Stable / wider optimum": {
        "optimal_lr": 7e-3,
        "descent_depth": 1.2,
        "curve_smoothness": 1.1,
        "divergence_power": 1.7,
        "noise_scale": 0.01,
    },
}


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def plot_lr_finder_curves() -> None:
    plt.figure(figsize=(10, 6))

    curves = {}
    for label, params in SCENARIOS.items():
        loss_curve = synthetic_loss_curve(LR_VALUES, **params)
        curves[label] = loss_curve
        plt.plot(LR_VALUES, loss_curve, label=label, linewidth=2)

    # Highlight optimal LR region for the "Well-behaved model"
    optimal_lr = SCENARIOS["Well-behaved model"]["optimal_lr"]
    optimal_region = (optimal_lr / 3, optimal_lr * 3)

    for boundary, text in zip(
        optimal_region,
        ("Optimal start", "Optimal end"),
    ):
        plt.axvline(boundary, color="gray", linestyle="--", alpha=0.5)
        plt.text(
            boundary,
            plt.gca().get_ylim()[0] * 0.98 + plt.gca().get_ylim()[1] * 0.02,
            text,
            rotation=90,
            verticalalignment="bottom",
            horizontalalignment="right" if boundary > optimal_lr else "left",
            fontsize=9,
            color="gray",
        )

    # Annotations for practitioner insights
    well_behaved_curve = curves["Well-behaved model"]
    min_loss_idx = np.argmin(well_behaved_curve)
    min_loss_lr = LR_VALUES[min_loss_idx]
    min_loss = well_behaved_curve[min_loss_idx]

    plt.annotate(
        "Too Low LR → Slow training",
        xy=(LR_VALUES[15], well_behaved_curve[15]),
        xytext=(LR_VALUES[5], min_loss + 0.8),
        arrowprops=dict(arrowstyle="->", color="tab:blue"),
        fontsize=10,
        color="tab:blue",
    )

    plt.annotate(
        "Optimal LR region → Steepest loss decline",
        xy=(min_loss_lr, min_loss),
        xytext=(min_loss_lr * 0.2, min_loss + 0.4),
        arrowprops=dict(arrowstyle="->", color="tab:green"),
        fontsize=10,
        color="tab:green",
    )

    plt.annotate(
        "Divergence → LR too high",
        xy=(LR_VALUES[-10], well_behaved_curve[-10]),
        xytext=(LR_VALUES[-30], min_loss + 1.2),
        arrowprops=dict(arrowstyle="->", color="tab:red"),
        fontsize=10,
        color="tab:red",
    )

    plt.xscale("log")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_lr_finder_curves()

