"""
Generate illustrative loss curve scenarios for deep learning trainings.

Each subplot highlights a common training outcome along with practical
annotations to help newcomers interpret what the curves suggest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Scenario:
    """Container for a single loss curve scenario."""

    name: str
    generator: Callable[[np.ndarray, np.random.Generator], tuple[np.ndarray, np.ndarray]]
    annotation: str


def generate_scenarios() -> list[Scenario]:
    """Define the collection of training scenarios to visualise."""
    return [
        Scenario(
            name="Ideal Training",
            generator=_ideal_training,
            annotation="Balanced decrease. Both curves going ↓ together → well-fitting model.",
        ),
        Scenario(
            name="Overfitting",
            generator=_overfitting,
            annotation="Val loss rising → Overfitting. Watch for early stop/regularisation.",
        ),
        Scenario(
            name="Underfitting",
            generator=_underfitting,
            annotation="Both losses high → Model too simple or under-trained.",
        ),
        Scenario(
            name="High Learning Rate / Noisy Training",
            generator=_high_lr_noisy,
            annotation="Loss zig-zags → Lower learning rate or add scheduling.",
        ),
        Scenario(
            name="Too Low Learning Rate",
            generator=_too_low_lr,
            annotation="Slow drift ↓ → Increase learning rate or train longer.",
        ),
        Scenario(
            name="Data Leakage",
            generator=_data_leakage,
            annotation="Val loss unrealistically low → Investigate leakage.",
        ),
        Scenario(
            name="Optimization Failure / Plateau",
            generator=_plateau,
            annotation="Loss stalls → Revisit optimiser, learning rate, or init.",
        ),
    ]


def _ideal_training(epochs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    train = 0.9 * np.exp(-epochs / 18) + 0.02 * rng.normal(size=epochs.size)
    val = 1.05 * np.exp(-epochs / 20) + 0.03 * rng.normal(size=epochs.size) + 0.02
    return np.maximum(train, 0.02), np.maximum(val, 0.03)


def _overfitting(epochs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    decay = np.exp(-epochs / 20)
    train = 0.8 * decay + 0.015 * rng.normal(size=epochs.size)
    val = 1.0 * decay + 0.03 * rng.normal(size=epochs.size)
    tipping_point = epochs.size // 2
    val[tipping_point:] += 0.02 * np.arange(1, epochs.size - tipping_point + 1)
    return np.maximum(train, 0.02), np.maximum(val, 0.05)


def _underfitting(epochs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    base = 1.4 - 0.1 * np.log1p(epochs)
    train = base + 0.05 * rng.normal(size=epochs.size)
    val = base + 0.06 * rng.normal(size=epochs.size) + 0.05
    return np.maximum(train, 0.5), np.maximum(val, 0.6)


def _high_lr_noisy(epochs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    trend = np.exp(-epochs / 15)
    noise = 0.12 * rng.normal(size=epochs.size)
    train = 0.9 * trend + noise
    val = 1.1 * trend + 0.15 * rng.normal(size=epochs.size) + 0.05
    return np.maximum(train, 0.03), np.maximum(val, 0.04)


def _too_low_lr(epochs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    train = 1.0 - 0.002 * epochs + 0.01 * rng.normal(size=epochs.size)
    val = 1.1 - 0.0015 * epochs + 0.012 * rng.normal(size=epochs.size)
    return np.maximum(train, 0.7), np.maximum(val, 0.8)


def _data_leakage(epochs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    train = 0.85 * np.exp(-epochs / 16) + 0.02 * rng.normal(size=epochs.size)
    val = train * 0.6 + 0.01 * rng.normal(size=epochs.size)
    return np.maximum(train, 0.02), np.maximum(val, 0.01)


def _plateau(epochs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    train = np.linspace(1.2, 0.7, epochs.size)
    plateau_start = epochs.size // 3
    train[plateau_start:] = 0.65 + 0.015 * rng.normal(size=epochs.size - plateau_start)
    val = train + 0.08 + 0.02 * rng.normal(size=epochs.size)
    return np.maximum(train, 0.6), np.maximum(val, 0.65)


def plot_loss_curves(num_epochs: int = 50) -> None:
    """Generate synthetic curves and render subplot visualisations."""
    rng = np.random.default_rng(seed=42)
    epochs = np.arange(1, num_epochs + 1)
    scenarios = generate_scenarios()

    n_rows, n_cols = 4, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 16), sharex=True)
    axes = axes.flatten()

    for ax, scenario in zip(axes, scenarios, strict=False):
        train_loss, val_loss = scenario.generator(epochs, rng)

        ax.plot(epochs, train_loss, label="Train Loss", color="tab:blue", linewidth=2)
        ax.plot(epochs, val_loss, label="Validation Loss", color="tab:orange", linewidth=2, linestyle="--")

        ax.set_title(scenario.name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        ax.text(
            0.03,
            0.95,
            scenario.annotation,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.3"),
        )

    # Hide any unused subplot axes
    for ax in axes[len(scenarios) :]:
        ax.axis("off")

    fig.tight_layout(pad=2.5, w_pad=2.0, h_pad=3.0)
    plt.show()


def main() -> None:
    """Entrypoint when executing as a script."""
    plot_loss_curves()


if __name__ == "__main__":
    main()

