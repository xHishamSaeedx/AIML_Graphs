"""
Generate illustrative metric curve scenarios to explain model behaviour across
vision, NLP, and speech tasks.

Each subplot highlights a different metric trend along with a short annotation
that teaches what to look for during training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class MetricScenario:
    """Container describing a metric trajectory scenario."""

    name: str
    metric_label: str
    generator: Callable[[np.ndarray, np.random.Generator], tuple[np.ndarray, np.ndarray]]
    annotation: str


def generate_metric_scenarios() -> list[MetricScenario]:
    """Create the list of metric curve scenarios."""
    return [
        MetricScenario(
            name="Accuracy Improving Normally",
            metric_label="Accuracy",
            generator=_accuracy_improving,
            annotation="Both curves ↑ steadily → Healthy learning and generalisation.",
        ),
        MetricScenario(
            name="Accuracy Overfitting",
            metric_label="Accuracy",
            generator=_accuracy_overfitting,
            annotation="Val accuracy stalls/drops → Consider regularisation or early stop.",
        ),
        MetricScenario(
            name="F1 Score Improving Smoothly",
            metric_label="F1 Score",
            generator=_f1_improving,
            annotation="Smooth ↑ F1 → Model balances precision & recall over time.",
        ),
        MetricScenario(
            name="BLEU Jump Then Stabilise",
            metric_label="BLEU Score",
            generator=_bleu_jump_stabilise,
            annotation="Early jump then plateau → Typical MT behaviour; watch plateau.",
        ),
        MetricScenario(
            name="ROUGE-L Slow Growth",
            metric_label="ROUGE-L",
            generator=_rouge_slow_growth,
            annotation="Gradual ↑ → Summaries improve slowly; training may need patience.",
        ),
        MetricScenario(
            name="WER Decreasing Normally",
            metric_label="Word Error Rate (WER)",
            generator=_wer_decreasing,
            annotation="WER ↓ is better → Both curves trending down shows progress.",
        ),
        MetricScenario(
            name="WER Overfitting",
            metric_label="Word Error Rate (WER)",
            generator=_wer_overfitting,
            annotation="Val WER stops ↓ → Speech model begins overfitting.",
        ),
        MetricScenario(
            name="Metric Saturation / Plateau",
            metric_label="Generic Metric",
            generator=_metric_plateau,
            annotation="Curves flatten → Optimiser stuck; adjust LR or architecture.",
        ),
    ]


def _clip_metric(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(values, lower, upper)


def _accuracy_improving(epochs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    train = 0.55 + 0.4 * (1 - np.exp(-epochs / 12)) + 0.01 * rng.normal(size=epochs.size)
    val = 0.5 + 0.35 * (1 - np.exp(-epochs / 15)) + 0.015 * rng.normal(size=epochs.size)
    return _clip_metric(train, 0.5, 0.99), _clip_metric(val, 0.45, 0.95)


def _accuracy_overfitting(epochs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    train = 0.6 + 0.38 * (1 - np.exp(-epochs / 10)) + 0.01 * rng.normal(size=epochs.size)
    val = 0.55 + 0.25 * (1 - np.exp(-epochs / 18)) + 0.02 * rng.normal(size=epochs.size)
    tipping = epochs.size // 2
    val[tipping:] -= 0.02 * np.linspace(0, 1, epochs.size - tipping)
    return _clip_metric(train, 0.6, 0.995), _clip_metric(val, 0.45, 0.92)


def _f1_improving(epochs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    train = 0.4 + 0.45 * (1 - np.exp(-epochs / 16)) + 0.012 * rng.normal(size=epochs.size)
    val = 0.38 + 0.4 * (1 - np.exp(-epochs / 20)) + 0.015 * rng.normal(size=epochs.size)
    return _clip_metric(train, 0.35, 0.9), _clip_metric(val, 0.3, 0.88)


def _bleu_jump_stabilise(epochs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    base = 0.2 + 0.25 * (1 - np.exp(-epochs / 6))
    train = base + 0.07 * np.exp(-epochs / 25) + 0.01 * rng.normal(size=epochs.size)
    val = base + 0.05 * np.exp(-epochs / 18) + 0.012 * rng.normal(size=epochs.size)
    return _clip_metric(train, 0.2, 0.65), _clip_metric(val, 0.18, 0.6)


def _rouge_slow_growth(epochs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    train = 0.25 + 0.3 * np.log1p(epochs) / np.log1p(epochs[-1]) + 0.012 * rng.normal(size=epochs.size)
    val = 0.22 + 0.28 * np.log1p(epochs) / np.log1p(epochs[-1]) + 0.015 * rng.normal(size=epochs.size)
    return _clip_metric(train, 0.2, 0.65), _clip_metric(val, 0.18, 0.6)


def _wer_decreasing(epochs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    train = 0.35 - 0.2 * (1 - np.exp(-epochs / 14)) + 0.01 * rng.normal(size=epochs.size)
    val = 0.4 - 0.16 * (1 - np.exp(-epochs / 18)) + 0.012 * rng.normal(size=epochs.size)
    return _clip_metric(train, 0.08, 0.4), _clip_metric(val, 0.12, 0.45)


def _wer_overfitting(epochs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    train = 0.4 - 0.22 * (1 - np.exp(-epochs / 12)) + 0.01 * rng.normal(size=epochs.size)
    val = 0.42 - 0.15 * (1 - np.exp(-epochs / 20)) + 0.015 * rng.normal(size=epochs.size)
    val[epochs > epochs.size / 2] += 0.015 * np.linspace(0, 1, np.sum(epochs > epochs.size / 2))
    return _clip_metric(train, 0.07, 0.4), _clip_metric(val, 0.1, 0.45)


def _metric_plateau(epochs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    base = 0.55 + 0.2 * (1 - np.exp(-epochs / 18))
    train = base + 0.01 * rng.normal(size=epochs.size)
    val = base - 0.03 + 0.012 * rng.normal(size=epochs.size)
    plateau_point = epochs.size // 3
    train[plateau_point:] = train[plateau_point] + 0.005 * rng.normal(size=epochs.size - plateau_point)
    val[plateau_point:] = val[plateau_point] + 0.006 * rng.normal(size=epochs.size - plateau_point)
    return _clip_metric(train, 0.5, 0.78), _clip_metric(val, 0.45, 0.75)


def plot_metric_curves(num_epochs: int = 50) -> None:
    """Generate synthetic metric curves and render subplot visualisations."""
    rng = np.random.default_rng(seed=123)
    epochs = np.arange(1, num_epochs + 1)
    scenarios = generate_metric_scenarios()

    n_rows, n_cols = 4, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 16), sharex=True)
    axes = axes.flatten()

    for ax, scenario in zip(axes, scenarios, strict=False):
        train_metric, val_metric = scenario.generator(epochs, rng)

        ax.plot(epochs, train_metric, label="Train", color="tab:blue", linewidth=2)
        ax.plot(epochs, val_metric, label="Validation", color="tab:orange", linewidth=2, linestyle="--")

        ax.set_title(scenario.name)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(scenario.metric_label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        ax.text(
            0.03,
            0.92,
            scenario.annotation,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.3"),
        )

    for ax in axes[len(scenarios) :]:
        ax.axis("off")

    fig.tight_layout(pad=2.5, w_pad=2.0, h_pad=3.0)
    plt.show()


def main() -> None:
    """Entrypoint when executing as a script."""
    plot_metric_curves()


if __name__ == "__main__":
    main()

