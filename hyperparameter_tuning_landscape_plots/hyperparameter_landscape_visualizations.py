"""
Synthetic hyperparameter landscape visualizations for educational use.

The script fabricates smooth-but-noisy performance surfaces across several
hyperparameters and visualizes them in the style of common tuning dashboards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib import pyplot as plt
from pandas.plotting import parallel_coordinates


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="talk")
RNG = np.random.default_rng(42)
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def smooth_peak(values: np.ndarray, center: float, width: float) -> np.ndarray:
    """Return a smooth bell-shaped response around a center."""
    return np.exp(-((values - center) ** 2) / (2 * width**2))


def simulate_lr_batch_performance() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a performance surface across learning rate and batch size."""
    lr_values = np.logspace(-5, -1, 40)
    batch_sizes = np.logspace(np.log10(16), np.log10(512), 40)
    lr_grid, batch_grid = np.meshgrid(lr_values, batch_sizes)

    log_lr = np.log10(lr_grid)
    log_batch = np.log2(batch_grid)

    lr_peak = smooth_peak(log_lr, -3, 0.35)
    batch_peak = smooth_peak(log_batch, 5.5, 1.0)
    ridge = np.exp(-((log_lr + log_batch / 8 + 0.45) ** 2) / (2 * 0.45**2))
    instability = smooth_peak(log_lr, -1.2, 0.25) + smooth_peak(log_batch, 3.5, 0.6)

    performance = (
        0.45
        + 0.35 * lr_peak
        + 0.2 * batch_peak
        + 0.15 * ridge
        - 0.18 * instability
        + RNG.normal(0, 0.015, size=lr_grid.shape)
    )
    performance = np.clip(performance, 0.2, 0.95)
    return lr_values, batch_sizes, performance


def simulate_lr_weight_decay_performance() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performance surface for learning rate vs. weight decay."""
    lr_values = np.logspace(-5, -1, 60)
    wd_values = np.logspace(-6, -2, 60)
    lr_grid, wd_grid = np.meshgrid(lr_values, wd_values)

    log_lr = np.log10(lr_grid)
    log_wd = np.log10(wd_grid)

    lr_u_shape = np.exp(-((log_lr + 3) ** 2) / (2 * 0.4**2))
    wd_monotonic = smooth_peak(log_wd, -4.5, 0.9)
    interaction = np.exp(-((log_lr + log_wd + 7.5) ** 2) / (2 * 0.55**2))

    performance = (
        0.4
        + 0.4 * lr_u_shape
        + 0.25 * wd_monotonic
        + 0.15 * interaction
        - 0.1 * smooth_peak(log_wd, -2.5, 0.4)
        + RNG.normal(0, 0.012, size=lr_grid.shape)
    )
    performance = np.clip(performance, 0.25, 0.97)
    return lr_values, wd_values, performance


def simulate_depth_width_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D surface for architecture hyperparameters."""
    depths = np.linspace(2, 24, 32)
    widths = np.linspace(64, 1024, 32)
    depth_grid, width_grid = np.meshgrid(depths, widths)

    normalized_depth = (depth_grid - 10) / 6
    normalized_width = (np.log2(width_grid) - 7) / 1.5

    valley = np.exp(-((normalized_depth) ** 2 + (normalized_width) ** 2))
    overfit_penalty = 0.4 * (normalized_depth**4 + normalized_width**4)
    ripple = 0.08 * np.sin(depth_grid / 3) * np.cos(np.log2(width_grid) / 1.4)

    performance = 0.78 + 0.15 * valley - overfit_penalty + ripple
    performance += RNG.normal(0, 0.01, size=performance.shape)
    performance = np.clip(performance, 0.1, 0.9)
    return depths, widths, performance


def sample_trial_dataframe(n_trials: int = 150) -> pd.DataFrame:
    """Generate a synthetic hyperparameter trial dataset."""
    learning_rate = 10 ** RNG.uniform(-5, -1, n_trials)
    batch_size = 2 ** RNG.uniform(4, 9, n_trials)
    depth = RNG.integers(2, 26, n_trials)
    width = 2 ** RNG.integers(6, 11, n_trials)
    weight_decay = 10 ** RNG.uniform(-6, -2, n_trials)

    log_lr = np.log10(learning_rate)
    log_bs = np.log2(batch_size)
    log_wd = np.log10(weight_decay)

    lr_score = np.exp(-((log_lr + 3) ** 2) / (2 * 0.35**2))
    bs_score = np.exp(-((log_bs - 5.5) ** 2) / (2 * 1.2**2))
    wd_score = np.exp(-((log_wd + 4.5) ** 2) / (2 * 0.8**2))
    depth_score = np.exp(-((depth - 10) ** 2) / (2 * 4.5**2))
    width_score = np.exp(-((np.log2(width) - 7) ** 2) / (2 * 1.3**2))

    performance = (
        0.45
        + 0.25 * lr_score
        + 0.15 * bs_score
        + 0.1 * depth_score
        + 0.08 * width_score
        + 0.12 * wd_score
        - 0.05 * ((depth > 18) & (width > 512))
        + RNG.normal(0, 0.02, size=n_trials)
    )
    performance = np.clip(performance, 0.25, 0.95)

    df = pd.DataFrame(
        {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "depth": depth,
            "width": width,
            "weight_decay": weight_decay,
            "score": performance,
        }
    )
    df["performance_bucket"] = pd.qcut(df["score"], 3, labels=["Baseline", "Good", "Great"])
    return df


def format_heatmap_ticks(ax: plt.Axes, values: np.ndarray, axis: str) -> None:
    """Attach readable tick labels on log-scaled heatmap axes."""
    tick_count = 6
    positions = np.linspace(0, len(values) - 1, tick_count).astype(int)
    labels = [f"{values[idx]:.0e}" if axis == "x" else f"{int(values[idx]):d}" for idx in positions]
    if axis == "x":
        ax.set_xticks(positions + 0.5)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    else:
        ax.set_yticks(positions + 0.5)
        ax.set_yticklabels(labels, rotation=0)


def plot_lr_batch_heatmap(lr_values, batch_values, performance) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        performance,
        ax=ax,
        cmap="viridis",
        cbar_kws={"label": "Validation accuracy"},
    )
    ax.set_title("Learning Rate vs Batch Size — Performance Landscape")
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel("Batch size")
    format_heatmap_ticks(ax, lr_values, axis="x")
    format_heatmap_ticks(ax, batch_values, axis="y")
    ax.text(
        0.03,
        0.92,
        "Diagonal ridge = stable region",
        transform=ax.transAxes,
        color="white",
        fontsize=11,
        bbox=dict(facecolor="black", alpha=0.35, pad=6),
    )
    ax.text(
        0.45,
        0.1,
        "High LR or tiny batches → instability",
        transform=ax.transAxes,
        color="white",
        fontsize=11,
        bbox=dict(facecolor="black", alpha=0.35, pad=6),
    )
    fig.tight_layout()
    return fig


def plot_lr_weight_decay_heatmap(lr_values, wd_values, performance) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        performance,
        ax=ax,
        cmap="magma",
        cbar_kws={"label": "Validation accuracy"},
    )
    ax.set_title("Learning Rate vs Weight Decay — Performance Landscape")
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel("Weight decay (log scale)")
    format_heatmap_ticks(ax, lr_values, axis="x")
    format_heatmap_ticks(ax, wd_values, axis="y")
    ax.text(
        0.05,
        0.88,
        "Sweet spot near moderate LR + moderate decay",
        transform=ax.transAxes,
        color="white",
        fontsize=11,
        bbox=dict(facecolor="black", alpha=0.35, pad=6),
    )
    ax.text(
        0.6,
        0.2,
        "Aggressive decay or LR → underfitting",
        transform=ax.transAxes,
        color="white",
        fontsize=11,
        bbox=dict(facecolor="black", alpha=0.35, pad=6),
    )
    fig.tight_layout()
    return fig


def plot_depth_width_surface(depths, widths, performance) -> plt.Figure:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    depth_grid, width_grid = np.meshgrid(depths, widths)
    surf = ax.plot_surface(
        depth_grid,
        width_grid,
        performance,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )
    ax.set_title("Architecture Search Landscape")
    ax.set_xlabel("Model depth (layers)")
    ax.set_ylabel("Model width (neurons)")
    ax.set_zlabel("Validation accuracy")
    fig.colorbar(surf, shrink=0.7, aspect=12, pad=0.1)
    ax.text2D(
        0.05,
        0.85,
        "Too deep/too wide → overfitting region",
        transform=ax.transAxes,
        color="darkred",
        bbox=dict(facecolor="white", alpha=0.7, pad=6),
    )
    ax.text2D(
        0.55,
        0.2,
        "Valley shows balanced architectures",
        transform=ax.transAxes,
        color="darkblue",
        bbox=dict(facecolor="white", alpha=0.7, pad=6),
    )
    fig.tight_layout()
    return fig


def plot_parallel_coordinates(trials: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    cols = ["learning_rate", "batch_size", "depth", "width", "weight_decay", "score"]
    pc_data = trials[["performance_bucket", *cols]].copy()
    pc_data["learning_rate"] = np.log10(pc_data["learning_rate"])
    pc_data["batch_size"] = np.log2(pc_data["batch_size"])
    pc_data["weight_decay"] = np.log10(pc_data["weight_decay"])
    parallel_coordinates(
        pc_data,
        class_column="performance_bucket",
        ax=ax,
        colormap="viridis",
        linewidth=2,
    )
    ax.set_title("Hyperparameter Sweep Overview")
    ax.set_ylabel("Scaled values (log for LR/WD/Batch)")
    ax.legend(title="Performance tier")
    ax.text(
        0.02,
        0.92,
        "Lines converging → consistent good configs",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, pad=6),
    )
    fig.tight_layout()
    return fig


def plot_trial_scatter(trials: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        trials["learning_rate"],
        trials["batch_size"],
        c=trials["score"],
        s=30 + (trials["depth"] - trials["depth"].min()) * 3,
        cmap="plasma",
        alpha=0.85,
        edgecolor="k",
        linewidth=0.4,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Learning rate (log)")
    ax.set_ylabel("Batch size (log)")
    ax.set_title("Hyperparameter Search Trials")
    fig.colorbar(scatter, ax=ax, label="Validation accuracy")
    ax.text(
        0.05,
        0.9,
        "Clustering shows promising search areas",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, pad=6),
    )
    ax.text(
        0.5,
        0.15,
        "High LR + low batch → divergence",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, pad=6),
    )
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, filename: str, dpi: int = 200) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {output_path}")


def main() -> None:
    lr_vals, batch_vals, lr_batch_perf = simulate_lr_batch_performance()
    lr_vals_wd, wd_vals, lr_wd_perf = simulate_lr_weight_decay_performance()
    depths, widths, arch_perf = simulate_depth_width_surface()
    trials = sample_trial_dataframe()

    figures = [
        (
            plot_lr_batch_heatmap(lr_vals, batch_vals, lr_batch_perf),
            "learning_rate_vs_batch_size_heatmap.png",
        ),
        (
            plot_lr_weight_decay_heatmap(lr_vals_wd, wd_vals, lr_wd_perf),
            "learning_rate_vs_weight_decay_heatmap.png",
        ),
        (
            plot_depth_width_surface(depths, widths, arch_perf),
            "architecture_search_surface.png",
        ),
        (
            plot_parallel_coordinates(trials),
            "hyperparameter_parallel_coordinates.png",
        ),
        (
            plot_trial_scatter(trials),
            "hyperparameter_trials_scatter.png",
        ),
    ]

    for fig, filename in figures:
        save_figure(fig, filename)

    plt.show()


if __name__ == "__main__":
    main()

