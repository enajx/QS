"""Plotting utilities for optimization results."""

import numpy as np
import matplotlib.pyplot as plt


def plot_training_curve(losses, output_path):
    """Plot loss over iterations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training Curve")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_expression_comparison(cell_positions, final_expression, target_mask, reporter, output_path):
    """Plot side-by-side comparison of target vs achieved expression."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    max_val = final_expression.max()
    if max_val > 1e-8:
        normalized_expression = final_expression / max_val
    else:
        normalized_expression = np.zeros_like(final_expression)

    cmap = "Greens" if reporter == "gfp" else "Reds"

    sc0 = axes[0].scatter(
        cell_positions[:, 0],
        cell_positions[:, 1],
        c=target_mask,
        cmap=cmap,
        s=50,
        vmin=0,
        vmax=1,
    )
    axes[0].set_title("Target Pattern")
    axes[0].set_aspect("equal")
    plt.colorbar(sc0, ax=axes[0])

    sc1 = axes[1].scatter(
        cell_positions[:, 0],
        cell_positions[:, 1],
        c=normalized_expression,
        cmap=cmap,
        s=50,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(f"Achieved {reporter.upper()} Expression")
    axes[1].set_aspect("equal")
    plt.colorbar(sc1, ax=axes[1])

    error = np.abs(normalized_expression - target_mask)
    sc2 = axes[2].scatter(
        cell_positions[:, 0],
        cell_positions[:, 1],
        c=error,
        cmap="hot",
        s=50,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Absolute Error")
    axes[2].set_aspect("equal")
    plt.colorbar(sc2, ax=axes[2])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_parameter_evolution(param_history, param_names, output_path):
    """Plot how parameters evolved during optimization."""
    param_history = np.array(param_history)
    n_params = param_history.shape[1]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(n_params):
        axes[i].plot(param_history[:, i], linewidth=2)
        axes[i].set_xlabel("Iteration")
        axes[i].set_ylabel(param_names[i])
        axes[i].set_title(param_names[i])
        axes[i].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
