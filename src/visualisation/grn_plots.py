import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap


def plot_simulation(history_states, history_field, cell_positions, grid_size, output_dir, show_cell_bg):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    steps_to_show = [0, len(history_field) // 2, len(history_field) - 1]
    for idx, step in enumerate(steps_to_show):
        ax = axes[0, idx]
        im = ax.imshow(history_field[step], origin="lower", cmap="YlOrRd", extent=[0, 1, 0, 1])
        if show_cell_bg:
            ax.scatter(cell_positions[:, 0], cell_positions[:, 1], c="lightgray", s=20, edgecolors="gray", linewidths=0.5)
        ax.set_title(f"AHL field (step {step})")
        plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    mean_gfp = history_states[:, :, 1].mean(axis=1)
    mean_rfp = history_states[:, :, 2].mean(axis=1)
    ax.plot(mean_gfp, "g-", label="GFP")
    ax.plot(mean_rfp, "r-", label="RFP")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean expression")
    ax.legend()
    ax.set_title("Reporter dynamics")

    ax = axes[1, 1]
    final_gfp = history_states[-1, :, 1]
    final_rfp = history_states[-1, :, 2]
    colors = np.zeros((len(cell_positions), 3))
    colors[:, 0] = final_rfp / (final_rfp.max() + 1e-6)
    colors[:, 1] = final_gfp / (final_gfp.max() + 1e-6)
    ax.scatter(cell_positions[:, 0], cell_positions[:, 1], c=colors, s=80, edgecolors="gray", linewidths=0.5)
    ax.set_title("Final reporter (R=RFP, G=GFP)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "grn_simulation.png", dpi=150)
    plt.show()


def animate_reporters(history_states, cell_positions, output_path, fps, show_cell_bg):
    n_steps = len(history_states)
    gfp_all = history_states[:, :, 1]
    rfp_all = history_states[:, :, 2]
    shared_max = max(gfp_all.max(), rfp_all.max(), 1e-6)

    cmap_gfp = LinearSegmentedColormap.from_list("white_green", ["white", "green"])
    cmap_rfp = LinearSegmentedColormap.from_list("white_red", ["white", "red"])

    fig, (ax_gfp, ax_rfp) = plt.subplots(1, 2, figsize=(10, 5))

    if show_cell_bg:
        ax_gfp.scatter(cell_positions[:, 0], cell_positions[:, 1], c="white", s=120, edgecolors="lightgray", linewidths=0.5)
    scatter_gfp = ax_gfp.scatter(cell_positions[:, 0], cell_positions[:, 1], c=np.zeros(len(cell_positions)), cmap=cmap_gfp, vmin=0, vmax=shared_max, s=100, edgecolors="gray", linewidths=0.5)
    ax_gfp.set_xlim(0, 1)
    ax_gfp.set_ylim(0, 1)
    ax_gfp.set_title("GFP")
    ax_gfp.set_aspect("equal")
    plt.colorbar(scatter_gfp, ax=ax_gfp)

    if show_cell_bg:
        ax_rfp.scatter(cell_positions[:, 0], cell_positions[:, 1], c="white", s=120, edgecolors="lightgray", linewidths=0.5)
    scatter_rfp = ax_rfp.scatter(cell_positions[:, 0], cell_positions[:, 1], c=np.zeros(len(cell_positions)), cmap=cmap_rfp, vmin=0, vmax=shared_max, s=100, edgecolors="gray", linewidths=0.5)
    ax_rfp.set_xlim(0, 1)
    ax_rfp.set_ylim(0, 1)
    ax_rfp.set_title("RFP")
    ax_rfp.set_aspect("equal")
    plt.colorbar(scatter_rfp, ax=ax_rfp)

    def update(frame):
        gfp = history_states[frame, :, 1]
        rfp = history_states[frame, :, 2]
        scatter_gfp.set_offsets(cell_positions)
        scatter_gfp.set_array(gfp)
        scatter_rfp.set_offsets(cell_positions)
        scatter_rfp.set_array(rfp)
        fig.suptitle(f"Step {frame}")
        return scatter_gfp, scatter_rfp

    anim = FuncAnimation(fig, update, frames=n_steps, blit=False)
    anim.save(output_path, writer="ffmpeg", fps=fps)
    plt.close()
