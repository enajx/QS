import numpy as np
import matplotlib.pyplot as plt


def plot_simulation(history_states, history_field, cell_positions, grid_size):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    steps_to_show = [0, len(history_field) // 2, len(history_field) - 1]
    for idx, step in enumerate(steps_to_show):
        ax = axes[0, idx]
        im = ax.imshow(history_field[step], origin="lower", cmap="YlOrRd", extent=[0, 1, 0, 1])
        ax.scatter(cell_positions[:, 0], cell_positions[:, 1], c="blue", s=20, alpha=0.7)
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
    ax.scatter(cell_positions[:, 0], cell_positions[:, 1], c=colors, s=100)
    ax.set_title("Final reporter (R=RFP, G=GFP)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig("results/grn_simulation.png", dpi=150)
    plt.show()
