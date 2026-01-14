import numpy as np
from scipy.integrate import solve_ivp


def hill(x, K, n):
    return x**n / (K**n + x**n)


def grn_ode(
    t,
    state,
    alpha_ahl,
    alpha_gfp,
    alpha_rfp,
    K_ahl,
    K_gfp,
    K_rfp,
    n_ahl,
    n_gfp,
    n_rfp,
    delta,
    ahl_ext,
):
    ahl, gfp, rfp = state
    total_ahl = ahl + ahl_ext
    d_ahl = alpha_ahl - delta * ahl
    d_gfp = alpha_gfp * hill(total_ahl, K_gfp, n_gfp) - delta * gfp
    d_rfp = alpha_rfp * (1 - hill(total_ahl, K_rfp, n_rfp)) - delta * rfp
    return [d_ahl, d_gfp, d_rfp]


def laplacian_2d(field, dx):
    lap = np.zeros_like(field)
    lap[1:-1, 1:-1] = (
        field[:-2, 1:-1]
        + field[2:, 1:-1]
        + field[1:-1, :-2]
        + field[1:-1, 2:]
        - 4 * field[1:-1, 1:-1]
    ) / (dx**2)
    return lap


def diffusion_step(ahl_field, D, mu, dt, dx, sources, source_positions, grid_size):
    lap = laplacian_2d(ahl_field, dx)
    ahl_field = ahl_field + dt * (D * lap - mu * ahl_field)
    for i, (px, py) in enumerate(source_positions):
        gx, gy = int(px * grid_size), int(py * grid_size)
        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            ahl_field[gy, gx] += dt * sources[i]
    return np.clip(ahl_field, 0, None)


def get_local_ahl(ahl_field, cell_positions, grid_size):
    local_ahl = np.zeros(len(cell_positions))
    for i, (px, py) in enumerate(cell_positions):
        gx, gy = int(px * grid_size), int(py * grid_size)
        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            local_ahl[i] = ahl_field[gy, gx]
    return local_ahl


def simulate(cell_positions, params, grid_size, dx, dt, n_steps):
    n_cells = len(cell_positions)
    cell_states = np.zeros((n_cells, 3))
    ahl_field = np.zeros((grid_size, grid_size))

    history_states = np.zeros((n_steps, n_cells, 3))
    history_field = np.zeros((n_steps, grid_size, grid_size))

    for step in range(n_steps):
        local_ahl = get_local_ahl(ahl_field, cell_positions, grid_size)

        for i in range(n_cells):
            sol = solve_ivp(
                grn_ode,
                [0, dt],
                cell_states[i],
                args=(
                    params["alpha_ahl"],
                    params["alpha_gfp"],
                    params["alpha_rfp"],
                    params["K_ahl"],
                    params["K_gfp"],
                    params["K_rfp"],
                    params["n_ahl"],
                    params["n_gfp"],
                    params["n_rfp"],
                    params["delta"],
                    local_ahl[i],
                ),
                method="RK45",
            )
            cell_states[i] = sol.y[:, -1]

        sources = cell_states[:, 0] * params["secretion_rate"]
        ahl_field = diffusion_step(
            ahl_field, params["D"], params["mu"], dt, dx, sources, cell_positions, grid_size
        )

        history_states[step] = cell_states.copy()
        history_field[step] = ahl_field.copy()

    return history_states, history_field


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from datetime import datetime

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from visualisation.grn_plots import plot_simulation, animate_reporters
    from models.shapes import generate_colony

    run_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    grid_size = 50
    shape = "square"  # "square", "circle", or "star"
    # shape = "star"  # "square", "circle", or "star"
    # shape = "circle"  # "square", "circle", or "star"
    cell_positions = generate_colony(shape, n_cells_per_side=20, center=(0.5, 0.5), size=0.6)

    params = {
        "alpha_ahl": 1.0,
        "alpha_gfp": 2.0,
        "alpha_rfp": 2.0,
        "K_ahl": 0.5,
        "K_gfp": 0.5,
        "K_rfp": 0.8,
        "n_ahl": 2.0,
        "n_gfp": 2.0,
        "n_rfp": 2.0,
        "delta": 0.1,
        "D": 0.1,
        "mu": 0.01,
        "secretion_rate": 0.5,
    }

    history_states, history_field = simulate(
        cell_positions, params, grid_size, dx=1.0, dt=0.1, n_steps=100
    )

    plot_simulation(
        history_states, history_field, cell_positions, grid_size, run_dir, show_cell_bg=False
    )
    animate_reporters(
        history_states, cell_positions, run_dir / "reporters.mp4", fps=10, show_cell_bg=True
    )
