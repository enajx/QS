"""CMA-ES optimization for GRN parameters to produce target shapes."""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import cma

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.QS_GRN_toy import simulate
from utils.plotting import plot_training_curve, plot_expression_comparison, plot_parameter_evolution
from visualisation.grn_plots import animate_reporters


def _point_in_star(x, y, center, size):
    """Check if point is inside star shape (same logic as shapes.py)."""
    dx = x - center[0]
    dy = y - center[1]
    r = np.sqrt(dx**2 + dy**2)
    if r < 1e-9:
        return True
    theta = np.arctan2(dy, dx)
    n_points = 5
    outer_radius = size / 2
    inner_radius = size / 5
    angle_per_point = 2 * np.pi / n_points
    theta_mod = theta % angle_per_point
    half_angle = angle_per_point / 2
    if theta_mod < half_angle:
        t = theta_mod / half_angle
        boundary_r = outer_radius * (1 - t) + inner_radius * t
    else:
        t = (theta_mod - half_angle) / half_angle
        boundary_r = inner_radius * (1 - t) + outer_radius * t
    return r <= boundary_r


def create_target_mask(cell_positions, target_shape, center, size):
    """Create binary mask: 1 inside shape, 0 outside (same logic as shapes.py).

    Args:
        cell_positions: Nx2 array of cell positions
        target_shape: "circle" or "star"
        center: (x, y) center of target shape
        size: size parameter (diameter for circle, bounding box for star)
    """
    n_cells = len(cell_positions)
    mask = np.zeros(n_cells)

    for i, (x, y) in enumerate(cell_positions):
        if target_shape == "circle":
            dx = x - center[0]
            dy = y - center[1]
            dist = np.sqrt(dx**2 + dy**2)
            radius = size / 2
            mask[i] = 1.0 if dist <= radius else 0.0

        elif target_shape == "star":
            mask[i] = 1.0 if _point_in_star(x, y, center, size) else 0.0

        else:
            raise ValueError(f"Unknown target shape: {target_shape}")

    return mask


def compute_loss(final_expression, target_mask):
    """MSE between normalized expression and target mask."""
    max_val = final_expression.max()
    if max_val < 1e-8:
        normalized = np.zeros_like(final_expression)
    else:
        normalized = final_expression / max_val
    return np.mean((normalized - target_mask) ** 2)


def make_objective(cell_positions, target_mask, reporter, fixed_params, sim_config):
    """
    Create objective function for CMA-ES.

    Args:
        cell_positions: Nx2 array of cell positions
        target_mask: binary mask for target shape
        reporter: "gfp" or "rfp" - which reporter should match the shape
        fixed_params: dict of non-trainable params
        sim_config: dict with grid_size, dx, dt, n_steps

    Returns:
        Callable that takes param vector [K_gfp, K_rfp, n_gfp, n_rfp, alpha_gfp, alpha_rfp]
        and returns loss
    """
    reporter_idx = 1 if reporter == "gfp" else 2

    def objective(x):
        K_gfp, K_rfp, n_gfp, n_rfp, alpha_gfp, alpha_rfp = x
        params = dict(fixed_params)
        params["K_gfp"] = K_gfp
        params["K_rfp"] = K_rfp
        params["n_gfp"] = n_gfp
        params["n_rfp"] = n_rfp
        params["alpha_gfp"] = alpha_gfp
        params["alpha_rfp"] = alpha_rfp

        history_states, _ = simulate(
            cell_positions,
            params,
            sim_config["grid_size"],
            sim_config["dx"],
            sim_config["dt"],
            sim_config["n_steps"],
            sim_config["ahl_init"],
        )
        final_expression = history_states[-1, :, reporter_idx]
        return compute_loss(final_expression, target_mask)

    return objective


def optimize_grn_params(
    cell_positions, target_shape, reporter, fixed_params, sim_config, cma_options, output_dir
):
    """
    Run CMA-ES optimization to find params matching target shape.

    Args:
        cell_positions: Nx2 array of cell positions
        target_shape: "circle" or "star"
        reporter: "gfp" or "rfp"
        fixed_params: dict of non-trainable params
        sim_config: dict with grid_size, dx, dt, n_steps
        cma_options: dict of CMA-ES options including:
            - x0: initial parameter guess [K_gfp, K_rfp, n_gfp, n_rfp, alpha_gfp, alpha_rfp]
            - sigma0: initial step size
            - lower_bounds: lower bounds for params
            - upper_bounds: upper bounds for params
            - maxiter, popsize, tolfun, tolx, seed, verbose, verb_disp, etc.
        output_dir: Path to save results

    Returns:
        best_params: dict with optimized K_gfp, K_rfp, n_gfp, n_rfp, alpha_gfp, alpha_rfp
        es: CMAEvolutionStrategy instance
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    center = (0.5, 0.5)
    target_size = cma_options.pop("target_size")
    target_mask = create_target_mask(cell_positions, target_shape, center, target_size)

    objective = make_objective(cell_positions, target_mask, reporter, fixed_params, sim_config)

    x0 = cma_options.pop("x0")
    sigma0 = cma_options.pop("sigma0")
    lower_bounds = cma_options.pop("lower_bounds")
    upper_bounds = cma_options.pop("upper_bounds")

    bounded_objective = cma.BoundDomainTransform(objective, [lower_bounds, upper_bounds])

    es = cma.CMAEvolutionStrategy(x0, sigma0, cma_options)

    losses = []
    param_history = []
    param_names = ["K_gfp", "K_rfp", "n_gfp", "n_rfp", "alpha_gfp", "alpha_rfp"]

    while not es.stop():
        X = es.ask()
        fitness_values = [bounded_objective(x) for x in X]
        es.tell(X, fitness_values)
        es.disp()

        best_idx = np.argmin(fitness_values)
        losses.append(min(fitness_values))
        param_history.append(bounded_objective.transform(X[best_idx]).tolist())

    es.result_pretty()

    x_best = es.result.xbest
    x_best_transformed = bounded_objective.transform(x_best)

    best_params = {
        "K_gfp": float(x_best_transformed[0]),
        "K_rfp": float(x_best_transformed[1]),
        "n_gfp": float(x_best_transformed[2]),
        "n_rfp": float(x_best_transformed[3]),
        "alpha_gfp": float(x_best_transformed[4]),
        "alpha_rfp": float(x_best_transformed[5]),
    }

    plot_training_curve(losses, output_dir / "training_curve.png")
    plot_parameter_evolution(param_history, param_names, output_dir / "parameter_evolution.png")

    final_params = dict(fixed_params)
    final_params.update(best_params)
    reporter_idx = 1 if reporter == "gfp" else 2
    history_states, _ = simulate(
        cell_positions,
        final_params,
        sim_config["grid_size"],
        sim_config["dx"],
        sim_config["dt"],
        sim_config["n_steps"],
        sim_config["ahl_init"],
    )
    final_expression = history_states[-1, :, reporter_idx]
    plot_expression_comparison(
        cell_positions,
        final_expression,
        target_mask,
        reporter,
        output_dir / "expression_comparison.png",
    )
    animate_reporters(
        history_states,
        cell_positions,
        output_dir / "reporters.mp4",
        sim_config["fps"],
        sim_config["show_cell_bg"],
    )

    results = {
        "target_shape": target_shape,
        "reporter": reporter,
        "best_params": best_params,
        "final_loss": float(es.result.fbest),
        "iterations": es.result.iterations,
        "evaluations": es.result.evaluations,
        "losses": losses,
        "fixed_params": fixed_params,
        "sim_config": sim_config,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_params, es


def main():
    from models.shapes import generate_colony

    cell_positions = generate_colony("circle", n_cells_per_side=20, center=(0.5, 0.5), size=0.6)

    fixed_params = {
        "alpha_ahl": 1.0,
        "K_ahl": 0.5,
        "n_ahl": 2.0,
        "delta": 0.1,
        "D": 0.1,
        "mu": 0.01,
        "secretion_rate": 0.5,
    }

    sim_config = {
        "grid_size": 80,
        "dx": 1.0,
        "dt": 0.1,
        "n_steps": 50,
        "ahl_init": 0.0,
        "fps": 10,
        "show_cell_bg": True,
    }

    cma_options = {
        "x0": [0.5, 0.8, 2.0, 2.0, 2.0, 2.0],
        "sigma0": 0.3,
        "lower_bounds": [0.1, 0.1, 1.0, 1.0, 0.5, 0.5],
        "upper_bounds": [2.0, 2.0, 4.0, 4.0, 5.0, 5.0],
        "target_size": 0.6,
        "maxiter": 5,
        "popsize": 6,
        "tolfun": 1e-12,
        "tolx": 1e-12,
        "verbose": 1,
        "verb_disp": 10,
    }

    output_dir = Path("results_optimisation") / datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Optimizing GRN params for circle shape (GFP reporter)")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    best_params, es = optimize_grn_params(
        cell_positions,
        target_shape="circle",
        reporter="gfp",
        fixed_params=fixed_params,
        sim_config=sim_config,
        cma_options=cma_options,
        output_dir=output_dir,
    )

    print("\nBest parameters found:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.4f}")
    print(f"Final loss: {es.result.fbest:.6f}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
