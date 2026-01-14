"""CMA-ES optimization for GRN parameters to produce target shapes."""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import cma

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.QS_GRN_toy import simulate
from utils.plotting import plot_training_curve, plot_parameter_evolution
from visualisation.grn_plots import animate_reporters


def compute_loss(gfp_expression, rfp_expression, gfp_high, rfp_high):
    """Loss for reporters being HIGH or LOW as specified.

    Args:
        gfp_expression: GFP levels for all cells
        rfp_expression: RFP levels for all cells
        gfp_high: if True, GFP should be high; if False, GFP should be low
        rfp_high: if True, RFP should be high; if False, RFP should be low
    """
    norm_gfp = gfp_expression / (gfp_expression.max() + 1e-8)
    norm_rfp = rfp_expression / (rfp_expression.max() + 1e-8)

    if gfp_high:
        loss_gfp = np.mean((1 - norm_gfp) ** 2)
    else:
        loss_gfp = np.mean(norm_gfp**2)

    if rfp_high:
        loss_rfp = np.mean((1 - norm_rfp) ** 2)
    else:
        loss_rfp = np.mean(norm_rfp**2)

    return loss_gfp + loss_rfp


def make_objective(circle_positions, star_positions, fixed_params, sim_config):
    """
    Create objective function for CMA-ES that runs both colony simulations.

    Args:
        circle_positions: Nx2 array of cell positions for circle colony
        star_positions: Mx2 array of cell positions for star colony
        fixed_params: dict of non-trainable params
        sim_config: dict with grid_size, dx, dt, n_steps

    Returns:
        Callable that takes param vector [K_gfp, K_rfp, n_gfp, n_rfp, alpha_gfp, alpha_rfp]
        and returns combined loss
    """

    def objective(x):
        K_gfp, K_rfp, n_gfp, n_rfp, alpha_gfp, alpha_rfp = x
        params = dict(fixed_params)
        params["K_gfp"] = K_gfp
        params["K_rfp"] = K_rfp
        params["n_gfp"] = n_gfp
        params["n_rfp"] = n_rfp
        params["alpha_gfp"] = alpha_gfp
        params["alpha_rfp"] = alpha_rfp

        history_circle, _ = simulate(
            circle_positions,
            params,
            sim_config["grid_size"],
            sim_config["dx"],
            sim_config["dt"],
            sim_config["n_steps"],
            sim_config["ahl_init"],
        )
        gfp_circle = history_circle[-1, :, 1]
        rfp_circle = history_circle[-1, :, 2]
        loss_circle = compute_loss(gfp_circle, rfp_circle, gfp_high=True, rfp_high=False)

        history_star, _ = simulate(
            star_positions,
            params,
            sim_config["grid_size"],
            sim_config["dx"],
            sim_config["dt"],
            sim_config["n_steps"],
            sim_config["ahl_init"],
        )
        gfp_star = history_star[-1, :, 1]
        rfp_star = history_star[-1, :, 2]
        loss_star = compute_loss(gfp_star, rfp_star, gfp_high=False, rfp_high=True)

        return loss_circle + loss_star

    return objective


def optimize_grn_params(
    circle_positions, star_positions, fixed_params, sim_config, cma_options, output_dir
):
    """
    Run CMA-ES optimization to find params for shape classification.

    Circle colony should express: high GFP, low RFP
    Star colony should express: low GFP, high RFP

    Args:
        circle_positions: Nx2 array of cell positions for circle colony
        star_positions: Mx2 array of cell positions for star colony
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

    objective = make_objective(circle_positions, star_positions, fixed_params, sim_config)

    x0 = cma_options.pop("x0")
    sigma0 = cma_options.pop("sigma0")
    lower_bounds = cma_options.pop("lower_bounds", None)
    upper_bounds = cma_options.pop("upper_bounds", None)

    if lower_bounds is not None and upper_bounds is not None:
        wrapped_objective = cma.BoundDomainTransform(objective, [lower_bounds, upper_bounds])
        transform_fn = wrapped_objective.transform
    else:
        wrapped_objective = objective
        transform_fn = lambda x: np.array(x)

    es = cma.CMAEvolutionStrategy(x0, sigma0, cma_options)

    losses = []
    param_history = []
    param_names = ["K_gfp", "K_rfp", "n_gfp", "n_rfp", "alpha_gfp", "alpha_rfp"]

    while not es.stop():
        X = es.ask()
        fitness_values = [wrapped_objective(x) for x in X]
        es.tell(X, fitness_values)
        es.disp()

        best_idx = np.argmin(fitness_values)
        losses.append(min(fitness_values))
        param_history.append(transform_fn(X[best_idx]).tolist())

    es.result_pretty()

    x_best = es.result.xbest
    x_best_transformed = transform_fn(x_best)

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

    history_circle, _ = simulate(
        circle_positions,
        final_params,
        sim_config["grid_size"],
        sim_config["dx"],
        sim_config["dt"],
        sim_config["n_steps"],
        sim_config["ahl_init"],
    )
    animate_reporters(
        history_circle,
        circle_positions,
        output_dir / "circle_reporters.mp4",
        sim_config["fps"],
        sim_config["show_cell_bg"],
    )

    history_star, _ = simulate(
        star_positions,
        final_params,
        sim_config["grid_size"],
        sim_config["dx"],
        sim_config["dt"],
        sim_config["n_steps"],
        sim_config["ahl_init"],
    )
    animate_reporters(
        history_star,
        star_positions,
        output_dir / "star_reporters.mp4",
        sim_config["fps"],
        sim_config["show_cell_bg"],
    )

    results = {
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

    circle_positions = generate_colony("circle", n_cells_per_side=20, center=(0.5, 0.5), size=0.6)
    star_positions = generate_colony("star", n_cells_per_side=20, center=(0.5, 0.5), size=0.6)

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
        "grid_size": 30,
        "dx": 1.0,
        "dt": 0.1,
        "n_steps": 5,
        "ahl_init": 0.0,
        "fps": 10,
        "show_cell_bg": True,
    }

    cma_options = {
        "x0": [0.5, 0.8, 2.0, 2.0, 2.0, 2.0],
        "sigma0": 1,
        "lower_bounds": [0.1, 0.1, 1.0, 1.0, 0.5, 0.5],
        "upper_bounds": [2.0, 2.0, 4.0, 4.0, 5.0, 5.0],
        "maxiter": 20,
        "popsize": 6,
        "tolfun": 1e-12,
        "tolx": 1e-12,
        "verbose": 1,
        "verb_disp": 10,
    }

    output_dir = Path("results_optimisation") / datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Shape Classifier: Circle->GFP, Star->RFP")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    best_params, es = optimize_grn_params(
        circle_positions,
        star_positions,
        fixed_params,
        sim_config,
        cma_options,
        output_dir,
    )

    print("\nBest parameters found:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.4f}")
    print(f"Final loss: {es.result.fbest:.6f}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
