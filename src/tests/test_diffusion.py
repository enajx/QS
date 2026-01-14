"""Test that AHL diffusion spreads locally from source."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.QS_GRN_toy import diffusion_step


def test_diffusion_spreads_locally():
    grid_size = 21
    dx = 1.0
    dt = 0.1
    D = 0.5
    mu = 0.01

    ahl_field = np.zeros((grid_size, grid_size))
    center = grid_size // 2
    source_position = np.array([[0.5, 0.5]])
    source_strength = np.array([10.0])

    for _ in range(50):
        ahl_field = diffusion_step(
            ahl_field, D, mu, dt, dx, source_strength, source_position, grid_size
        )

    center_value = ahl_field[center, center]
    assert center_value > 0, "Center should have AHL"

    for dist in [1, 2, 3, 4]:
        neighbor_value = ahl_field[center + dist, center]
        closer_value = ahl_field[center + dist - 1, center]
        assert neighbor_value < closer_value, (
            f"AHL at distance {dist} ({neighbor_value:.4f}) should be less than "
            f"at distance {dist-1} ({closer_value:.4f})"
        )

    corner_value = ahl_field[1, 1]
    assert center_value > corner_value * 5, (
        f"Center ({center_value:.4f}) should be much higher than corner ({corner_value:.4f})"
    )

    print(f"Center AHL: {center_value:.4f}")
    print(f"Distance 1: {ahl_field[center+1, center]:.4f}")
    print(f"Distance 2: {ahl_field[center+2, center]:.4f}")
    print(f"Distance 3: {ahl_field[center+3, center]:.4f}")
    print(f"Corner AHL: {corner_value:.4f}")
    print("Diffusion test PASSED: AHL spreads locally from source")


if __name__ == "__main__":
    test_diffusion_spreads_locally()
