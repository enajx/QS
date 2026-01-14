import numpy as np


def generate_colony(shape, n_cells_per_side, center, size):
    if shape == "square":
        return _square_grid(n_cells_per_side, center, size)
    elif shape == "circle":
        return _circle_grid(n_cells_per_side, center, size)
    elif shape == "star":
        return _star_grid(n_cells_per_side, center, size)
    else:
        raise ValueError(f"Unknown shape: {shape}")


def _square_grid(n_cells_per_side, center, size):
    x = np.linspace(center[0] - size/2, center[0] + size/2, n_cells_per_side)
    y = np.linspace(center[1] - size/2, center[1] + size/2, n_cells_per_side)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _circle_grid(n_cells_per_side, center, size):
    radius = size / 2
    spacing = size / (n_cells_per_side - 1)
    points = [np.array(center)]
    r = spacing
    while r <= radius:
        circumference = 2 * np.pi * r
        n_points = max(1, int(circumference / spacing))
        for i in range(n_points):
            theta = 2 * np.pi * i / n_points
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            points.append(np.array([x, y]))
        r += spacing
    return np.array(points)


def _star_grid(n_cells_per_side, center, size):
    square = _square_grid(n_cells_per_side, center, size)
    mask = np.array([_point_in_star(p[0], p[1], center, size) for p in square])
    return square[mask]


def _point_in_star(x, y, center, size):
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
