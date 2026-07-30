"""Derives the per-cell surface-angle field from cloth vertex positions.

The spec's model never receives or predicts a normal vector -- the angle
field is "already-computed" before it reaches the model. This module is
where that computation happens: given the (N, N, 3) grid of cloth vertex
world positions, compute each interior cell's local surface tilt angle
relative to a fixed flat reference (the world up-axis), and reduce it to a
single signed scalar per cell.

Border cells (row/col 0 or N-1) can't form a centered normal from both
neighbors, so they reuse their nearest interior neighbor's angle -- keeping
the field the same fixed [N, N] shape without fabricating an artificial edge
value from a one-sided estimate.
"""

from __future__ import annotations

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


def _cell_normal(vertices: np.ndarray, i: int, j: int) -> np.ndarray:
    """Local surface normal at interior cell (i, j) via central-difference
    tangent vectors along both grid axes, then their cross product.
    """
    n = vertices.shape[0]
    tangent_row = vertices[min(i + 1, n - 1), j] - vertices[max(i - 1, 0), j]
    tangent_col = vertices[i, min(j + 1, n - 1)] - vertices[i, max(j - 1, 0)]
    normal = np.cross(tangent_row, tangent_col)
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return UP.copy()
    return normal / norm


def compute_angle_field(vertices: np.ndarray, signed: bool = True) -> np.ndarray:
    """vertices: [N, N, 3] cloth vertex world positions (row-major grid, as
    stored by ClothFoldEnv's cloth_{i} bodies reshaped to [N, N, 3]).

    Returns angle: [N, N] float32, the angle between each cell's local
    surface normal and the fixed world up-axis (the "zero-degree reference").

    signed=True returns a signed angle in [-pi, pi] using the normal's
    horizontal tilt direction (atan2 of the in-plane tilt component vs. the
    vertical component) so fold direction is distinguishable, not just
    magnitude. signed=False returns the unsigned deviation in [0, pi]
    (arccos of normal . up), collapsing tilt direction to magnitude only.
    """
    n = vertices.shape[0]
    if vertices.shape != (n, n, 3):
        raise ValueError(f"vertices must be [N, N, 3], got {vertices.shape}")

    angle = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            normal = _cell_normal(vertices, i, j)
            if signed:
                # in-plane tilt magnitude (how far the normal leans off vertical)
                # signed by the x-component of the tilt, a fixed, consistent
                # reference direction so the same physical fold always gets
                # the same sign.
                vertical = float(np.dot(normal, UP))
                horizontal = float(np.linalg.norm(normal - vertical * UP))
                tilt_sign = np.sign(normal[0]) if abs(normal[0]) > 1e-9 else 1.0
                angle[i, j] = np.arctan2(tilt_sign * horizontal, vertical)
            else:
                cos_angle = np.clip(float(np.dot(normal, UP)), -1.0, 1.0)
                angle[i, j] = np.arccos(cos_angle)
    return angle


def vertices_grid_from_flat(flat_xpos: np.ndarray, grid_size: int) -> np.ndarray:
    """flat_xpos: [N*N, 3] row-major cloth vertex positions (as returned by
    env.data.xpos[env._cloth_body_ids]) -> [N, N, 3].
    """
    n2 = grid_size * grid_size
    if flat_xpos.shape != (n2, 3):
        raise ValueError(f"flat_xpos must be [{n2}, 3], got {flat_xpos.shape}")
    return flat_xpos.reshape(grid_size, grid_size, 3)
