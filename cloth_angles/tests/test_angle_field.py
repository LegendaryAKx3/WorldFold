import numpy as np
import pytest

from cloth_angles.data.angle_field import compute_angle_field, vertices_grid_from_flat


def _flat_grid(n=5, spacing=0.03, z=0.5):
    xs, ys = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing, indexing="ij")
    zs = np.full((n, n), z)
    return np.stack([xs, ys, zs], axis=-1).astype(np.float64)


def test_flat_cloth_has_near_zero_angle_everywhere():
    vertices = _flat_grid()
    angle = compute_angle_field(vertices, signed=True)
    assert angle.shape == (5, 5)
    assert np.allclose(angle, 0.0, atol=1e-6)


def test_flat_cloth_unsigned_angle_is_zero():
    vertices = _flat_grid()
    angle = compute_angle_field(vertices, signed=False)
    assert np.allclose(angle, 0.0, atol=1e-6)


def test_tilted_cloth_has_nonzero_angle():
    n = 5
    vertices = _flat_grid(n=n)
    # tilt the whole sheet by lifting one edge (x=0 row) up in z
    tilt = np.linspace(0.05, 0.0, n)
    vertices[:, :, 2] += tilt[:, None]
    angle = compute_angle_field(vertices, signed=True)
    assert not np.allclose(angle, 0.0, atol=1e-4)
    unsigned = compute_angle_field(vertices, signed=False)
    assert np.all(unsigned >= 0.0)
    assert np.all(unsigned <= np.pi)


def test_signed_angle_flips_sign_for_opposite_tilt():
    n = 5
    up_tilt = _flat_grid(n=n)
    tilt = np.linspace(0.05, 0.0, n)
    up_tilt[:, :, 2] += tilt[:, None]

    down_tilt = _flat_grid(n=n)
    down_tilt[:, :, 2] -= tilt[:, None]

    angle_up = compute_angle_field(up_tilt, signed=True)
    angle_down = compute_angle_field(down_tilt, signed=True)
    # opposite physical tilts should not collapse to the same signed angle
    assert not np.allclose(angle_up, angle_down, atol=1e-3)


def test_vertices_grid_from_flat_reshapes_row_major():
    flat = np.arange(9 * 3).reshape(9, 3).astype(np.float64)
    grid = vertices_grid_from_flat(flat, grid_size=3)
    assert grid.shape == (3, 3, 3)
    assert np.array_equal(grid[0, 0], flat[0])
    assert np.array_equal(grid[2, 2], flat[8])


def test_vertices_grid_from_flat_rejects_wrong_shape():
    flat = np.zeros((10, 3))
    with pytest.raises(ValueError):
        vertices_grid_from_flat(flat, grid_size=3)


def test_compute_angle_field_rejects_non_square_input():
    with pytest.raises(ValueError):
        compute_angle_field(np.zeros((3, 4, 3)))
