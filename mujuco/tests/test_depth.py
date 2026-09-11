import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim_main import ClothFoldEnv, DEPTH_MAX, DEPTH_MIN, TABLE_TOP_Z, CAMERA_POS  # noqa: E402

def test_state_mode_has_no_depth():
    env = ClothFoldEnv(observation_mode="state")
    obs, _ = env.reset(seed=0)
    assert "depth" not in obs
    assert "depth" not in env.observation_space.spaces

def test_pixels_mode_depth_shape_range_and_noise():
    env = ClothFoldEnv(observation_mode="pixels", image_size=(48, 64))
    obs, _ = env.reset(seed=0)
    depth = obs["depth"]
    assert env.observation_space["depth"].contains(depth)
    assert depth.shape == (48, 64, 1)
    assert depth.dtype == np.float32
    assert obs["image"].shape == (48, 64, 3)

    # out-of-range pixels are dropped to 0, everything else inside the sensor range
    valid = depth[depth > 0]
    assert valid.size > 0
    assert valid.min() >= DEPTH_MIN
    assert valid.max() <= DEPTH_MAX

    # the table top is roughly the camera-to-target distance away
    expected = np.linalg.norm(np.array(CAMERA_POS) - np.array([0.0, 0.0, TABLE_TOP_Z]))
    center = depth[24, 32, 0]
    assert abs(center - expected) < 0.15

    # noise is seeded: same seed reproduces, different seed does not
    obs_same, _ = ClothFoldEnv(observation_mode="pixels", image_size=(48, 64)).reset(seed=0)
    obs_other, _ = ClothFoldEnv(observation_mode="pixels", image_size=(48, 64)).reset(seed=1)
    assert np.array_equal(depth, obs_same["depth"])
    assert not np.array_equal(depth, obs_other["depth"])
