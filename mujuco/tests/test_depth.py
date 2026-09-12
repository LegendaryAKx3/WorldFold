import numpy as np

import mujuco.sim_main as sm
from mujuco.sim_main import ClothFoldEnv, DEPTH_MAX, DEPTH_MIN, TABLE_TOP_Z, CAMERA_POS, CAMERA_TARGET


def test_move_camera_points_at_target():
    env = ClothFoldEnv(observation_mode="pixels", image_size=(48, 64))
    env.reset(seed=0)
    new_pos = (0.0, -0.9, 1.0)
    env.move_camera(new_pos)
    cid = env.model.camera("main").id
    assert np.allclose(env.data.cam_xpos[cid], new_pos, atol=1e-6)
    # mujoco cameras look down their local -z axis
    axis = -env.data.cam_xmat[cid].reshape(3, 3)[:, 2]
    want = np.array(CAMERA_TARGET) - np.array(new_pos)
    want = want / np.linalg.norm(want)
    assert np.allclose(axis, want, atol=1e-6)


def test_calibrate_picks_the_view_that_sees_the_cloth(monkeypatch):
    far_away = (4.0, -4.0, 4.0)         # cloth beyond DEPTH_MAX: every vertex is a hole
    monkeypatch.setattr(sm, "CANDIDATE_POSITIONS", [far_away, CAMERA_POS])
    monkeypatch.setattr(sm, "BEST_POS", far_away)
    env = ClothFoldEnv(observation_mode="pixels", image_size=(48, 64))
    env.reset(seed=0)
    best, scores = env.test_calibrate_camera()
    assert tuple(best) == tuple(CAMERA_POS)
    assert tuple(sm.BEST_POS) == tuple(CAMERA_POS)
    assert scores[1] < scores[0]


def test_render_uses_best_pos_only_in_testing_mode(monkeypatch):
    other = (0.0, -0.9, 1.0)
    monkeypatch.setattr(sm, "BEST_POS", other)
    env = ClothFoldEnv(observation_mode="pixels", image_size=(48, 64))
    env.reset(seed=0)
    cid = env.model.camera("main").id
    env._render_image(testing=True)
    assert np.allclose(env.data.cam_xpos[cid], other, atol=1e-6)
    env._render_image(testing=False)
    assert np.allclose(env.data.cam_xpos[cid], CAMERA_POS, atol=1e-6)

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
