"""Checkpoints must carry their schema and reject incompatible loads."""

import pytest

from cloth_angles.model.checkpoint import ExpectedSchema, IncompatibleCheckpointError, load_checkpoint, save_checkpoint
from cloth_angles.model.world_model import WorldModel

MODEL_CONFIG = dict(encoder_hidden=[16, 8], h_dim=16, n_categoricals=2, n_classes=2,
                     mlp_hidden=16, kl_free_bits=1.0, kl_weight=1.0, huber_delta=1.0)


def _make_model(grid_size=4, action_dim=2, angle_convention="signed"):
    kwargs = dict(MODEL_CONFIG)
    kwargs["encoder_hidden"] = tuple(kwargs["encoder_hidden"])
    return WorldModel(grid_size=grid_size, action_dim=action_dim, angle_convention=angle_convention, **kwargs)


def test_round_trip_matches_schema(tmp_path):
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, step=1, grid_size=4, action_dim=2,
                     angle_unit="radians", angle_convention="signed", model_config=MODEL_CONFIG)

    expected = ExpectedSchema(grid_size=4, action_dim=2, angle_unit="radians", angle_convention="signed")
    loaded_model, checkpoint = load_checkpoint(path, expected)
    assert checkpoint["grid_size"] == 4
    assert loaded_model.grid_size == 4


@pytest.mark.parametrize("mismatch_field,expected_kwargs", [
    ("grid_size", dict(grid_size=8, action_dim=2, angle_unit="radians", angle_convention="signed")),
    ("action_dim", dict(grid_size=4, action_dim=14, angle_unit="radians", angle_convention="signed")),
    ("angle_unit", dict(grid_size=4, action_dim=2, angle_unit="degrees", angle_convention="signed")),
    ("angle_convention", dict(grid_size=4, action_dim=2, angle_unit="radians", angle_convention="unsigned")),
])
def test_rejects_incompatible_schema(tmp_path, mismatch_field, expected_kwargs):
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, step=1, grid_size=4, action_dim=2,
                     angle_unit="radians", angle_convention="signed", model_config=MODEL_CONFIG)

    expected = ExpectedSchema(**expected_kwargs)
    with pytest.raises(IncompatibleCheckpointError, match=mismatch_field):
        load_checkpoint(path, expected)


def test_rejects_wrong_normalizer_version(tmp_path):
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, step=1, grid_size=4, action_dim=2,
                     angle_unit="radians", angle_convention="signed", model_config=MODEL_CONFIG,
                     normalizer_version=2)

    expected = ExpectedSchema(grid_size=4, action_dim=2, angle_unit="radians", angle_convention="signed",
                               normalizer_version=1)
    with pytest.raises(IncompatibleCheckpointError, match="normalizer_version"):
        load_checkpoint(path, expected)


def test_rejects_checkpoint_missing_required_fields(tmp_path):
    import torch
    path = tmp_path / "bare.pt"
    torch.save({"model_state_dict": {}}, path)

    expected = ExpectedSchema(grid_size=4, action_dim=2, angle_unit="radians", angle_convention="signed")
    with pytest.raises(IncompatibleCheckpointError, match="missing required fields"):
        load_checkpoint(path, expected)
