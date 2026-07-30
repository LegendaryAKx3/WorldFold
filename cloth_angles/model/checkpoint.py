"""Checkpoint save/load with schema validation.

Every checkpoint stores the grid resolution, action dimension, angle unit,
angle convention, model config, and normalizer version (per spec). Loading
against an incompatible schema -- e.g. a checkpoint trained on a 16x16 grid
being loaded for an 8x8 evaluation -- must raise, not silently produce a
shape-mismatched or semantically wrong model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cloth_angles.model.world_model import WorldModel

CURRENT_NORMALIZER_VERSION = 1


class IncompatibleCheckpointError(ValueError):
    pass


@dataclass
class ExpectedSchema:
    grid_size: int
    action_dim: int
    angle_unit: str
    angle_convention: str
    normalizer_version: int = CURRENT_NORMALIZER_VERSION


def save_checkpoint(path, model: WorldModel, step: int, grid_size: int, action_dim: int,
                     angle_unit: str, angle_convention: str, model_config: dict,
                     normalizer_version: int = CURRENT_NORMALIZER_VERSION) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": step,
        "grid_size": grid_size,
        "action_dim": action_dim,
        "angle_unit": angle_unit,
        "angle_convention": angle_convention,
        "model_config": model_config,
        "normalizer_version": normalizer_version,
    }, path)


def _check_field(checkpoint: dict, field: str, expected) -> str | None:
    actual = checkpoint.get(field)
    if actual != expected:
        return f"{field}: checkpoint has {actual!r}, expected {expected!r}"
    return None


def load_checkpoint(path, expected: ExpectedSchema, device: torch.device | None = None) -> tuple[WorldModel, dict]:
    """Loads a checkpoint and validates it against `expected`. Raises
    IncompatibleCheckpointError (rather than constructing a mismatched model)
    if the grid resolution, action dimension, angle unit/convention, or
    normalizer version don't match.
    """
    checkpoint = torch.load(path, map_location=device or "cpu")

    required_fields = ("grid_size", "action_dim", "angle_unit", "angle_convention",
                        "model_config", "normalizer_version")
    missing = [f for f in required_fields if f not in checkpoint]
    if missing:
        raise IncompatibleCheckpointError(f"checkpoint {path} is missing required fields: {missing}")

    errors = [
        _check_field(checkpoint, "grid_size", expected.grid_size),
        _check_field(checkpoint, "action_dim", expected.action_dim),
        _check_field(checkpoint, "angle_unit", expected.angle_unit),
        _check_field(checkpoint, "angle_convention", expected.angle_convention),
        _check_field(checkpoint, "normalizer_version", expected.normalizer_version),
    ]
    errors = [e for e in errors if e is not None]
    if errors:
        raise IncompatibleCheckpointError(
            f"checkpoint {path} is incompatible with the expected schema:\n" + "\n".join(errors)
        )

    model_cfg = checkpoint["model_config"]
    model = WorldModel(
        grid_size=checkpoint["grid_size"],
        action_dim=checkpoint["action_dim"],
        angle_convention=checkpoint["angle_convention"],
        encoder_hidden=tuple(model_cfg["encoder_hidden"]),
        h_dim=model_cfg["h_dim"],
        n_categoricals=model_cfg["n_categoricals"],
        n_classes=model_cfg["n_classes"],
        mlp_hidden=model_cfg["mlp_hidden"],
        kl_free_bits=model_cfg["kl_free_bits"],
        kl_weight=model_cfg["kl_weight"],
        huber_delta=model_cfg["huber_delta"],
    )
    if device is not None:
        model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
